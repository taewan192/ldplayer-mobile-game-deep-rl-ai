"""Choose a productive next step when an individual action is unavailable."""
import time
import re
import math
from .exploration import ExplorationMemory
from .exploration import SWEEP_RAYS


class ActivityMixin:
    def wide_exploration_enabled(self):
        return (self.config.get('wide_exploration',True) and self.config.get('autonomous',True)
                and self.config.get('attack_enabled',True))

    def sweep_view_matches(self):
        path=self.data/'sweep-view.png'
        if not path.exists():return False
        from PIL import Image
        with Image.open(path) as source:reference=source.convert('RGB')
        if self.vision.same_viewport(self.image,reference):return True
        # The map's easing can settle a few pixels after the last capture.
        # This is route continuity only; enemy coordinates still use strict CV.
        matrix=self.vision.viewport_transform(reference,self.image)
        if matrix is None:
            changed,mean=self.vision.sweep_frame_delta(reference,self.image)
            return changed<.05 and mean<2
        return all(math.dist(self.vision.transform_point(matrix,p),p)<40
                   for p in ((300,200),(1000,500),(640,350)))

    def center_sweep_main(self):
        """Use the actual house pin in our city list, regardless of current sort/page."""
        self.map();self.tap(1047,665,delay=.4)
        if not self.vision.match(self.image,'city_list_title',.9,area=(200,105,360,165)):
            self.map();return False
        self.tap(959,122,delay=.25)
        page=self.city_list_page()
        if not page or page[0]>13:self.map();return False
        for _ in range(page[0]-1):self.tap(574,578,delay=.25)
        if self.city_list_page()!=(1,page[1]):self.map();return False
        home=self.vision.match(self.image,'city_list_home',.9,area=(235,235,296,545))
        if not home:self.map();return False
        name=self.region({'city_name':(638,home[1]-21,834,home[1]+20)})['city_name'].strip()
        if not name:self.map();return False
        self.tap(*home[:2],delay=.55);self.map()
        # Traveling to a verified own-city pin does not need an army reading.
        # Dispatch callers keep the default strict troop requirement.
        selected=self.select_owned((640,330),require_troops=False)
        from .strategy import city_key
        if not selected or city_key(selected['name'])!=city_key(name):
            selected=self.select_owned_named((640,330),name)
        self.map()
        if not selected or city_key(selected['name'])!=city_key(name):return False
        self.wait(.3);self.capture()
        self.exploration_memory().sweep_bind_main(name)
        self.state['exploration_main_city']=name
        self.log(f'광역 탐색 기준 메인 성 확인: {name}')
        return True

    def sweep_pan(self,drag):
        """One observed drag. Never count a failed input or a fixed map edge."""
        self.sweep_interrupted=False
        self.sweep_stationary=False
        self.sweep_pan_issued=False
        if self.pending or self.muster or self.defense_busy() or self.incoming():
            self.sweep_interrupted=True;return False
        reference=self.image.copy()
        self.driver.pan_map(*drag);self.sweep_pan_issued=True;self.wait(.25);self.capture()
        # Save the settled camera, rather than an intermediate easing frame
        # that would force a return to the main castle on every next step.
        for _ in range(2):
            previous=self.image.copy();self.wait(.25);self.capture()
            if self.vision.sweep_frame_delta(previous,self.image)[0]<.15:break
        for attempt in range(3):
            matrix=self.vision.viewport_transform(reference,self.image)
            if matrix is not None:
                point=self.vision.transform_point(matrix,(640,350))
                dx,dy=point[0]-640,point[1]-350
                if math.hypot(dx,dy)>12 and dx*drag[0]+dy*drag[1]>0:return True
            if matrix is None:
                changed,mean=self.vision.sweep_motion_delta(reference,self.image)
                if (changed>.45 and mean>18 and
                        self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)) and
                        not self.vision.match(self.image,'move_troops',.84) and
                        not self.vision.match(self.image,'move_troops_city88',.84)):return True
            if attempt<2:self.wait(.3);self.capture()
        # A long 3D-camera pan can break a single homography. Confirm a large
        # terrain change on the actual map, without using it as target coords.
        reference.save(self.data/'sweep-pan-before.png');self.image.save(self.data/'sweep-pan-after.png')
        changed,mean=self.vision.sweep_frame_delta(reference,self.image)
        self.sweep_stationary=changed<.1 and mean<8
        return False

    def sweep_travel(self,drag):
        """Count the requested camera gestures even over featureless sea.
        Keep visual verification separate, and never use this as combat geometry.
        A blocked input, a menu, or a defense interruption still stops travel.
        """
        verified=self.sweep_pan(drag)
        self.sweep_motion_verified=bool(verified)
        if verified:return True
        if not getattr(self,'sweep_pan_issued',False) or getattr(self,'sweep_interrupted',False):return False
        if (self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)) and
                not self.vision.match(self.image,'move_troops',.84) and
                not self.vision.match(self.image,'move_troops_city88',.84)):
            self.log('지도 이동 명령 완료 · 바다/경계에서 장면 변화 미확인 · 다음 순번 계속, 전투 좌표로 사용 안 함')
            return True
        return False

    def explore_wide(self,reason):
        memory=self.exploration_memory();route=memory.state['sweep']
        matches=self.sweep_view_matches()
        if route['needs_home'] or not matches:
            # The cursor survives attacks, defense, other camera moves and restart.
            memory.sweep_reanchor()
            if not self.center_sweep_main():
                self.exploration_pause_until=time.time()+15
                self.decide('메인 성 재확인','도시 목록의 메인 성·이름을 확인한 뒤 광역 탐색 재개')
                return False
            point=memory.sweep_next()
            for _ in range(point['step']-1):
                if not self.sweep_travel(point['drag']):return self.sweep_pan_failed('기존 탐색 위치 복귀 실패')
            route['needs_home']=False;memory.save()
        point=memory.sweep_next()
        self.decide('주변 탐색',f"메인 {route.get('main','')} 기준 {point['direction']} {point['step']}/9 · "
                    f"전체 {point['cursor']+1}/72 · {point['cycle']}회차 · {reason}")
        if not self.sweep_travel(point['drag']):return self.sweep_pan_failed('지도 이동/새 구역 확인 실패')
        visited=memory.sweep_commit(verified=self.sweep_motion_verified)
        self.image.save(self.data/'sweep-view.png')
        self.state['exploration_region']=f"{route['main']}/sweep:{visited['cycle']}:{visited['ray']}:{visited['step']}"
        self.state['wide_sweep']=visited
        self.counts['exploration_commands']=self.counts.get('exploration_commands',0)+1
        self.counts['explorations']+=int(visited['movement_verified'])
        label='이동 확인' if visited['movement_verified'] else '이동 명령 완료 (장면 변화 미확인)'
        self.log(f"광역 탐색 {label} · {visited['direction']} {visited['step']}/9 · 전체 {visited['cursor']+1}/72")
        return True

    def sweep_pan_failed(self,reason):
        memory=self.exploration_memory();route=memory.state['sweep']
        if getattr(self,'sweep_interrupted',False):memory.sweep_reanchor();return False
        route['failures']=route.get('failures',0)+1;memory.sweep_reanchor()
        if route['failures']>=3 and getattr(self,'sweep_stationary',False):
            self.log(f"{reason} 3회 · 지도 정지 확인, 현재 방향을 미완료로 기록하고 다음 방향으로 이동")
            memory.sweep_skip_ray(reason)
        else:
            self.log(f'{reason} · 이동 횟수에 넣지 않고 메인 성에서 위치 복구')
        self.exploration_pause_until=time.time()+(10 if route.get('failures',0)>=3 else 3)
        return False

    def locate_owned_city(self,expected,max_pages=12):
        """Confirm an uncertain conquest through our own city list and pin.
        A missing name is not a defeat; never inspect an enemy's city list."""
        from .strategy import city_key
        if not expected:return None
        self.map();self.tap(1047,665,delay=.5)
        if not self.vision.match(self.image,'city_list_title',.9,area=(200,105,360,165)):
            self.map();return None
        self.tap(959,122,delay=.3)
        page=self.city_list_page()
        if page is None:self.map();return None
        current,total=page
        pages=list(range(current,total+1))+list(range(1,current))
        for wanted in pages[:max_pages]:
            for _ in range(abs(wanted-current)):
                self.tap(705 if wanted>current else 574,578,delay=.3)
            current=wanted
            if self.city_list_page()!=(current,total):self.map();return None
            ys=(264,328,392,456,520)
            names=self.region({str(i):(638,y-21,834,y+20) for i,y in enumerate(ys)})
            for i,y in enumerate(ys):
                if city_key(names[str(i)])!=city_key(expected):continue
                pin=(self.vision.match(self.image,'city_list_pin',.88,area=(235,y-25,296,y+25)) or
                     self.vision.match(self.image,'city_list_home',.9,area=(235,y-25,296,y+25)))
                if not pin:continue
                self.tap(*pin[:2],delay=.7);self.map();self.capture()
                actual=self.select_owned((640,330))
                if not actual or city_key(actual['name'])!=city_key(expected):
                    actual=self.select_owned_named((640,330),expected)
                if actual and city_key(actual['name'])==city_key(expected):
                    self.log(f'{expected} 보유 도시 목록·이름·내 성 조작 버튼 대조로 실제 소유 확인')
                    return actual
                self.map();return None
        self.map();return None

    def city_list_page(self):
        values=self.region({'page':(616,565,664,595),'page_wide':(610,563,671,594)})
        for value in values.values():
            match=re.fullmatch(r'\s*(\d+)\s*/\s*(\d+)\s*',value)
            if match and 1<=int(match[1])<=int(match[2])<=40:return tuple(map(int,match.groups()))
        return None

    def exploration_memory(self):
        if not hasattr(self,'exploration'):
            self.exploration=ExplorationMemory(getattr(self,'data',None))
        return self.exploration

    def next_region(self):
        if self.pending or self.muster or self.defense_busy() or self.incoming():return False
        if self.wide_exploration_enabled():return self.explore_wide('다음 광역 탐색 구역으로 이동')
        memory=self.exploration_memory()
        self.map();self.capture()
        if not memory.recently_seen(self.image,self.vision):
            memory.remember(self.state.get('exploration_region','이전 구역'),self.image)
        for _ in range(3):
            self.map();self.tap(1047,665,delay=.6)
            if not self.vision.match(self.image,'city_list_title',.9,area=(200,105,360,165)):
                self.map();break
            self.tap(959,122,delay=.3)  # Own cities tab, never bookmarks or another player's pin.
            page=self.city_list_page()
            if page is None:
                self.map();break
            current,pages=page;wanted,row=memory.reserve_anchor(pages)
            if abs(wanted-current)>12:self.map();continue
            for _ in range(abs(wanted-current)):
                self.tap(705 if wanted>current else 574,578,delay=.45)
            self.wait(.25);self.capture()
            if self.city_list_page()!=(wanted,pages):self.map();continue
            y=(264,328,392,456,520)[row]
            pin=(self.vision.match(self.image,'city_list_pin',.88,area=(235,y-25,296,y+25)) or
                 self.vision.match(self.image,'city_list_home',.9,area=(235,y-25,296,y+25)))
            if not pin:self.map();continue
            name=self.region({'city_name':(638,y-21,834,y+20)})['city_name'].strip()
            if not name:self.map();continue
            self.decide('다른 지역 탐색',f'{name} 주변으로 이동 · 도시 목록 {wanted}/{pages}페이지')
            self.tap(*pin[:2],delay=.8);self.map();self.capture()
            # A list pin belongs to our own-cities tab; also verify own controls
            # on the destination before treating it as a staging area.
            sources=self.vision.own_sources(self.image,self.state.get('owner'))
            if not any(math.dist(pos,(640,330))<95 for pos,_ in sources):
                selected=self.select_owned((640,330))
                self.map()
                if not selected:continue
            if memory.recently_seen(self.image,self.vision):
                self.log(f'{name}: 최근 확인한 동일 화면이라 다음 보유 도시로 이동')
                continue
            memory.remember(name,self.image)
            self.state['exploration_region']=name
            self.counts['regions']=self.counts.get('regions',0)+1
            self.exploration_step=0;self.missing_sources=0
            self.log(f'새 탐색 거점 확인: {name} · 누적 지역 {self.counts["regions"]}곳')
            return True
        self.exploration_pause_until=time.time()+15
        self.decide('지역 이동 재확인','이동할 보유 도시를 확인하지 못해 15초 뒤 다음 순번 재시도')
        return False

    def cycle_delay(self):
        if (getattr(self,'muster',None) and self.muster.get('version',1)>=2 and
            self.muster['status'] in ('discovering','preparing')):return .4
        exploring=self.state.get('decision',{}).get('action')=='주변 탐색'
        factor=self.config.get('exploration_speed',4) if exploring else self.config.get('decision_speed',2.5)
        return max(.25,self.config.get('interval',12)/max(1,min(4,float(factor))))

    def return_to_territory(self):
        """Use the observed city-list home control; never infer home from swipe offsets."""
        if self.pending or self.muster or self.defense_busy() or self.incoming(): return False
        self.decide('내 영토 복귀','주변 탐색 범위를 벗어나 주요 도시로 지도 재정렬')
        self.map(); self.tap(1047,665,delay=.6)
        fields=self.region({'title':(208,108,500,160)})
        if ('도시목록' not in ''.join(fields['title'].split()) or
                not self.vision.match(self.image,'city_list_home',.9,area=(230,235,300,290))):
            self.map(); self.exploration_pause_until=time.time()+30
            self.emit({'alert':'주요 도시 이동 버튼을 확인하지 못해 추가 지도 이동을 30초 보류합니다.'})
            return False
        self.tap(265,264,delay=.8); self.map()
        sources=self.vision.own_sources(self.image,self.state.get('owner'))
        if not sources:
            self.exploration_pause_until=time.time()+30
            self.emit({'alert':'주요 도시 복귀 후 내 성 인식을 확인하지 못해 추가 탐색을 30초 보류합니다.'})
            return False
        self.exploration_step=0; self.missing_sources=0
        self.log('주요 도시 주변 복귀 확인 · 가까운 영역부터 다시 탐색')
        return True

    def decide(self,action,reason):
        decision={'action':action,'reason':reason,'ts':time.time()}
        changed=(self.state.get('decision',{}).get('action'),self.state.get('decision',{}).get('reason'))!=(action,reason)
        self.state['decision']=decision
        self.emit({'decision':f'{action} · {reason}'})
        if changed: self.log(f'판단: {action} · {reason}')

    def explore(self,reason):
        if (not self.config.get('autonomous',True) or not self.config.get('attack_enabled',True) or
            self.pending or self.muster or self.defense_busy()): return False
        self.map(); self.capture()
        if self.incoming(): return False
        if self.wide_exploration_enabled():return self.explore_wide(reason)
        # The global direction cursor survives every city return and restart.
        step=getattr(self,'exploration_step',0)
        if step>=4:
            return self.next_region()
        else:
            self.decide('주변 탐색',reason)
            self.driver.pan_map(*self.exploration_memory().next_direction())
            self.wait(.2); self.capture()
        self.exploration_step=step+1
        self.counts['explorations']+=1
        return True

    def progress_cycle(self):
        if self.config.get('focus_defense'):
            self.decide('집중 방어','새 공격·광역 탐색 중단 · 침입 감시');return
        if getattr(self,'reinforcement_policy',lambda:None)() and self.config.get('autonomous',True):
            return self.reinforcement_progress()
        if time.time()<getattr(self,'exploration_pause_until',0):
            self.decide('지역 이동 재확인','이전 화면 이동을 보류했으며 다음 탐색 거점을 곧 재확인')
            return
        if time.time()<getattr(self,'alignment_retry_at',0):
            self.decide('인식 복구 대기','연속 좌표 인식 실패로 잠시 쉬며 방어 감시만 유지')
            return
        wide=self.wide_exploration_enabled()
        startup_outcome=None
        if wide and getattr(self,'startup_combat_check',False):
            self.startup_combat_check=False
            if not (self.pending or self.muster or self.defense_busy()) and self.vision.own_flags(self.image):
                self.decide('현재 지도 공격 기회 확인','탐색 위치 복귀 전에 현재 보이는 성의 정찰·공격 가능 여부 우선 확인')
                startup_outcome=self.attack_nearby()
                if self.pending or self.muster:
                    self.decide('공격 진행 확인' if self.pending else '병력 집결','실제 정찰·병력으로 현재 지도에서 가능한 출병 우선 처리')
                    return
        if wide and not (self.pending or self.muster or self.defense_busy()) and not self.sweep_view_matches():
            self.explore('중단했던 광역 탐색 순서에서 계속 진행')
            return
        if wide and self.state.get('wide_sweep',{}).get('movement_verified') is False:
            self.explore('동일/미확인 장면에서 정찰을 반복하지 않고 다음 이동 순번 진행')
            return
        outcome=startup_outcome if startup_outcome is not None else self.attack_nearby()
        if self.pending or self.muster:
            self.decide('공격 진행 확인' if self.pending else '병력 집결', '도착 결과를 확인하면서 방어 감시')
            return
        if outcome=='defense' or self.defense_busy():
            self.decide('방어 우선','피격 성 지원/도착 확인이 끝난 뒤 다른 행동 재평가')
            return
        if outcome in ('camera_unstable','unknown_view'):
            self.alignment_failures=getattr(self,'alignment_failures',0)+1
            self.decide('좌표 재확인','지도 이동을 멈추고 화면이 안정된 뒤 다시 관측')
            self.wait(1.5); self.capture()
            if self.alignment_failures>=3:
                self.alignment_retry_at=time.time()+30
                self.alignment_failures=0
                self.emit({'alert':'좌표 인식이 연속 실패해 30초간 추가 탐색/정찰을 보류합니다.'})
            return
        self.alignment_failures=0
        if outcome=='no_sources':
            if wide:
                self.explore('이 구역에 내 성이 없어 다음 광역 탐색 위치로 이동')
                return
            self.missing_sources=getattr(self,'missing_sources',0)+1
            # Empty surroundings do not justify walking farther away or another
            # expensive investment scan over the same empty viewport.
            if self.missing_sources>=2:
                self.next_region()
                return
            self.explore('내 성이 보이지 않아 인접 구역 확인 후 다른 보유 도시로 이동')
            return
        self.missing_sources=0
        if not wide and getattr(self,'outcome_policy',None) and self.config.get('batch_scout',True) and outcome in ('blocked','no_targets','report_unreadable'):
            if self.goto_combat_bank():return
        upgraded=False
        if not wide or time.time()-getattr(self,'last_wide_upgrade_check',0)>=120:
            self.last_wide_upgrade_check=time.time()
            self.decide('핵심 성 투자 검토','가능한 공격이 없어 방어력·병력·투자 손실 조건 확인')
            try:upgraded=self.upgrade()
            except RuntimeError as exc:
                if not wide:raise
                # An optional investment inspection must not restart scouting
                # this same map on every recovery and starve the sweep cursor.
                self.log(f'강화 후보 인식 보류 · 현재 탐색 순서는 계속 진행: {exc}')
                if not self.recover_navigation(exc):raise
        if upgraded:
            self.decide('투자 결과 추적','실제 레벨 상승 확인, 이후 소유권·손실 평가 대기')
            return
        reasons={'no_sources':'출발 가능한 내 성을 찾기 위해 주변 확인',
                 'no_targets':'화면에 정찰 가능한 적 성이 없어 주변 확인',
                 'cooldown':'최근 정찰한 성을 반복하지 않고 다른 방향 확인',
                 'shielded':'원형 보호막 성을 클릭하지 않고 다른 구역의 공격 후보 탐색',
                 'blocked':'현재 후보는 출병/집결 조건을 충족하지 못해 다른 후보 탐색',
                 'report_unreadable':'보고서 숫자 인식 실패 성을 제외하고 다른 후보 탐색',
                 'report_pending':'정찰 보고서가 아직 없어 해당 성 재시도를 보류',
                 'report_screen':'보고서 화면 확인 실패 성을 제외하고 다른 후보 탐색',
                 'no_scout_button':'정찰할 수 없는 대상이라 다른 성 탐색'}
        if not self.explore(reasons.get(outcome,'현재 공격·강화 조건을 충족하는 다른 성 탐색')):
            self.decide('감시 대기','설정 또는 방어 조건으로 추가 행동 보류, 무료 수급·위협 감시 계속')
