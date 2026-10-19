"""Observe several opponents before committing troops; no long speculative gather.

Coordinates in a batch are always projected from one fixed screenshot. Every
troop source is reopened and measured before its capacity enters a plan.
"""
from __future__ import annotations
import json
import math
import time
from PIL import Image
from .planner import required_attack,plan_muster,MusterDonor
from .strategy import city_key
from .vision import pretty


class BatchMixin:
    def ready_enemy_identity(self,identity,expected_city,expected_owner):
        def matches(value):
            return bool(value and ':' in value and
                city_key(value.split(':',1)[0])==city_key(expected_owner) and
                city_key(value.split(':',1)[1])==city_key(expected_city))
        for _ in range(2):
            if matches(identity):return identity
            self.wait(.4);self.capture()
            button=self.vision.match(self.image,'scout',.84)
            if not button:return None
            identity=self.vision.enemy_identity(self.image,button,expected_city,expected_owner)
        return identity

    def ready_scout_report(self,values):
        # Report totals animate after the title/city banner. Re-read this same
        # report before sending another scout or abandoning a valid target.
        def complete(v):
            return bool(v and v.get('defense') is not None and v.get('city','').strip() and v.get('owner','').strip())
        for _ in range(2):
            if complete(values):return values
            self.wait(.5);self.capture();values=self.vision.scout_report(self.image)
        if not complete(values) and getattr(self,'data',None):
            self.image.save(self.data/'unreadable-batch-report.png')
        return values

    def enemy_interaction(self,point):
        adjusted=self.vision.enemy_click_point(self.image,point)
        return adjusted if isinstance(adjusted,(tuple,list)) and len(adjusted)==2 else point

    def settled_map(self):
        self.map();self.wait(.35);self.capture()

    def fixed_point(self,reference,point):
        matrix=self.align_view(reference)
        return self.vision.transform_point(matrix,point) if matrix is not None else None

    def verify_sources(self,limit=3):
        self.settled_map();reference=self.image.copy()
        estimates=sorted(self.vision.own_sources(self.image,self.state.get('owner')),key=lambda p:-p[1])
        unknown=[]
        for cx,cy,scale,_ in self.vision.own_flags(self.image):
            point=(cx+round((640-cx)*.07),cy+round(95*scale))
            if self.world_position(point) and all(math.dist(point,p)>35 for p,_ in estimates):unknown.append((point,0))
        # Missing map OCR is not evidence that the city has no army. Reserve a
        # measurement slot for a real blue flag with an unreadable number.
        selected=(estimates[:max(0,limit-1)]+unknown[:1] if unknown and limit<=3 else (estimates+unknown)[:limit])
        verified=[];names=set()
        for position,estimate in selected:
            point=self.fixed_point(reference,position)
            if not point or not self.world_position(point):
                self.log(f'출발 성 좌표 재확인 실패 · {position} → {point}');continue
            city=self.select_owned(point)
            if not city:
                self.log(f"출발 성 병력/이름 인식 실패 · {point} / {getattr(self.vision,'last_owned_read',None)}")
                continue
            selected_view=self.image.copy()
            selected_view.save(self.data/'batch-source-last.png')
            self.settled_map()
            matrix=self.align_view(selected_view)
            if matrix is None:
                self.log(f"출발 성 선택 후 지도 대조 실패 · {city['name']}")
                selected_view.save(self.data/'source-measure-selected.png');self.image.save(self.data/'source-measure-current.png')
                continue
            actual=self.vision.transform_point(matrix,city['position'])
            key=city_key(city['name'])
            if key in names:continue
            if estimate>max(city['troops']*2,1e6):
                self.log(f"지도 병력 추정 폐기 · {city['name']} 추정 {pretty(estimate)} → 실제 {pretty(city['troops'])}")
            names.add(key)
            # Return in the fixed reference's coordinates, not coordinates from
            # different city-selection frames mixed together.
            current_to_fixed=self.vision.muster_transform(self.image,reference)
            if current_to_fixed is None:
                self.log(f"출발 성 병력은 확인했으나 원래 정찰 지도와 대조 실패 · {city['name']} {pretty(city['troops'])}")
                reference.save(self.data/'source-measure-reference.png');self.image.save(self.data/'source-measure-current.png')
                continue
            verified.append({'name':city['name'],'troops':city['troops'],
                'position':self.vision.transform_point(current_to_fixed,actual)})
            self.log(f"출발 성 실측 · {city['name']} {pretty(city['troops'])}")
        return reference,verified

    def batch_has_capacity(self,reports,sources):
        visible=[s for s in sources if self.world_position(s['position'])]
        if not visible:return False
        origin=max(visible,key=lambda s:s['troops'])
        fraction=math.floor((self.config['max_send_fraction']+1e-9)*10)/10
        donors=[MusterDonor(tuple(s['position']),int(s['troops']),math.dist(s['position'],origin['position']))
                for s in visible if s['name']!=origin['name']]
        for report in reports:
            if not self.world_position(report['point']):continue
            needed=required_attack(report['defense'],self.state['attack_bonus'],self.config['combat_margin'])
            if needed<=origin['troops']*fraction:return True
            if plan_muster(needed,origin['troops'],donors,self.config['max_send_fraction'],'fewest',quantum=.1):return True
        return False

    def extend_batch_sources(self,reports,sources):
        """At most two adjacent views, retaining strict observed coordinates.

        A short frontline at the edge must not repeatedly scout the same towns
        while a donor is just outside the viewport. No new scouts or transfers
        are sent here; each added source is opened and measured normally.
        """
        if self.batch_has_capacity(reports,sources):return reports,sources
        for dx,dy in ((240,0),(0,-120)):
            if self.incoming():return None
            self.decide('인접 지원 병력 확인','현재 병력 부족 · 정찰 재전송 없이 최대 두 인접 화면에서 실제 병력 실측')
            pieces=max(math.ceil(abs(dx)/110),math.ceil(abs(dy)/70))
            for _ in range(pieces):
                self.settled_map();reference=self.image.copy()
                self.driver.pan_map(dx/pieces,dy/pieces);self.wait(.5);self.capture()
                matrix=self.align_view(reference)
                if matrix is None:return None
                reports=[{**r,'point':self.vision.transform_point(matrix,r['point'])} for r in reports]
                sources=[{**s,'position':self.vision.transform_point(matrix,s['position'])} for s in sources]
            reference=self.image.copy()
            source_reference,added=self.verify_sources(limit=4)
            self.settled_map()
            matrix=self.align_view(reference);sm=self.align_view(source_reference)
            if matrix is None or sm is None:return None
            reports=[{**r,'point':self.vision.transform_point(matrix,r['point'])} for r in reports]
            current={city_key(s['name']):{**s,'position':self.vision.transform_point(matrix,s['position'])} for s in sources}
            for item in added:
                fresh={**item,'position':self.vision.transform_point(sm,item['position'])}
                previous=next((key for key,s in current.items() if key==city_key(fresh['name']) or
                    math.dist(s['position'],fresh['position'])<50),None)
                old=current.pop(previous,{}) if previous is not None else {}
                current[city_key(fresh['name'])]={**old,**fresh}
            # Perspective can separate projections of the same city by >50px.
            # Two projections resolving to one observed blue flag are one army.
            sources=[];flag_indices={}
            for source in current.values():
                flag=self.vision.own_flag_for_castle(self.image,source['position'])
                key=tuple(flag) if isinstance(flag,(tuple,list)) and len(flag)==2 else None
                if key is not None and key in flag_indices:
                    i=flag_indices[key];sources[i]={**sources[i],**source}
                else:
                    if key is not None:flag_indices[key]=len(sources)
                    sources.append(source)
            if self.batch_has_capacity(reports,sources):break
        return reports,sources

    def start_batch_muster(self,report,origin,sources,needed,before):
        if getattr(self,'config',{}).get('focus_defense'):return False
        # Stage controls must be clear of the top HUD before reading the city.
        # Project every donor from the same pre-pan frame, rather than dropping
        # an offscreen donor or mixing coordinates from different camera views.
        pair=self.centre_combat_pair(report['point'],origin['position'])
        if not pair:return False
        projected=[dict(s) for s in sources]
        for matrix in self.combat_pan_matrices:
            projected=[{**s,'position':self.vision.transform_point(matrix,s['position'])} for s in projected]
        self.last_scout_city=report['name'];self.last_scout_owner=report['owner']
        self.expected_origin_name=origin['name'];self.prevalidated_bulk=True
        self.verified_source_names={tuple(s['position']):s['name'] for s in projected}
        return self.begin_muster(pair[0],pair[1],needed,
            [(s['position'],s['troops']) for s in projected],before)

    def inspect_batch_targets(self,reference,points):
        targets=[]
        self.decide('정찰 후보 신원 확인',f'주변 후보 {min(5,len(points))}곳 · 실드와 제외 유저 확인')
        for original in points[:5]:
            self.settled_map();point=self.fixed_point(reference,original)
            if not point or not self.world_position(point) or self.skip_shielded(point):continue
            if self.targets.excluded(self.image,point,self.vision):continue
            if getattr(self,'exclusions',None):
                map_owner=self.vision.map_enemy_owner(self.image,point)
                if isinstance(map_owner,str) and map_owner and self.exclusions.blocked(map_owner):
                    self.targets.add(reference,original,map_owner+':',seconds=0)
                    self.log(f'지도 닉네임으로 제외 유저 확인 · 선택/정찰 생략: {map_owner}')
                    continue
            self.tap(*self.enemy_interaction(point),delay=.6)
            scout=self.vision.match(self.image,'scout',.84)
            if not scout:continue
            identity=self.vision.enemy_identity(self.image,scout)
            if not identity or self.offensive_excluded(identity):continue
            owner,name=identity.split(':',1)
            self.targets.add(reference,original,identity,seconds=0)
            targets.append({'point':original,'owner':owner,'name':name})
        self.settled_map()
        return targets

    def native_multi_scout(self,reference,targets):
        if getattr(self,'config',{}).get('focus_defense'):return False
        if len(targets)<2:return False
        self.settled_map()
        # Do not issue an area command when a known excluded player is visible
        # anywhere in this view. Targeted parallel scouts handle that case.
        for entry,point in self.targets.projected(self.image,self.vision):
            if self.targets._excluded_entry(entry) and self.world_position(point):
                self.log('다중 정찰 범위에 제외 유저 기록 · 대상 지정 병렬 정찰 사용');return False
        all_points=self.vision.enemy_candidates(self.image)
        if len(all_points)>len(targets):
            self.log(f'다중 정찰 사전 신원 확인 {len(targets)}/{len(all_points)} · 확인한 대상만 병렬 정찰');return False
        sources=self.vision.own_sources(self.image,self.state.get('owner'))
        if not sources:return False
        matrix=self.align_view(reference)
        if matrix is None:return False
        projected=[self.vision.transform_point(matrix,t['point']) for t in targets]
        centre=tuple(sum(p[i] for p in projected)/len(projected) for i in (0,1))
        source=min(sources,key=lambda p:math.dist(p[0],centre))[0]
        city=self.select_owned(source)
        if not city:self.log('다중 정찰 출발 도시 확인 실패');return False
        button=self.vision.match(self.image,'multi_scout',.84)
        if not button:self.log('다중 정찰 아이콘 재인식 필요');self.settled_map();return False
        self.tap(*button[:2],delay=.7)
        mode=self.region({'prompt':(440,5,840,76)})['prompt']
        if '다중' not in mode or '정찰' not in mode:
            self.log(f'다중 정찰 모드 제목 확인 실패 · {mode!r}');self.back();self.settled_map();return False
        confirm=self.vision.match(self.image,'operation_confirm',.87)
        if not confirm:self.log('다중 정찰 확인 버튼 재인식 필요');self.back();self.settled_map();return False
        # This is the game's multi-scout confirmation, never regrouping. The
        # regroup command sends 100% and is deliberately not used here.
        self.save_batch_intent(targets,'native_multi_scout')
        self.tap(*confirm[:2],delay=.8)
        self.log(f'게임 다중 정찰 전송 · 확인한 후보 {len(targets)}곳 · 정찰 완료 한 번만 대기')
        self.wait(12);self.settled_map()
        return True

    def save_batch_intent(self,targets,mode):
        record={'ts':time.time(),'mode':mode,'status':'intent','targets':targets}
        path=self.data/'scout-batch-intent.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2),'utf-8');tmp.replace(path)

    def targeted_parallel_scout(self,reference,targets):
        if getattr(self,'config',{}).get('focus_defense'):return []
        self.save_batch_intent(targets,'targeted_parallel')
        submitted=[]
        for target in targets:
            if getattr(self,'config',{}).get('focus_defense'):break
            self.settled_map();point=self.fixed_point(reference,target['point'])
            if not point or not self.world_position(point):
                self.log(f"병렬 정찰 좌표 재확인 필요 · {target['name']} / {point}");continue
            if self.skip_shielded(point):continue
            self.tap(*self.enemy_interaction(point),delay=.6)
            scout=self.vision.match(self.image,'scout',.84)
            identity=self.vision.enemy_identity(self.image,scout,target['name']) if scout else None
            if not identity:
                self.image.save(self.data/'batch-scout-unreadable.png')
                self.log(f"병렬 정찰 대상 재인식 필요 · {target['name']} / 버튼 {bool(scout)}");continue
            if self.offensive_excluded(identity):continue
            self.tap(*scout[:2],delay=.4);submitted.append(target)
            self.counts['scouts']+=1
        if submitted:
            self.log(f'대상 지정 병렬 정찰 {len(submitted)}곳 전송 · 중간 완료 대기 없이 한꺼번에 확인')
            self.wait(12)
        self.settled_map()
        return submitted

    def read_batch_reports(self,reference,targets):
        reports=[]
        self.decide('다중 정찰 결과 비교',f'{len(targets)}곳의 실제 방어력과 소유자 확인')
        for target in targets:
            self.settled_map();point=self.fixed_point(reference,target['point'])
            if not point or not self.world_position(point) or self.skip_shielded(point):continue
            self.tap(*self.enemy_interaction(point),delay=.6)
            icon=self.vision.match(self.image,'report',.84)
            if not icon:continue
            self.tap(*icon[:2])
            values=self.vision.scout_report(self.image)
            if values:
                owner,name=city_key(values['owner']),city_key(values['city'])
                if owner and name:
                    self.targets.add(reference,target['point'],owner+':'+name,seconds=0)
                    if self.offensive_excluded(owner+':'+name):continue
            first=values;values=self.ready_scout_report(values)
            if values is not first and values and values.get('owner') and values.get('city'):
                owner,name=city_key(values['owner']),city_key(values['city'])
                self.targets.add(reference,target['point'],owner+':'+name,seconds=0)
                if self.offensive_excluded(owner+':'+name):continue
            if not values or values['defense'] is None or not values.get('city','').strip() or not values.get('owner','').strip():
                self.log(f"정찰 보고서 재확인 필요 · {target['name']} / {values}");continue
            owner,name=city_key(values['owner']),city_key(values['city'])
            if name!=target['name']:
                self.log(f"지도/보고서 이름 재대조 · {target['name']} → 보고서 {name}")
                self.settled_map();point=self.fixed_point(reference,target['point'])
                if not point:continue
                self.tap(*self.enemy_interaction(point),delay=.6)
                scout=self.vision.match(self.image,'scout',.84)
                verified=self.vision.enemy_identity(self.image,scout,name) if scout else None
                if not verified or verified.rsplit(':',1)[-1]!=name:continue
            reports.append({**target,'name':name,'owner':owner,'defense':values['defense'],
                'observed':time.time(),'region':self.state.get('exploration_region')})
            self.log(f"다중 정찰 비교 · {name} 방어력 {pretty(values['defense'])}")
        self.settled_map()
        reference.save(self.data/'batch-report-view.png')
        (self.data/'batch-reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),'utf-8')
        return reports

    def centre_combat_pair(self,target,origin):
        self.combat_pan_matrices=[]
        for _ in range(6):
            if (220<target[0]<1060 and 170<target[1]<550 and
                    220<origin[0]<1060 and 300<origin[1]<470):return target,origin
            dx=max(-110,min(110,640-(target[0]+origin[0])/2))
            dy=max(-70,min(70,350-origin[1]))
            reference=self.image.copy();self.driver.pan_map(dx,dy);self.wait(.6);self.capture()
            matrix=self.align_view(reference)
            if matrix is None:
                self.log('공격 기점 화면 조정 중 지형 대조 실패 · 출병/이동 없이 다음 후보 확인')
                return None
            self.combat_pan_matrices.append(matrix)
            target,origin=(self.vision.transform_point(matrix,p) for p in (target,origin))
        return None

    def attack_screen_batch(self):
        if getattr(self,'config',{}).get('focus_defense'):return 'defense'
        self.settled_map();reference=self.image.copy()
        # Wide rays intentionally continue beyond our border. Without a visible
        # owned source there is no local attack route: do not waste minutes
        # scouting enemies before discovering that no troops can depart here.
        if (getattr(self,'wide_exploration_enabled',lambda:False)() and
                not self.vision.own_flags(self.image)):
            return 'no_sources'
        points=self.vision.enemy_candidates(self.image)
        if not points:return 'no_targets'
        # A recent report may be used for an attack even though re-scouting the
        # same city is on cooldown. Previously that cooldown blocked both.
        reports=[];report_reference=None
        path=self.data/'batch-reports.json'
        if path.exists() and (self.data/'batch-report-view.png').exists():
            cached=json.loads(path.read_text('utf-8'))
            with Image.open(self.data/'batch-report-view.png') as source:old=source.convert('RGB')
            matrix=self.vision.muster_transform(old,self.image)
            if matrix is not None:
                reports=[{**r,'point':self.vision.transform_point(matrix,r['point'])} for r in cached
                    if time.time()-r['observed']<120 and not r.get('used')
                    and r.get('region')==self.state.get('exploration_region')
                    and self.world_position(self.vision.transform_point(matrix,r['point']))
                    and not self.offensive_excluded(r['owner']+':'+r['name'])]
                report_reference=reference
        if not reports:
            candidates=[p for p in points if not self.targets.blocked(self.image,p,self.vision)]
            if not candidates:return 'cooldown'
            targets=self.inspect_batch_targets(reference,candidates)
            if not targets:return 'no_targets'
            if self.native_multi_scout(reference,targets):
                self.counts['scouts']+=len(targets)
            else:targets=self.targeted_parallel_scout(reference,targets)
            reports=self.read_batch_reports(reference,targets)
            report_reference=reference
        if not reports:return 'report_unreadable'
        self.decide('공격·집결 병력 실측','지도 추정값 대신 출발 성을 열어 실제 병력 확인')
        source_reference,sources=self.verify_sources(limit=3)
        if not sources:return 'no_sources'
        self.settled_map()
        matrix=self.align_view(source_reference);target_matrix=self.align_view(report_reference)
        if matrix is None or target_matrix is None:return 'unknown_view'
        sources=[{**s,'position':self.vision.transform_point(matrix,s['position'])} for s in sources]
        reports=[{**r,'point':self.vision.transform_point(target_matrix,r['point'])} for r in reports]
        expanded=self.extend_batch_sources(reports,sources)
        if expanded is None:return 'unknown_view'
        reports,sources=expanded
        sources=[s for s in sources if self.world_position(s['position'])]
        reports=[r for r in reports if self.world_position(r['point'])]
        if not sources:return 'no_sources'
        if not reports:return 'no_targets'
        # A long scout/source pass can cross the refresh interval. Update the
        # bonuses before choosing troop quantities, reprojecting all coordinates
        # after the profile round-trip rather than using stale screen positions.
        if time.time()-getattr(self,'last_stats_update',time.time())>=60:
            reference=self.image.copy()
            self.sync_stats();self.settled_map()
            matrix=self.align_view(reference)
            if matrix is None:return 'unknown_view'
            sources=[{**s,'position':self.vision.transform_point(matrix,s['position'])} for s in sources]
            reports=[{**r,'point':self.vision.transform_point(matrix,r['point'])} for r in reports]
        deep=getattr(self,'reinforcement_policy',lambda:None)()
        self.rl_origin_choice=None
        origin=max(sources,key=lambda s:s['troops'])
        if deep:
            origins=[]
            for s in sources:
                cap=s['troops']*(math.floor((self.config['max_send_fraction']+1e-9)*10)/10)
                donors=[MusterDonor(tuple(d['position']),int(d['troops']),math.dist(d['position'],s['position'])) for d in sources if d['name']!=s['name']]
                feasible=any(required_attack(r['defense'],self.state['attack_bonus'],self.config['combat_margin'])<=cap or
                    plan_muster(required_attack(r['defense'],self.state['attack_bonus'],self.config['combat_margin']),s['troops'],donors,self.config['max_send_fraction'],'fewest',quantum=.1) is not None for r in reports)
                if feasible:origins.append(s)
            if not origins:return 'blocked'
            choices=[{'domain':'attack_origin','context':'verified_sources','action':'source:'+str(i),
                      'features':{'troops':s['troops'],'distance':min(math.dist(s['position'],r['point']) for r in reports)}} for i,s in enumerate(origins)]
            selected,_=deep.order(choices,[s['troops']/max(origin['troops'],1) for s in origins])[0]
            self.rl_origin_choice=selected;origin=origins[int(selected['action'].split(':')[1])]
            self.rl_origin_choice['origin_name']=origin['name']
        allowed=origin['troops']*(math.floor((self.config['max_send_fraction']+1e-9)*10)/10)
        for r in reports:r['needed']=required_attack(r['defense'],self.state['attack_bonus'],self.config['combat_margin'])
        if deep:
            donors=[MusterDonor(tuple(s['position']),int(s['troops']),math.dist(s['position'],origin['position'])) for s in sources if s['name']!=origin['name']]
            reports=[r for r in reports if r['needed']<=allowed or plan_muster(r['needed'],origin['troops'],donors,self.config['max_send_fraction'],'fewest',quantum=.1) is not None]
            if not reports:return 'blocked'
        self.rl_target_reports=reports
        ranked=self.rank_attack_targets([r['point'] for r in reports],origin['position'],origin['troops'])
        valid_points={tuple(p) for p in ranked}
        deep=getattr(self,'reinforcement_policy',lambda:None)()
        order={tuple(p):i for i,p in enumerate(ranked)}
        feasible=sorted((r for r in reports if r['needed']<=allowed and tuple(r['point']) in valid_points),
            key=(lambda r:-order[tuple(r['point'])]) if deep else
                (lambda r:self.attack_rank_choices[tuple(r['point'])][1]['score']-.7*r['needed']/max(allowed,1)),reverse=True)
        if deep and ranked and not any(tuple(r['point'])==tuple(ranked[0]) for r in feasible):
            # A learned preference for a feasible gathered attack must actually
            # reach the gathering executor, rather than always losing to the
            # old unconditional direct-attack-first branch.
            feasible=[]
        if feasible:
            chosen=feasible[0]
            self.last_scout_city=chosen['name'];self.last_scout_owner=chosen['owner']
            self.attack_choices=[self.attack_rank_choices[tuple(chosen['point'])][0]]
            needed,self.attack_margin=self.choose_attack_requirement(chosen['defense'],origin['troops'],allowed)
            (self.data/'batch-reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),'utf-8')
            self.image.save(self.data/'batch-report-view.png')
            pair=self.centre_combat_pair(chosen['point'],origin['position'])
            if pair is None:return 'unknown_view'
            target,position=pair;self.expected_origin_name=origin['name']
            self.hud();before=self.state.copy()
            self.decide('공격 가능한 목표 선택',f"{chosen['name']} 필요 {pretty(needed)} / {origin['name']} 동원 가능 {pretty(allowed)} · 강한 후보보다 출병 가능한 성 우선")
            sent=self.dispatch_attack(target,position,origin['troops'],needed,allowed,before)
            if sent:
                for r in reports:r['used']=r['name']==chosen['name']
                # All reports and coordinates use the current frame on save.
                # The dispatched target is also durably tracked by its episode.
                (self.data/'batch-reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),'utf-8')
            return 'dispatched' if sent else 'blocked'
        # No speculative 24-pan search: prove whole-plan capacity before moving.
        for chosen in sorted(reports,key=(lambda r:order.get(tuple(r['point']),999)) if deep else (lambda r:r['needed'])):
            donors=[MusterDonor(tuple(s['position']),int(s['troops']),math.dist(s['position'],origin['position']))
                    for s in sources if s['name']!=origin['name']]
            plan=plan_muster(chosen['needed'],origin['troops'],donors,self.config['max_send_fraction'],'fewest',quantum=.1)
            if plan is None:continue
            if tuple(chosen['point']) not in valid_points:continue
            self.attack_choices=[self.attack_rank_choices[tuple(chosen['point'])][0]]
            self.attack_margin=self.config['combat_margin']
            self.hud()
            if self.start_batch_muster(chosen,origin,sources,chosen['needed'],self.state.copy()):return 'muster'
        for r in reports:
            self.targets.add(self.image,r['point'],r['owner']+':'+r['name'],seconds=180)
        self.log('정찰 후보 전부 비교 · 실제 병력으로 출병/완전 집결 가능한 목표 없음 → 다른 구역으로 이동')
        return 'blocked'

    def goto_combat_bank(self):
        """Select a measured large garrison from the troop-sorted owned-city list."""
        self.settled_map();self.tap(1047,665,delay=.6)
        if not self.vision.match(self.image,'city_list_title',.9,area=(200,105,360,165)):
            self.settled_map();return False
        self.tap(959,122,delay=.4)
        pixel=self.image.getpixel((508,193))
        if pixel[2]<150 or pixel[1]<100:self.tap(550,201,delay=.5)
        fields={}
        for i,y in enumerate((264,328,392,456,520)):
            fields['army'+str(i)]=(489,y-18,566,y+18)
            fields['city'+str(i)]=(638,y-21,834,y+20)
        values=self.region(fields)
        from .vision import amount
        visits=getattr(self,'bank_visits',{})
        candidates=[]
        for i,y in enumerate((264,328,392,456,520)):
            count=amount(values['army'+str(i)]);name=values['city'+str(i)].strip()
            pin=(self.vision.match(self.image,'city_list_pin',.88,area=(235,y-25,296,y+25)) or
                 self.vision.match(self.image,'city_list_home',.9,area=(235,y-25,296,y+25)))
            if count and name and pin:candidates.append((count,name,pin))
        candidates.sort(reverse=True,key=lambda c:c[0])
        # Visit every bank before revisiting the first one. Otherwise two
        # 90-second scout cycles made the 180-second cooldown loop forever.
        eligible=[c for c in candidates if time.time()-visits.get(city_key(c[1]),0)>180]
        choice=min(eligible,key=lambda c:(visits.get(city_key(c[1]),0),-c[0]),default=None)
        if not choice:self.settled_map();return False
        army,name,pin=choice;visits[city_key(name)]=time.time();self.bank_visits=visits
        self.decide('공격 병력 보유 도시로 이동',f'{name} 실제 도시 목록 병력 {pretty(army)} · 작은 기점에서 무리한 집결 탐색 종료')
        self.tap(*pin[:2],delay=.9);self.settled_map()
        self.exploration_step=0;self.missing_sources=0
        self.state['exploration_region']=name
        self.counts['regions']=self.counts.get('regions',0)+1
        return True
