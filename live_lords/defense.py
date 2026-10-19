"""Deadline-aware, journaled support. Unknown attack power never becomes zero."""
from __future__ import annotations

import json
import math
import time

from PIL import Image

from .planner import SupportRoute,defense_shortfall,support_plan
from .strategy import StrategyLedger,city_key
from .vision import pretty
from .outcomes import bucket,decision
from .driver import GameUnavailable,EmulatorBridgeUnavailable


class DefenseMixin:
    def init_defense(self):
        self.strategy=StrategyLedger(self.data/'strategy-journal.json',self.learner)
        path=self.data/'defense-journal.json'
        self.defenses=json.loads(path.read_text('utf-8')) if path.exists() else []
        # After a crash, inspect troop totals and wait. Never replay a send intent.
        for job in self.defenses:
            job['purpose']='defense'
            if job['status']=='preparing': job['status']='waiting' if job['moves'] else 'unverified'
        self.last_defense_poll=0
        self.defense_unknown_until=0
        self.active_defense=None
        self.threatened_names=set()
        self.defense_read_waiting=False
        self.defense_read_retry_at=0
        self.defense_read_failures=0
        self.defense_read_last_log=0
        self.defense_focus_failures=[]

    def save_defenses(self):
        path=self.data/'defense-journal.json'; temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.defenses,ensure_ascii=False,indent=2),'utf-8')
        temporary.replace(path)

    def rebase_defense(self):
        job=self.active_defense
        if not job: return
        path=self.data/job['view']
        with Image.open(path) as source: reference=source.convert('RGB')
        matrix=self.vision.viewport_transform(reference,self.image)
        if matrix is None: raise RuntimeError('방어 지도 좌표 재확인 실패. 같은 지원은 반복하지 않습니다.')
        for key in ('staging','target'):
            job[key]=self.vision.transform_point(matrix,job[key])
        for donor in job.get('donors',[]): donor['position']=self.vision.transform_point(matrix,donor['position'])
        self.image.save(path); self.save_defenses()

    def defense_position(self,job):
        with Image.open(self.data/job['view']) as source: reference=source.convert('RGB')
        matrix=self.vision.viewport_transform(reference,self.image)
        if matrix is None: return None
        position=self.vision.transform_point(matrix,job['staging'])
        return position if self.world_position(position) else None

    def read_owned_details(self,position,expected=None):
        selected=self.select_owned(position)
        if not selected: return None
        if expected and city_key(selected['name'])!=city_key(expected): self.map(); return None
        # The info node is centered below the move radial, as observed on owned cities.
        mx,my=selected['move_button']
        self.tap(mx-87,my+70)
        city=self.vision.own_city_details(self.image,expected or selected['name'])
        if not city: self.image.save(self.data/'unreadable-own-city-details.png')
        self.map()
        if city and (not expected or city_key(city['name'])==city_key(expected)):
            self.strategy.observe_city(city)
            return city
        return None

    def inspect_defense_rows(self,focus=None,reuse=False):
        """Poll the actual movement menu, including off-screen destinations via its pin."""
        retained=(reuse and self.vision.march_panel(self.image) and
                  self.vision.defense_tab_selected(self.image))
        if not retained:
            button=self.vision.march_button(self.image)
            if not button: return []
            self.tap(*button[:2],delay=.3)
        if not retained and not self.vision.march_panel(self.image):
            for _ in range(4):
                self.wait(.3);self.capture()
                if self.vision.march_panel(self.image):break
                # The last outgoing march can arrive between the screenshot
                # and the tap, removing this shortcut. This is not an enemy
                # attack and must not block conquest verification for 30s.
                if (self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)) and
                        not self.vision.march_button(self.image) and
                        not self.vision.incoming_button(self.image)):
                    return []
            if not self.vision.march_panel(self.image):
                raise RuntimeError('행군 진행 상황 창을 확인하지 못했습니다.')
        if not retained:self.tap(1036,57,delay=.3)
        if not self.vision.defense_tab_selected(self.image):
            for _ in range(3):
                self.wait(.3);self.capture()
                if not self.vision.march_panel(self.image):
                    raise RuntimeError('방어 탭 확인 중 화면 변경 · 지도 스크롤 없이 다시 관측합니다.')
                if self.vision.defense_tab_selected(self.image):break
        if not self.vision.defense_tab_selected(self.image):
            self.back(); return []  # Disabled tab: only our own outgoing marches exist.
        # The game remembers list scroll. Reset it before reading any wave.
        # If a later page failed, rescan from the top so earlier threatened
        # cities cannot accidentally become support donors on the retry.
        reset=self.vision.march_list_scrollable(self.image)
        for _ in range(3 if reset and not (retained and not getattr(self,'defense_read_page',0)) else 0):
            self.driver.scroll_marches(to_top=True); self.wait(.2); self.capture()
        result=[]
        for page in range(5):
            self.defense_read_page=page
            snapshot=time.time()
            rows=self.vision.defense_rows(self.image)
            if not rows:
                self.image.save(self.data/'unreadable-defense-list.png')
                raise RuntimeError('활성 방어 목록을 읽지 못해 공격/강화를 보류합니다.')
            for row in rows: row['impact']=snapshot+row['remaining']
            if focus and any(city_key(r['city'])==focus for r in rows): return rows
            new=[r for r in rows if not any(city_key(r['city'])==city_key(old['city']) and
                 r['attacker']==old['attacker'] and r['troops']==old['troops'] and
                 abs(r['impact']-old['impact'])<12 for old in result)]
            if not new:
                self.defense_list_complete=True
                return [] if focus else result
            result.extend(new)
            if len(rows)<6:
                self.defense_list_complete=True
                return [] if focus else result
            self.driver.scroll_marches(); self.wait(.3); self.capture()
        self.defense_list_complete=False
        self.log('방어 목록이 5화면을 초과했습니다. 지원 성의 피격 여부를 확정하지 못해 출병을 보류합니다.')
        return [] if focus else result

    def inspect_attack_power(self,row):
        # The troop column (including exact counts) is not a verified details
        # button. Never click it. Only an explicit total already on screen may
        # provide power; a hidden troop range always stays unknown.
        if row.get('troops_exact') is not True:return None
        return self.vision.labeled_attack_power(self.image,row['city'],row['attacker'])

    def focus_defense_city(self,row):
        """Use a freshly matched location pin, then wait for camera arrival."""
        if not (self.vision.march_panel(self.image) and self.vision.defense_tab_selected(self.image)):
            return False
        x,y=row['pin']
        # Match only the pin's narrow column, never the adjacent troop text.
        marker=self.vision.match(self.image,'march_pin',.85,
                                 area=(max(620,x-28),max(100,y-35),min(685,x+28),min(650,y+35)))
        if not marker or math.dist(marker[:2],(x,y))>8:return False
        self.log(f"방어 위치 이동: {row['city']} · 위치 핀 ({marker[0]}, {marker[1]}) · 병력 숫자는 클릭하지 않음")
        self.tap(*marker[:2],delay=1.2)
        previous=None
        for _ in range(5):
            if (self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)) and
                    not self.vision.march_panel(self.image)):
                if previous is not None and self.vision.same_viewport(self.image,previous):
                    return True
                previous=self.image.copy()
            self.wait(.4);self.capture()
        self.log(f"{row['city']} 위치 이동 완료 미확인 · 성 선택 없이 다음 관측까지 대기")
        return False

    def defense_focus_retry(self,group,impact,failed=False):
        """Do not reopen the same failed city on every defense poll."""
        now=time.time()
        records=[r for r in getattr(self,'defense_focus_failures',[]) if r['impact']+20>now]
        signature=sorted((r['attacker'],r['troops']) for r in group)
        record=next((r for r in records if r['city']==city_key(group[0]['city']) and
                     abs(r['impact']-impact)<12 and r['signature']==signature),None)
        if failed:
            if record is None:
                record={'city':city_key(group[0]['city']),'impact':impact,'signature':signature,'attempts':0}
                records.append(record)
            record['attempts']+=1
            record['retry_at']=now+min(60,10*2**min(record['attempts']-1,3))
        self.defense_focus_failures=records
        return record is None or (record['attempts']<3 and now>=record['retry_at'])

    def defer_defense_read(self,reason):
        """Retain the current defense view and back off without replaying taps."""
        now=time.time()
        self.defense_read_failures=getattr(self,'defense_read_failures',0)+1
        delay=min(2,.25*2**min(self.defense_read_failures-1,3)) if self.config.get('focus_defense') else min(30,2**min(self.defense_read_failures,5))
        self.defense_read_waiting=True
        self.defense_read_retry_at=now+delay
        self.defense_unknown_until=self.defense_read_retry_at+10
        message=f'{reason} 방어 창 유지 · {delay}초 뒤 화면만 재인식 (같은 숫자 클릭 안 함)'
        if self.defense_read_failures==1 or now-getattr(self,'defense_read_last_log',0)>=60:
            self.log(message);self.defense_read_last_log=now
        self.emit({'notice':message})
        return True

    def poll_defense(self,force=False):
        now=time.time()
        focused=getattr(self,'troop_defense_enabled',lambda:self.config.get('focus_defense',False))()
        if getattr(self,'defense_read_waiting',False) and now<self.defense_read_retry_at:
            return True
        if not force and now-self.last_defense_poll<(.25 if focused else 8/self.config.get('decision_speed',2.5)):
            return self.defense_busy()
        self.last_defense_poll=now
        retained=(getattr(self,'defense_read_waiting',False) and self.vision.march_panel(self.image) and
                  self.vision.defense_tab_selected(self.image))
        if focused and not retained and not self.vision.incoming_button(self.image):
            return self.defense_busy()
        if not retained:self.map()
        try: rows=self.inspect_defense_rows(reuse=True) if retained else self.inspect_defense_rows()
        except (GameUnavailable,EmulatorBridgeUnavailable):
            raise  # Let Bot.run enter game/Android recovery, not an OCR retry.
        except RuntimeError as exc:
            return self.defer_defense_read(str(exc))
        uncertain=any(row.get('time_uncertain') for row in rows)
        if not uncertain:
            self.defense_read_waiting=False
            self.defense_read_retry_at=0
            self.defense_read_failures=0
        if not rows:
            if self.vision.incoming_button(self.image):
                self.defense_unknown_until=max(self.defense_unknown_until,now+10)
                self.log('방어 방패 알림은 있으나 목록 미확인 · 새 공격/강화 보류 후 재확인')
            else:
                # A verified inactive tab / expired shortcut ends the failed-
                # OCR hold; an outgoing-only list must not renew the hold.
                self.defense_unknown_until=0
            return self.defense_busy()
        self.defense_unknown_until=max(self.defense_unknown_until,now+10)
        self.threatened_names={city_key(r['city']) for r in rows}
        if self.muster: self.cancel_muster('방어 우선')
        # One job per city/wave; all visible attacks are summed, not just the first.
        keys=list(dict.fromkeys(city_key(row['city']) for row in rows))
        if focused:keys.sort(key=lambda key:self.defense_group_priority([r for r in rows if city_key(r['city'])==key]))
        for group_key in keys:
            group=[r for r in rows if city_key(r['city'])==group_key]
            if any(r.get('time_uncertain') for r in group):
                continue
            impact=min(r['impact'] for r in group)
            minimum=15 if focused else 35
            if impact-time.time()<minimum:
                self.log(f"{group[0]['city']} 도착까지 {minimum}초 미만 · 늦은 지원을 보내지 않고 다른 피격 성 확인")
                continue
            fingerprint=sorted((r['attacker'],r['troops']) for r in group)
            existing=next((j for j in self.defenses if city_key(j['staging_name'])==group_key and
                           abs(j['impact']-impact)<10 and j['status'] not in ('confirmed','lost')),None)
            if existing and sorted(tuple(r) for r in existing.get('fingerprint',[]))==fingerprint:
                if focused and existing.get('incoming_rows'):
                    if existing.get('needed',0)>0 and now-existing.get('focus_last_attempt',0)>=10:
                        self.map()
                        if self.defense_position(existing) is not None:self.prepare_defense(existing)
                    continue
                retry=(existing['status']=='insufficient' and now-existing['started']>30 and
                       impact-now>35 and not existing['moves'])
                if not retry: continue
            if existing:
                if getattr(self,'outcome_policy',None) and existing.get('episode_id'):
                    self.outcome_policy.mark(existing['episode_id'],'abandoned')
                existing['status']='superseded'; self.save_defenses()
            if not self.defense_focus_retry(group,impact):continue
            powers=[]
            for row in ([] if focused else group):
                if row.get('troops_exact') is False: break
                power=self.inspect_attack_power(row)
                if power is None: break
                powers.append(power)
            power_known=len(powers)==len(group)
            if focused:
                self.log(f"{group[0]['city']} 집중 방어 · 상대 공격력 계산 생략 · 병력 범위 상한의 99.9% 기준으로 넉넉한 지원")
            elif not power_known:
                self.log(f"{group[0]['city']} 적 공격력 비공개 · 예비병력을 유지하는 제한 지원으로 전환합니다.")
                self.emit({'alert':'적 병력/공격력이 비공개입니다. 승리를 보장할 수 없는 제한 지원을 검토합니다.'})
            self.map(); current=self.inspect_defense_rows(focus=group_key)
            target=next((r for r in current if city_key(r['city'])==group_key),None)
            if not target: self.map(); continue
            if not self.focus_defense_city(target):
                self.defense_focus_retry(group,impact,failed=True)
                continue
            # The observed map-pin action centers its city at (640,330), but
            # camera arrival AND the selected city's name must both be verified.
            self.map(); reference=self.image.copy()
            city=self.read_owned_details((640,330),target['city'])
            if not city:
                self.defense_focus_retry(group,impact,failed=True)
                self.defense_unknown_until=max(self.defense_unknown_until,impact+15)
                self.log('피격 성 상세정보 확인 실패 · 새 공격/강화를 보류하고 방어 재확인')
                continue
            matrix=self.vision.viewport_transform(reference,self.image)
            if matrix is None: continue
            position=self.vision.transform_point(matrix,(640,330))
            if getattr(self,'outcome_policy',None):self.hud()
            job={'id':str(time.time_ns()),'status':'preparing','staging':position,'target':position,
                 'staging_name':city['name'],'impact':impact,'attack':sum(powers) if power_known else None,'before':city,
                 'fingerprint':fingerprint,'attackers':[r['attacker'] for r in group],
                 'started':time.time(),'moves':[],'donors':[],'view':'defense-'+str(time.time_ns())+'.png',
                 'economy_before':self.state.copy(),'purpose':'defense','incoming_rows':group}
            self.defenses.append(job); self.image.save(self.data/job['view']); self.save_defenses()
            if power_known: self.strategy.threat(city['name'],job['attack'],impact)
            self.prepare_defense(job)
        if uncertain:
            names=', '.join(dict.fromkeys(r['city'] for r in rows if r.get('time_uncertain')))
            return self.defer_defense_read(f'{names} 방어 도착시간 재인식 필요 · 확인된 다른 피격 성 우선 처리 완료.')
        self.map()
        return True

    def defense_busy(self):
        return (time.time()<self.defense_unknown_until or
                any(j['status'] in ('preparing','waiting','insufficient') or
                    j['status']=='unverified' and j['impact']>time.time() for j in self.defenses))

    def defense_snapshot_valid(self,job):
        self.map()
        focused=getattr(self,'troop_defense_enabled',lambda:self.config.get('focus_defense',False))()
        rows=self.inspect_defense_rows() if focused else self.inspect_defense_rows(focus=city_key(job['staging_name']))
        self.map()
        if focused:
            if not getattr(self,'defense_list_complete',True):return False
            self.threatened_names.update(city_key(r['city']) for r in rows)
        current=[r for r in rows if city_key(r['city'])==city_key(job['staging_name'])]
        if not current: return False
        signature=sorted((r['attacker'],r['troops']) for r in current)
        expected=sorted(tuple(r) for r in job['fingerprint'])
        return signature==expected and abs(min(r['impact'] for r in current)-job['impact'])<10

    def prepare_defense(self,job):
        if not getattr(self,'defense_list_complete',True):
            job['status']='insufficient'; self.save_defenses(); return
        if getattr(self,'troop_defense_enabled',lambda:self.config.get('focus_defense',False))():return self.prepare_focus_defense(job)
        if job['attack'] is None:
            return self.prepare_precautionary_defense(job)
        self.active_defense=job
        try:
            needed=defense_shortfall(job['attack'],job['before']['defense'],job['before']['city_bonus'],
                                     job['before']['guardian_bonus'],self.config.get('defense_margin',1.3))
            needed=max(0,needed-sum(m['troops'] for j in self.defenses if j is not job and
                city_key(j['staging_name'])==city_key(job['staging_name']) for m in j['moves']
                if m['status']=='sent' and time.time()<m['arrival']<job['impact']-20))
            job['needed']=needed
            if not needed:
                self.ensure_defense_episode(job,decision('defense_route','sufficient','hold'))
                job['status']='waiting'; self.save_defenses()
                self.log(f"{job['staging_name']} 현재 방어력 충분 · 공격 도착 후 결과 확인"); return
            source_view=self.image.copy()
            sources=self.vision.own_sources(self.image,self.state.get('owner'))
            self.capture(); self.rebase_defense()
            matrix=self.vision.viewport_transform(source_view,self.image)
            if matrix is None: job['status']='unverified'; self.save_defenses(); return
            sources=[(self.vision.transform_point(matrix,pos),troops) for pos,troops in sources]
            job['donors']=[{'position':pos,'troops':troops} for pos,troops in sources
                if math.dist(pos,job['staging'])>55]
            self.save_defenses()
            routes=[]
            for donor in job['donors']:
                # Preflight uses a real route, a real slider quantity and its displayed ETA.
                if time.time()+30>=job['impact']: break
                p=self.prepare_transfer(donor['position'],min(needed,donor['troops']*.1),job,self.rebase_defense)
                if p:
                    d=p['donor']
                    donor['name']=d['name']
                    # Do not empty another city already known to be under attack.
                    threatened=(city_key(d['name']) in self.threatened_names or any(
                        city_key(j['staging_name'])==city_key(d['name']) and j['impact']>time.time() for j in self.defenses))
                    if not threatened:
                        routes.append(SupportRoute(d['name'],tuple(donor['position']),int(d['troops']),p['duration']))
                    self.back(); self.map(); self.rebase_defense()
            policy=getattr(self,'outcome_policy',None)
            if policy:
                plans={}
                now=time.time()
                for margin_name,margin in (('base',self.config.get('defense_margin',1.3)),
                                           ('buffered',self.config.get('defense_margin',1.3)*1.15)):
                    extra=defense_shortfall(job['attack'],job['before']['defense'],job['before']['city_bonus'],job['before']['guardian_bonus'],margin)
                    original=defense_shortfall(job['attack'],job['before']['defense'],job['before']['city_bonus'],
                                               job['before']['guardian_bonus'],self.config.get('defense_margin',1.3))
                    extra=needed+max(0,extra-original)
                    for route_kind in ('fastest','fewest'):
                        candidate=support_plan(extra,job['impact'],now,routes,self.config['max_send_fraction'],
                            buffer=self.config.get('defense_buffer',20),setup_seconds=18,strategy=route_kind)
                        if candidate:plans[route_kind+':'+margin_name]=candidate
                context='pressure:'+bucket(needed/max(job['before'].get('troops',1),1),(.3,1,3))+'|deadline:'+bucket(job['impact']-now,(60,180,600))
                if plans:
                    choice=policy.choose('defense_route',context,list(plans),
                        {a:{'donors':len(p),'troops':sum(n for _,n in p),
                            'duration':max(r.travel_seconds for r,_ in p),'needed':needed} for a,p in plans.items()})
                    job['policy']=choice['action'].split(':',1)[0]
                    self.ensure_defense_episode(job,choice)
                    plan=plans[choice['action']]
                else:plan=None
            else:
                job['policy']=self.learner.choose('defense_route',['fastest','fewest'])
                plan=support_plan(needed,job['impact'],time.time(),routes,self.config['max_send_fraction'],
                                  buffer=self.config.get('defense_buffer',20),setup_seconds=18,strategy=job['policy'])
                if plan is None and job['policy']!='fastest':
                    job['policy']='fastest'
                    plan=support_plan(needed,job['impact'],time.time(),routes,self.config['max_send_fraction'],
                                      buffer=self.config.get('defense_buffer',20),setup_seconds=18,strategy='fastest')
            if not plan:
                job['status']='insufficient'; self.save_defenses()
                self.log(f"{job['staging_name']} 방어 보류: 도착 전 지원 가능 병력 부족 · 필요 {pretty(needed)}"); return
            for route,take in plan:
                if not self.defense_snapshot_valid(job):
                    self.log('적 공격 구성이 바뀌어 추가 지원을 보류하고 다시 관측합니다.'); break
                self.rebase_defense()
                # Project measured source position against the current terrain.
                donor=next((d for d in job['donors'] if d.get('name')==route.source),None)
                if not donor: break
                p=self.prepare_transfer(donor['position'],take,job,self.rebase_defense)
                if not p: break
                # Deadline is checked AGAIN after OCR/slider work, immediately before send.
                if (time.time()-p['observed']>5 or
                    time.time()+p['duration']+self.config.get('defense_buffer',20)>=job['impact'] or
                    city_key(p['donor']['name'])!=city_key(route.source)):
                    self.back(); self.log('지원 전송 취소: 재확인한 도착시간/출발 성 불일치'); break
                move={'source':route.source,'troops':p['troops'],'arrival':time.time()+p['duration'],
                      'status':'intent','sent_at':time.time()}
                job['moves'].append(move); self.save_defenses()
                self.tap(*p['button'],delay=.5)
                self.capture()
                if not self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)):
                    self.log('지원 결과 불명확: 같은 전송은 반복하지 않습니다.'); break
                move['status']='sent'; self.rebase_defense(); self.save_defenses()
                self.counts['supports']+=1
                self.log(f"방어 지원 {route.source} → {job['staging_name']} {pretty(move['troops'])}, 도착 여유 {int(job['impact']-move['arrival'])}초")
            job['status']='waiting'; self.save_defenses()
            job['sent_total']=sum(m['troops'] for m in job['moves'] if m['status']=='sent')
            self.save_defenses()
        finally:
            self.active_defense=None
            self.map()

    def prepare_precautionary_defense(self,job):
        """Hidden enemy totals permit one bounded reinforcement, never a made-up power.

        Keep at least 70% at the donor, exclude threatened donors, and verify the
        destination and actual free ETA. This does not claim the city can hold.
        """
        self.active_defense=job
        try:
            sources=self.vision.own_sources(self.image,self.state.get('owner'))
            job['donors']=[{'position':pos,'troops':troops} for pos,troops in
                sorted(sources,key=lambda s:math.dist(s[0],job['staging'])) if math.dist(pos,job['staging'])>55]
            policy=getattr(self,'outcome_policy',None)
            choice=None
            if policy and job['donors']:
                choice=policy.choose('defense_blind','deadline:'+bucket(job['impact']-time.time(),(60,180,600)),
                                     ['nearest']+(['largest'] if len(job['donors'])>1 else []))
                if choice['action']=='largest':job['donors'].sort(key=lambda d:-d['troops'])
            self.save_defenses()
            for donor in job['donors']:
                if time.time()+35>=job['impact']: break
                # A single wave cannot request more than its current garrison;
                # the UI's 10% minimum and configured reserve are still enforced.
                take=min(job['before']['troops'],donor['troops']*min(.3,self.config['max_send_fraction']))
                p=self.prepare_transfer(donor['position'],take,job,self.rebase_defense)
                if not p: continue
                name=city_key(p['donor']['name'])
                unsafe=(name in self.threatened_names or time.time()-p['observed']>5 or
                    time.time()+p['duration']+self.config.get('defense_buffer',20)>=job['impact'])
                if unsafe:
                    self.back(); self.map(); self.rebase_defense(); continue
                # Revalidate the wave, then re-open the same measured route.
                self.back(); self.map()
                if not self.defense_snapshot_valid(job): break
                self.rebase_defense()
                p=self.prepare_transfer(donor['position'],take,job,self.rebase_defense)
                if not p: continue
                if (city_key(p['donor']['name'])!=name or time.time()-p['observed']>5 or
                    time.time()+p['duration']+self.config.get('defense_buffer',20)>=job['impact']):
                    self.back(); break
                move={'source':p['donor']['name'],'troops':p['troops'],'arrival':time.time()+p['duration'],
                      'status':'intent','sent_at':time.time()}
                if choice:self.ensure_defense_episode(job,choice)
                job['moves'].append(move); self.save_defenses()
                self.tap(*p['button'],delay=.5)
                if self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)):
                    move['status']='sent'; self.counts['supports']+=1
                    self.log(f"제한 방어 지원 {move['source']} → {job['staging_name']} {pretty(move['troops'])} · 적 공격력 비공개, 결과 확인 대기")
                break
            job['status']='waiting' if job['moves'] else 'insufficient'
            self.save_defenses()
            if not job['moves']: self.log(f"{job['staging_name']} 기한 내 지원 가능한 비피격 성을 확인하지 못했습니다.")
        finally:
            self.active_defense=None
            self.map()

    def ensure_defense_episode(self,job,choice):
        policy=getattr(self,'outcome_policy',None)
        if not policy or job.get('episode_id'):return
        deep=getattr(self,'reinforcement_policy',lambda:None)()
        if deep and not choice.get('_rl_id'):
            choice=deep.choose(choice['domain'],choice['context'],[choice['action']])
        job['episode_id']=policy.begin('defense',[choice],job.get('economy_before',{}),
            {'city':job['staging_name'],'level':job['before'].get('level'),'impact':job['impact']},event_id='defense-'+job['id'],now=job['started'])
        self.save_defenses()

    def check_defense_results(self):
        for job in self.defenses:
            if job['status'] not in ('waiting','insufficient','unverified') or time.time()<job['impact']+10: continue
            if time.time()-job.get('last_result_check',0)<45: continue
            if time.time()>job['impact']+1800:
                if getattr(self,'outcome_policy',None) and job.get('episode_id'):
                    self.outcome_policy.mark(job['episode_id'],'abandoned')
                job['status']='unknown'; self.save_defenses(); continue
            job['last_result_check']=time.time()
            self.map(); pos=self.defense_position(job)
            if pos is None:
                job['status']='unverified'; self.save_defenses(); continue
            city=self.read_owned_details(pos,job['staging_name'])
            if city:
                # Ownership alone is not a victory: the attacker could cancel. A battle
                # report with explicit result and matching city is needed for a reward.
                job['ownership_after']='held'; job['troops_after']=city['troops']
            else: job['ownership_after']='unknown'
            latest_position=self.defense_position(job)
            lost_owner=bool(not city and latest_position and self.confirm_city_lost(latest_position,job['staging_name']))
            result=(False if lost_owner
                    else self.read_defense_result(job))
            if result is not None:
                job['status']='confirmed' if result else 'lost'
                sent=sum(m['troops'] for m in job['moves'] if m['status']=='sent')
                initial=job['before']['troops']
                lower_loss=max(0,initial+sent-city['troops']) if city else None
                reward=((1-min(.9,lower_loss/max(initial+sent,1))) if lower_loss is not None else .2) if result else -2
                self.learner.record('defense_route',job.get('policy','no_support'),reward,
                    job['before'],{'held':result,'sent':sum(m['troops'] for m in job['moves']),
                                   'troops_after':job.get('troops_after'),'troop_loss_lower_bound':lower_loss,
                                   'city_vp_exposure':job['before'].get('vp')},'실제 성 상실 확인' if lost_owner else '도시/공격자 일치 방어 보고서 결과')
                policy=getattr(self,'outcome_policy',None)
                if policy:
                    self.map();self.hud()
                    if job.get('episode_id'):
                        capital=sum(i['cost'] for i in self.strategy.state['investments']
                                    if city_key(i['name'])==city_key(job['staging_name']) and i['status'] in ('intent','pending','held')) if lost_owner else 0
                        learned=policy.settle(job['episode_id'],result,
                            {**self.state,'troops_exposed':initial+sent,'troop_loss_lower_bound':lower_loss,
                             'capital_lost':capital},'실제 성 상실 확인' if lost_owner else '시각/도시/공격자 일치 방어 보고서')
                        if learned is not None:
                            self.log(f"실전 결과 학습 · 방어 {'성공' if result else '패배'} · 평가 {learned:+.2f} · 지원 경로/병력 선택에 반영")
                            self.publish_state()
                    if lost_owner:self.learn_city_loss(job['staging_name'],'방어 결과 실제 성 상실')
                self.strategy.outcome(job['staging_name'],result,'실제 성 상실 확인' if lost_owner else '방어 보고서로 투자 성 보유/상실 확인',event_id=job['id'])
                self.counts['defenses']+=int(result)
                self.log(f"{job['staging_name']} 방어 {'성공' if result else '실패'} 보고서 확인 · 학습 저장")
            else:
                job['status']='unverified'
                self.log(f"{job['staging_name']} 도착 후 보유={job['ownership_after']} · 보고서 결과 미확인, 승리 학습 보류")
            job['last_result_check']=time.time(); self.save_defenses()

    def read_defense_result(self,job):
        self.map(); self.tap(1224,670)
        title=self.region({'title':(213,110,570,156)})['title']
        if '보고서' not in title: self.map(); return None
        # The last-open report may be a detail; return to the observed list once.
        if '세부' in title:
            self.tap(228,129)
        for row in range(3):
            y=299+row*93
            fields=self.region({'city_name':(405,y-20,537,y+13),'attacker':(848,y-34,1004,y+2)})
            if city_key(fields['city_name'])!=city_key(job['staging_name']): continue
            if city_key(fields['attacker']) not in {city_key(a) for a in job.get('attackers',[])}: continue
            from .planner import march_seconds
            age=march_seconds(self.region({'remaining':(244,y+9,302,y+35)})['remaining'])
            if age is None or abs((time.time()-age)-job['impact'])>75: continue
            self.tap(825,y)
            self.image.save(self.data/f"defense-result-{job['id']}.png")
            doc=self.vision.ocr.read(self.data/'latest.png','ko')['text']
            compact=''.join(doc.split())
            # Explicit text only; no reward from an ambiguous icon or a scout report.
            if city_key(job['staging_name']) not in city_key(compact) or '방어' not in compact:
                self.map(); return None
            won=any(w in compact for w in ('방어성공','승리','방어성공했습니다'))
            lost=any(w in compact for w in ('방어실패','패배','도시를잃'))
            self.map()
            return won if won!=lost else None
        self.map(); return None

    def confirm_city_lost(self,position,expected):
        self.map(); self.tap(*position)
        scout=self.vision.match(self.image,'scout',.84)
        attack=self.vision.match(self.image,'attack',.84)
        if not scout or not attack: self.map(); return False
        self.tap(scout[0]+85,scout[1]+68)
        if self.vision.navigation_panel(self.image)!='도시 상세정보': self.map(); return False
        fields=self.region({'city_name':(270,113,525,153),'owner':(302,465,510,496)})
        self.map()
        return (city_key(fields['city_name'])==city_key(expected) and bool(city_key(fields['owner'])) and
                city_key(fields['owner'])!=city_key(self.state.get('owner','')))

    def strategic_upgrade(self):
        """Invest in a defendable garrison, with delayed ownership rewards and loss locks."""
        if not self.config.get('upgrade_enabled',True) or time.time()-getattr(self,'last_upgrade_check',0)<60: return
        self.last_upgrade_check=time.time()
        self.map(); sources=self.vision.own_sources(self.image,self.state.get('owner'))
        if not sources: return
        # Troops are a screening heuristic only. Actual walls/defense are read before spending.
        position,_=max(sources,key=lambda source:source[1]); reference=self.image.copy()
        city=self.read_owned_details(position)
        if not city: return
        matrix=self.vision.viewport_transform(reference,self.image)
        if matrix is None: return
        position=self.vision.transform_point(matrix,position)
        self.hud()
        allowed,reason=self.strategy.investment_allowed(city,city.get('cost'),self.state.get('gold'),
            self.config['gold_reserve'],self.config.get('defense_margin',1.3))
        if not allowed or city['level']>=self.config['target_level']:
            self.log(f"{city['name']} 강화 보류: {reason}"); return
        threat=self.strategy.reference_attack()
        context='upgrade_safety:'+('unmeasured' if threat is None else str(min(5,int(city['defense']/threat))))
        rows=[r for r in self.learner.summary()['policy'] if r['context']==context and r['action']=='upgrade']
        if rows and rows[0]['n']>=3 and rows[0]['mean']<0:
            self.log('해당 방어 조건의 강화 학습 순손실: 자동 투자 보류'); return
        action='upgrade' if getattr(self,'reinforcement_policy',lambda:None)() else self.learner.choose(context,['defer','upgrade'])
        if action=='defer':
            self.learner.record(context,'defer',0,city,city,'투자 대기. 미실행 강화의 가상 수익을 기록하지 않음'); return
        reference=self.image.copy()
        selected=self.select_owned(position)
        if not selected or city_key(selected['name'])!=city_key(city['name']): self.map(); return
        marker=self.vision.match(self.image,'upgrade',.84)
        if not marker: self.map(); return
        from .vision import amount
        cost=amount(self.region({'cost':(marker[0]-47,marker[1]+22,marker[0]+25,marker[1]+51)})['cost'])
        if cost is None or abs(cost-city['cost'])>max(1,city['cost']*.015): self.map(); return
        entry=self.strategy.invest(city['name'],cost,context,{**self.state,'city_level':city['level'],'city_vp':city.get('vp')})
        entry['position']=position; entry['view']='investment-'+entry['id']+'.png'
        reference.save(self.data/entry['view']); self.strategy.save()
        self.tap(*marker[:2],delay=.5)
        self.map()
        transform=self.vision.viewport_transform(reference,self.image)
        if transform is None: self.log('강화 결과 위치 미확인. 같은 강화를 반복하지 않습니다.'); return
        updated=self.read_owned_details(self.vision.transform_point(transform,position),city['name'])
        if not updated or updated['level']!=city['level']+1:
            self.log('강화 결과 미확인. 추가 투자 중단.'); return
        entry['status']='pending'; self.strategy.save(); self.counts['upgrades']+=1
        self.learner.record('upgrade_expense','upgrade',0,entry['before'],{'cost':cost,'level':updated['level']},
                            '지출과 레벨만 확인. 지연된 성 보유/손실 평가 전에는 성공 보상 없음')
        self.log(f"{city['name']} 방어 검증 후 1레벨 투자 {pretty(cost)} · 보유/손실 평가 대기")
        return True

    def audit_investments(self):
        entries=[e for e in self.strategy.state['investments'] if e['status'] in ('pending','held','intent') and
                 time.time()-e['ts']>=900 and time.time()-e.get('last_audit',0)>120 and e.get('view')]
        for entry in sorted(entries,key=lambda e:e.get('last_audit',0)):
            self.map()
            with Image.open(self.data/entry['view']) as source: reference=source.convert('RGB')
            transform=self.vision.viewport_transform(reference,self.image)
            if transform is None: continue
            pos=self.vision.transform_point(transform,entry['position'])
            if not self.world_position(pos): continue
            entry['last_audit']=time.time(); self.strategy.save()
            city=self.read_owned_details(pos,entry['name'])
            if city:
                if entry['status']=='intent':
                    if city['level']<entry['before']['city_level']+1:
                        entry['status']='unverified'; self.strategy.save(); return
                    entry['status']='pending'; self.strategy.save()
                self.strategy.outcome(entry['name'],True,'15분 후 실제 성 보유 재확인 (전투 승리/순이익 증명 아님)')
            else:
                transform=self.vision.viewport_transform(reference,self.image)
                if transform is not None and self.confirm_city_lost(self.vision.transform_point(transform,entry['position']),entry['name']):
                    self.strategy.outcome(entry['name'],False,'강화 성의 새 소유자를 화면에서 확인. 누적 투자 골드 손실')
                    if getattr(self,'outcome_policy',None):self.hud();self.learn_city_loss(entry['name'],'강화 성의 실제 새 소유자 확인')
                    self.log(f"{entry['name']} 상실 확인 · 투자 골드 손실 학습, 24시간 재투자 제한")
            return  # One real ownership audit per cycle, leaving time for defense polling.
