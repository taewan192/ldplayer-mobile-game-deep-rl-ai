"""Measured city-list donors and acknowledged defensive transfers."""
import math
import re
import time
import hashlib

import numpy as np

from PIL import Image

from .strategy import city_key
from .vision import numeric_amount, pretty, march_countdown


def explicit_army(text):
    # 81.1G can OCR as 81.16. Never treat that as 81 troops, or guess G.
    if not re.search(r'[kKmMgGtT]',text):return None
    value=numeric_amount(text)
    return value if isinstance(value,(int,float)) and math.isfinite(value) and value>0 else None


def defense_waves(rows):
    """Separate simultaneous waves from attacks many minutes in the future."""
    result=[]
    valid=[r for r in rows if not r.get('time_uncertain') and isinstance(r.get('impact'),(int,float)) and math.isfinite(r['impact'])]
    for row in sorted(valid,key=lambda r:r['impact']):
        if not result or row['impact']-result[-1][0]['impact']>30:result.append([])
        result[-1].append(row)
    return result


class DefenseLogisticsMixin:
    def remember_defense_view(self,name,image):
        folder=self.data/'defense-atlas';folder.mkdir(exist_ok=True)
        path=folder/(hashlib.sha256(city_key(name).encode()).hexdigest()[:16]+'.png')
        image.save(path)

    def linked_defense_transform(self,reference,current):
        """Join overlapping, observed city views; no invented world coordinates."""
        folder=self.data/'defense-atlas'
        if not folder.exists():return None
        pairs=[];deadline=time.monotonic()+3
        for path in sorted(folder.glob('*.png'),key=lambda p:-p.stat().st_mtime)[:8]:
            if time.monotonic()>deadline:return None
            with Image.open(path) as source:middle=source.convert('RGB')
            first=self.vision.muster_transform(reference,middle)
            last=self.vision.muster_transform(middle,current)
            if first is not None and last is not None:
                first=np.vstack((first,[0,0,1])) if first.shape==(2,3) else first
                last=np.vstack((last,[0,0,1])) if last.shape==(2,3) else last
                return last@first
            pairs.append((middle,first,last))
        for left,first,_ in pairs:
            if first is None:continue
            for right,_,last in pairs:
                if time.monotonic()>deadline:return None
                if last is None:continue
                bridge=self.vision.muster_transform(left,right)
                if bridge is None:continue
                matrices=[np.vstack((m,[0,0,1])) if m.shape==(2,3) else m for m in (first,bridge,last)]
                return matrices[2]@matrices[1]@matrices[0]
        return None

    def defense_visible_sources(self):
        return self.vision.own_sources_batch(self.image,self.state.get('owner'))

    def support_arrival_feasible(self,duration,impact):
        if time.time()+duration+5<impact:return True
        ratio=self.strategy.state.get('free_support_speedup_ratio')
        return (self.config.get('defense_acceleration_enabled',True) and isinstance(ratio,(int,float)) and
                0<ratio<1 and time.time()+duration*ratio+8<impact)

    def read_defense_donor(self,position):
        # Reinforcing needs a verified name and army, not every wall/bonus OCR
        # field in the slower full-details panel.
        selected=self.select_owned(position)
        if not selected:return None
        cached=next((c for c in getattr(self,'defense_inventory',[]) if city_key(c['name'])==city_key(selected['name'])),{})
        known=self.strategy.state.get('cities',{}).get(city_key(selected['name']),{})
        self.map()
        return {**selected,'level':cached.get('level',known.get('level'))}

    def read_focus_defense_target(self,row):
        # Troop-only defense does not need wall, guardian or bonus fields.
        # An unrelated unreadable stat must not prevent protecting this city.
        selected=self.select_owned_named((640,330),row['city'])
        if not selected:return None
        known=self.strategy.state.get('cities',{}).get(city_key(selected['name']),{})
        city={**selected,'level':row.get('level') if row.get('level') is not None else known.get('level')}
        self.map();self.strategy.observe_city(city)
        return city

    def defense_donor_capacity(self,city):
        troops=city['troops'];level=city.get('level')
        fraction=.1 if level is None or 110<=level<=115 else .7
        available=math.floor(troops*fraction)
        if city_key(city['name']) in self.threatened_names:
            # Being attacked is not by itself evidence that an 81G army has
            # no surplus. Preserve its independently observed defensive goal.
            if time.time()-getattr(self,'latest_defense_rows_at',0)>10:return 0
            rows=[r for r in getattr(self,'latest_defense_rows',[]) if city_key(r['city'])==city_key(city['name'])]
            if not rows or any(r.get('time_uncertain') for r in rows):return 0
            from .focus_defense import troop_goal
            try:reserve=troop_goal(rows,level)
            except (ValueError,KeyError,TypeError):return 0
            available=min(available,max(0,troops-reserve))
        return math.floor((available/max(troops,1)+1e-9)*10)/10*troops

    def refresh_defense_garrison(self,job):
        self.map();self.rebase_defense()
        selected=self.select_owned_named(job['staging'],job['staging_name'])
        if not selected:
            self.log('피격 성 현재 주둔 병력 재확인 실패 · 과거 지원 병력을 확보량으로 재사용하지 않음')
            self.map();return False
        job['garrison_observed']=selected['troops'];job['garrison_at']=time.time()
        self.map();self.rebase_defense();return True

    def reconcile_defense_absence(self,now):
        if (self.vision.march_button(self.image) or
                not self.vision.match(self.image,'map_home',.72,area=(0,615,100,720))):
            self.defense_absent_since=None;return
        since=getattr(self,'defense_absent_since',None)
        if since is None:self.defense_absent_since=now;return
        if now-since<3:return
        self.defense_read_waiting=False;self.defense_read_failures=0;self.defense_unknown_until=0
        changed=False
        for job in self.defenses:
            if (job['impact']>now and job['status'] in ('preparing','waiting','insufficient','unverified') and
                    not any(m['status'] in ('sent','intent') for m in job['moves'])):
                job['status']='cancelled';changed=True
                policy=getattr(self,'outcome_policy',None)
                if policy and job.get('episode_id'):policy.mark(job['episode_id'],'abandoned')
        if changed:
            self.defense_unknown_until=0;self.threatened_names.clear();self.save_defenses()
            self.log('연속 지도 관측에서 침입/행군 알림 해제 확인 · 미출병 방어 대기 해제 · 승리 학습 안 함')

    def defense_city_inventory(self,max_pages=2):
        """Read our troop-sorted list; the pinned main city is not rank one."""
        self.map();self.tap(1047,665,delay=.45)
        if not self.vision.match(self.image,'city_list_title',.9,area=(200,105,360,165)):
            self.map();return []
        self.tap(959,122,delay=.25)
        pixel=self.image.getpixel((508,193))
        if pixel[2]<150 or pixel[1]<100:self.tap(550,201,delay=.35)
        page=self.city_list_page()
        if page is None:self.map();return []
        current,total=page
        for _ in range(current-1):self.tap(574,578,delay=.2)
        result=[]
        for index in range(min(total,max_pages)):
            if self.city_list_page()!=(index+1,total):break
            boxes={}
            for i,y in enumerate((264,328,392,456,520)):
                boxes[f'army{i}']=(488,y-18,568,y+18)
                boxes[f'name{i}']=(638,y-21,834,y+20)
                boxes[f'level{i}']=(399,y-17,438,y+15)
            values=self.region(boxes)
            for i,y in enumerate((264,328,392,456,520)):
                n=explicit_army(values.get(f'army{i}',''))
                if n is None:
                    n=self.vision.city_list_army(self.image,y)
                name=values.get(f'name{i}','').strip()
                level=values.get(f'level{i}','').strip()
                if n and name and not any(city_key(c['name'])==city_key(name) for c in result):
                    result.append({'name':name,'troops':n,'level':int(level) if level.isdigit() else None})
            if index+1<min(total,max_pages):self.tap(705,578,delay=.25)
        self.map()
        result.sort(key=lambda c:-c['troops'])
        self.defense_inventory=result;self.defense_inventory_at=time.time()
        for city in result:
            key=city_key(city['name'])
            self.strategy.state['cities'][key]={**self.strategy.state['cities'].get(key,{}),**city,'observed':time.time()}
        self.strategy.save()
        self.log('방어 지원 도시 목록 · 병력 내림차순 · '+', '.join(c['name']+' '+pretty(c['troops']) for c in result[:4]))
        return result

    def canonical_defense_rows(self,rows):
        known=[c.get('name','') for c in self.strategy.state.get('cities',{}).values()]
        known.extend(c['name'] for c in getattr(self,'defense_inventory',[]))
        fold=lambda n:city_key(n).replace('l','i')
        for row in rows:
            exact={n for n in known if city_key(n)==city_key(row['city'])}
            matches=exact or {n for n in known if fold(n)==fold(row['city'])}
            if len(matches)==1:row['city']=next(iter(matches))
        return rows

    def open_defense_list_donor(self,job,candidate):
        """A list pin may jump the camera; old target coordinates cannot survive it."""
        with Image.open(self.data/job['view']) as source:reference=source.convert('RGB')
        self.remember_defense_view(job['staging_name'],reference)
        selected=self.locate_owned_city(candidate['name'],max_pages=3)
        if not selected:return None
        self.map();self.capture()
        self.remember_defense_view(candidate['name'],self.image)
        matrix=self.vision.muster_transform(reference,self.image)
        if matrix is None:matrix=self.linked_defense_transform(reference,self.image)
        if matrix is None:
            self.log(candidate['name']+' · 방어 목적지와 지도 연결 미확인, 추측 경로 출병 안 함')
            return None
        for key in ('staging','target'):job[key]=self.vision.transform_point(matrix,job[key])
        for donor in job.get('donors',[]):
            if donor.get('position') is not None:donor['position']=self.vision.transform_point(matrix,donor['position'])
        self.image.save(self.data/job['view'])
        selected=self.select_owned_named(selected['position'],candidate['name'])
        if not selected:return None
        self.map();self.rebase_defense()
        return {**selected,'level':candidate.get('level')}

    def restore_defense_target(self,job):
        selected=self.locate_owned_city(job['staging_name'],max_pages=5)
        if not selected:return False
        self.map()
        try:self.rebase_defense()
        except RuntimeError:
            # Exact owned-city list verification permits a new anchor, never
            # reuse a donor's old screen coordinates after a remote jump.
            job['staging']=job['target']=selected['position']
            for donor in job.get('donors',[]):donor['position']=None
            self.image.save(self.data/job['view']);self.save_defenses()
        return True

    def defense_destination_in_view(self,job):
        """Bring a registered destination into view while target-select mode remains open."""
        for _ in range(18):
            x,y=job['staging']
            if 260<x<1020 and 180<y<500:return True
            self.driver.pan_map(max(-180,min(180,640-x)),max(-110,min(110,330-y)))
            self.wait(.3);self.capture()
            try:self.rebase_defense()
            except RuntimeError:return False
        return False

    def confirm_defense_dispatch(self,job,move,preflight):
        """A map after a tap is not an acknowledgement. Inspect outgoing marches."""
        for _ in range(3):
            if not self.vision.match(self.image,'send_transfer',.84,area=(830,490,935,590)):break
            self.wait(.25);self.capture()
        if self.vision.match(self.image,'send_transfer',.84,area=(830,490,935,590)):
            move['status']='not_sent';self.save_defenses()
            self.log('지원 전송 버튼 뒤 병력 창이 유지됨 · 전송 실패로 기록, 다음 출발 성 검토')
            self.back();self.map();return False
        self.map()
        button=self.vision.march_button(self.image)
        if not button:
            self.log('지원 전송 확인 대기 · 행군 목록 미확인, 중복 출병 금지');return False
        self.tap(*button[:2],delay=.3)
        if not self.vision.march_panel(self.image):self.map();return False
        self.tap(959,57,delay=.3)
        if getattr(self,'data',None):self.image.save(self.data/'outgoing-support-check.png')
        for _ in range(2):
            for row in self.vision.outgoing_support_rows(self.image):
                if (city_key(row['city'])==city_key(job['staging_name']) and
                        abs(row['troops']-move['troops'])<=max(1,move['troops']*.025) and
                        abs(row['remaining']-preflight['duration'])<max(20,preflight['duration']*.15)):
                    move['status']='sent';move['arrival']=time.time()+row['remaining']
                    move['confirmed_by']='outgoing_city_troops_eta'
                    move['march_pin']=row['pin'];self.save_defenses()
                    self.log('지원 출병 확인 · '+move['source']+' → '+job['staging_name']+' '+pretty(move['troops']))
                    free=self.vision.match(self.image,'march_speedup_free',.9,
                        area=(275,max(95,row['row']-35),360,min(660,row['row']+40)))
                    if move['arrival']+5>=job['impact'] or (free and row['remaining']>30):
                        self.accelerate_defense_march(job,move,row)
                    move['late']=move['arrival']+5>=job['impact']
                    self.map();return True
            self.wait(.3);self.capture()
        self.map();self.log('지원 전송 결과 미확인 · intent 유지, 같은 병력 재전송 금지')
        return False

    def accelerate_defense_march(self,job,move,row):
        # The user confirmed speedups live in the outgoing march menu. Never
        # mistake a troop label or location pin for a speedup control.
        if not self.config.get('defense_acceleration_enabled',True):return False
        observed=row.get('observed',time.time())
        free=self.vision.match(self.image,'march_speedup_free',.9,
                               area=(275,max(95,row['row']-35),360,min(660,row['row']+40)))
        if free:
            move['acceleration_intent']=time.time();self.save_defenses()
            self.tap(*free[:2],delay=.4)
            if getattr(self,'data',None):self.image.save(self.data/'free-acceleration-result.png')
            for _ in range(3):
                for actual in self.vision.outgoing_support_rows(self.image):
                    if (city_key(actual['city'])==city_key(job['staging_name']) and
                            abs(actual['troops']-move['troops'])<=max(1,move['troops']*.025) and
                            actual['remaining']<row['remaining']-(time.time()-observed)-3):
                        move['arrival']=time.time()+actual['remaining'];move['accelerated']=True
                        ratio=min(.95,(actual['remaining']+time.time()-observed+3)/max(1,row['remaining']))
                        self.strategy.state['free_support_speedup_ratio']=ratio;self.strategy.save();self.save_defenses()
                        self.log(f"무료 행군 가속 확인 · {row['remaining']}초 → {actual['remaining']}초")
                        return True
                self.wait(.25);self.capture()
            self.log('무료 가속 결과 미확인 · 같은 가속 버튼 반복 사용 안 함');return False
        boost=self.vision.march_acceleration_button(self.image,row['row'])
        if not boost:
            self.log('지원 도착 지연 · 해당 행군의 가속 버튼 미확인 · 다른 지원 경로 우선')
            return False
        self.tap(*boost[:2],delay=.35)
        if getattr(self,'data',None):self.image.save(self.data/'acceleration-dialog.png')
        option=self.vision.acceleration_option(self.image)
        if not option:
            self.log('가속 보류 · 보유 가속 아이템/적용 버튼을 확인하지 못함');self.back();return False
        if option['remaining_after']+time.time()+5>=job['impact']:
            self.log('가속 후에도 방어 기한 초과 · 아이템 사용 보류');self.back();return False
        # Consumption intent is persisted before applying a verified item.
        move['acceleration_intent']=time.time();self.save_defenses()
        self.tap(*option['button'],delay=.4)
        # The outgoing countdown must change before treating the boost as real.
        for actual in self.vision.outgoing_support_rows(self.image):
            if city_key(actual['city'])==city_key(job['staging_name']) and actual['remaining']<row['remaining']:
                move['arrival']=time.time()+actual['remaining'];move['accelerated']=True;self.save_defenses();return True
        self.log('가속 적용 결과 미확인 · 가속 아이템 반복 사용 안 함');return False

    def prepare_defense_standby(self):
        """Idle-only local prepositioning, with durable intent and no fake win reward."""
        if not self.config.get('defense_standby_enabled',True):return
        now=time.time()
        if now-getattr(self,'standby_last_check',0)<300:return
        if self.defense_busy() or self.vision.incoming_button(self.image):return
        previous=self.strategy.state.get('standby_move')
        if previous and (previous['status']=='intent' or previous.get('arrival',0)>now):return
        self.standby_last_check=now
        self.defense_city_inventory()
        self.map()
        cities=list(self.strategy.state.get('cities',{}).values())
        targets=[c for c in cities if isinstance(c.get('level'),(int,float)) and 110<=c['level']<=115
                 and c.get('troops',float('inf'))<1e9 and now-c.get('observed',0)<1800]
        if not targets:return
        target=min(targets,key=lambda c:c['troops'])
        selected=self.locate_owned_city(target['name'],max_pages=3)
        if not selected:return
        self.map()
        sources=self.defense_visible_sources()
        sources=sorted((s for s in sources if math.dist(s[0],selected['position'])>55),key=lambda s:-s[1])
        if not sources:return
        job={'purpose':'defense','staging_name':target['name'],'staging':selected['position'],
             'target':selected['position'],'view':'defense-standby-view.png','donors':[{'position':sources[0][0]}],'support_fraction':.1,
             'strict_destination':True}
        self.image.save(self.data/job['view']);self.active_defense=job
        try:
            source=self.read_defense_donor(sources[0][0])
            if not source or city_key(source['name']) in self.threatened_names:return
            self.rebase_defense()
            requested=min(max(0,1e9-selected['troops']),source['troops']*.1)
            if requested<1e8:return
            p=self.prepare_transfer(job['donors'][0]['position'],requested,job,self.rebase_defense)
            if not p or p['duration']>180:self.map();return
            move={'source':source['name'],'destination':target['name'],'troops':p['troops'],
                  'status':'intent','arrival':time.time()+p['duration']}
            self.strategy.state['standby_move']=move;self.strategy.save()
            job.update(impact=time.time()+180,moves=[move])
            self.tap(*p['button'],delay=.4)
            if self.confirm_defense_dispatch(job,move,p):
                self.counts['standby_transfers']=self.counts.get('standby_transfers',0)+1
            self.strategy.save()
            self.log('방어 사전 배치 요청 · '+source['name']+' → '+target['name']+' '+pretty(p['troops'])+' · 전투 승리로 학습하지 않음')
        finally:self.active_defense=None;self.map()
