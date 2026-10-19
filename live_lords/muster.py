"""Observed free troop transfers and persisted gather/wait/re-scout/attack workflow."""
from __future__ import annotations

import json
import math
import re
import time
import numpy as np

from PIL import Image

from .planner import MusterDonor,plan_muster,required_attack,march_seconds
from .vision import amount,pretty
from .strategy import city_key
from .outcomes import bucket


def muster_sweep():
    """A bounded expanding square around the saved staging city, not a new target."""
    steps=[]
    directions=((120,0),(0,80),(-120,0),(0,-80))
    for leg,length in enumerate((1,1,2,2,3,3,4,4,4)):
        steps.extend([directions[leg%4]]*length)
    return steps


class MusterMixin:
    @staticmethod
    def world_position(position):
        return 135<position[0]<1145 and 120<position[1]<615

    def init_muster(self):
        self.muster=None
        path=self.data/'muster-journal.json'
        if path.exists():
            saved=json.loads(path.read_text('utf-8'))
            if saved.get('status') in ('discovering','preparing','waiting'):
                # Submitted/uncertain moves are never sent again after restart.
                saved['status']=('waiting' if saved.get('version',1)<2 or
                    any(m['status']=='intent' for m in saved['moves']) else saved['status'])
                self.muster=saved
                if saved.get('version',1)>=2:
                    saved['donors']=[d for d in saved.get('donors',[]) if d['troops']>0]
                    if saved.get('bulk_once'):
                        saved['single_transfer']=False
                        if any(move['status']=='intent' for move in saved['moves']):saved['status']='waiting'
                        elif any(not d.get('done') for d in saved['donors']):saved['status']='preparing'
                    else:
                        saved['single_transfer']=True;saved['policy']='single'
                        if saved['moves']:saved['status']='waiting'
                        elif len(saved['donors'])!=1:
                            saved['donors']=[];saved['status']='discovering'
            elif saved.get('status')=='dispatched' and saved.get('attack_arrival'):
                self.pending={'target':saved['target'],'arrival':saved['attack_arrival'],
                              'before':saved['before'],'muster':saved}

    def save_muster(self,status=None):
        if status: self.muster['status']=status
        path=self.data/'muster-journal.json'
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.muster,ensure_ascii=False,indent=2),'utf-8')
        temporary.replace(path)

    def cancel_muster(self,reason):
        if not self.muster: return
        self.log('집결 계획 종료: '+reason+' · 이미 이동한 병력은 그대로 둡니다.')
        self.save_muster('cancelled')
        self.muster=None
        self.emit({'muster':'대기'})

    def verify_muster_view(self):
        with Image.open(self.data/'muster-view.png') as source:
            reference=source.convert('RGB')
        return self.vision.same_viewport(self.image,reference)

    def rebase_muster(self):
        """Track camera movement caused by a verified city selection, not guessed screen offsets."""
        if not self.muster: return
        if self.muster.get('fixed_reference'):
            m=self.muster
            with Image.open(self.data/m['fixed_reference']) as source:reference=source.convert('RGB')
            matrix=self.vision.muster_transform(reference,self.image)
            if matrix is None and (self.data/'muster-view.png').exists():
                # A selected city's large banner can hide most of the original
                # anchor. Register the small observed change from the last
                # verified frame, then compose its unrounded matrix. All points
                # still project from their original canonical coordinates.
                with Image.open(self.data/'muster-view.png') as source:previous=source.convert('RGB')
                delta=self.vision.viewport_transform(previous,self.image)
                if delta is not None and all(math.dist(self.vision.transform_point(delta,p),p)<220
                        for p in ((300,200),(1000,200),(300,500),(1000,500))):
                    if delta.shape==(2,3):delta=np.vstack((delta,[0,0,1]))
                    matrix=delta@np.asarray(m['anchor_matrix'])
            if matrix is None:
                self.image.save(self.data/'muster-tracking-failed.png')
                raise RuntimeError('고정 집결 지도 대조 실패 · 추측 좌표로 이동하지 않음')
            m['anchor_matrix']=matrix.tolist()
            for key,point in m['anchors'].items():m[key]=self.vision.transform_point(matrix,point)
            for collection in ('donors','candidates','moves','destination_candidates'):
                for item in m.get(collection,[]):
                    if item.get('anchor_position') is not None:
                        item['position']=self.vision.transform_point(matrix,item['anchor_position'])
            self.image.save(self.data/'muster-view.png');self.save_muster();return
        with Image.open(self.data/'muster-view.png') as source: reference=source.convert('RGB')
        matrix=self.vision.muster_transform(reference,self.image)
        if matrix is None:
            self.wait(.7);self.capture();matrix=self.vision.muster_transform(reference,self.image)
        if matrix is None: raise RuntimeError('집결 지도 좌표 재확인 실패. 이동을 반복하지 않습니다.')
        for key in ('target','staging','transfer_origin'):
            if key not in self.muster:continue
            self.muster[key]=self.vision.transform_point(matrix,self.muster[key])
        for collection in ('donors','candidates','moves','destination_candidates'):
            for donor in self.muster.get(collection,[]):
                if donor.get('position') is not None:
                    donor['position']=self.vision.transform_point(matrix,donor['position'])
        self.image.save(self.data/'muster-view.png')
        self.save_muster()

    def select_owned(self,position,require_troops=True):
        if not self.world_position(position): return None
        self.map()
        self.tap(*position,delay=1.5)
        def read():
            if require_troops:return self.vision.selected_own_city(self.image,self.state.get('owner'))
            return self.vision.selected_own_city(self.image,self.state.get('owner'),require_troops=False)
        selected=read()
        if not selected:
            self.map(); self.capture()
            flag=self.vision.own_flag_for_castle(self.image,position)
            if flag:
                self.tap(*flag,delay=1)
                selected=read()
        if not selected:
            self.map()
            return None
        return selected

    def select_owned_named(self,position,expected):
        """Resolve overlapping banners by clicking observed owned flags and
        requiring the stored city name, rather than accepting the neighbour."""
        self.map();self.capture();reference=self.image.copy()
        selected=self.select_owned(position)
        if selected:
            if city_key(selected['name'])==city_key(expected):return selected
            if selected.get('move_button'):
                x,y=selected['move_button']
                values=self.region({'scout_city_padded':(x-90,y-207,min(1280,x+60),y-176),
                                    'owned_name_tight':(x-85,y-205,min(1280,x+65),y-176)})
                if any(city_key(value)==city_key(expected) for value in values.values()):
                    return {**selected,'name':expected}
        self.map();self.capture();reference=self.image.copy()
        flags=[]
        for cx,cy,scale,_ in self.vision.own_flags(self.image):
            body=(cx+round((640-cx)*.07),cy+round(95*scale))
            distance=math.dist(body,position)
            if distance<160:flags.append((distance,(cx,cy)))
        for _,flag in sorted(flags)[:3]:
            self.map();matrix=self.vision.muster_transform(reference,self.image)
            if matrix is None:return None
            point=self.vision.transform_point(matrix,flag)
            if not (135<point[0]<1145 and 55<point[1]<615):continue
            self.tap(*point,delay=.7)
            actual=self.vision.selected_own_city(self.image,self.state.get('owner'))
            if actual and city_key(actual['name'])==city_key(expected):return actual
            if actual and actual.get('move_button'):
                x,y=actual['move_button']
                values=self.region({'scout_city_padded':(x-90,y-207,min(1280,x+60),y-176)})
                if any(city_key(value)==city_key(expected) for value in values.values()):
                    return {**actual,'name':expected}
        self.map();return None

    def begin_muster(self,target,origin_position,needed,sources,before):
        if self.config.get('focus_defense'):return False
        if not self.config.get('muster_enabled',True): return False
        self.map()
        reference=self.image.copy()
        staging=self.select_owned(origin_position)
        if (getattr(self,'prevalidated_bulk',False) and getattr(self,'expected_origin_name',None) and
                (not staging or city_key(staging['name'])!=city_key(self.expected_origin_name))):
            staging=self.select_owned_named(origin_position,self.expected_origin_name)
        if not staging:
            self.log('집결 기점 선택 실패 · 원래 후보 좌표와 현재 내 성 위치를 다시 확인합니다.')
            return False
        if (getattr(self,'prevalidated_bulk',False) and getattr(self,'expected_origin_name',None) and
                city_key(staging['name'])!=city_key(self.expected_origin_name)):
            self.prevalidated_bulk=False
            self.log('집결 기점 이름이 검증한 출발 성과 달라 계획 취소');self.map();return False
        selected_view=self.image.copy()
        self.map()
        transform=self.vision.muster_transform(reference,self.image)
        if transform is None:
            self.log('집결 보류: 도시 선택 중 지도 위치가 변경되었습니다.'); return False
        selected_transform=self.vision.muster_transform(selected_view,self.image)
        if selected_transform is None:
            self.log('집결 기점 선택 후 지형 좌표 확인 실패');return False
        origin_position=(self.vision.transform_point(selected_transform,staging['position']) if staging.get('position')
                         else self.vision.transform_point(transform,origin_position))
        target=self.vision.transform_point(transform,target)
        sources=[(self.vision.transform_point(transform,pos),troops) for pos,troops in sources]
        fraction=self.config['max_send_fraction']
        bulk=bool(getattr(self,'prevalidated_bulk',False));self.prevalidated_bulk=False
        policy='fewest' if bulk else 'single'
        donors=[MusterDonor(tuple(pos),int(troops),math.dist(pos,origin_position))
                for pos,troops in sources if math.dist(pos,origin_position)>50]
        plan=plan_muster(needed,staging['troops'],donors,fraction,policy,quantum=.1)
        if plan is not None and not plan.contributions: return False
        if bulk and plan is None:
            self.log('실제 병력으로 전체 집결 계획을 완성하지 못해 탐색 없이 다른 공격 후보 선택');return False
        self.image.save(self.data/'muster-view.png')
        self.muster={'version':4 if bulk else 3,'single_transfer':not bulk,'bulk_once':bulk,'strict_amount':bulk,'status':'discovering' if plan is None else 'preparing','target':target,'staging':origin_position,
                     'staging_name':staging['name'],'staging_before':staging['troops'],
                     'reserve':math.ceil(staging['troops']*(1-fraction)),'required':needed,'before':before,
                     'target_name':getattr(self,'last_scout_city',None),'scan_step':0,
                     'target_owner':getattr(self,'last_scout_owner',None),
                     'gather_required':sum(c.troops for c in plan.contributions) if plan else None,
                     'policy':policy,'started':time.time(),'arrival':time.time(),
                     'learning_choices':list(getattr(self,'attack_choices',[])),
                     'root_choice':getattr(self,'rl_root_choice',None),
                     'origin_choice':getattr(self,'rl_origin_choice',None),
                     'attack_margin':getattr(self,'attack_margin',self.config.get('combat_margin',1.5)),
                     'moves':[], 'candidates':[{'position':d.position,'troops':d.troops} for d in donors],
                     'donors':[{'position':c.position,'troops':c.troops} for c in plan.contributions] if plan else []}
        if getattr(self,'outcome_policy',None) and plan is not None:
            plan=self.muster_plan()
            if plan is None:
                self.muster=None
                self.log('집결 정책의 전체 병력 검증 실패 · 병력 이동 없이 다른 후보 확인');return False
            self.muster['donors']=[{'position':c.position,'troops':c.troops} for c in plan.contributions]
            self.muster['gather_required']=sum(c.troops for c in plan.contributions)
        if bulk:
            m=self.muster
            names=[(self.vision.transform_point(transform,pos),name) for pos,name in getattr(self,'verified_source_names',{}).items()]
            for d in m['donors']:
                d['expected_name']=next((name for pos,name in names if math.dist(pos,d['position'])<45),None)
            m['fixed_reference']='muster-anchor.png';self.image.save(self.data/m['fixed_reference'])
            m['anchors']={key:list(m[key]) for key in ('target','staging')}
            m['anchor_matrix']=np.eye(3).tolist()
            for collection in ('donors','candidates'):
                for item in m[collection]:item['anchor_position']=list(item['position'])
        self.save_muster()
        if plan:
            self.log(f"{staging['name']} 기점 고정 · 부족분 {pretty(self.muster['gather_required'])}을 {len(plan.contributions)}개 성에서 집결")
        else:
            self.log(f"{staging['name']} 기점 고정 · 필요 공격 병력 {pretty(needed)}, 현재 화면 밖의 인접 지원 성 추가 탐색")
        self.emit({'muster':f"{staging['name']} 집결 계획 유지 · 지원 성 {'탐색' if plan is None else '이동 준비'}"})
        return True

    def focus_muster(self,position,selecting=False):
        """Pan using successive observed terrain transforms, including move-target mode."""
        if not selecting:
            self.map(); self.rebase_muster()
        for _ in range(14):
            x,y=position()
            if 250<x<1020 and 300<y<450: return True
            dx=max(-110,min(110,640-x));dy=max(-70,min(70,330-y))
            if not self.pan_muster(dx,dy):return False
        self.log('집결 좌표를 화면 안으로 가져오지 못해 이번 이동을 보류합니다.')
        return False

    def pan_muster(self,dx,dy):
        self.driver.pan_map(dx,dy);self.wait(.4);self.capture()
        try:self.rebase_muster()
        except RuntimeError:
            # A reverse swipe only moves the camera. No city or send button is
            # touched until terrain registration has succeeded again.
            self.driver.pan_map(-dx,-dy);self.wait(.7);self.capture();self.rebase_muster()
            self.log('집결 탐색 지형 추적이 어려운 구역 · 직전 화면 복구 후 다른 구역 확인')
            return False
        return True

    def muster_plan(self):
        m=self.muster
        base=m.get('planning_base',m['staging_before'])
        sent=sum(move['troops'] for move in m['moves'][m.get('planning_move_index',0):]
                 if move['status'] in ('sent','intent'))
        used=[move['position'] for move in m['moves'] if move.get('position')]
        donors=[MusterDonor(tuple(d['position']),int(d['troops']),math.dist(d['position'],m['staging']))
                for d in m['candidates'] if not d.get('unavailable') and
                not any(math.dist(d['position'],pos)<50 for pos in used)]
        policy=getattr(self,'outcome_policy',None)
        if policy and (m.get('single_transfer') or m.get('bulk_once')):
            variants=({'largest_bulk':'single','nearest_bulk':'single_nearest'} if m.get('bulk_once') else
                      {'nearest_bulk':'single_nearest','largest_bulk':'single'})
            plans={name:plan_muster(m['required'],base+sent,donors,self.config['max_send_fraction'],
                      strategy,quantum=.1,staging_reserve=m['reserve']) for name,strategy in variants.items()}
            if m.get('bulk_once') and not any(plan is not None for plan in plans.values()):
                plans={name:plan_muster(m['required'],base+sent,donors,self.config['max_send_fraction'],
                    strategy,quantum=.1,staging_reserve=m['reserve']) for name,strategy in
                    {'largest_bulk':'fewest','nearest_bulk':'nearest'}.items()}
            feasible=[name for name,plan in plans.items() if plan is not None]
            if not feasible:return None
            context='shortfall:'+bucket(m['required']/max(base,1),(.3,1,3))
            if not m.get('muster_choice'):
                features={name:{'donors':len(plans[name].contributions),
                    'troops':sum(c.troops for c in plans[name].contributions),
                    'needed':m['required'],
                    'distance':sum(math.dist(c.position,m['staging']) for c in plans[name].contributions)} for name in feasible}
                m['muster_choice']=policy.choose('muster_source',context,feasible,features)
            selected=m['muster_choice']['action']
            # Keep the strategy; every planned donor is used at most once.
            return plans.get(selected)
        return plan_muster(m['required'],base+sent,donors,self.config['max_send_fraction'],
                           m['policy'],quantum=.1,staging_reserve=m['reserve'])

    def discover_muster(self):
        m=self.muster
        if m.get('bulk_once') and m['status']=='discovering':
            self.cancel_muster('고정 일괄 계획이 부족해 장시간 지도 탐색 대신 다른 목표 선택');return False
        for pos,troops in self.vision.own_sources(self.image,self.state.get('owner')):
            if math.dist(pos,m['staging'])<55:continue
            existing=next((d for d in m['candidates'] if math.dist(pos,d['position'])<45),None)
            if existing:
                existing['position']=pos;existing['troops']=troops
            else:m['candidates'].append({'position':pos,'troops':troops})
        plan=self.muster_plan()
        if plan is not None:
            m['donors']=[{'position':c.position,'troops':c.troops} for c in plan.contributions]
            m['gather_required']=sum(c.troops for c in plan.contributions)
            self.save_muster('preparing' if plan.contributions else 'waiting')
            self.log(f"{m['staging_name']} 집결 확보: 지원 성 {len(plan.contributions)}곳, 추가 {pretty(m['gather_required'])}")
            return True
        sweep=muster_sweep();step=m.get('scan_step',0)
        if step>=len(sweep):
            available=sum(d['troops']*.3 for d in m['candidates'] if not d.get('unavailable'))
            self.log(f"{m['staging_name']} 주변 {len(sweep)}구역·지원 후보 {len(m['candidates'])}곳 확인, 예비병력 제외 약 {pretty(available)} · 필요 {pretty(m['required'])}")
            self.targets.add(self.image,m['target'],seconds=900) if self.world_position(m['target']) else None
            self.cancel_muster('인접 탐색 범위 전체의 병력이 실제로 부족함');return False
        self.decide('집결 지원 성 탐색',f"기점 {m['staging_name']} 유지 · {step+1}/{len(sweep)}구역, 후보 {len(m['candidates'])}곳")
        self.pan_muster(*sweep[step])
        m['scan_step']=step+1;self.save_muster()
        return True

    def advance_muster(self):
        m=self.muster
        if m.get('single_transfer') and m['moves']:
            self.save_muster('waiting');return True
        if m['status']=='discovering':return self.discover_muster()
        current=self.muster_plan()
        if current is not None and not current.contributions:
            self.save_muster('waiting');return True
        for donor in m['donors']:
            if donor.get('done'):continue
            if any(move.get('position') and math.dist(move['position'],donor['position'])<50 for move in m['moves']):
                donor['done']=True;continue
            if not self.focus_muster(lambda:donor['position']):
                donor['focus_attempts']=donor.get('focus_attempts',0)+1
                if donor['focus_attempts']<3:
                    self.save_muster();self.log(f'지원 성 위치 복구 재시도 {donor["focus_attempts"]}/3 · 병력 전송 없음')
                    return True
                donor['done']=True;break
            self.decide('한 번에 집결 전송',f"{m['staging_name']}에 부족분 {pretty(donor['troops'])}을 한 성에서 전송 준비")
            try:success=self.transfer_donor(donor['position'],donor['troops'])
            except RuntimeError as exc:
                donor['attempts']=donor.get('attempts',0)+1
                self.map();self.save_muster()
                if donor['attempts']<3:
                    self.log(f'지원 성 이동 재확인 {donor["attempts"]}/3 · {exc}');return True
                success=False
            donor['done']=True
            if not success:
                for candidate in m['candidates']:
                    if math.dist(candidate['position'],donor['position'])<50:candidate['unavailable']=True
                self.log('해당 지원 성 이동 실패 · 기점을 유지하고 다른 지원 성 확인')
            self.save_muster()
            if m.get('single_transfer') and m['moves']:self.save_muster('waiting')
            return True
        plan=self.muster_plan()
        if m.get('bulk_once') and m['moves']:
            self.save_muster('waiting')
        elif plan is not None and not plan.contributions:
            self.save_muster('waiting');self.emit({'muster':f"{m['staging_name']} 집결 병력 도착 대기"})
        else:
            self.save_muster('discovering')
        return True

    def prepare_transfer(self,position,requested,mission,rebase,navigate_destination=None):
        """Inspect a free route and selected quantity without sending any troops."""
        if self.config.get('focus_defense') and mission.get('purpose')!='defense':return None
        donor=self.select_owned(position)
        if not donor:
            self.log(f'지원 출발 성 인식 실패: {position}');return None
        if city_key(donor['name'])==city_key(mission['staging_name']):
            self.log('기점 성 자체는 지원 출발 성에서 제외합니다.');self.map();return None
        rebase()
        fraction=mission.get('support_fraction',self.config['max_send_fraction']) if mission.get('purpose')=='defense' else self.config['max_send_fraction']
        if not .1<=fraction<=.7:raise ValueError('지원 병력 비율 범위 오류')
        capacity=math.floor(donor['troops']*(math.floor((fraction+1e-9)*10)/10))
        if capacity<=0: self.map(); return None
        if (mission.get('single_transfer') or mission.get('strict_amount')) and requested>capacity:
            self.log(f"{donor['name']} 실제 동원 가능 {pretty(capacity)} < 필요 {pretty(requested)} · 소량 전송 없이 다른 큰 성 탐색")
            self.map();return None
        requested=min(requested,capacity)
        self.tap(*donor['move_button'])
        prompt=self.region({'prompt':(480,30,805,90)})['prompt']
        if '목표' not in prompt or '선택' not in prompt:
            self.back(); return None
        # The destination-selection screen can settle after the source camera pan.
        rebase()
        visible=[]
        if navigate_destination:
            visible=[pos for pos,_ in self.vision.own_sources(self.image,self.state.get('owner'))
                if self.world_position(pos) and math.dist(pos,donor['position'])>110]
        # Prefer actual visible blue castles. Perspective can put the saved
        # terrain projection outside the screen while the destination is in
        # view. Selecting a city only opens the form; its exact name is still
        # mandatory before any transfer can be submitted.
        if navigate_destination and not visible and not navigate_destination():
            self.back();return None
        if not visible and not self.world_position(mission['staging']):
            self.back(); return None
        normalize=lambda s:re.sub(r'[^a-z0-9]','',s.lower())
        mission['destination_candidates']=[{'position':pos} for pos in
            sorted(visible,key=lambda pos:math.dist(pos,mission['staging']))]
        if self.world_position(mission['staging']):mission['destination_candidates'].append({'position':mission['staging']})
        if navigate_destination and not visible:
            nearby=sorted(self.vision.own_sources(self.image,self.state.get('owner')),
                          key=lambda s:math.dist(s[0],mission['staging']))
            mission['destination_candidates'].extend({'position':pos} for pos,_ in nearby
                if math.dist(pos,mission['staging'])<220 and math.dist(pos,mission['staging'])>25)
        if mission.get('fixed_reference'):
            inverse=np.linalg.inv(np.asarray(mission['anchor_matrix']))
            for candidate in mission['destination_candidates']:
                candidate['anchor_position']=self.vision.transform_point(inverse,candidate['position'])
        verified=False
        for candidate in mission['destination_candidates'][:5]:
            if not self.world_position(candidate['position']):continue
            actual_destination=candidate['position']
            self.tap(*actual_destination,delay=1)
            button=None
            # LDPlayer shows a player/city transition before the troop form.
            # A missing button in this animation is not a failed route.
            for _ in range(8):
                button=self.vision.match(self.image,'send_transfer',.84,area=(830,490,935,590))
                if button:break
                self.wait(.6);self.capture()
            if not button:
                prompt=self.region({'prompt':(480,30,805,90)})['prompt']
                if '목표' not in prompt:self.back();return None
                rebase();continue
            destination=self.region({'city_name':(1012,506,1136,538)})['city_name']
            names=[destination]
            if normalize(destination)!=normalize(mission['staging_name']):
                names.extend(self.region({'city_tight':(1015,506,1100,533),
                                          'city_wide':(1000,500,1160,540)}).values())
            if any(normalize(value)==normalize(mission['staging_name']) for value in names):
                mission['staging']=actual_destination
                if mission.get('fixed_reference'):mission['anchors']['staging']=candidate['anchor_position']
                verified=True;break
            self.image.save(self.data/'transfer-destination-mismatch.png')
            self.log(f"지원 목적지 이름 불일치: {destination} / {mission['staging_name']} · 인접 내 성 재확인")
            self.back();rebase()
        if not verified:
            self.log('집결 목적지 확인 실패 · 다른 지원 경로에서 기점 재확인');self.map();return None
        if not self.vision.match(self.image,'transfer_toggle_off',.87,area=(885,391,950,435)):
            self.log('병력 이동 기본 설정 인식 실패'); self.image.save(self.data/'unrecognized-transfer-options.png')
            self.back(); return None
        selected=self.select_troops(requested,donor['troops'],capacity)
        if not selected:
            self.back(); return None
        troops,duration=selected
        # The second switch disappears on the observed <30s free route. For
        # longer marches it must explicitly be off before sending.
        if duration>=30 and not self.vision.match(self.image,'transfer_toggle_off',.87,area=(885,444,950,488)):
            self.image.save(self.data/'unrecognized-transfer-options.png')
            self.log('이동 보류: 추가 자원 사용 옵션이 꺼져 있지 않습니다.')
            self.back(); return None
        button=self.vision.match(self.image,'send_transfer',.84,area=(830,490,935,590))
        if not button: self.back(); return None
        return {'donor':donor,'troops':troops,'duration':duration,'button':button[:2],
                'observed':time.time()}

    def transfer_donor(self,position,requested):
        self.muster['transfer_origin']=position
        if self.muster.get('fixed_reference'):
            inverse=np.linalg.inv(np.asarray(self.muster['anchor_matrix']))
            self.muster['anchors']['transfer_origin']=self.vision.transform_point(inverse,position)
        prepared=self.prepare_transfer(position,requested,self.muster,self.rebase_muster,
            navigate_destination=lambda:self.focus_muster(lambda:self.muster['staging'],selecting=True))
        if not prepared: return False
        donor=prepared['donor']; troops=prepared['troops']; duration=prepared['duration']
        planned=next((d for d in self.muster['donors'] if math.dist(d['position'],self.muster['transfer_origin'])<50),None)
        if planned and planned.get('expected_name') and city_key(planned['expected_name'])!=city_key(donor['name']):
            self.log('지원 출발 성 이름이 검증 계획과 다름 · 전송 취소');self.back();self.map();return False
        for candidate in self.muster.get('candidates',[]):
            if math.dist(candidate['position'],self.muster['transfer_origin'])<50:
                candidate['troops']=donor['troops'];candidate['name']=donor['name']
        if any(city_key(move['source'])==city_key(donor['name']) for move in self.muster['moves']):
            self.log('동일한 성에서 이미 집결 이동 기록이 있어 중복 이동을 생략합니다.')
            self.back();self.map();return False
        button=prepared['button']
        if self.config.get('focus_defense'):self.back();self.map();return False
        # Journal intent before the click; a crash/unknown input outcome must not duplicate it.
        move={'source':donor['name'],'position':self.muster['transfer_origin'],'troops':troops,
              'arrival':time.time()+duration+10,'status':'intent'}
        if self.muster.get('fixed_reference'):move['anchor_position']=self.muster['anchors']['transfer_origin']
        self.muster['moves'].append(move)
        self.muster['arrival']=max(self.muster['arrival'],move['arrival'])
        self.save_muster()
        self.tap(*button[:2])
        self.wait(1.5); self.capture()
        if not self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)):
            self.log('병력 이동 결과가 불확실합니다. 같은 이동을 반복하지 않습니다.')
            return False
        move['status']='sent'
        self.rebase_muster()
        self.save_muster()
        self.counts['transfers']+=1
        self.log(f"{donor['name']} → {self.muster['staging_name']} {pretty(troops)} 이동 · 약 {duration}초")
        self.record('transfer_sent',{'city':donor['name'],'troops':donor['troops']},
                    {'destination':self.muster['staging_name'],'sent':troops},False,'집결 이동 전송, 점령/VP 획득 아님')
        return True

    def select_troops(self,needed,total,allowed):
        if not total or needed<=0 or needed>allowed: return None
        self.capture()
        # The observed game slider selects one tenth of the army at a time.
        percent=max(.1,math.ceil((needed/total-1e-9)*10)/10)
        for attempt in range(5):
            if percent>allowed/total+1e-6: return None
            handle=self.vision.troop_slider_handle(self.image)
            if handle is None:
                for _ in range(5):
                    self.wait(.6);self.capture();handle=self.vision.troop_slider_handle(self.image)
                    if handle is not None:break
                if handle is None:
                    self.log('병력 수량 조절 손잡이를 확인할 수 없습니다.'); return None
            # Observed detents: 10%=419, 20%=457.5, 40%=534.5.
            # Cross the detent by three pixels; stopping just short leaves the
            # previous quantity selected, and a long gesture can stop early.
            goal=381+385*percent
            end=goal+(3 if goal>handle else -3)
            self.driver.drag_troop_slider(handle,max(360,min(795,end)))
            self.wait(.6); self.capture()
            values=self.region({'send':(323,468,780,511),'time':(323,512,780,538)})
            troops=amount(values['send']); duration=march_seconds(values['time'])
            if (troops is None or troops>allowed*1.001 or troops>total*(percent+.015) or
                    troops>needed+max(total*.101,needed*.015)):
                retried=self.vision.troop_selection_amount(self.image)
                if isinstance(retried,(int,float)) and math.isfinite(retried):troops=retried
            if duration is None:
                retry_time=self.vision.troop_selection_timer(self.image)
                if (isinstance(retry_time,(int,float)) and not isinstance(retry_time,bool) and
                        math.isfinite(retry_time) and 0<=retry_time<=86400):duration=retry_time
            if not troops or duration is None:
                if getattr(self,'data',None):self.image.save(self.data/'unreadable-troop-selection.png')
                self.log(f"병력/도착시간 인식 실패: {values}"); return None
            # Respect source reserves and round up only one selectable game unit.
            if (troops>allowed*1.001 or troops>total*(percent+.015) or
                    troops>needed+max(total*.101,needed*.015)):
                if attempt<4:
                    self.log(f'병력 슬라이더 재조정: 선택 {pretty(troops)} / 필요 {pretty(needed)}')
                    continue
                self.log(f'병력 선택량이 필요량/예비병력 한도를 초과: {pretty(troops)}'); return None
            if troops>=needed*.999:
                return troops,duration
            if troops>=total*(percent-.015): percent+=.1
        return None

    def check_muster(self):
        if self.config.get('focus_defense'):
            if self.muster:self.cancel_muster('집중 방어 · 공격용 집결 중단')
            return False
        if not self.muster: return False
        exclusions=getattr(self,'exclusions',None)
        if exclusions and (exclusions.blocked(self.muster.get('target_owner','')) or
                           exclusions.blocked_city(self.muster.get('target_name',''))):
            self.cancel_muster('제외 유저 대상 집결 취소');return False
        self.map()
        if self.muster.get('version',1)>=2:
            try:self.rebase_muster()
            except RuntimeError:
                self.muster['tracking_failures']=self.muster.get('tracking_failures',0)+1
                if self.muster['tracking_failures']>=3:
                    self.cancel_muster('지형 좌표 연속 확인 실패 · 이미 전송한 병력은 재전송하지 않음');return False
                self.save_muster();self.log('집결 지형 좌표 재확인 · 기점과 이동 기록 유지');return True
            self.muster['tracking_failures']=0
            if self.muster['status'] in ('discovering','preparing'):
                return self.advance_muster()
        elif not self.verify_muster_view():
            self.cancel_muster('도착 대기 중 지도 위치 변경'); return False
        if time.time()<self.muster['arrival']:
            seconds=math.ceil(self.muster['arrival']-time.time())
            self.emit({'muster':f'집결 도착 대기 {seconds}초','log':f'집결 병력 도착 대기 · 약 {seconds}초'})
            self.wait(min(2,max(.3,self.muster['arrival']-time.time()))); return True
        if self.muster.get('version',1)>=2 and not self.focus_muster(lambda:self.muster['staging']):return True
        staging=self.select_owned(self.muster['staging'])
        if not staging or city_key(staging['name'])!=city_key(self.muster['staging_name']):
            staging=self.select_owned_named(self.muster['staging'],self.muster['staging_name'])
        if not staging or city_key(staging['name'])!=city_key(self.muster['staging_name']):
            self.map(); self.cancel_muster('집결 성 확인 실패'); return False
        actual=staging['troops']
        if actual<self.muster['staging_before']:
            self.map(); self.cancel_muster('집결 성 병력 감소'); return False
        self.log(f"집결 성 {staging['name']} 현재 병력 {pretty(actual)} · 도착 후 화면 확인")
        self.map()
        self.rebase_muster()
        self.sync_stats()
        self.rebase_muster()
        if self.muster.get('version',1)>=2 and not self.focus_muster(lambda:self.muster['target']):return True
        defense=self.scout_target(self.muster['target'],force=True)
        if defense is None:
            self.muster['scout_retries']=self.muster.get('scout_retries',0)+1
            if self.muster['scout_retries']<3:
                self.save_muster('waiting')
                self.log(f'집결 후 재정찰 인식 재시도 {self.muster["scout_retries"]}/3 · 추가 병력 이동 없음')
                return True
            self.cancel_muster('재정찰 결과를 확인하지 못함'); return False
        if (self.muster.get('target_name') and getattr(self,'last_scout_city',None) and
            self.muster['target_name']!=self.last_scout_city):
            self.cancel_muster('재정찰 도시 이름이 저장한 공격 목표와 다름');return False
        needed=required_attack(defense,self.state['attack_bonus'],self.muster.get('attack_margin',self.config['combat_margin']))
        # Re-read the staging army AFTER scouting; scouts and enemy actions can change it.
        if self.muster.get('version',1)>=2 and not self.focus_muster(lambda:self.muster['staging']):return True
        staging=self.select_owned(self.muster['staging'])
        if not staging or city_key(staging['name'])!=city_key(self.muster['staging_name']):
            staging=self.select_owned_named(self.muster['staging'],self.muster['staging_name'])
        if not staging or city_key(staging['name'])!=city_key(self.muster['staging_name']):
            self.map(); self.cancel_muster('재정찰 후 집결 성 확인 실패'); return False
        allowed=max(0,staging['troops']-self.muster['reserve'])
        self.map()
        self.rebase_muster()
        mission=self.muster
        if needed>allowed:
            if mission.get('single_transfer') or mission.get('bulk_once'):
                self.cancel_muster('한 번 집결 후 재정찰에서 병력 부족 · 소량 추가 이동은 하지 않음');return False
            if mission.get('version',1)>=2:
                mission['required']=needed;mission['planning_base']=staging['troops']
                mission['planning_move_index']=len(mission['moves'])
                self.save_muster('discovering')
                self.log(f"{mission['staging_name']} 재정찰 후 부족분 추가 집결 · 목표를 유지합니다.")
                return True
            self.cancel_muster('재정찰 방어력 대비 병력 부족. 추가 이동/공격 보류'); return False
        self.record('muster_arrived',{'troops':mission['staging_before']},
                    {'troops':staging['troops'],'needed':needed},True,'실제 집결 성 병력 확인. 점령 성공 아님')
        if mission.get('version',1)>=2 and not self.focus_muster(lambda:self.muster['target']):return True
        sent=self.dispatch_attack(mission['target'],mission['staging'],staging['troops'],
                                  needed,allowed,mission['before'],mission)
        if self.muster is None: return self.defense_busy()
        if sent: self.muster['attack_arrival']=self.pending['arrival']
        self.save_muster('dispatched' if sent else 'cancelled')
        self.muster=None
        self.emit({'muster':'공격 도착 대기' if sent else '집결 후 공격 보류'})
        return bool(sent)
