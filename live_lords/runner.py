from __future__ import annotations

import json
import re
import time
import math
from subprocess import TimeoutExpired
from pathlib import Path

from .driver import LDPlayer,OCR,GameUnavailable,EmulatorBridgeUnavailable
from .recovery import GameRecoveryMixin
from .learning import Learner
from .planner import attack_power,required_attack
from .vision import Vision,amount,pretty
from .muster import MusterMixin
from .defense import DefenseMixin
from .activity import ActivityMixin
from .targets import TargetMemory
from .strategy import city_key
from .exclusions import PlayerExclusions
from .adaptive import AdaptiveMixin
from .batch import BatchMixin
from .focus_defense import FocusDefenseMixin
from .reinforcement import ReinforcementMixin


class Stopped(Exception): pass


class DefensePreempted(Exception): pass


class NavigationBlocked(RuntimeError): pass


class Bot(ReinforcementMixin,FocusDefenseMixin,GameRecoveryMixin,ActivityMixin,DefenseMixin,MusterMixin,AdaptiveMixin,BatchMixin):
    def __init__(self,config,data,stop,emit):
        self.config,self.data,self.stop,self.emit=config,Path(data),stop,emit
        self.driver=LDPlayer(Path(config['install']),config['name'],self.data)
        self.vision=Vision(OCR(),self.data)
        self.learner=Learner(self.data/'learning.sqlite3',neural=True,mode=config.get('learning_mode','train'))
        self.learner.deep.observer=self.reinforcement_observation
        self.image=None
        self.state={}
        previous=self.data/'status.json'
        if previous.exists():
            try:
                owner=json.loads(previous.read_text('utf-8')).get('state',{}).get('owner')
                if isinstance(owner,str): self.state['owner']=owner
            except (ValueError,OSError): pass
        self.seen={}
        self.targets=TargetMemory(self.data)
        self.exclusions=PlayerExclusions(self.data)
        self.targets.exclusions=self.exclusions
        self.counts={'upgrades':0,'scouts':0,'attacks':0,'conquests':0,'recruit_claims':0,'transfers':0,'supports':0,'defenses':0,'explorations':0}
        self.last_recruit_check=0
        self.last_stats_update=0
        self.pending=None
        self.game_recovering=False
        self.startup_combat_check=True
        self.origin_troops=None
        self.init_muster()
        self.init_adaptive()
        self.init_defense()
        if self.muster and self.muster.get('version',1)<4 and not self.muster.get('moves'):
            self.cancel_muster('검증되지 않은 지도 추정으로 만든 이전 집결 계획 재평가')

    def log(self,text):
        self.emit({'log':text})
        with (self.data/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'ts':time.time(),'text':text},ensure_ascii=False)+'\n')

    def wait(self,seconds):
        focused=self.config.get('focus_defense',False)
        if (seconds<=(.5 if focused else 4) or not self.config.get('defense_enabled',True) or
            getattr(self,'active_defense',None) or getattr(self,'_polling_defense',False)):
            if self.stop.wait(seconds): raise Stopped()
            return
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if self.stop.wait(min(.25 if focused else 2,max(0,deadline-time.monotonic()))): raise Stopped()
            self.capture()
            if self.vision.incoming_button(self.image):
                self.log('방어 알림 감지 → 정찰/탐색 대기 중단, 방어 우선 처리')
                raise DefensePreempted()

    def capture(self):
        if self.stop.is_set(): raise Stopped()
        self.image=self.driver.capture()
        self.emit({'image':str(self.data/'latest.png')})
        return self.image

    def tap(self,x,y,delay=None):
        if self.stop.is_set(): raise Stopped()
        # Protect the march troop column from EVERY caller, including a stale
        # profile/stats sequence interrupted by a different game panel.
        if (680<=x<=820 and 100<=y<=660 and self.vision.match(
                self.image,'march_sort',.86,area=(760,35,850,95))):
            raise NavigationBlocked('행군 목록의 병력 숫자 영역 클릭 차단 · 현재 화면부터 다시 판단')
        self.driver.tap(x,y)
        self.wait(max(.3,.8/self.config.get('decision_speed',2.5)) if delay is None else delay)
        self.capture()

    def back(self):
        if self.stop.is_set(): raise Stopped()
        self.driver.back(); self.wait(max(.24,.6/self.config.get('decision_speed',2.5))); self.capture()

    def region(self,boxes): return self.vision.regions(self.image,boxes)

    def align_view(self,reference):
        for attempt in range(3):
            matrix=self.vision.muster_transform(reference,self.image)
            if matrix is not None: return matrix
            if attempt<2: self.wait(.4); self.capture()
        reference.save(self.data/'alignment-before.png')
        self.image.save(self.data/'alignment-after.png')
        return None

    def map(self):
        # Known panels close normally. Unknown views go to bounded recovery;
        # they never authorize blind purchase/confirmation taps.
        for attempt in range(9):
            panel=self.vision.navigation_panel(self.image)
            if panel:
                if attempt==8: break
                self.log(f'{panel} 닫기 → 지도 복귀')
                self.back(); continue
            if self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)):
                if (self.vision.match(self.image,'upgrade',.84) or self.vision.match(self.image,'scout',.84) or
                        self.vision.match(self.image,'move_troops',.84) or self.vision.match(self.image,'move_troops_city88',.84) or
                        self.vision.match(self.image,'city_info',.9,area=(135,160,1145,615))):
                    if attempt==8: break
                    self.back(); continue
                return
            panel=self.vision.navigation_panel_text(self.image)
            if panel:
                if attempt==8: break
                self.log(f'{panel} 닫기 → 지도 복귀')
                self.back(); continue
            text=self.region({'popup':(330,142,950,185)})['popup']
            if '모집' in text and '보너스' in text:
                button=self.vision.match(self.image,'receive',.88,area=(540,490,745,575))
                pixel=self.image.getpixel((580,521))
                if not button or not (pixel[1]>pixel[0]*1.3 and pixel[1]>100):
                    self.back(); return
                claimed=self.region({'troops':(553,398,666,440)})['troops']
                before=self.state.copy()
                self.tap(640,530,2)
                self.tap(640,570)
                self.counts['recruit_claims']+=1
                self.record('collect_recruiter',before,{**self.state,'claimed':amount(claimed)},True,'수령 버튼과 보상 화면 처리')
                self.log('모집관 보너스 수령: '+claimed)
                self.last_recruit_check=time.time()
                continue
            prompt=''.join(self.region({'prompt':(480,30,805,90)}).get('prompt','').split())
            if '도시선택' in prompt and ('목표' in prompt or '출발' in prompt):
                if attempt==8:break
                self.back();continue
            raise NavigationBlocked('화면을 확인할 수 없어 지도 복귀를 재시도합니다.')
        raise NavigationBlocked('패널 닫기 실패')

    def recover_navigation(self,error):
        """Return without replaying the interrupted action; preserve submitted marches."""
        try:
            if not self.driver.foreground(): return False
            self.capture()
            for attempt in range(3):
                try:
                    self.map()
                    self.seen.clear()
                    self.log(f'화면 복구 완료 → 지도에서 다시 판단: {error}')
                    return True
                except NavigationBlocked as exc:
                    if attempt==2 or str(exc)=='패널 닫기 실패': break
                    if not self.driver.foreground(): return False
                    # Android Back dismisses a menu/confirmation, never accepts
                    # a purchase, deletes an account, or repeats a troop send.
                    self.log('미등록 게임 창 닫기 시도 → 새 화면 확인')
                    self.back()
            self.image.save(self.data/'navigation-unrecognized.png')
        except (RuntimeError,OSError,TimeoutExpired):
            return False
        return False

    def hud(self):
        self.state.update(self.vision.hud(self.image,verify=True))
        self.state['defense_pending']=sum(j['status'] in ('preparing','waiting','insufficient') for j in self.defenses)
        self.state['investment_pending']=sum(i['status'] in ('pending','intent') for i in self.strategy.state['investments'])
        self.state['investment_gold_lost']=sum(i['cost'] for i in self.strategy.state['investments'] if i['status']=='lost')
        self.publish_state()

    def publish_state(self):
        self.emit({'state':self.state.copy(),'counts':self.counts.copy(),'learning':self.learner.summary()})
        (self.data/'status.json').write_text(json.dumps({'state':self.state,'counts':self.counts,'learning':self.learner.summary()},ensure_ascii=False,indent=2),'utf-8')

    def record(self,action,before,after,success,detail):
        # The level reward is an explicit proxy. HUD VP is separately retained, with display rounding.
        dvp=0
        if before.get('vp') is not None and after.get('vp') is not None:
            dvp=(after['vp']-before['vp'])/1e6
        reward=0. if getattr(self,'outcome_policy',None) else float(success)+max(-10,min(10,dvp))
        self.learner.record('live:'+str(self.driver.index),action,reward,before,after,detail)
        self.emit({'learning':self.learner.summary()})

    def sync_stats(self):
        self.map()
        self.tap(70,55)
        # Never continue the fixed profile click sequence on a march/guild/map
        # view, nor overwrite our owner identity from unrelated list text.
        for attempt in range(4):
            title=self.region({'title':(208,108,650,160)})['title']
            if '플레이어정보' in re.sub(r'\s+','',title):break
            if self.vision.match(self.image,'march_sort',.86,area=(760,35,850,95)):
                raise NavigationBlocked('프로필 대신 행군 목록 감지 · 능력치 고정 좌표 클릭 중단')
            if attempt==3:raise NavigationBlocked('플레이어 정보 화면 미확인 · 능력치 고정 좌표 클릭 중단')
            self.wait(.3);self.capture()
        owner=self.region({'name':(553,234,744,264)})['name']
        if not owner.strip():raise NavigationBlocked('플레이어 이름 미확인 · 기존 소유자 유지')
        self.tap(728,120)
        self.tap(424,545)
        title=self.region({'title':(270,113,480,153)})['title']
        if '능력치' not in title: raise RuntimeError('능력치 화면 확인 실패')
        for attempt in range(3):
            try:
                bonuses=self.vision.combat_bonuses(self.image)
                break
            except RuntimeError:
                if attempt==2:raise
                self.wait(.3);self.capture()
                if '능력치' not in self.region({'title':(270,113,480,153)})['title']:
                    raise NavigationBlocked('능력치 재인식 중 화면 변경 · 이전 좌표 클릭 없이 재확인')
        previous={key:self.state.get(key+'_bonus') for key in ('attack','defense')}
        self.state.update({key+'_bonus':value for key,value in bonuses.items()})
        self.state['owner']=owner
        self.last_stats_update=time.time()
        self.state['stats_updated_at']=self.last_stats_update
        if all(previous[key] is not None for key in previous) and previous!=bonuses:
            self.log(f"스탯 변경 반영 · 공격 +{previous['attack']}% → +{bonuses['attack']}% · 방어 +{previous['defense']}% → +{bonuses['defense']}% · 이후 병력 계산에 적용")
        self.log(f"공격 +{self.state['attack_bonus']}%, 방어 +{self.state['defense_bonus']}% 화면에서 확인")
        self.publish_state()
        self.map()

    def incoming(self):
        if self.config.get('defense_enabled',True):
            self._polling_defense=True
            try: return self.poll_defense()
            finally: self._polling_defense=False
        warning=self.vision.has_incoming_warning(self.image)
        if warning:
            self.log('공격 행군으로 의심되는 빨간 타이머 감지: '+warning)
            self.emit({'alert':'침입 의심: 방어 기능이 꺼져 있어 새 공격을 보류합니다.'})
            # Do not guess an unobserved destination or spend gems to gather troops.
            self.wait(10)
            return True
        return False

    def upgrade(self):
        return self.strategic_upgrade()

    def attack_nearby(self):
        if self.config.get('focus_defense'):return 'defense'
        if not self.config['attack_enabled']: return 'disabled'
        self.map()
        if self.incoming():
            self.decide('방어 우선','공격 작업보다 피격 성 지원과 도착 결과 확인을 우선합니다.')
            return 'defense'
        if getattr(self,'outcome_policy',None) and self.config.get('batch_scout',True):return self.attack_screen_batch()
        source_view=self.image.copy()
        sources=self.vision.own_sources(self.image,self.state.get('owner'))
        self.capture()
        transform=self.align_view(source_view)
        if transform is None: return 'unknown_view'
        sources=[(self.vision.transform_point(transform,pos),troops) for pos,troops in sources]
        if not sources: return 'no_sources'
        origin_position,origin=max(sources,key=lambda pair:pair[1])
        candidates=self.vision.enemy_candidates(self.image)
        shielded=getattr(self.vision,'last_shield_exclusions',0)
        if isinstance(shielded,int):self.state['shielded_visible']=shielded
        if not candidates: return 'shielded' if isinstance(shielded,int) and shielded else 'no_targets'
        if getattr(self,'outcome_policy',None):
            self.hud()
            candidates=[p for p in candidates if not self.targets.blocked(self.image,p,self.vision)]
            candidates=self.rank_attack_targets(candidates,origin_position,origin)
        for target in candidates[:5]:
            if self.targets.blocked(self.image,target,self.vision): continue
            if getattr(self,'outcome_policy',None):
                self.attack_choices=[self.attack_rank_choices[tuple(target)][0]]
            self.decide('정찰',f'출발 성 병력 {pretty(origin)}, 적 방어력 확인 후 출병 여부 판단')
            before=self.state.copy()
            defense=self.scout_target(target)
            if defense is None:
                failure=getattr(self,'last_scout_failure','report_unreadable')
                if failure in ('no_scout_button','cooldown','shielded','excluded_player'): continue
                return failure
            transform=self.last_scout_transform
            if transform is None: return 'unknown_view'
            target=self.vision.transform_point(transform,target)
            origin_position=self.vision.transform_point(transform,origin_position)
            sources=[(self.vision.transform_point(transform,pos),troops) for pos,troops in sources]
            if time.time()-self.last_stats_update>60:
                reference=self.image.copy()
                self.sync_stats()
                transform=self.align_view(reference)
                if transform is None: return 'camera_unstable'
                target=self.vision.transform_point(transform,target)
                origin_position=self.vision.transform_point(transform,origin_position)
                sources=[(self.vision.transform_point(transform,pos),troops) for pos,troops in sources]
            # The game slider has 10% detents. A 35% limit only permits 30%.
            allowed=origin*(math.floor((self.config['max_send_fraction']+1e-9)*10)/10)
            needed,self.attack_margin=self.choose_attack_requirement(defense,origin,allowed)
            if needed>allowed:
                if self.begin_muster(target,origin_position,needed,sources,before): return 'muster'
                self.log('공격 보류: 집결 가능 병력으로도 조건을 충족하지 못했습니다.')
                return 'blocked'
            return 'dispatched' if self.dispatch_attack(target,origin_position,origin,needed,allowed,before) else 'blocked'
        return 'cooldown'

    def skip_shielded(self,target):
        if not self.vision.is_shielded(self.image,target):return False
        self.last_scout_failure='shielded'
        self.counts['shield_skips']=self.counts.get('shield_skips',0)+1
        self.decide('실드 성 제외','원형 보호막을 화면에서 확인해 해당 성 클릭·정찰·공격 생략')
        return True

    def scout_target(self,target,force=False):
        if getattr(self,'config',{}).get('focus_defense'):return None
        self.last_scout_failure='camera_unstable'
        self.last_scout_city=None
        self.last_scout_owner=None
        if not self.world_position(target): return None
        original_target=target
        self.map()
        if force and getattr(self,'muster',None) and self.muster.get('fixed_reference'):
            # Terrain projections drift with the 3D camera. Snap only to a
            # nearby observed enemy flag; the exact stored city/owner is then
            # checked below before issuing any fresh scout.
            nearby=[p for p in self.vision.enemy_candidates(self.image) if math.dist(p,target)<110]
            if nearby:
                import numpy as np
                target=min(nearby,key=lambda p:math.dist(p,target))
                self.muster['target']=target
                inverse=np.linalg.inv(np.asarray(self.muster['anchor_matrix']))
                self.muster['anchors']['target']=self.vision.transform_point(inverse,target)
                self.save_muster();original_target=target
        if self.skip_shielded(target):return None
        if self.targets.excluded(self.image,target,self.vision):
            self.last_scout_failure='excluded_player';return None
        self.last_scout_reference=self.image.copy()
        self.last_scout_transform=None
        self.tap(*self.enemy_interaction(target),delay=.8)
        if self.muster:
            self.rebase_muster()
            target=self.muster['target']
        else:
            transform=self.align_view(self.last_scout_reference)
            if transform is None: self.map(); return None
            target=self.vision.transform_point(transform,target)
        scout=self.vision.match(self.image,'scout',.84)
        if not scout:
            entry=self.targets.add(self.last_scout_reference,original_target,seconds=120)
            entry['reason']='no_scout_button'; self.targets.save()
            self.last_scout_failure='no_scout_button'; self.map(); return None
        identity=(self.vision.enemy_identity(self.image,scout,self.muster.get('target_name'),self.muster.get('target_owner'))
                  if force and self.muster else self.vision.enemy_identity(self.image,scout))
        if force and self.muster and identity:
            expected_city=self.muster.get('target_name');expected_owner=self.muster.get('target_owner')
            identity=self.ready_enemy_identity(identity,expected_city,expected_owner)
            if not identity:
                self.last_scout_failure='target_identity_unreadable';self.map();return None
            scout=self.vision.match(self.image,'scout',.84)
            if not scout:
                self.last_scout_failure='no_scout_button';self.map();return None
            owner,name=identity.split(':',1)
            if ((expected_city and city_key(name)!=city_key(expected_city)) or
                    (expected_owner and city_key(owner)!=city_key(expected_owner))):
                self.last_scout_failure='target_identity_mismatch'
                if getattr(self,'data',None):self.image.save(self.data/'muster-rescout-identity.png')
                self.log(f'집결 재정찰 대상 이름/소유자 불일치 · {identity} / {expected_owner}:{expected_city} · 정찰 전송 없이 재관측')
                self.map();return None
        if self.offensive_excluded(identity):
            self.targets.add(self.last_scout_reference,original_target,identity)
            self.last_scout_failure='excluded_player';self.map();return None
        if not identity:
            self.log('상대 닉네임 확인 실패 · 제외 유저 여부를 검증할 수 없어 정찰 보류')
            self.targets.add(self.last_scout_reference,original_target,seconds=120)
            self.last_scout_failure='no_scout_button';self.map();return None
        self.last_scout_owner=identity.split(':',1)[0]
        if not force and self.targets.identity_blocked(identity):
            self.last_scout_failure='cooldown'; self.map(); return None
        entry=self.targets.add(self.last_scout_reference,original_target,identity)
        before=self.state.copy()
        scout_view=self.image.copy()
        self.tap(*scout[:2],delay=12)
        self.counts['scouts']+=1
        if self.muster:
            self.rebase_muster()
            target=self.muster['target']
        else:
            transform=self.align_view(scout_view)
            if transform is None: self.map(); return None
            target=self.vision.transform_point(transform,target)
        self.log('주변 성 정찰 전송. 정찰 완료 후 보고서 확인')
        if self.skip_shielded(target):return None
        self.tap(*self.enemy_interaction(target))
        if self.muster: self.rebase_muster()
        report=self.vision.match(self.image,'report',.84)
        if not report:
            self.last_scout_failure='report_pending'; self.map(); return None
        self.tap(*report[:2])
        values=self.ready_scout_report(self.vision.scout_report(self.image))
        if values is None or not values.get('city','').strip() or not values.get('owner','').strip():
            self.last_scout_failure='report_screen'; self.map(); return None
        defense=values['defense']
        self.last_scout_owner=values.get('owner') or self.last_scout_owner
        if values['owner'] and values['city']:
            entry['identities'].append(city_key(values['owner'])+':'+city_key(values['city']))
        if self.offensive_excluded(str(self.last_scout_owner)+':'+str(values.get('city',''))):
            entry['reason']='excluded_player';self.targets.save()
            self.last_scout_failure='excluded_player';self.map();return None
        if defense is not None: self.last_scout_city=city_key(values['city'])
        entry['defense']=defense; entry['reason']='report_unreadable' if defense is None else 'scouted'
        self.targets.save()
        if defense is None:
            self.last_scout_failure='report_unreadable'
            self.image.save(self.data/'unreadable-scout-report.png')
            self.log(f"정찰 보고서 숫자 인식 실패: {values['raw']!r} · 같은 성 재정찰은 10분간 보류")
        self.record('scout',before,{**self.state,'enemy_defense':defense},defense is not None,'정찰 보고서 총 방어력')
        if defense is not None: self.log(f'정찰 총 방어력: {pretty(defense)}')
        self.map()
        if self.muster: self.rebase_muster()
        else: self.last_scout_transform=self.align_view(self.last_scout_reference)
        return defense

    def offensive_excluded(self,identity):
        exclusions=getattr(self,'exclusions',None)
        if exclusions and (exclusions.blocked_identity(identity) or
                           bool(identity and ':' in identity and exclusions.blocked_city(identity.rsplit(':',1)[-1]))):
            self.log('패널티/제외 유저 · 정찰 및 공격 생략: '+str(identity))
            return True
        return False

    def reject_restricted_attack(self,owner,city):
        reason=self.vision.deployment_restriction(self.image)
        if not reason:return False
        if owner:
            self.exclusions.exclude(owner,reason,city=city)
        for alias in getattr(self,'attack_owner_aliases',()):
            if alias:self.exclusions.exclude(alias,reason,city=city)
        self.image.save(self.data/'restricted-player.png')
        self.log(f'병력 배치 제한 감지 · {owner or "닉네임 미확인"} 출병 취소 및 유저 제외: {reason}')
        self.back();self.map();return True

    def dispatch_attack(self,target,origin_position,origin,needed,allowed,before,mission=None):
        if getattr(self,'config',{}).get('focus_defense'):return False
        owner=(mission or {}).get('target_owner') or getattr(self,'last_scout_owner',None)
        self.attack_owner_aliases={name for name in (owner,getattr(self,'last_scout_owner',None)) if name}
        if self.offensive_excluded(str(owner or '')+':'):return False
        extended=bool(mission and mission.get('version',1)>=2)
        if extended:
            if not self.focus_muster(lambda:mission['staging']):return False
            target,origin_position=mission['target'],mission['staging']
        elif not self.world_position(target) or not self.world_position(origin_position): return False
        self.map()
        if self.world_position(target) and self.skip_shielded(target):return False
        reference=self.image.copy()
        actual=self.select_owned(origin_position)
        expected=(mission or {}).get('staging_name') or getattr(self,'expected_origin_name',None)
        if expected and (not actual or city_key(actual['name'])!=city_key(expected)):
            actual=self.select_owned_named(origin_position,expected)
        if not actual:
            self.log(f'공격 보류: 출발 성 선택/실제 병력 확인 실패 {origin_position}')
            return False
        if (not mission and getattr(self,'expected_origin_name',None) and
                city_key(actual['name'])!=city_key(self.expected_origin_name)):
            self.log('공격 출발 성 이름 불일치 · 다른 성의 병력을 사용하지 않고 재관측');self.map();return False
        if mission and city_key(actual['name'])!=city_key(mission['staging_name']):
            self.log('공격 보류: 실제 출발 성 이름이 저장한 집결 기점과 다릅니다.');self.map();return False
        self.map()
        matrix=(self.vision.muster_transform(reference,self.image) if extended else
                self.vision.viewport_transform(reference,self.image))
        if matrix is None:
            self.log('공격 보류: 출발 성 확인 후 지도 좌표 재정렬 실패'); return False
        target=self.vision.transform_point(matrix,target)
        origin_position=self.vision.transform_point(matrix,origin_position)
        if not extended and (not self.world_position(target) or not self.world_position(origin_position)): return False
        origin=actual['troops']
        allowed=min(allowed,max(0,origin-mission['reserve']) if mission else origin*self.config['max_send_fraction'])
        if needed>allowed:
            self.log('출발 성 실제 병력 재확인 후 예비병력 부족: 공격 보류'); return False
        if self.muster:
            self.rebase_muster()
            target=self.muster['target']; origin_position=self.muster['staging']
        if self.incoming(): return False
        if extended:
            if not self.focus_muster(lambda:mission['target']):return False
            target,origin_position=mission['target'],mission['staging']
        self.image.save(self.data/'pending-attack-view.png')
        attack_reference_target=target
        if self.skip_shielded(target):return False
        self.tap(*self.enemy_interaction(target),delay=.8)
        if self.muster:
            self.rebase_muster()
            origin_position=self.muster['staging']
        if getattr(self,'last_scout_city',None):
            scout=self.vision.match(self.image,'scout',.84)
            identity=self.vision.enemy_identity(self.image,scout,self.last_scout_city) if scout else None
            if not identity or identity.rsplit(':',1)[-1]!=self.last_scout_city:
                self.wait(.5); self.capture()
                if self.muster: self.rebase_muster(); origin_position=self.muster['staging']
                scout=self.vision.match(self.image,'scout',.84)
                identity=self.vision.enemy_identity(self.image,scout,self.last_scout_city) if scout else None
            if not identity or identity.rsplit(':',1)[-1]!=self.last_scout_city:
                self.image.save(self.data/'attack-target-mismatch.png')
                self.log(f'공격 대상 이름 검증 실패: 보고서 {self.last_scout_city} / 지도 {identity} · 출병 보류')
                self.map(); return False
            if self.offensive_excluded(identity):self.map();return False
            observed_owner=identity.split(':',1)[0]
            self.attack_owner_aliases.add(observed_owner)
            if not owner:owner=observed_owner
        attack=self.vision.match(self.image,'attack',.84)
        if not attack:
            self.log('공격 보류: 선택한 성의 공격 버튼을 확인하지 못함'); self.map(); return False
        self.tap(*attack[:2])
        prompt=self.region({'prompt':(480,30,805,90)})['prompt']
        if '출발' not in prompt:
            self.log(f'공격 보류: 출발 성 선택 화면 확인 실패 {prompt!r}'); self.back(); return False
        if extended:
            if not self.focus_muster(lambda:mission['staging'],selecting=True):self.back();return False
            origin_position=mission['staging']
        self.tap(*origin_position,delay=3)
        if self.reject_restricted_attack(owner,getattr(self,'last_scout_city',None)):return False
        selected=self.select_troops(needed,origin,allowed)
        if not selected:
            self.log('필요 병력/예비병력 검증 실패: 공격 취소'); self.back(); return False
        if self.vision.match(self.image,'loss_warning',.82):
            self.log('게임 내 패배 경고 감지: 공격 취소'); self.back(); return False
        troops,duration=selected
        if self.reject_restricted_attack(owner,getattr(self,'last_scout_city',None)):return False
        button=self.vision.match(self.image,'send_attack',.83,area=(830,490,933,590))
        if not button: self.back(); return False
        if self.config.get('focus_defense'):
            self.log('집중 방어 전환 · 새 공격 전송 취소');self.back();return False
        pending={'target':target,'arrival':time.time()+duration+8,'before':before,'muster':mission,
                 'reference_target':attack_reference_target,'view':'pending-attack-view.png',
                 'target_name':getattr(self,'last_scout_city',None)}
        episode_id=self.attack_intent(pending,before,troops,origin,duration,mission)
        if episode_id:pending['episode_id']=episode_id
        self.pending=pending
        self.tap(*button[:2])
        if episode_id:self.outcome_policy.mark(episode_id,'sent')
        if self.muster:
            self.wait(1.5); self.capture(); self.rebase_muster()
            target=self.muster['target']
        self.counts['attacks']+=1
        self.log(f'공격 전송 {pretty(troops)} / 전투력 {pretty(attack_power(troops,self.state["attack_bonus"]))}')
        self.record('dispatch_attack',before,{**self.state,'troops_sent':troops},False,'전송만 기록, 점령 성공 아님')
        self.pending['target']=target
        return True

    def check_pending(self):
        if not self.pending: return False
        if time.time()<self.pending['arrival']:
            self.emit({'log':'공격 도착 대기 · 화면에서 위협 감시 중'})
            self.wait(2)
            return True
        before=self.pending['before']
        if self.pending.get('muster') and not self.pending.get('episode_id') and not self.verify_muster_view():
            self.log('공격 도착 후 지도 위치가 바뀌어 자동 점령 판정을 보류합니다.')
            mission=self.pending['muster']
            self.muster=mission; self.save_muster('unverified'); self.muster=None
            self.learn_attack_result(self.pending,None)
            self.pending=None
            return False
        target=self.pending['target'];expected=self.pending.get('target_name');actual=None
        if (self.pending.get('episode_id') or not self.pending.get('muster')) and self.pending.get('view'):
            from PIL import Image
            with Image.open(self.data/self.pending['view']) as source: reference=source.convert('RGB')
            matrix=self.vision.viewport_transform(reference,self.image)
            projected=self.vision.transform_point(matrix,self.pending['reference_target']) if matrix is not None else None
            if projected is None or not self.world_position(projected):
                actual=self.locate_owned_city(expected) if expected else None
                if actual is None:
                    self.log('지도·보유 도시 목록에서 공격 결과 미확인. 점령 보상을 기록하지 않습니다.')
                    self.learn_attack_result(self.pending,None);self.pending=None;return False
            else:target=projected
        if actual is None:actual=self.select_owned(target)
        if expected and actual and city_key(actual['name'])!=city_key(expected):actual=None
        if actual is None and expected:actual=self.select_owned_named(target,expected)
        if actual is None:
            self.map(); self.capture()
            sources=self.vision.own_sources(self.image,self.state.get('owner'))
            nearby=sorted((pos for pos,_ in sources if math.dist(pos,target)<45),key=lambda pos:math.dist(pos,target))
            if nearby: actual=self.select_owned(nearby[0])
        if actual is None:
            self.wait(.6); self.capture()
            actual=self.vision.selected_own_city(self.image,self.state.get('owner'))
        expected=self.pending.get('target_name')
        if expected and actual and city_key(actual['name'])!=city_key(expected):actual=None
        if actual is None and expected:actual=self.locate_owned_city(expected)
        success=actual is not None
        self.hud()
        if success:
            self.counts['conquests']+=1
            self.log('점령 성공: 도착 후 내 도시 강화 버튼 확인')
        else:
            result=None
            if self.pending.get('episode_id'):
                result=self.read_attack_result(self.pending)
                self.map();self.hud()
            if result is True:
                self.counts['conquests']+=1
                self.record('conquest_result',before,self.state.copy(),True,'도시·유저·시각이 일치하는 공격 승리 보고서')
                self.learn_attack_result(self.pending,True,detail='도시·유저·시각이 일치하는 공격 승리 보고서')
                self.log('공격 승리 보고서 확인 · 실제 전투 결과 학습 반영')
                if mission:=self.pending.get('muster'):
                    self.muster=mission;self.save_muster('completed');self.muster=None
                self.pending=None;self.map();return False
            self.learn_attack_result(self.pending,False if result is False else None,detail='도시·유저·도착 시각이 일치하는 공격 패배 보고서')
            self.log('공격 패배 보고서 확인 · 해당 선택의 평가 하락' if result is False else
                     '공격 결과 미확인: 소유권을 읽지 못해 승패 학습을 보류합니다.')
            if mission:=self.pending.get('muster'):
                self.muster=mission; self.save_muster('cancelled' if result is False else 'unverified'); self.muster=None
            self.pending=None; self.map()
            return False
        self.record('conquest_result',before,self.state.copy(),success,'공격 도착 뒤 화면 확인')
        self.learn_attack_result(self.pending,True,actual)
        if mission:=self.pending.get('muster'):
            delay=min(.5,(time.time()-mission['started'])/1200)
            self.learner.record('muster_route',mission['policy'],0 if getattr(self,'outcome_policy',None) else (1 if success else -1)-delay,
                                mission['before'],self.state.copy(),'집결 경로: 실제 공격 도착 뒤 점령 확인 결과')
            self.muster=mission; self.save_muster('completed'); self.muster=None
        self.pending=None
        self.map()
        return False

    def run(self):
        try:
            initialized=False; recovery_failures=0
            while not self.stop.is_set():
                try:
                    if getattr(self,'game_recovering',False):
                        if not self.recover_game():continue
                        initialized=False
                    self.capture()
                    if getattr(self,'config',{}).get('focus_defense'):
                        self.focus_defense_tick();self.reinforcement_tick();continue
                    # A failed incoming-list read stays on that list. Reopening
                    # it every cycle only repeats inputs and loses the clock.
                    if getattr(self,'defense_read_waiting',False) and self.incoming():
                        self.decide('방어 재인식 대기','방어 창을 유지하고 화면만 다시 확인 중')
                        self.wait(2);continue
                    self.map(); self.hud()
                    if not initialized:
                        if self.config.get('defense_enabled',True) and self.vision.incoming_button(self.image):
                            self.incoming(); self.wait(2); continue
                        self.sync_stats(); self.hud()
                        self.state['initial_vp']=self.state.get('vp')
                        initialized=True
                        self.log('로컬 자동 플레이 시작. 정책과 경험은 인스턴스별로 저장됩니다.')
                    recovery_failures=0
                    if not self.muster:self.check_defense_results()
                    if self.incoming():
                        self.decide('방어 감시','피격 성과 지원 도착 결과 확인 중')
                        self.hud(); self.wait(3); continue
                    if time.time()-getattr(self,'last_stats_update',time.time())>=60:
                        self.sync_stats();self.hud()
                    if getattr(self,'outcome_policy',None):self.restore_outcome_attack()
                    if self.check_pending(): continue
                    if self.check_muster(): continue
                    self.audit_investments()
                    self.audit_learned_holdings()
                    if time.time()-self.last_recruit_check>60:
                        self.last_recruit_check=time.time()
                        self.tap(176,135)
                        self.map()
                    self.reinforcement_tick()
                    self.progress_cycle()
                    self.hud()
                    (self.data/'status.json').write_text(json.dumps({'state':self.state,'counts':self.counts,'learning':self.learner.summary()},ensure_ascii=False,indent=2),'utf-8')
                    self.wait(self.cycle_delay())
                except (GameUnavailable,EmulatorBridgeUnavailable):
                    self.begin_game_recovery()
                    continue
                except DefensePreempted:
                    self.last_defense_poll=0
                    continue
                except (RuntimeError,OSError,TimeoutExpired) as exc:
                    if self.recover_navigation(exc):
                        recovery_failures=0
                        self.decide('화면 복구 완료','지도에서 진행 중인 행군을 확인한 뒤 판단 재개')
                        self.wait(1)
                    else:
                        recovery_failures+=1
                        delay=min(30,2**min(recovery_failures,5))
                        self.decide('화면 복구 대기',f'{delay}초 후 재확인 · 자동 플레이 연결 유지')
                        self.emit({'notice':f'화면 복구 대기: {exc} · {delay}초 후 재시도'})
                        if recovery_failures==1: self.log(f'화면 인식/연결 복구 대기: {exc}')
                        self.recovery_pause(delay)
        except Stopped: self.log('중지됨. 학습 기록 저장 완료')
        except Exception as exc:
            self.log('자동 조작 중단: '+str(exc))
            self.emit({'error':str(exc)})
        finally:
            self.learner.close()
            ocr=getattr(getattr(self,'vision',None),'ocr',None)
            if ocr is not None:ocr.close()
            self.emit({'stopped':True})
