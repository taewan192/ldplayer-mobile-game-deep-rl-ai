"""Connect verified gameplay outcomes to the next safe decisions."""
from __future__ import annotations
import math
import time
from PIL import Image
from .outcomes import bucket,decision,choice_label
from .planner import required_attack
from .strategy import city_key


class AdaptiveMixin:
    def init_adaptive(self):
        self.outcome_policy=self.learner.outcomes
        self.attack_choices=[]
        self.restore_outcome_attack()

    def restore_outcome_attack(self):
        if self.pending:
            event_id=self.pending.get('episode_id') or (self.pending.get('muster') or {}).get('episode_id')
            entry=self.outcome_policy.get(event_id) if event_id else None
            if entry and entry['reward'] is None:self.pending={**entry['meta']['pending'],'episode_id':event_id}
            return
        if getattr(self,'muster',None):
            # A crash may happen after the attack intent commits but before the
            # gather journal changes from waiting to dispatched. Audit, never
            # re-enter the gather/attack send path in that case.
            entries=self.outcome_policy.pending('attack')
            for entry in entries:
                previous=(entry['meta'].get('pending') or {}).get('muster') or {}
                if previous.get('started')==self.muster.get('started') and previous.get('staging_name')==self.muster.get('staging_name'):
                    self.pending={**entry['meta']['pending'],'episode_id':entry['id']}
                    self.muster=None;return
            unresolved=next((entry for entry in entries if entry['status']=='unverified' and
                time.time()-entry['meta'].get('last_result_check',0)>=45 and entry['meta'].get('pending')),None)
            if unresolved and not self.muster.get('moves'):
                self.cancel_muster('이전 공격의 실제 결과를 먼저 확인 · 아직 전송하지 않은 새 계획 재평가')
                self.pending={**unresolved['meta']['pending'],'episode_id':unresolved['id']}
            return
        now=time.time()
        for episode in self.outcome_policy.pending('attack'):
            meta=episode['meta']
            if now-episode['started']>1800:
                self.outcome_policy.mark(episode['id'],'abandoned');continue
            if now-meta.get('last_result_check',0)<45:continue
            if meta.get('pending'):
                self.pending={**meta['pending'],'episode_id':episode['id']}
                break

    def rank_attack_targets(self,candidates,origin_position,origin):
        policy=getattr(self,'outcome_policy',None)
        if policy is None:return candidates
        levels=self.vision.target_levels(self.image,candidates)
        context='army:'+bucket(origin,(1e7,1e8,1e9,1e10))
        items=[]
        for p in candidates:
            distance=math.dist(p,origin_position)
            action='level:'+bucket(levels.get(tuple(p)),(60,90,110))+'|distance:'+bucket(distance,(180,360))
            choice=decision('attack_target',context,action)
            choice['features']={'distance':distance,'level':levels.get(tuple(p)),'troops':origin}
            report=next((r for r in getattr(self,'rl_target_reports',[]) if tuple(r['point'])==tuple(p)),{})
            choice['features'].update(defense=report.get('defense'),needed=report.get('needed'))
            prior=.15/(1+distance/250)
            if report.get('needed') is not None:
                prior-=.4*report['needed']/max(origin*self.config.get('max_send_fraction',.35),1)
            items.append((p,choice,prior))
        ranked=policy.rank(items)
        self.attack_rank_choices={tuple(p):(choice,estimate) for p,choice,estimate in ranked}
        allowed=[p for p,c,e in ranked if not e['avoid']]
        if ranked:
            p,c,e=next(((p,c,e) for p,c,e in ranked if not e['avoid']),ranked[0])
            if getattr(policy,'deep',None):
                deep=policy.deep
                self.log(f"강화학습 공격 후보 정렬 · {choice_label(c['domain'],c['action'])} / 신경망 갱신 {deep.online.step}회 / {'예측 Q' if deep.online.step else '초기 우선순위'} {e['score']:+.2f}")
            else:
                self.log(f"결과 학습으로 공격 후보 정렬 · {choice_label(c['domain'],c['action'])} / 실전 {e['samples']}건 / 평균 평가 {e['mean']:+.2f}")
        return allowed

    def choose_attack_requirement(self,defense,origin,allowed):
        configured=self.config['combat_margin']
        needed=required_attack(defense,self.state['attack_bonus'],configured)
        policy=getattr(self,'outcome_policy',None)
        if policy is None:return needed,configured
        options={'configured':(needed,configured)}
        safer=configured*1.15
        buffered=required_attack(defense,self.state['attack_bonus'],safer)
        if buffered<=allowed:options['buffered']=(buffered,safer)
        context='need:'+bucket(needed/max(origin,1),(.1,.3,.6))
        choice=policy.choose('attack_margin',context,list(options),
            {a:{'needed':n,'margin':margin,'defense':defense,'troops':origin} for a,(n,margin) in options.items()})
        self.attack_choices.append(choice)
        return options[choice['action']]

    def attack_intent(self,pending,before,troops,origin,duration,mission):
        policy=getattr(self,'outcome_policy',None)
        if policy is None:return None
        choices=list((mission or {}).get('learning_choices') or self.attack_choices)
        root=(mission or {}).get('root_choice') or getattr(self,'rl_root_choice',None)
        if root and root.get('action')=='attack':choices.insert(0,root)
        origin_choice=(mission or {}).get('origin_choice') or getattr(self,'rl_origin_choice',None)
        actual_origin=(mission or {}).get('staging_name') or getattr(self,'expected_origin_name',None)
        if origin_choice and origin_choice.get('origin_name')==actual_origin:choices.insert(1,origin_choice)
        if mission and mission.get('muster_choice') and any(m['status']=='sent' for m in mission['moves']):
            choices.append(mission['muster_choice'])
        if not choices:choices=[decision('attack_target','unknown','unclassified')]
        meta={'pending':pending,'troops_sent':troops,'origin_troops':origin,'duration':duration,
              'target_name':getattr(self,'last_scout_city',None),'target_owner':getattr(self,'last_scout_owner',None)}
        event_id=policy.begin('attack',choices,before,meta,now=(mission or {}).get('started',time.time()))
        if mission:mission['episode_id']=event_id
        view='episode-'+event_id+'.png'
        with Image.open(self.data/'pending-attack-view.png') as source:source.save(self.data/view)
        pending['view']=view
        # Persist the intent before the irreversible send. A restart only audits it.
        policy.mark(event_id,'intent',{'pending':pending})
        if mission and getattr(self,'muster',None) is mission:self.save_muster()
        return event_id

    def learn_attack_result(self,pending,won,actual=None,detail='실제 점령 소유권 확인'):
        policy=getattr(self,'outcome_policy',None)
        if policy is None or not pending.get('episode_id'):return
        event_id=pending['episode_id'];entry=policy.get(event_id)
        if won is None:
            policy.mark(event_id,'unverified',{'last_result_check':time.time()});return
        sent=entry['meta']['troops_sent']
        loss=None
        if won and actual and actual.get('troops') is not None and actual['troops']<=sent*1.05:
            loss=max(0,sent-actual['troops'])
        after={**self.state,'troops_exposed':sent,'troop_loss_lower_bound':loss,
               'loss_measurement':'lower_bound_with_production' if loss is not None else 'unknown'}
        reward=policy.settle(event_id,won,after,detail,hold=won)
        if reward is not None:
            self.log(f"실전 결과 학습 · 공격 {'점령' if won else '패배'} · 평가 {reward:+.2f} · 대상/공격 병력/집결 선택에 반영")
            self.publish_state()

    def learn_city_loss(self,name,detail):
        policy=getattr(self,'outcome_policy',None)
        if policy is None:return
        for episode in policy.holdings():
            if city_key(episode['meta'].get('target_name') or '')!=city_key(name):continue
            capital=sum(i['cost'] for i in self.strategy.state['investments']
                        if city_key(i['name'])==city_key(name) and i['ts']>=episode['started'])
            reward=policy.settle(episode['id'],False,{**self.state,'capital_lost':capital},detail)
            if reward is not None:
                self.log(f"점령 후 성 상실 · 기존 성공 평가를 {reward:+.2f}로 수정 · 투자 골드 손실 {capital:,.0f}")
                self.publish_state()

    def audit_learned_holdings(self):
        if not getattr(self,'outcome_policy',None) or self.pending or self.muster or self.defense_busy():return
        now=time.time()
        for episode in self.outcome_policy.holdings():
            meta=episode['meta']
            if now-episode['updated']<900 or now-meta.get('last_audit',0)<900:continue
            self.outcome_policy.mark(episode['id'],'holding',{'last_audit':now})
            pending=meta['pending'];name=meta.get('target_name')
            if not name:continue
            with Image.open(self.data/pending['view']) as source:reference=source.convert('RGB')
            matrix=self.vision.viewport_transform(reference,self.image)
            if matrix is None:return
            pos=self.vision.transform_point(matrix,pending['reference_target'])
            if not self.world_position(pos):return
            actual=self.select_owned(pos)
            if actual and city_key(actual['name'])==city_key(name):
                if now-episode['started']>=86400:self.outcome_policy.mark(episode['id'],'complete')
            elif self.confirm_city_lost(pos,name):
                self.map();self.hud();self.learn_city_loss(name,'점령 후 실제 새 소유자 확인')
            self.map();return

    def read_attack_result(self,pending):
        """A defeat needs a matching, recent attack report; unreadable != loss."""
        entry=self.outcome_policy.get(pending['episode_id'])
        name=entry['meta'].get('target_name');rival=entry['meta'].get('target_owner')
        if not name or not rival:return None
        self.map();self.tap(1224,670)
        title=self.region({'title':(213,110,570,156)})['title']
        if '보고서' not in title:self.map();return None
        if '세부' in title:self.tap(228,129)
        from .planner import march_seconds
        for row in range(3):
            y=299+row*93
            fields=self.region({'city_name':(405,y-20,537,y+13),'attacker':(848,y-34,1004,y+2),'remaining':(244,y+9,302,y+35)})
            if city_key(fields['city_name'])!=city_key(name):continue
            if city_key(fields['attacker']) not in {city_key(rival),city_key(self.state.get('owner',''))}:continue
            age=march_seconds(fields['remaining'])
            arrival=pending['arrival']-8
            if age is None or abs(time.time()-age-arrival)>75:continue
            self.tap(825,y)
            text=self.vision.ocr.read(self.data/'latest.png','ko')['text']
            compact=''.join(text.split())
            from .outcomes import combat_result
            result=combat_result(compact,name,rival,'공격')
            self.map();return result
        self.map();return None
