"""Live environment adapter and hierarchical skill scheduler for visual RL."""
import math
import time

from .deep_rl import DeepPolicy
from .outcomes import decision


class ReinforcementMixin:
    def reinforcement_policy(self):
        deep=getattr(getattr(self,'learner',None),'deep',None)
        return deep if isinstance(deep,DeepPolicy) else None

    def reinforcement_observation(self):
        state={**self.state}
        state.update(gold_reserve=self.config.get('gold_reserve'),
                     max_send_fraction=self.config.get('max_send_fraction'),
                     focus_defense=self.config.get('focus_defense',False),
                     pending_attacks=int(bool(self.pending)))
        active=[j for j in self.defenses if j['impact']>time.time() and j['status'] not in ('complete','unknown','superseded')]
        state['incoming_troops']=sum(r['troops'] for j in active for r in j.get('incoming_rows',[]) if isinstance(r.get('troops'),(int,float)))
        state['impact_seconds']=min((j['impact']-time.time() for j in active),default=None)
        state['critical_threats']=sum(isinstance(j['before'].get('level'),(int,float)) and 110<=j['before']['level']<=115 for j in active)
        state['pending_supports']=sum(m['status'] in ('intent','sent') for j in active for m in j['moves'])
        job=getattr(self,'active_defense',None) or getattr(self,'rl_decision_job',None) or self.muster
        if job:
            state.update(city_level=job.get('before',{}).get('level'),
                         city_troops=job.get('before',{}).get('troops',job.get('staging_before')),
                         needed=job.get('needed',job.get('required')),
                         elapsed=time.time()-job['started'])
        return self.image,state

    def reinforcement_tick(self):
        deep=self.reinforcement_policy()
        if not deep:return
        deep.mode=self.config.get('learning_mode','train')
        # Training never runs inside march OCR, transfer setup or defense work.
        # One small CPU minibatch per idle cycle keeps incoming polls responsive.
        if not self.pending and not self.muster and not self.defense_busy() and not self.vision.incoming_button(self.image):
            deep.train_step()
        if time.time()-getattr(self,'rl_last_publish',0)>10:
            self.publish_state();self.rl_last_publish=time.time()

    def reinforcement_progress(self):
        """The policy chooses skills; verified executors choose actual UI inputs."""
        deep=self.reinforcement_policy()
        if self.pending or self.muster or self.defense_busy() or self.incoming():return
        deep.mode=self.config.get('learning_mode','train')
        now=time.time()
        if now<max(getattr(self,'exploration_pause_until',0),getattr(self,'alignment_retry_at',0)):return
        self.map()
        sources=self.vision.own_sources(self.image,self.state.get('owner'))
        candidates=self.vision.enemy_candidates(self.image) if self.config.get('attack_enabled') else []
        candidates=[p for p in candidates if not self.targets.blocked(self.image,p,self.vision)]
        self.state.update(own_cities=len(sources),own_troops=sum(n or 0 for _,n in sources),
                          origin_troops=max((n or 0 for _,n in sources),default=0),enemy_candidates=len(candidates))
        choices=[];priors=[]
        retry_view=getattr(self,'rl_attack_view',None)
        cooling=(now<getattr(self,'rl_attack_retry_at',0) and retry_view is not None and
                 self.vision.same_viewport(self.image,retry_view))
        if sources and candidates and not cooling:
            choices.append(decision('strategy','map','attack'));priors.append(1.)
        if self.config.get('upgrade_enabled') and now-getattr(self,'last_wide_upgrade_check',0)>=120 and sources:
            gold=self.state.get('gold')
            if isinstance(gold,(int,float)) and gold>self.config.get('gold_reserve',0):
                choices.append(decision('strategy','map','invest'));priors.append(.1)
        if self.config.get('attack_enabled') and self.config.get('autonomous',True):
            choices.append(decision('strategy','map','explore'));priors.append(.2)
        if not choices:choices=[decision('strategy','map','wait')];priors=[0.]
        choice,q=deep.order(choices,priors)[0]
        deep.root_choice=choice
        self.rl_root_choice=choice
        self.state['rl_choice']={'skill':choice['action'],'q':q,'epsilon':deep.epsilon,'trained':deep.online.step>0}
        self.decide('강화학습 판단',{'attack':'정찰 결과 비교 → 출병 또는 병력 집결',
            'invest':'방어·골드 한도 내 핵심 성 투자 검토','explore':'저장된 8방향 순서로 다음 구역 탐색',
            'wait':'실행 가능한 행동이 없어 침입 감시'}[choice['action']]+f" · {'Q' if deep.online.step else '초기 우선순위'} {q:+.3f}")
        investment_before=len(self.strategy.state['investments'])
        try:
            if choice['action']=='attack':
                outcome=self.attack_nearby()
                if outcome in ('unknown_view','camera_unstable'):
                    self.alignment_retry_at=time.time()+2
                elif not self.pending and not self.muster and outcome!='defense':
                    # The observed lack of a feasible attack masks this same
                    # viewport briefly, avoiding a scout/assembly retry loop.
                    self.rl_attack_retry_at=time.time()+20
                    self.rl_attack_view=self.image.copy()
            elif choice['action']=='invest':
                self.last_wide_upgrade_check=now;self.upgrade()
            elif choice['action']=='explore':self.explore('강화학습 정책이 다음 구역 탐색 선택')
        finally:
            invested=len(self.strategy.state['investments'])>investment_before
            if not (self.pending or self.muster or invested):
                deep.finish_navigation(choice,time.time()-now)
