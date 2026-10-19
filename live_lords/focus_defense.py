"""Defense-only monitoring and generous, explicitly troop-based reinforcements."""
import math
import time

from .outcomes import decision
from .strategy import city_key
from .vision import pretty


def protected_level(level):
    return isinstance(level,(int,float)) and 110<=level<=115


def incoming_troops(row):
    upper=row.get('troops');lower=row.get('troops_min',upper)
    if upper is None or lower is None or not all(math.isfinite(v) and v>0 for v in (upper,lower)) or lower>upper:
        raise ValueError('확인되지 않은 침입 병력 범위')
    return upper if row.get('troops_exact',lower==upper) else upper*.999


def troop_goal(rows,level):
    return math.ceil(sum(incoming_troops(row) for row in rows)*(1.5 if protected_level(level) else 1.3))


class FocusDefenseMixin:
    def troop_defense_enabled(self):
        return self.config.get('focus_defense',False) or bool(getattr(self,'reinforcement_policy',lambda:None)())

    def defense_support_enabled(self):
        return self.config.get('focus_defense',False) or (bool(getattr(self,'reinforcement_policy',lambda:None)()) and self.config.get('defense_enabled',True))

    def focus_defense_tick(self):
        if self.muster:self.cancel_muster('집중 방어 전환 · 공격용 집결 중단, 이미 전송한 행군 기록 유지')
        self.state['focus_defense']=True
        if not getattr(self,'defense_read_waiting',False):self.map()
        if not self.state.get('owner'):self.sync_stats()
        self.incoming()
        if not self.vision.incoming_button(self.image) and not getattr(self,'defense_read_waiting',False):
            self.check_defense_results()
            if hasattr(self,'prepare_defense_standby'):self.prepare_defense_standby()
            # Existing attacks are observed only; no new attack is dispatched.
            if self.pending and time.time()>=self.pending['arrival']:self.check_pending()
        if time.time()-getattr(self,'focus_last_hud',0)>=15:
            self.hud();self.publish_state();self.focus_last_hud=time.time()
        self.decide('집중 방어','새 공격·공격용 집결·광역 탐색 중단 · Lv.110~115 우선 보호 · 침입 감시')
        self.wait(max(.25,min(1,float(self.config.get('interval',.25)))))

    def defense_group_priority(self,rows):
        cities=getattr(getattr(self,'strategy',None),'state',{}).get('cities',{})
        level=next((r.get('level') for r in rows if r.get('level') is not None),None)
        if level is None:level=cities.get(city_key(rows[0]['city']),{}).get('level')
        return (not protected_level(level),min(r['impact'] for r in rows))

    def prepare_focus_defense(self,job):
        """Send useful partial support too; never invent an enemy attack bonus."""
        try:goal=troop_goal(job['incoming_rows'],job['before'].get('level'))
        except (ValueError,TypeError,KeyError):
            job['status']='unverified';self.save_defenses()
            self.log('집중 방어 병력 범위 미확인 · 임의 계산/전송 없이 재관측');return
        choice=None
        deep=getattr(self,'reinforcement_policy',lambda:None)()
        if deep:
            choice=job.get('focus_rl_choice')
            if not choice:
                base=1.5 if protected_level(job['before'].get('level')) else 1.3
                options=[f'{route}:{factor}' for route in ('largest','nearest') for factor in (base,base+.3,base+.6)]
                self.rl_decision_job=job
                try:
                    choice=deep.choose('defense_focus','critical' if protected_level(job['before'].get('level')) else 'ordinary',options,
                        {a:{'margin':float(a.split(':')[1]),'duration':job['impact']-time.time(),
                            'troops':job['before']['troops'],'needed':goal} for a in options})
                finally:self.rl_decision_job=None
                job['focus_rl_choice']=choice
            goal=math.ceil(sum(incoming_troops(r) for r in job['incoming_rows'])*float(choice['action'].split(':')[1]))
        self.active_defense=job
        job['purpose']='defense';job['focus_last_attempt']=time.time()
        job['troop_goal']=goal
        job['threat_basis']='troops_only';job['attack']=None
        buffer=5
        try:
            if hasattr(self,'refresh_defense_garrison') and not self.refresh_defense_garrison(job):
                job['status']='unverified';self.save_defenses();return
            garrison=job.get('garrison_observed',job['before']['troops'])
            observed_at=job.get('garrison_at',job.get('started',0))
            supplied=sum(m['troops'] for other in self.defenses
                if city_key(other['staging_name'])==city_key(job['staging_name'])
                for m in other['moves'] if m['status']=='sent' and
                observed_at<m.get('arrival',float('inf'))<job['impact']-buffer)
            needed=max(0,job['troop_goal']-garrison-supplied)
            job['needed']=needed;self.save_defenses()
            self.ensure_defense_episode(job,choice or decision('defense_blind','troops_only','largest'))
            if not needed:
                job['status']='waiting';self.save_defenses()
                self.log(f"집중 방어 {job['staging_name']} · 병력 기준 목표 {pretty(job['troop_goal'])} 확보 · 실제 결과 대기")
                return
            self.log(f"집중 방어 {'핵심 성 ' if protected_level(job['before'].get('level')) else ''}{job['staging_name']} · 침입 병력 기준 목표 {pretty(job['troop_goal'])}, 추가 필요 {pretty(needed)} · 공격력 추정 안 함")
            self.map();self.rebase_defense()
            sources=(self.defense_visible_sources() if hasattr(self,'defense_visible_sources')
                     else self.vision.own_sources(self.image,self.state.get('owner')))
            order=(lambda s:math.dist(s[0],job['staging'])) if choice and choice['action'].startswith('nearest:') else (lambda s:-s[1])
            job['donors']=[{'position':p,'troops':n} for p,n in sorted(sources,key=order)
                           if math.dist(p,job['staging'])>55]
            if hasattr(self,'defense_city_inventory'):
                local=job['donors']
                inventory=self.defense_city_inventory()
                self.map();self.rebase_defense()
                minimum=min(needed*.02,1e8)
                listed=[{**c,'from_city_list':True} for c in inventory
                        if city_key(c['name'])!=city_key(job['staging_name']) and c['troops']>=minimum]
                job['donors']=listed[:4]+[d for d in local if d['troops']>=minimum][:4]
            self.save_defenses()
            for donor in job['donors'][:8]:
                if not self.defense_support_enabled() or time.time()+15>=job['impact'] or needed<=0:break
                if donor.get('from_city_list'):
                    city=self.open_defense_list_donor(job,donor)
                    if not city:
                        self.restore_defense_target(job);continue
                    donor['position']=city['position']
                else:
                    if donor.get('position') is None:continue
                    self.rebase_defense()
                    city=(self.read_defense_donor(donor['position']) if hasattr(self,'read_defense_donor')
                          else self.read_owned_details(donor['position']))
                self.rebase_defense()
                if not city:continue
                threatened=city_key(city['name']) in self.threatened_names
                if threatened and not hasattr(self,'defense_donor_capacity'):continue
                if threatened and not self.defense_snapshot_valid(job):break
                if any(city_key(m['source'])==city_key(city['name']) for other in self.defenses
                       if other['impact']>time.time() for m in other['moves'] if m['status'] in ('intent','sent')):continue
                # Keep 90% in another protected castle. Other donors retain 30%.
                fraction=.1 if city.get('level') is None or protected_level(city.get('level')) else .7
                capacity=(self.defense_donor_capacity(city) if hasattr(self,'defense_donor_capacity')
                          else math.floor(city['troops']*fraction))
                requested=min(needed,capacity)
                if requested<=0:continue
                fraction=min(fraction,capacity/city['troops'])
                job['support_fraction']=fraction
                if not self.defense_snapshot_valid(job):
                    self.log('집중 방어 · 공격 구성 변경/종료 · 전송 없이 다시 관측');break
                if (city_key(city['name']) in self.threatened_names and
                        (not hasattr(self,'defense_donor_capacity') or self.defense_donor_capacity(city)<requested)):continue
                self.rebase_defense()
                if donor.get('from_city_list'):
                    job['strict_destination']=True
                    p=self.prepare_transfer(donor['position'],requested,job,self.rebase_defense,
                        navigate_destination=lambda:self.defense_destination_in_view(job))
                else:p=self.prepare_transfer(donor['position'],requested,job,self.rebase_defense)
                if not p:continue
                # The real route, source name, amount and ETA are rechecked at send time.
                unsafe=(not self.defense_support_enabled() or time.time()-p['observed']>5 or
                    city_key(p['donor']['name'])!=city_key(city['name']) or
                    (not self.support_arrival_feasible(p['duration'],job['impact']) if hasattr(self,'support_arrival_feasible')
                     else time.time()+p['duration']+buffer>=job['impact']) or
                    p['troops']>p['donor']['troops']*fraction+1)
                if unsafe:self.back();self.map();continue
                move={'source':city['name'],'troops':p['troops'],'arrival':time.time()+p['duration'],
                      'status':'intent','sent_at':time.time(),'reserve_fraction':1-fraction}
                job['moves'].append(move);self.save_defenses()
                self.tap(*p['button'],delay=.25)
                if hasattr(self,'confirm_defense_dispatch'):
                    if not self.confirm_defense_dispatch(job,move,p):break
                    self.counts['supports']+=1
                    if not move.get('late'):needed=max(0,needed-move['troops'])
                    job['needed']=needed
                    self.restore_defense_target(job);self.save_defenses();continue
                if not self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)):
                    self.log('집중 방어 · 지원 결과 미확인 · 같은 병력 전송 반복 안 함');break
                move['status']='sent';self.counts['supports']+=1
                needed=max(0,needed-move['troops']);job['needed']=needed
                self.map();self.rebase_defense();self.save_defenses()
                self.log(f"집중 방어 지원 {city['name']} → {job['staging_name']} {pretty(move['troops'])} · 도착 여유 {int(job['impact']-move['arrival'])}초")
            job['status']='waiting' if job['moves'] or needed<=0 else 'insufficient'
            if needed>0:
                text=f"{'핵심 성 ' if protected_level(job['before'].get('level')) else ''}{job['staging_name']} 방어 병력 목표 미달 {pretty(needed)} · 보유 병력/도착 기한 부족 · 성 보유 보장 불가"
                self.log(text);self.emit({'alert':text})
            self.save_defenses();self.publish_state()
        finally:
            self.active_defense=None;self.map()
