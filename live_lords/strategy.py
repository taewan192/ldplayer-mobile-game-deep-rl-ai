"""Persistent delayed outcomes. An upgrade click is an expense, never a win."""
from __future__ import annotations

import json
import time
import math


def city_key(name):
    import re
    return re.sub(r'[^a-z0-9가-힣]','',name.casefold())


class StrategyLedger:
    def __init__(self,path,learner):
        self.path,self.learner=path,learner
        self.state=json.loads(path.read_text('utf-8')) if path.exists() else {'cities':{},'investments':[],'threats':[],'outcomes':[]}

    def save(self):
        temporary=self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state,ensure_ascii=False,indent=2),'utf-8')
        temporary.replace(self.path)

    def observe_city(self,city):
        key=city_key(city['name'])
        self.state['cities'][key]={**self.state['cities'].get(key,{}),**city,'observed':time.time()}
        self.save()

    def threat(self,city,attack,impact):
        # Duplicate countdown observations update one event rather than manufacturing samples.
        key=city_key(city)
        existing=next((t for t in self.state['threats'] if t['city']==key and abs(t['impact']-impact)<10),None)
        if existing: existing['attack']=max(existing['attack'],attack)
        else: self.state['threats'].append({'city':key,'attack':attack,'impact':impact,'ts':time.time()})
        self.state['threats']=self.state['threats'][-200:]
        self.save()

    def reference_attack(self,now=None):
        now=time.time() if now is None else now
        return max((t['attack'] for t in self.state['threats'] if 0<=now-t['ts']<86400),default=None)

    def investment_allowed(self,city,cost,gold,reserve,margin=1.3,now=None):
        now=time.time() if now is None else now
        key=city_key(city['name']); attack=self.reference_attack(now)
        history=self.state['cities'].get(key,{})
        if history.get('loss_until',0)>now: return False,'최근 점령 손실로 투자 보류'
        if any(i['city']==key and i['status']=='intent' for i in self.state['investments']):
            return False,'결과 미확인 강화 지출이 있어 중복 투자 보류'
        limit=.02
        defense=city.get('defense')
        if not isinstance(defense,(int,float)) or not math.isfinite(defense):return False,'도시 총 방어력 재인식 필요'
        if attack is None:
            # Learn with bounded capital in one garrison, without pretending an
            # unobserved enemy is weak or spreading investment across every city.
            values=[city.get(k) for k in ('defense','troops','walls')]
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
                return False,'핵심 성의 병력/성벽/총 방어력 실측 필요'
            if self.state.get('core_city') and self.state['core_city']!=key:
                return False,'위협 기록 확보 전에는 지정 핵심 성에만 투자'
            if any(i['status'] in ('pending','intent') for i in self.state['investments']):
                return False,'핵심 성 시험 투자 결과 확인 대기'
            if sum(now-i['ts']<86400 and i['context']=='upgrade_safety:unmeasured' for i in self.state['investments'])>=2:
                return False,'위협 미관측 단계의 하루 시험 투자 2회 한도'
            limit=.01
        elif city.get('defense',0)*.99<attack*margin: return False,'관측 적 공격력을 버틸 방어력 부족'
        if any(t['city']==key and t['impact']>now for t in self.state['threats']): return False,'공격 도착 전 강화 보류'
        # Bound both each expense and unresolved investment in this city.
        unresolved=sum(i['cost'] for i in self.state['investments'] if i['city']==key and i['status']=='pending')
        if gold is None or cost is None or cost<=0 or cost>gold-reserve or unresolved+cost>gold*limit:
            return False,f'골드 예비금/시험 투자 {limit:.0%} 한도'
        return True,('관측 위협 대비 방어력과 투자 한도 충족' if attack is not None else '위협 미관측: 핵심 성 한 곳에 한정한 1% 시험 투자')

    def invest(self,city,cost,context,before):
        entry={'id':str(time.time_ns()),'city':city_key(city),'name':city,'cost':cost,'context':context,
               'before':before,'ts':time.time(),'status':'intent'}
        deep=getattr(self.learner,'deep',None)
        if deep is not None and isinstance(getattr(deep,'root_choice',None),dict) and deep.root_choice.get('action')=='invest':
            entry['rl_choice']=deep.root_choice
        self.state['investments'].append(entry); self.save()
        if context=='upgrade_safety:unmeasured':
            self.state['core_city']=entry['city']; self.save()
        return entry

    def outcome(self,city,held,detail,now=None,event_id=None):
        now=time.time() if now is None else now
        if event_id and event_id in self.state['outcomes']: return False
        key=city_key(city)
        for entry in self.state['investments']:
            if entry['city']!=key or entry['status'] not in (('pending',) if held else ('pending','held')): continue
            # Real VP attribution is unavailable. Held is a delayed ownership proxy;
            # lost capital is charged explicitly and is not hidden by unrelated VP gains.
            reward=(.5 if held else -2)-min(2,entry['cost']/max(entry['before'].get('gold',entry['cost']),1)*(10 if held else 100))
            self.learner.record(entry['context'],'upgrade',reward,entry['before'],
                                {'held':held,'gold_spent':entry['cost'],'capital_lost':0 if held else entry['cost'],
                                 'city_vp_at_investment':entry['before'].get('city_vp')},detail)
            if entry.get('rl_choice') and getattr(self.learner,'deep',None):
                # Retention is only an ownership proxy. Cost is always charged;
                # it is not recorded as a won battle or a passive-income win.
                self.learner.deep.investment(entry,held,(-min(2,entry['cost']/max(entry['before'].get('gold',entry['cost']),1)*20)) if held else reward)
            entry['status']='held' if held else 'lost'; entry['resolved']=now
        if not held:
            self.state['cities'].setdefault(key,{})['loss_until']=now+86400
        if event_id: self.state['outcomes']=(self.state['outcomes']+[event_id])[-500:]
        self.save(); return True
