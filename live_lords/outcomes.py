"""Local outcome-based contextual policies. UI events never train these models.

Each persisted episode credits the decisions actually used, once. Global HUD
changes are explicitly correlated observations, not exact combat attribution.
Unverified battles remain pending; later confirmed city loss replaces a win.
"""
from __future__ import annotations
import json
import math
import time
import uuid
from .strategy import city_key


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def bucket(value,edges):
    if not finite(value):return 'unknown'
    return str(sum(value>=edge for edge in edges))


def decision(domain,context,action):
    return {'domain':domain,'context':context,'action':action}


def choice_label(domain,action):
    labels={'configured':'설정된 여유 병력','buffered':'여유 병력 15% 추가',
            'nearest_bulk':'부족분 전체를 보낼 수 있는 가까운 성','largest_bulk':'부족분 전체를 보낼 수 있는 병력 많은 성',
            'nearest':'가까운 지원 성부터','largest':'병력 많은 지원 성부터','hold':'현재 주둔 병력으로 방어'}
    if action in labels:return labels[action]
    if domain=='attack_target' and '|distance:' in action:
        level,distance=action.split('|distance:',1);level=level.split(':',1)[-1]
        return {'0':'성 Lv.1~59','1':'성 Lv.60~89','2':'성 Lv.90~109','3':'성 Lv.110 이상'}.get(level,'성 레벨 미확인')+' · '+{'0':'가까운 목표','1':'중간 거리 목표','2':'먼 목표'}.get(distance,'거리 미확인')
    if domain=='defense_route' and ':' in action:
        route,margin=action.split(':',1)
        return ('빠른 지원 우선' if route=='fastest' else '지원 성 수 최소')+' · '+('추가 여유 병력' if margin=='buffered' else '설정 방어 여유')
    return action


def combat_result(text,city,rival,role):
    compact=''.join(text.split())
    if role not in compact or '정찰' in compact:return None
    if not city or not rival or city_key(city) not in city_key(compact) or city_key(rival) not in city_key(compact):return None
    won=any(word in compact for word in ('승리',role+'성공'))
    lost=any(word in compact for word in ('패배',role+'실패'))
    return won if won!=lost else None


def economic_reward(before,after,won,elapsed=0,confounded=False):
    if not isinstance(won,bool):raise ValueError('A verified outcome is required')
    components={'battle':1. if won else -2.,'vp':0.,'gold':0.,'troops':0.,'time':-min(.4,max(0,elapsed)/3600)}
    observed={'economy_scope':'global_window_correlated','confounded':confounded}
    for resource in ('vp','gold'):
        a,b=before.get(resource),after.get(resource)
        delta=b-a if finite(a) and finite(b) and a>=0 and b>=0 else None
        observed[resource+'_delta']=delta
        if delta is None:continue
        scale=max(a*(.01 if resource=='vp' else .05),1e5 if resource=='vp' else 1e6)
        if resource=='vp':
            # Long/overlapping windows cannot attribute passive VP growth to us.
            credit=min(1,delta/scale) if won and not confounded and elapsed<=300 else 0
            components['vp']=.6*(max(-2,delta/scale) if delta<0 else max(0,credit))
        else:
            # Passive income never earns a reward. Verified lost investment is
            # charged even when unrelated income hides the fall in the HUD.
            capital=after.get('capital_lost',0)
            decline=max(0,-delta,capital if finite(capital) else 0)
            components['gold']=-min(2,decline/scale)
    if observed['gold_delta'] is None and finite(after.get('capital_lost')):
        components['gold']=-min(2,after['capital_lost']/max(before.get('gold',0) or 0,1e6)*20)
    loss,exposed=after.get('troop_loss_lower_bound'),after.get('troops_exposed')
    if finite(loss) and finite(exposed) and exposed>0:
        components['troops']=-min(1,max(0,loss)/exposed)
    observed['troop_loss_lower_bound']=loss if finite(loss) else None
    observed['components']=components
    return max(-6,min(3,sum(components.values()))),observed


class OutcomePolicy:
    def __init__(self,db,deep=None):
        self.db,self.deep=db,deep
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS outcome_models(domain TEXT,context TEXT,action TEXT,n INTEGER,mean REAL,PRIMARY KEY(domain,context,action))')
            db.execute('CREATE TABLE IF NOT EXISTS outcome_episodes(id TEXT PRIMARY KEY,kind TEXT,status TEXT,started REAL,updated REAL,choices TEXT,before_json TEXT,meta_json TEXT,after_json TEXT,reward REAL,result INTEGER,detail TEXT)')

    def estimate(self,choice,prior=0):
        row=self.db.execute('SELECT n,mean FROM outcome_models WHERE domain=? AND context=? AND action=?',
                            (choice['domain'],choice['context'],choice['action'])).fetchone()
        n,mean=row or (0,0.)
        total=self.db.execute('SELECT COALESCE(SUM(n),0) FROM outcome_models WHERE domain=? AND context=?',
                              (choice['domain'],choice['context'])).fetchone()[0]
        score=mean+.45*math.sqrt(math.log(total+2)/(n+1))+prior
        return {'samples':n,'mean':mean,'score':score,'avoid':n>=3 and mean<-.75}

    def choose(self,domain,context,allowed,features=None):
        """Caller must filter capacity, exclusions, reserves and deadlines first."""
        if not allowed:raise ValueError('No feasible actions')
        if self.deep:return self.deep.choose(domain,context,allowed,features)
        choices=[decision(domain,context,a) for a in allowed]
        rows=[self.estimate(c) for c in choices]
        # Cold actions are explored only inside the caller's safe feasible set.
        unseen=[i for i,r in enumerate(rows) if r['samples']==0]
        i=unseen[0] if unseen else max(range(len(rows)),key=lambda i:rows[i]['score'])
        return choices[i]

    def rank(self,items):
        """items = [(payload, decision descriptor, small cold-start prior)]."""
        if self.deep and items:
            # Preserve distinct candidate coordinates/features even when their
            # old level/distance buckets had the same string action.
            choices=[dict(choice,_rank_index=i) for i,(_,choice,_) in enumerate(items)]
            ranked=self.deep.order(choices,[prior for _,_,prior in items])
            return [(items[c['_rank_index']][0],c,{'samples':0,'mean':q,'score':q,'avoid':False,'neural':True}) for c,q in ranked]
        ranked=[(payload,choice,self.estimate(choice,prior)) for payload,choice,prior in items]
        return sorted(ranked,key=lambda row:row[2]['score'],reverse=True)

    def begin(self,kind,choices,before,meta=None,event_id=None,now=None):
        now=time.time() if now is None else now
        event_id=event_id or uuid.uuid4().hex
        if kind not in ('attack','defense'):raise ValueError('Unknown episode kind')
        unique=list({(c['domain'],c['context'],c['action']):c for c in choices}.values())
        if not unique:raise ValueError('An episode needs an actual decision')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO outcome_episodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (event_id,kind,'intent',now,now,json.dumps(unique),json.dumps(before),json.dumps(meta or {}),None,None,None,''))
        return event_id

    def get(self,event_id):
        row=self.db.execute('SELECT id,kind,status,started,updated,choices,before_json,meta_json,after_json,reward,result,detail FROM outcome_episodes WHERE id=?',(event_id,)).fetchone()
        if not row:return None
        keys=('id','kind','status','started','updated','choices','before','meta','after','reward','result','detail')
        entry=dict(zip(keys,row))
        for key in ('choices','before','meta','after'):
            if entry[key] is not None:entry[key]=json.loads(entry[key])
        return entry

    def mark(self,event_id,status,meta=None):
        if status not in ('intent','sent','unverified','abandoned','holding','complete'):raise ValueError('Unknown episode status')
        entry=self.get(event_id)
        if not entry:return
        if entry['reward'] is not None and status not in ('holding','complete'):return
        merged={**entry['meta'],**(meta or {})}
        with self.db:self.db.execute('UPDATE outcome_episodes SET status=?,updated=?,meta_json=? WHERE id=?',
                                     (status,time.time(),json.dumps(merged),event_id))

    def settle(self,event_id,won,after,detail,now=None,hold=False):
        if won is None:
            self.mark(event_id,'unverified');return None
        now=time.time() if now is None else now
        entry=self.get(event_id)
        if not entry:raise ValueError('Unknown episode')
        # A confirmed later loss may replace a provisional conquest reward.
        correction=entry['reward'] is not None
        if correction and not (entry['status']=='holding' and entry['result']==1 and won is False):return None
        overlap=self.db.execute('SELECT COUNT(*) FROM outcome_episodes WHERE id<>? AND started<=? AND (reward IS NULL OR updated>=?) AND status<>?',
                                (event_id,now,entry['started'],'abandoned')).fetchone()[0]
        reward,signals=economic_reward(entry['before'],after,won,now-entry['started'],confounded=overlap>0)
        final={**after,'signals':signals,'revision':correction}
        with self.db:
            for c in entry['choices']:
                key=(c['domain'],c['context'],c['action'])
                n,mean=self.db.execute('SELECT n,mean FROM outcome_models WHERE domain=? AND context=? AND action=?',key).fetchone() or (0,0.)
                if correction:
                    if not n:raise RuntimeError('Missing model for correction')
                    mean+=(reward-entry['reward'])/n
                else:n+=1;mean+=(reward-mean)/n
                self.db.execute('INSERT OR REPLACE INTO outcome_models VALUES(?,?,?,?,?)',(*key,n,mean))
            self.db.execute('UPDATE outcome_episodes SET status=?,updated=?,after_json=?,reward=?,result=?,detail=? WHERE id=?',
                ('holding' if hold and won else 'complete',now,json.dumps(final),reward,int(won),detail,event_id))
            if self.deep:self.deep.credit({**entry,'updated':now},reward,revision=correction)
        return reward

    def pending(self,kind=None):
        query='SELECT id FROM outcome_episodes WHERE reward IS NULL AND status IN (?,?,?)'
        args=['intent','sent','unverified']
        if kind:query+=' AND kind=?';args.append(kind)
        return [self.get(row[0]) for row in self.db.execute(query,args)]

    def correct_resource_observation(self,event_id,resource,value,evidence):
        """Repair a proven OCR error without creating another battle/sample.

        The caller must re-read the original result frame and retain its
        evidence. Combat result, elapsed time and overlapping observations are
        preserved; only this observed resource and its resulting credit change.
        """
        if resource not in ('gold','vp') or not finite(value) or value<0 or not evidence:
            raise ValueError('A verified resource value and evidence are required')
        entry=self.get(event_id)
        if not entry or entry['reward'] is None or not entry['after']:
            raise ValueError('A verified episode is required')
        after=dict(entry['after']);previous=after.get(resource)
        if previous==value:return entry['reward']
        after[resource]=value
        reward,signals=economic_reward(entry['before'],after,bool(entry['result']),
            entry['updated']-entry['started'],confounded=after['signals']['confounded'])
        after['signals']=signals
        after['observation_corrections']=[*after.get('observation_corrections',[]),
            {'resource':resource,'previous':previous,'verified':value,'evidence':evidence,'ts':time.time()}]
        with self.db:
            for choice in entry['choices']:
                key=(choice['domain'],choice['context'],choice['action'])
                row=self.db.execute('SELECT n,mean FROM outcome_models WHERE domain=? AND context=? AND action=?',key).fetchone()
                if not row or not row[0]:raise RuntimeError('Missing model for observation correction')
                n,mean=row
                self.db.execute('UPDATE outcome_models SET mean=? WHERE domain=? AND context=? AND action=?',
                    (mean+(reward-entry['reward'])/n,*key))
            self.db.execute('UPDATE outcome_episodes SET after_json=?,reward=? WHERE id=?',
                (json.dumps(after),reward,event_id))
            if self.deep:self.deep.credit(entry,reward,revision=True)
        return reward

    def holdings(self):
        return [self.get(row[0]) for row in self.db.execute("SELECT id FROM outcome_episodes WHERE status='holding'")]

    def summary(self):
        episodes,evaluated,reward=self.db.execute('SELECT COUNT(*),COUNT(reward),COALESCE(SUM(reward),0) FROM outcome_episodes').fetchone()
        models=[dict(domain=d,context=c,action=a,samples=n,mean=m) for d,c,a,n,m in self.db.execute('SELECT * FROM outcome_models')]
        recent=[{'id':i,'kind':k,'result':bool(r),'reward':v,'detail':d,'signals':json.loads(a).get('signals',{})}
                for i,k,r,v,d,a in self.db.execute('SELECT id,kind,result,reward,detail,after_json FROM outcome_episodes WHERE reward IS NOT NULL ORDER BY updated DESC LIMIT 8')]
        return {'episodes':episodes,'evaluated':evaluated,'pending':self.db.execute("SELECT COUNT(*) FROM outcome_episodes WHERE reward IS NULL AND status IN ('intent','sent','unverified')").fetchone()[0],
                'net_reward':reward,'revisions':self.db.execute("SELECT COUNT(*) FROM outcome_episodes WHERE after_json LIKE '%\"revision\": true%'").fetchone()[0],
                'models':models,'recent':recent}
