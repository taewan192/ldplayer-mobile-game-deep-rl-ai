"""Constrained contextual bandit. Stores actual screen transitions, not simulator games."""
import json
import math
import sqlite3
import time
from .outcomes import OutcomePolicy


class Learner:
    def __init__(self,path,neural=False,mode='train'):
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS experience(id INTEGER PRIMARY KEY, ts REAL, context TEXT, action TEXT, reward REAL, before_json TEXT, after_json TEXT, detail TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS policy(context TEXT, action TEXT, n INTEGER, mean REAL, PRIMARY KEY(context,action))')
        self.db.commit()
        self.deep=None
        if neural:
            from .deep_rl import DeepPolicy
            self.deep=DeepPolicy(self.db,path.parent/'reinforcement',mode)
        self.outcomes=OutcomePolicy(self.db,self.deep)

    def choose(self,context,allowed):
        if not allowed: raise ValueError('No allowed actions')
        rows={a:(n,m) for a,n,m in self.db.execute('SELECT action,n,mean FROM policy WHERE context=?',(context,))}
        total=sum(n for n,_ in rows.values())+1
        return max(allowed,key=lambda a: (float('inf') if a not in rows else rows[a][1]+math.sqrt(2*math.log(total+1)/rows[a][0])))

    def record(self,context,action,reward,before,after,detail):
        with self.db:
            self.db.execute('INSERT INTO experience(ts,context,action,reward,before_json,after_json,detail) VALUES(?,?,?,?,?,?,?)',
                            (time.time(),context,action,reward,json.dumps(before),json.dumps(after),detail))
            row=self.db.execute('SELECT n,mean FROM policy WHERE context=? AND action=?',(context,action)).fetchone()
            n,mean=row or (0,0.)
            self.db.execute('INSERT OR REPLACE INTO policy VALUES(?,?,?,?)',(context,action,n+1,mean+(reward-mean)/(n+1)))

    def summary(self):
        count,total=self.db.execute('SELECT COUNT(*),COALESCE(SUM(reward),0) FROM experience').fetchone()
        return {'updates':count,'reward':total,'outcomes':self.outcomes.summary(),
                'reinforcement':self.deep.summary() if self.deep else None,
                'policy':[dict(context=c,action=a,n=n,mean=m) for c,a,n,m in self.db.execute('SELECT * FROM policy')]}

    def close(self):
        try:
            if self.deep:self.deep.save()
        finally:self.db.close()
