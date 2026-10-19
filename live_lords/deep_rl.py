"""Visual, parameterised Double DQN over verified game skills.

The environment is the live game, not a fabricated combat simulator. Skills
retain UI/identity/deadline checks. A combat trajectory is only admitted to
replay when the existing result verifier confirms it. Navigation transitions
have an explicit time cost, never a scouting/victory bonus.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = 1
PIXELS = 16 * 12 * 3
KEYS = ('vp', 'gold', 'attack_bonus', 'defense_bonus', 'origin_troops',
        'own_troops', 'own_cities', 'enemy_candidates', 'incoming_troops',
        'impact_seconds', 'critical_threats', 'pending_attacks', 'pending_supports',
        'gold_reserve', 'max_send_fraction', 'focus_defense', 'city_level',
        'city_troops', 'needed', 'defense', 'travel_seconds', 'target_level',
        'distance', 'elapsed')
OBS = PIXELS * 2 + len(KEYS) * 2
ACTION = 64
CAPACITY = 12000


def number(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(value)


def pack(value):
    return np.asarray(value, dtype='<f4').tobytes()


def unpack(value):
    return np.frombuffer(value, dtype='<f4').copy()


def action_vector(choice):
    """Stable semantic hashing plus observed continuous candidate features."""
    result = np.zeros(ACTION, np.float32)
    for field in ('domain', 'context', 'action'):
        text = str(choice.get(field, ''))
        for token in [text, *text.replace('|', ' ').replace(':', ' ').split()]:
            digest = hashlib.sha256((field + '=' + token).encode()).digest()
            result[int.from_bytes(digest[:2], 'little') % 48] += .25 if digest[2] & 1 else -.25
    features = choice.get('features', {})
    for i, key in enumerate(('distance', 'level', 'troops', 'needed', 'defense',
                              'duration', 'donors', 'margin')):
        value = features.get(key)
        if number(value):
            result[48 + i] = np.sign(value) * min(1, math.log1p(abs(value)) / 32)
            result[56 + i] = 1
    return result


class QNetwork:
    """Two trainable visual/state/action hidden layers, Huber loss and Adam.

    NumPy keeps deployment local on CPU without a GPU/runtime download. Every
    input, including RGB pixels and temporal differences, participates in
    backpropagation. This is a small MLP visual encoder, not a pretrained CNN.
    """
    def __init__(self, seed=481):
        rng = np.random.default_rng(seed)
        self.weights = {}
        for i, (a, b) in enumerate(((OBS + ACTION, 64), (64, 32), (32, 1))):
            self.weights['w'+str(i)] = (rng.normal(0, math.sqrt(2/a), (a, b))).astype(np.float32)
            self.weights['b'+str(i)] = np.zeros(b, np.float32)
        self.m = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.step = 0

    def forward(self, x, cache=False):
        x = np.asarray(x, np.float32)
        a = np.tanh(x @ self.weights['w0'] + self.weights['b0'])
        b = np.tanh(a @ self.weights['w1'] + self.weights['b1'])
        q = (b @ self.weights['w2'] + self.weights['b2']).ravel()
        return (q, (x, a, b)) if cache else q

    def fit(self, x, target, importance):
        q, (x, a, b) = self.forward(x, True)
        error = q - target
        loss = np.mean(importance * np.where(abs(error) <= 1, .5*error**2, abs(error)-.5))
        d = (np.clip(error, -1, 1)*importance/len(q))[:, None]
        grads = {'w2': b.T @ d, 'b2': d.sum(0)}
        d = (d @ self.weights['w2'].T) * (1-b*b)
        grads.update(w1=a.T @ d, b1=d.sum(0))
        d = (d @ self.weights['w1'].T) * (1-a*a)
        grads.update(w0=x.T @ d, b0=d.sum(0))
        norm = math.sqrt(sum(float((g*g).sum()) for g in grads.values()))
        self.step += 1
        for key, g in grads.items():
            g *= min(1., 5/max(norm, 1e-8))
            self.m[key] = .9*self.m[key] + .1*g
            self.v[key] = .999*self.v[key] + .001*g*g
            self.weights[key] -= .0007*(self.m[key]/(1-.9**self.step))/(np.sqrt(self.v[key]/(1-.999**self.step))+1e-8)
        return float(loss), abs(error)


class DeepPolicy:
    algorithm = 'Visual hierarchical Double DQN + prioritized replay'

    def __init__(self, db, directory, mode='train', seed=481):
        self.db, self.directory = db, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.online, self.target = QNetwork(), QNetwork()
        self.observation = None
        self.previous_pixels = None
        self.observer = None
        self.root_choice = None
        self.last_loss = None
        self.last_td = None
        self.dirty = False
        self.checkpoint_error = None
        self.last_save = time.monotonic()
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS rl_decisions(id TEXT PRIMARY KEY, ts REAL, domain TEXT, choice TEXT, observation BLOB, candidates BLOB, episode TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS rl_replay(id TEXT PRIMARY KEY, episode TEXT, source TEXT, ts REAL, observation BLOB, action BLOB, reward REAL, next_observation BLOB, next_actions BLOB, discount REAL, terminal INTEGER, priority REAL, ready INTEGER, revision INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS rl_meta(key TEXT PRIMARY KEY,value TEXT)')
        self.load()

    def observe(self, image, state):
        if not isinstance(image, Image.Image):
            self.observation = None
            return
        pixels = np.asarray(image.convert('RGB').resize((16, 12), Image.Resampling.BOX), np.float32).ravel()/255
        delta = np.zeros(PIXELS, np.float32) if self.previous_pixels is None else pixels-self.previous_pixels
        numbers, valid = [], []
        for key in KEYS:
            value = state.get(key)
            valid.append(float(number(value)))
            numbers.append(np.sign(float(value))*min(1, math.log1p(abs(float(value)))/32) if number(value) else 0.)
        self.observation = np.concatenate((pixels-.5, delta, numbers, valid)).astype(np.float32)
        self.previous_pixels = pixels

    def refresh(self):
        if self.observer:
            image, state = self.observer()
            self.observe(image, state)
        return self.observation is not None

    def rows(self):
        return self.db.execute('SELECT COUNT(*) FROM rl_replay WHERE ready=1').fetchone()[0]

    @property
    def epsilon(self):
        if self.mode != 'train' or self.online.step == 0:return 0.
        return max(.01, .08*math.exp(-self.rows()/1500))

    def order(self, choices, priors=None, explore=True):
        """Only receives the caller's currently legal alternatives. No new actions."""
        if not choices:raise ValueError('No legal RL actions')
        if not self.refresh():raise RuntimeError('강화학습 화면 관측이 없어 행동을 선택하지 않습니다.')
        actions = np.stack([action_vector(c) for c in choices])
        self.link_navigation(self.observation, actions, choices[0]['domain'])
        inputs = np.concatenate((np.repeat(self.observation[None], len(actions), axis=0), actions), axis=1)
        q = self.online.forward(inputs)
        # Cold start is explicitly a feasible-skill prior, not a trained policy.
        scores = np.asarray(priors if priors is not None else [0.]*len(choices), np.float32) if self.online.step == 0 else q
        order = list(np.argsort(-scores, kind='stable'))
        if explore and len(order)>1 and self.rng.random()<self.epsilon:
            selected=int(self.rng.integers(len(order)));order.remove(selected);order.insert(0, selected)
        result=[]
        now=time.time()
        with self.db:
            for i in order:
                choice={**choices[i], '_rl_id':uuid.uuid4().hex}
                self.db.execute('INSERT INTO rl_decisions VALUES(?,?,?,?,?,?,NULL)',
                    (choice['_rl_id'], now, choice['domain'], json.dumps(choice), pack(self.observation), pack(actions)))
                result.append((choice, float(scores[i])))
            # Unexecuted scout candidates are not training data and are bounded.
            self.db.execute('DELETE FROM rl_decisions WHERE episode IS NULL AND ts<?', (now-86400,))
        return result

    def choose(self, domain, context, allowed, features=None):
        choices=[dict(domain=domain, context=context, action=a, features=(features or {}).get(a, {})) for a in allowed]
        return self.order(choices)[0][0]

    def set_mode(self,mode):
        if mode not in ('train','evaluate'):raise ValueError('Unknown learning mode')
        self.mode=mode

    def link_navigation(self, observation, actions, domain):
        if domain != 'strategy':return
        # Successor action masks come from a fresh actual decision, never from
        # imaginary options at a closed/menu/loading screen.
        with self.db:
            self.db.execute('UPDATE rl_replay SET next_observation=?,next_actions=?,ready=1 WHERE source IN (?,?) AND ready=0',
                            (pack(observation), pack(actions), 'navigation', 'investment'))

    def finish_navigation(self, choice, elapsed):
        if not choice or not choice.get('_rl_id'):return
        row=self.db.execute('SELECT observation,ts FROM rl_decisions WHERE id=?', (choice['_rl_id'],)).fetchone()
        if not row:return
        # Costs only verified elapsed live interaction, no reward for finding
        # pixels or sending scouts. Future actual rewards propagate through TD.
        reward=-min(.4, max(0, elapsed)/3600)
        with self.db:
            self.insert('nav:'+choice['_rl_id'], 'nav:'+choice['_rl_id'], 'navigation', row[0],
                        pack(action_vector(choice)), reward, None, None, .97**(max(1,elapsed)/60), False, False)

    def insert(self, identity, episode, source, obs, action, reward, next_obs, next_actions, discount, terminal, ready):
        self.db.execute('INSERT OR IGNORE INTO rl_replay(id,episode,source,ts,observation,action,reward,next_observation,next_actions,discount,terminal,priority,ready) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (identity,episode,source,time.time(),obs,action,float(reward),next_obs,next_actions,discount,int(terminal),1.,int(ready)))

    def credit(self, entry, reward, revision=False, source='combat'):
        """A verified terminal option reward trains the sequence leading to it.

        Intermediate decisions have zero immediate reward; Double DQN learns
        their discounted future value. No image/choice = no invented sample.
        """
        traces=[]
        for choice in entry['choices']:
            identity=choice.get('_rl_id')
            if not identity:continue
            row=self.db.execute('SELECT ts,choice,observation,candidates,episode FROM rl_decisions WHERE id=?', (identity,)).fetchone()
            if row and (row[4] is None or row[4]==entry['id']):traces.append((identity,*row[:4]))
        traces=list({row[0]:row for row in traces}.values())
        traces.sort(key=lambda row:row[1])
        if not traces:return
        with self.db:
            for i, (identity, ts, choice_json, obs, candidates) in enumerate(traces):
                terminal=i==len(traces)-1
                successor=traces[i+1] if not terminal else None
                elapsed=max(1, (entry.get('updated',time.time()) if terminal else successor[1])-ts)
                self.insert(entry['id']+':'+identity,entry['id'],source,obs,pack(action_vector(json.loads(choice_json))),
                    reward if terminal else 0.,successor[3] if successor else None,
                    successor[4] if successor else None,.97**(elapsed/60),terminal,True)
                self.db.execute('UPDATE rl_decisions SET episode=? WHERE id=?', (entry['id'],identity))
            if revision:
                self.db.execute('UPDATE rl_replay SET reward=?,priority=6,revision=revision+1 WHERE id=?', (reward,entry['id']+':'+traces[-1][0]))
                # Old gradients cannot be subtracted. Reset and relearn from
                # corrected retained replay, including the negative outcome.
                if self.mode=='train':self.reset_model()
                else:self.db.execute('INSERT OR REPLACE INTO rl_meta VALUES(?,?)',('rebuild_pending','1'))
            self.prune()

    def investment(self, entry, held, reward):
        choice=entry.get('rl_choice')
        if not choice:return
        identity='investment:'+entry['id']
        self.credit({'id':identity,'choices':[choice],'updated':time.time()},reward,
                    revision=not held and entry.get('status')=='held',source='investment')
        # Ownership after 15 minutes is a proxy, not a combat victory; retain
        # the cost, and bootstrap only on the next observed strategy mask.
        with self.db:
            self.db.execute('UPDATE rl_replay SET terminal=0,ready=0 WHERE episode=?', (identity,))

    def prune(self):
        self.db.execute('DELETE FROM rl_replay WHERE id IN (SELECT id FROM rl_replay ORDER BY ts DESC LIMIT -1 OFFSET ?)', (CAPACITY,))
        # Preserve decisions for delayed result corrections (up to a day).
        self.db.execute('DELETE FROM rl_decisions WHERE episode IS NOT NULL AND ts<? AND episode NOT IN (SELECT episode FROM rl_replay)', (time.time()-172800,))

    def reset_model(self):
        self.online,self.target=QNetwork(),QNetwork()
        self.dirty=True
        n=int(self.meta('rebuilds','0'))+1
        self.db.execute('INSERT OR REPLACE INTO rl_meta VALUES(?,?)',('rebuilds',str(n)))
        self.db.execute('INSERT OR REPLACE INTO rl_meta VALUES(?,?)',('rebuild_pending','0'))

    def targets(self, samples):
        targets=[];blocks=[];ranges=[];offset=0
        for row in samples:
            reward,next_obs,next_actions,discount,terminal=row[3:8]
            targets.append(reward)
            if not terminal:
                actions=unpack(next_actions).reshape(-1,ACTION)
                inputs=np.concatenate((np.repeat(unpack(next_obs)[None],len(actions),axis=0),actions),axis=1)
                blocks.append(inputs);ranges.append((len(targets)-1,offset,offset+len(inputs),discount));offset+=len(inputs)
        if blocks:
            inputs=np.concatenate(blocks)
            online=self.online.forward(inputs);frozen=self.target.forward(inputs)
            for index,start,end,discount in ranges:
                # Online network selects; a distinct frozen target evaluates.
                selected=start+int(np.argmax(online[start:end]))
                targets[index]+=discount*float(frozen[selected])
        return np.asarray(targets,np.float32)

    def train_step(self, batch=16):
        if self.mode != 'train':return False
        if self.meta('rebuild_pending','0')=='1':
            with self.db:self.reset_model()
        metadata=self.db.execute('SELECT id,priority FROM rl_replay WHERE ready=1 ORDER BY ts').fetchall()
        if len(metadata)<8:return False
        priorities=np.asarray([max(1e-4,r[1]) for r in metadata],np.float64)**.6
        probabilities=priorities/priorities.sum()
        indices=self.rng.choice(len(metadata),size=min(batch,len(metadata)),p=probabilities,replace=True)
        selected=[self.db.execute('SELECT id,observation,action,reward,next_observation,next_actions,discount,terminal,priority FROM rl_replay WHERE id=?', (metadata[int(i)][0],)).fetchone() for i in indices]
        importance=(len(metadata)*probabilities[indices])**(-min(1.,.4+self.online.step/10000*.6))
        importance=(importance/importance.max()).astype(np.float32)
        inputs=np.stack([np.concatenate((unpack(r[1]),unpack(r[2]))) for r in selected])
        self.last_loss,td=self.online.fit(inputs,self.targets(selected),importance)
        self.last_td=float(td.mean())
        if not math.isfinite(self.last_loss) or any(not np.isfinite(v).all() for v in self.online.weights.values()):
            self.reset_model();raise RuntimeError('강화학습 수치 오류: 모델을 초기화하고 검증된 경험을 유지했습니다.')
        with self.db:
            self.db.executemany('UPDATE rl_replay SET priority=? WHERE id=?', [(float(e)+.001,r[0]) for r,e in zip(selected,td)])
            self.db.execute('INSERT OR REPLACE INTO rl_meta VALUES(?,?)',('lifetime_updates',str(int(self.meta('lifetime_updates','0'))+1)))
        if self.online.step%100==0:self.target.weights={k:v.copy() for k,v in self.online.weights.items()}
        self.dirty=True
        if time.monotonic()-self.last_save>30:self.save()
        return True

    def meta(self,key,default):
        row=self.db.execute('SELECT value FROM rl_meta WHERE key=?',(key,)).fetchone()
        return row[0] if row else default

    def save(self):
        path=self.directory/'visual-policy.npz'
        temporary=path.with_suffix('.tmp')
        arrays={prefix+k:v for prefix,values in (('online_',self.online.weights),('target_',self.target.weights),('adam_m_',self.online.m),('adam_v_',self.online.v)) for k,v in values.items()}
        with temporary.open('wb') as f:np.savez_compressed(f, schema=SCHEMA,step=self.online.step,generation=int(self.meta('rebuilds','0')),**arrays)
        temporary.replace(path)
        self.dirty=False;self.last_save=time.monotonic()

    def load(self):
        path=self.directory/'visual-policy.npz'
        if not path.exists():return
        try:
            with path.open('rb') as stream, np.load(stream,allow_pickle=False) as data:
                if int(data['schema'])!=SCHEMA:raise ValueError('model schema mismatch')
                if int(data.get('generation',0))!=int(self.meta('rebuilds','0')):raise ValueError('replay was corrected after this checkpoint')
                groups=(('online_',self.online.weights),('target_',self.target.weights),('adam_m_',self.online.m),('adam_v_',self.online.v))
                loaded=[]
                for prefix,values in groups:
                    group={k:data[prefix+k].copy() for k in values}
                    if any(group[k].shape!=v.shape or not np.isfinite(group[k]).all() for k,v in values.items()):raise ValueError('invalid model weights')
                    loaded.append(group)
                step=int(data['step'])
                if step<0:raise ValueError('invalid optimizer step')
                for (_,values),group in zip(groups,loaded):values.update(group)
                self.online.step=step
        except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as exc:
            self.checkpoint_error=str(exc)
            path.replace(path.with_name('visual-policy-invalid-'+str(time.time_ns())+'.npz'))
            self.online,self.target=QNetwork(),QNetwork()

    def summary(self):
        ready=self.rows()
        sources=dict(self.db.execute('SELECT source,COUNT(*) FROM rl_replay GROUP BY source'))
        battles=self.db.execute("SELECT COUNT(DISTINCT episode) FROM rl_replay WHERE source='combat'").fetchone()[0]
        rows=self.db.execute("SELECT e.result,e.reward,e.after_json FROM outcome_episodes e WHERE e.id IN (SELECT episode FROM rl_replay WHERE source='combat') AND e.reward IS NOT NULL ORDER BY e.updated DESC LIMIT 40").fetchall() if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='outcome_episodes'").fetchone() else []
        def performance(items):
            return {'battles':len(items),'win_rate':sum(r[0] for r in items)/len(items) if items else None,
                    'mean_reward':sum(r[1] for r in items)/len(items) if items else None}
        return {'algorithm':self.algorithm,'mode':self.mode,'gradient_updates':self.online.step,
                'transitions':ready,'combat_episodes':battles,'sources':sources,
                'lifetime_updates':int(self.meta('lifetime_updates','0')),
                'epsilon':self.epsilon,'loss':self.last_loss,'td_error':self.last_td,
                'rebuilds':int(self.meta('rebuilds','0')),'checkpoint_error':self.checkpoint_error,
                'performance':{'recent':performance(rows[:20]),'previous':performance(rows[20:])},
                'phase':('경험 수집' if self.mode=='train' else '미학습 · 기본 실행') if self.online.step==0 else ('학습 중' if self.mode=='train' else '학습된 정책 실행'),
                'checkpoint':str(self.directory/'visual-policy.npz')}
