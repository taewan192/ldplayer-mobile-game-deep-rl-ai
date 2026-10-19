"""Player-wide exclusions survive restarts and cannot be bypassed by re-scouting."""
import json
import re
import time
from pathlib import Path


def player_key(name):
    return re.sub(r'[^a-z0-9가-힣]','',str(name).casefold())


def close_key(a,b,limit):
    if abs(len(a)-len(b))>limit:return False
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]<=limit


class PlayerExclusions:
    def __init__(self,data):
        self.data=Path(data);self.path=self.data/'excluded-players.json'
        self.players=json.loads(self.path.read_text('utf-8')) if self.path.exists() else {}
        if 'prabowo' not in self.players:
            self.exclude('-PRABOWO-','사용자가 제공한 병력 배치 제한 화면',city='Astev')

    def exclude(self,name,reason,city=None,vp=None):
        key=player_key(name)
        if not key:return
        old=self.players.get(key,{})
        cities=set(old.get('cities',[]))
        if city:cities.add(player_key(city))
        self.players[key]={**old,'name':name,'reason':reason,'cities':sorted(cities),'updated':time.time()}
        if vp is not None:self.players[key]['vp']=vp
        tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(self.players,ensure_ascii=False,indent=2),'utf-8');tmp.replace(self.path)

    def blocked(self,name):
        key=player_key(name)
        if key in self.players:return True
        for known in self.players:
            if min(len(key),len(known))>=6 and close_key(key,known,2):return True
        return False

    def blocked_identity(self,identity):
        return bool(identity and self.blocked(identity.split(':',1)[0]))

    def blocked_city(self,city):
        key=player_key(city)
        return bool(key and any(key==city or min(len(key),len(city))>=4 and close_key(key,city,1)
                    for record in self.players.values() for city in record.get('cities',[])))
