"""Persist region order independently of a return-to-city camera reset."""
import json
import re
import time
from pathlib import Path

from PIL import Image


# Camera drags are opposite to the direction being explored on the map.
SWEEP_RAYS=(('아래',0,-180),('위',0,180),('왼쪽',260,0),('오른쪽',-260,0),
            ('왼쪽 위',260,180),('오른쪽 위',-260,180),
            ('왼쪽 아래',260,-180),('오른쪽 아래',-260,-180))


class ExplorationMemory:
    def __init__(self,data=None):
        self.data=Path(data) if data is not None else None
        self.path=self.data/'exploration.json' if self.data else None
        self.state=json.loads(self.path.read_text('utf-8')) if self.path and self.path.exists() else {}
        self.state.setdefault('cursor',1)
        self.state.setdefault('direction',0)
        self.state.setdefault('visits',[])
        self.state.setdefault('sweep',{'version':1,'cursor':0,'cycle':1,'needs_home':True,'visits':[]})

    def sweep_next(self):
        route=self.state['sweep'];cursor=route['cursor']
        name,dx,dy=SWEEP_RAYS[cursor//9]
        return {'direction':name,'drag':(dx,dy),'step':cursor%9+1,
                'ray':cursor//9,'cursor':cursor,'cycle':route['cycle']}

    def sweep_commit(self,verified=True):
        route=self.state['sweep'];point=self.sweep_next()
        point['movement_verified']=bool(verified)
        route['visits'].append({**point,'ts':time.time()})
        route['visits']=route['visits'][-144:]
        route['cursor']=(route['cursor']+1)%72
        if route['cursor']==0:route['cycle']+=1
        route['needs_home']=route['cursor']%9==0
        route['failures']=0
        self.save()
        return point

    def sweep_reanchor(self):
        self.state['sweep']['needs_home']=True;self.save()

    def sweep_bind_main(self,name):
        route=self.state['sweep'];previous=route.get('main')
        if previous and previous!=name:
            route.update(cursor=0,cycle=route['cycle']+1,visits=[])
        route['main']=name;self.save()

    def sweep_skip_ray(self,reason):
        route=self.state['sweep'];point=self.sweep_next()
        route.setdefault('skipped',[]).append({**point,'reason':reason,'ts':time.time()})
        route['skipped']=route['skipped'][-16:]
        route['cursor']=((route['cursor']//9+1)*9)%72
        if route['cursor']==0:route['cycle']+=1
        route['needs_home']=True;route['failures']=0;self.save()

    def save(self):
        if self.path:
            temp=self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.state,ensure_ascii=False,indent=2),'utf-8');temp.replace(self.path)

    def reserve_anchor(self,pages):
        # Visit a different page first, then a different row on each page.
        # Advance BEFORE navigation so a failed city cannot pin the cursor.
        cursor=self.state['cursor']%(pages*5)
        page,row=cursor%pages+1,(cursor//pages)%5
        self.state['cursor']=(cursor+1)%(pages*5);self.save()
        return page,row

    def next_direction(self):
        directions=[(230,0),(0,160),(-230,0),(0,-160),(-230,0),(0,160),(230,0),(0,-160)]
        step=self.state['direction'];self.state['direction']=step+1;self.save()
        return directions[step%len(directions)]

    def recently_seen(self,image,vision,seconds=180):
        if not self.data:return False
        for visit in self.state['visits'][-8:]:
            if time.time()-visit['ts']>seconds:continue
            path=self.data/visit['view']
            if not path.exists():continue
            with Image.open(path) as source:reference=source.convert('RGB')
            if vision.same_viewport(image,reference):return True
        return False

    def remember(self,name,image):
        if not self.data:return
        filename=f'exploration-view-{time.time_ns()}.png';image.save(self.data/filename)
        self.state['visits'].append({'city':name,'view':filename,'ts':time.time()})
        old=self.state['visits'][:-8];self.state['visits']=self.state['visits'][-8:];self.save()
        for item in old:
            if re.fullmatch(r'exploration-view-\d+\.png',item['view']):(self.data/item['view']).unlink(missing_ok=True)
