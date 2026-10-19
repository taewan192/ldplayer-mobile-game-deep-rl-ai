"""Persistent scout cooldowns in terrain coordinates, rather than screen pixels."""
import json
import math
import re
import time
import hashlib
from pathlib import Path

from PIL import Image


class TargetMemory:
    def __init__(self,data):
        self.data=Path(data); self.path=self.data/'scout-targets.json'
        self.entries=json.loads(self.path.read_text('utf-8')) if self.path.exists() else []
        self._projection_key=None;self._projections=[]

    def save(self):
        self._projection_key=None
        ordinary=[e for e in self.entries if not self._excluded_entry(e)][-48:]
        old=[e for e in self.entries if e not in ordinary and not self._excluded_entry(e)]
        self.entries=[e for e in self.entries if e in ordinary or self._excluded_entry(e)]
        temporary=self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.entries,ensure_ascii=False,indent=2),'utf-8'); temporary.replace(self.path)
        for entry in old:
            name=entry.get('view','')
            if re.fullmatch(r'scout-view-\d+\.png',name): (self.data/name).unlink(missing_ok=True)

    def add(self,image,point,identity=None,seconds=600):
        name=f'scout-view-{time.time_ns()}.png'; image.save(self.data/name)
        entry={'view':name,'point':point,'until':time.time()+seconds,'identities':[identity] if identity else [],'reason':'scouted'}
        self.entries.append(entry); self.save(); return entry

    def identity_blocked(self,identity):
        return bool(identity and any(identity in e.get('identities',[]) and e['until']>time.time() for e in self.entries))

    def blocked(self,image,point,vision):
        return any(math.dist(projected,point)<32 and (entry['until']>time.time() or self._excluded_entry(entry))
                   for entry,projected in self.projected(image,vision))

    def _excluded_entry(self,entry):
        exclusions=getattr(self,'exclusions',None)
        return bool(exclusions and any(exclusions.blocked_identity(i) or exclusions.blocked_city(i.rsplit(':',1)[-1])
                    for i in entry.get('identities',[])))

    def excluded(self,image,point,vision):
        return any(self._excluded_entry(entry) and math.dist(projected,point)<32
                   for entry,projected in self.projected(image,vision))

    def projected(self,image,vision):
        if not any(e['until']>time.time() or self._excluded_entry(e) for e in self.entries):return []
        exclusions=getattr(self,'exclusions',None)
        revision=tuple(sorted((name,record.get('updated')) for name,record in exclusions.players.items())) if exclusions else ()
        key=(hashlib.blake2b(image.tobytes(),digest_size=16).digest(),revision)
        if key==self._projection_key:return self._projections
        projected_entries=[]
        for entry in reversed(self.entries):
            if entry['until']<=time.time() and not self._excluded_entry(entry): continue
            path=self.data/entry['view']
            if not path.exists(): continue
            with Image.open(path) as source: reference=source.convert('RGB')
            matrix=vision.viewport_transform(reference,image)
            if matrix is None: continue
            projected=vision.transform_point(matrix,entry['point'])
            projected_entries.append((entry,projected))
        self._projection_key=key;self._projections=projected_entries
        return projected_entries
