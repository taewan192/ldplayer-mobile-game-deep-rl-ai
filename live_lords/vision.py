from __future__ import annotations

import re
import hashlib
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageOps


def amount(text):
    """Only accept explicit units; never silently turn OCR G into digit 6."""
    # Read the leading value only. A troop icon can OCR as '4t' after '78.1M';
    # searching for any later token would turn 78.1M4t into a fictitious 4T army.
    match = re.match(r'^\s*(\d+(?:[.,]\d+)?)\s*([kKmMgGtT])(?=$|[\W\d])', text)
    if match:
        return float(match[1].replace(',', '.')) * {'K':1e3,'M':1e6,'G':1e9,'T':1e12}[match[2].upper()]
    clean = text.strip().replace(',', '')
    return float(clean) if re.fullmatch(r'\d+(?:\.\d+)?', clean) else None


def pretty(value):
    if value is None: return '—'
    for suffix, scale in [('T',1e12),('G',1e9),('M',1e6),('K',1e3)]:
        if abs(value) >= scale: return f'{value/scale:.3f}{suffix}'
    return f'{value:,.0f}'


def numeric_amount(text):
    """Repair confusable digits only in a leading decimal with an explicit unit."""
    clean=text.strip()
    match=re.match(r'^([\dIl|]+[.,][\dOoIl|]+)\s*([kKmMgGtT])(?=$|[\W\d])',clean)
    if match:
        digits=match[1].translate(str.maketrans({'I':'1','l':'1','|':'1','O':'0','o':'0'}))
        clean=digits+match[2]+clean[match.end():]
    return amount(clean)


def troop_bounds(text):
    text=re.sub(r'\s+','',text)
    parts=re.split(r'[-–—~]',text)
    # In this numeric column, 10G-100G commonly OCRs as LOG-IOOG.
    # Repair digits only inside a complete number+unit token; never accept
    # arbitrary words, or turn a hidden range into an exact enemy army.
    digits=str.maketrans({'I':'1','l':'1','L':'1','|':'1','O':'0','o':'0'})
    parts=[p.translate(digits) if re.fullmatch(r'[\dIlL|Oo]+(?:[.,][\dIlL|Oo]+)?[kKmMgGtT]',p)
           else p for p in parts]
    if not all(re.fullmatch(r'(?:\d+(?:[.,]\d+)?[kKmMgGtT]|\d+)',p) for p in parts):
        return None,None
    if len(parts)==1:
        value=numeric_amount(parts[0]); return (value,value)
    if len(parts)==2:
        lower,upper=map(numeric_amount,parts)
        if lower is not None and upper is not None and 0<lower<=upper: return lower,upper
    return None,None


def march_countdown(text):
    """Accept a whole clock, never extract '3s' from OCR garbage such as '3so'."""
    text=re.sub(r'\s+','',text).lower().replace('rn','m')
    # Only substitute confusable digits in the numeric slots of a whole clock.
    text=re.sub(r'[il|](?=[hms])','1',text)
    text=re.sub(r'(?<=[\dm])s(?=s$)','5',text)
    # In the seconds slot, the live small font's 9 can be read as lowercase g.
    text=re.sub(r'(?<=m)([0-5]?)g(?=s$)',r'\g<1>9',text)
    # The live small font commonly turns the final 's' into 3 or 8.
    text=re.sub(r'(?<=m)([0-5]?\d)[38]$',r'\1s',text)
    match=re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?',text)
    if not match or not any(match.groups()): return None
    h,m,s=(int(v or 0) for v in match.groups())
    if m>=60 or s>=60: return None
    return h*3600+m*60+s


def percentage(text):
    text=re.sub(r'\s+','',text)
    # A complete percent glyph may OCR as 0/0 or 0/6. Consume that
    # terminal glyph, never strip a genuine trailing digit from 930%.
    match=re.fullmatch(r'\+?(\d+(?:[.,]\d+)?)(?:%|0/[06])',text)
    return float(match[1].replace(',','.')) if match else None


class Vision:
    def __init__(self, ocr, data):
        self.ocr, self.data = ocr, Path(data)
        self.assets = Path(__file__).with_name('assets')
        self.ocr_cache=OrderedDict()
        self.feature_cache=OrderedDict()
        self.templates = {p.stem: cv2.imdecode(np.frombuffer(p.read_bytes(),dtype=np.uint8), cv2.IMREAD_COLOR)
                          for p in self.assets.glob('*.png')}

    def match(self, image, name, threshold=.87, area=None):
        template = self.templates.get(name)
        if template is None: return None
        array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        x0,y0,x1,y1 = area or (0,0,1280,720)
        array = array[y0:y1,x0:x1]
        scores = cv2.matchTemplate(array, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(scores)
        if score < threshold: return None
        if name=='map_home':
            patch=array[loc[1]:loc[1]+template.shape[0],loc[0]:loc[0]+template.shape[1]]
            if patch.mean()<template.mean()*.74: return None
        return (loc[0]+x0+template.shape[1]//2, loc[1]+y0+template.shape[0]//2, score)

    def regions(self, image, boxes):
        """Read isolated enlarged crops; independent rows prevent OCR layout merging."""
        results={}; pending=[]
        for name,box in boxes.items():
            crop=image.crop(box)
            scale=min(2 if name in {'march_city_padded','scout_city_padded'} else 3 if name in {'remaining','march_city','scout_power_ink'} else 7,
                      1050/crop.width,240/crop.height)
            size=(round(crop.width*scale),round(crop.height*scale))
            language='ko' if name in {'title','prompt','popup','warning'} or name.startswith('restriction') else 'en-US'
            padding=24 if name in {'march_city_padded','scout_city_padded','scout_power_ink'} else 0
            ink=name=='scout_power_ink'
            key=(language,box,crop.mode,size,padding,ink,hashlib.blake2b(crop.tobytes(),digest_size=16).digest())
            if key in self.ocr_cache:
                results[name]=self.ocr_cache[key]; self.ocr_cache.move_to_end(key)
                continue
            path=self.data/f'ocr-region-{len(pending)}.png'
            prepared=(Image.fromarray(np.where(np.asarray(crop).max(axis=2)>140,0,255).astype('uint8')) if ink else crop)
            enlarged=prepared.resize(size,Image.Resampling.LANCZOS)
            if padding:enlarged=ImageOps.expand(enlarged,border=padding,fill='white')
            enlarged.save(path)
            pending.append((name,crop,size,language,path,key))
        requests=[(item[4],item[3]) for item in pending]
        reader=getattr(self.ocr,'read_many',None)
        raw=reader(requests) if reader and requests else [self.ocr.read(*request) for request in requests]
        retries=[]
        for item,result in zip(pending,raw):
            name,crop,size,language,path,key=item
            text=result['text'].strip()
            if not text and language=='en-US' and name!='scout_power_ink':
                brightness=np.array(crop.convert('L'))
                clean=Image.fromarray(np.where(brightness>170,0,255).astype('uint8'))
                clean.resize(size,Image.Resampling.LANCZOS).save(path)
                retries.append(item)
            results[name]=text
        requests=[(item[4],item[3]) for item in retries]
        raw=reader(requests) if reader and requests else [self.ocr.read(*request) for request in requests]
        for item,result in zip(retries,raw): results[item[0]]=result['text'].strip()
        for name,_,_,_,_,key in pending:
            if results[name]: self.ocr_cache[key]=results[name]
        while len(self.ocr_cache)>256: self.ocr_cache.popitem(last=False)
        return {name:results[name] for name in boxes}

    def hud(self, image,verify=False):
        values = self.regions(image, {'vp':(308,35,365,61),'gold':(46,183,118,212)})
        def resource(text):
            # Gold is integral. An OCR string like '3.956' is not 3.956 gold:
            # its suffix may have been misread. Require a suffix for decimals.
            return amount(text) if re.search(r'[kKmMgGtT]\b',text) or re.fullmatch(r'\s*\d+\s*',text) else None
        gold=resource(values['gold'])
        if gold is None and self.match(image,'hud_gold_g',.96,area=(74,181,114,216)):
            number=re.fullmatch(r'\s*(\d+[.,]\d{2})6?\s*',values['gold'])
            if number: gold=float(number[1].replace(',','.'))*1e9
        if gold is None:
            retry=self.regions(image,{'gold':(48,185,113,210)})['gold']
            gold=resource(retry)
        if verify:
            checked=resource(self.regions(image,{'gold_verify':(47,185,113,210)})['gold_verify'])
            if gold is None or checked is None or gold!=checked:
                third=resource(self.regions(image,{'gold_confirm':(45,184,117,212)})['gold_confirm'])
                readings=[value for value in (gold,checked,third) if value is not None]
                gold=next((value for value in readings if readings.count(value)>=2),None)
        return {'vp':resource(values['vp']),'gold':gold}

    def combat_bonuses(self,image):
        boxes={}
        for name in ('attack','defense'):
            icon=self.match(image,'stat_'+name,.84,area=(225,210,270,575))
            if not icon: raise RuntimeError(f'{name} 능력치 아이콘 확인 실패')
            boxes[name]=(528,icon[1]-17,612,icon[1]+18)
        raw=self.regions(image,boxes)
        result={name:percentage(value) for name,value in raw.items()}
        checks={name:(box[0]-8,box[1]-3,box[2]+3,box[3]+3) for name,box in boxes.items()
                if result[name] is None or re.search(r'0/[06]$',re.sub(r'\s+','',raw[name]))}
        if checks:
            verified=self.regions(image,{name+'_verify':box for name,box in checks.items()})
            for name in checks:
                value=percentage(verified[name+'_verify'])
                if value is None or (result[name] is not None and value!=result[name]):
                    raise RuntimeError(f'공격/방어 능력치 재확인 불일치: {raw} / {verified}')
                result[name]=value
        if any(value is None for value in result.values()):
            raise RuntimeError(f'공격/방어 능력치 OCR 확인 실패: {raw}')
        return result

    def scout_report(self,image):
        values=self.regions(image,{'title':(265,110,500,157),'defense':(746,348,842,381),
                                  'city_name':(590,262,693,293),'attacker':(860,175,979,202)})
        if '보고서' not in values['title']: return None
        if not re.search(r'[\w가-힣]',values['city_name']):
            values['city_name']=self.regions(image,{'scout_city_padded':(590,260,693,293)})['scout_city_padded']
        value=numeric_amount(values['defense'])
        if value is None:
            alternatives=self.regions(image,{'defense':(761,349,837,379)})
            value=numeric_amount(alternatives['defense'])
        if value is None:
            alternate=self.regions(image,{'scout_power_ink':(752,351,839,379)})['scout_power_ink']
            value=numeric_amount(alternate)
        return {'defense':value,'city':values['city_name'],'owner':values['attacker'],'raw':values['defense']}

    def enemy_identity(self,image,scout,expected_city=None,expected_owner=None):
        x,y=scout[:2]
        if x+225>1280 or y-242<0: return None
        fields=self.regions(image,{'city_name':(x+125,y-204,x+220,y-180),
                                  'attacker':(x-70,y-242,x+225,y-209)})
        normalize=lambda s:re.sub(r'[^a-z0-9가-힣]','',s.casefold())
        city,owner=normalize(fields['city_name']),normalize(fields['attacker'])
        if not owner:
            alternatives=self.regions(image,{'owner_middle':(x,y-238,x+180,y-212),
                                              'owner_tight':(x-25,y-240,x+175,y-215)})
            owner=next((normalize(value) for value in alternatives.values() if normalize(value)),'')
        if expected_owner and owner!=normalize(expected_owner):
            alternatives=self.regions(image,{'owner_middle':(x,y-238,x+180,y-212),
                                              'owner_tight':(x-25,y-240,x+175,y-215),
                                              'owner_baseline':(x+24,y-241,x+164,y-218)})
            if any(normalize(value)==normalize(expected_owner) for value in alternatives.values()):
                owner=normalize(expected_owner)
        if not expected_city:
            ink_name=self.enemy_city_ink(image,scout)
            if ink_name:city=ink_name
        if expected_city and city!=expected_city:
            alternatives=self.regions(image,{'city_wide':(x+75,y-207,x+225,y-176),
                                             'city_tight':(x+145,y-207,x+225,y-176),
                                             'city_baseline':(x+90,y-202,x+220,y-181)})
            if any(normalize(value)==expected_city for value in alternatives.values()): city=expected_city
            elif image is not None:
                # Gold city letters have a dark outline over the coloured
                # banner. Isolate that ink when the outline merges glyphs.
                label=np.array(image.crop((x+75,y-207,x+225,y-176))).astype('float32')
                r,g,b=label[:,:,0],label[:,:,1],label[:,:,2]
                ink=(r>165)&(g>150)&(r>=g)&(b<g*.85)
                yy,xx=np.where(ink)
                if len(xx)>=15 and np.ptp(xx)>5 and np.ptp(yy)>4:
                    clean=Image.fromarray(np.where(ink,0,255).astype('uint8'))
                    clean=clean.crop((max(0,int(xx.min())-3),max(0,int(yy.min())-3),
                                      min(clean.width,int(xx.max())+4),min(clean.height,int(yy.max())+4)))
                    path=self.data/'ocr-city-name.png'
                    clean.resize((clean.width*7,clean.height*7),Image.Resampling.LANCZOS).save(path)
                    value=self.ocr.read(path,'en-US')['text']
                    if normalize(value)==expected_city: city=expected_city
        return owner+':'+city if city and owner else None

    def enemy_city_ink(self,image,scout):
        x,y=scout[:2]
        label=np.asarray(image.crop((x+75,y-207,x+225,y-176))).astype('float32')
        r,g,b=label[:,:,0],label[:,:,1],label[:,:,2]
        ink=(r>165)&(g>150)&(r>=g)&(b<g*.85)
        yy,xx=np.where(ink)
        if len(xx)<15 or np.ptp(xx)<=5 or np.ptp(yy)<=4:return None
        clean=Image.fromarray(np.where(ink,0,255).astype('uint8'))
        clean=clean.crop((max(0,int(xx.min())-3),max(0,int(yy.min())-3),min(clean.width,int(xx.max())+4),min(clean.height,int(yy.max())+4)))
        path=self.data/'ocr-city-name-initial.png';clean.resize((clean.width*7,clean.height*7),Image.Resampling.LANCZOS).save(path)
        text=self.ocr.read(path,'en-US')['text']
        value=re.sub(r'[^a-z]','',text.casefold())
        return value if len(value)>=2 else None

    def navigation_panel(self,image):
        """Recognize read-only game panels, including titles containing player/city names."""
        for asset,label in [('panel_harvester','수확자'),('panel_settings','설정'),('panel_news','게임 소식/공지')]:
            if self.match(image,asset,.9,area=(200,108,370,163)): return label
        if (self.match(image,'panel_guild_members',.9,area=(715,173,790,220)) and
                self.match(image,'panel_guild_points',.9,area=(820,173,985,220))): return '길드'
        if self.match(image,'panel_shop_back',.9,area=(5,3,82,74)): return '상점/상자 전체화면 메뉴'
        if self.match(image,'city_list_title',.9,area=(200,105,360,165)):
            return '도시 목록'
        if self.match(image,'march_sort',.86,area=(760,35,850,95)):
            return '행군 진행 상황'
        if (self.match(image,'own_city_total',.86,area=(540,170,750,230)) and
                self.match(image,'city_details_stats',.86,area=(540,310,735,375))):
            return '내 도시 상세정보'
        if (self.match(image,'city_details_defense',.86,area=(540,170,750,230)) and
                self.match(image,'city_details_stats',.86,area=(540,310,735,375))):
            return '도시 상세정보'
        if (self.match(image,'enemy_profile_info',.86,area=(300,165,560,230)) and
                self.match(image,'enemy_profile_kingdom',.86,area=(790,165,970,230))):
            return '상대 프로필'
        return None

    def navigation_panel_text(self,image):
        # OCR fallback for scales or titles not covered by the paired visual landmarks.
        title=self.regions(image,{'title':(208,108,650,160)})['title']
        compact=re.sub(r'\s+','',title)
        if any(t in compact for t in ('보고서','능력치','플레이어정보','도시목록','도시상세','공격진행','진행상황',
                                      '공격정보','방어정보','행군정보','부대정보','공격상세','방어상세',
                                      '수확자','상점','설정','공지','게임소식','우편','길드','동맹','연맹',
                                      '상자','보관함','인벤토리','임무','퀘스트','이벤트','업적','보상')):
            return title
        english=self.regions(image,{'profile':(256,108,1000,160)})['profile']
        if re.search(r'\bprofile\b',english,re.I): return '상대 프로필'
        other=self.regions(image,{'title':(100,72,1110,132),'prompt':(340,200,940,255)})
        combined=''.join(other.values())
        if re.search(r'수확자|상점|설정|공지|게임\s*소식|상자|길드|연맹|인벤토리|퀘스트|이벤트|사파이어|도시\s*스킨',combined):
            return combined[:100]
        if re.search(r'\b(harvester|settings|shop|store|chests?|guild|alliance|news|inventory|quests?)\b',combined,re.I):
            return combined[:100]
        return None

    def own_city_details(self,image,expected_name=None):
        if not self.match(image,'own_city_total',.86,area=(540,170,750,230)): return None
        title=np.array(image)[112:155,270:900].astype('float32')
        white=(np.min(title,axis=2)>175)&((np.max(title,axis=2)-np.min(title,axis=2))<60)
        columns=np.where(np.sum(white,axis=0)>2)[0]
        name_box=(max(262,int(columns.min())+260),112,min(910,int(columns.max())+282),155) if len(columns) else (267,112,525,155)
        fields=self.regions(image,{'city_name':name_box,'defense':(970,181,1051,215),
            'troops':(630,247,744,279),'guardian':(874,248,948,280),
            'city_bonus':(628,412,688,440),'walls':(631,494,704,524),
            'level':(944,318,983,350),'city_vp':(813,323,866,355),'cost':(914,552,995,585)})
        if expected_name:
            normalize=lambda text:re.sub(r'[^a-z0-9가-힣]','',text.casefold())
            if normalize(fields['city_name'])!=normalize(expected_name):
                alternatives=self.regions(image,{'short_name':(267,112,445,155),'long_name':(267,112,565,155)})
                if not any(normalize(v)==normalize(expected_name) for v in alternatives.values()): return None
            fields['city_name']=expected_name
        if numeric_amount(fields['troops']) is None:
            # At large garrisons the leading 1 is clipped by the old crop.
            # Read it from pixels; never invent a missing leading digit.
            fields['troops']=self.regions(image,{'defense_troops':(626,245,720,280)})['defense_troops']
        result={'name':fields['city_name'],'defense':amount(fields['defense']),
                'troops':numeric_amount(fields['troops']),'city_bonus':percentage(fields['city_bonus']),
                'guardian_bonus':percentage(fields['guardian']),'walls':amount(fields['walls']),
                'vp':amount(fields['city_vp']),'cost':amount(fields['cost'])}
        nums=re.fullmatch(r'\s*(\d+)\s*',fields['level'])
        result['level']=int(nums[1]) if nums else None
        if not result['name'] or any(result[k] is None for k in ('defense','troops','city_bonus','guardian_bonus','level')): return None
        # Validate the displayed sum against its components when walls are readable.
        if result['walls'] is not None:
            expected=result['walls']+result['troops']*(1+(result['city_bonus']+result['guardian_bonus'])/100)
            if abs(expected-result['defense'])>max(expected*.04,1): return None
        return result

    def march_panel(self,image):
        # This title and paired tab locations were observed on the live march list.
        if self.match(image,'march_sort',.86,area=(760,35,850,95)): return '행군 진행 상황'
        title=self.regions(image,{'title':(210,46,510,89)})['title']
        return title if any(t in title for t in ('이동','공격','방어','정찰')) else None

    def march_button(self,image):
        area=(1180,500,1280,655)
        return (self.match(image,'march_defense_button',.82,area=area) or
                self.match(image,'march_button',.84,area=area) or
                self.match(image,'march_mixed_button',.86,area=area))

    def incoming_button(self,image):
        shield=self.match(image,'march_defense_button',.82,area=(1180,500,1280,655))
        if shield:return shield
        mixed=self.match(image,'march_mixed_button',.86,area=(1180,500,1280,655))
        if mixed:
            # The mixed swords shortcut includes our outgoing moves. It is
            # incoming only when its observed upper-right red count is present.
            x,y=mixed[:2]
            a=np.asarray(image)[max(0,y-44):y-20,x+21:min(1280,x+46)].astype('int16')
            if a.size and np.mean((a[:,:,0]>150)&(a[:,:,0]>a[:,:,1]*1.5)&(a[:,:,0]>a[:,:,2]*1.4))>.2:
                return mixed
        return None

    @staticmethod
    def defense_tab_selected(image):
        a=np.array(image)[40:70,1020:1052].astype('float32')
        return bool(np.mean((a[:,:,1]>90)&(a[:,:,2]>130)&(a[:,:,2]>a[:,:,0]*1.5))>.45)

    def march_list_scrollable(self,image):
        """Short lists at their normal top position need no scroll gestures."""
        from PIL import ImageDraw
        work=image.copy();positions=[]
        for _ in range(6):
            pin=self.match(work,'march_pin',.85,area=(620,100,685,660))
            if not pin:break
            positions.append(pin[1])
            ImageDraw.Draw(work).rectangle((620,pin[1]-40,684,pin[1]+42),fill='black')
        return bool(positions and (len(positions)>=6 or abs(min(positions)-148)>9))

    def defense_rows(self,image):
        """Read ONLY the selected defense list, never turn outgoing moves into threats."""
        if not self.defense_tab_selected(image): return []
        result=[]; work=image.copy()
        from PIL import ImageDraw
        from .planner import march_seconds
        for _ in range(6):
            pin=self.match(work,'march_pin',.85,area=(620,100,685,650))
            if not pin: break
            x,y,_=pin
            fields=self.regions(image,{'march_city':(x-165,y-21,x-31,y+17),
                'remaining':(x-435,y+1,x-340,y+31),'troops':(x+33,y+3,x+135,y+29),
                  'attacker':(x+140,y-28,x+297,y+3),'level':(x-215,y-20,x-170,y+20)})
            fields['city_name']=fields.pop('march_city')
            # Very short names (Iqi) disappear in the wide level/name banner,
            # or become punctuation. Use a tight crop with a larger OCR scale.
            if not re.search(r'[\w가-힣]',fields['city_name']):
                fields['city_name']=self.regions(image,{'defense_city_tight':(x-140,y-20,x-45,y+14)})['defense_city_tight']
            if not re.search(r'[\w가-힣]',fields['city_name']):
                fields['city_name']=self.regions(image,{'march_city_padded':(x-145,y-23,x-40,y+18)})['march_city_padded']
            if not re.search(r'[\w가-힣]',fields['city_name']):
                fields['city_name']=self.regions(image,{'warning':(440,y-20,624,y+24)})['warning']
                fields['city_name']=re.sub(r'^\d+\s*','',fields['city_name']).strip()
            fields['city_name']=re.sub(r'(?<=[a-zA-Z])\\$', 'l',fields['city_name'])
            remaining=march_countdown(fields['remaining']); lower,troops=troop_bounds(fields['troops'])
            if troops is None:
                retry=self.regions(image,{'troops':(x+30,y+3,x+138,y+30)})['troops']
                lower,troops=troop_bounds(retry)
            if remaining is None:
                retry=self.regions(image,{'remaining':(215,y+6,330,y+40)})['remaining']
                remaining=march_countdown(retry)
            if remaining is None:
                retry=self.regions(image,{'defense_clock':(x-430,y+5,x-360,y+33)})['defense_clock']
                remaining=march_countdown(retry)
            if troops is None or not re.search(r'[\w가-힣]',fields['city_name']) or not fields['attacker']:
                image.save(self.data/'unreadable-defense-list.png')
                raise RuntimeError(f'방어 행군 읽기 실패 ({y}): {fields}. 출병/강화를 보류합니다.')
            result.append({'city':fields['city_name'],'remaining':remaining or 0,'time_uncertain':remaining is None,'troops':troops,
                             'troops_min':lower,'troops_exact':lower==troops,
                             'attacker':fields['attacker'],'pin':(x,y),'row':y,
                             'level':int(fields['level']) if re.fullmatch(r'\d{1,3}',fields.get('level','').strip()) and 1<=int(fields['level'])<=200 else None})
            ImageDraw.Draw(work).rectangle((620,y-40,684,y+42),fill='black')
        return sorted(result,key=lambda row:row['remaining'])

    def labeled_attack_power(self,image,city=None,attacker=None):
        """Require an explicit power label. The troop column is NOT attack power."""
        path=self.data/'attack-detail-ocr.png'; image.save(path)
        doc=self.ocr.read(path,'ko')
        lines=doc.get('lines',[])
        normalized=lambda text:re.sub(r'[^a-z0-9가-힣]','',text.casefold())
        all_text=normalized(doc.get('text',' '.join(line['text'] for line in lines)))
        if any(value and normalized(value) not in all_text for value in (city,attacker)): return None
        for line in lines:
            text=re.sub(r'\s+','',line['text'])
            label=re.search(r'(?:총공격력|합계공격력|공격력|Attackpower)[:：]?',text,re.I)
            if not label: continue
            value=amount(text[label.end():])
            if value and value>0: return value
            words=line.get('words',[])
            if not words: continue
            right=max(w['x']+w['w'] for w in words); y=sum(w['y']+w['h']/2 for w in words)/len(words)
            # A separate value on the same row, or directly below a short title.
            candidates=[]
            for other in lines:
                if other is line or not other.get('words'): continue
                ow=other['words']; ox=min(w['x'] for w in ow); oy=sum(w['y']+w['h']/2 for w in ow)/len(ow)
                v=amount(other['text'])
                if v and ((ox>=right-5 and abs(oy-y)<25) or (0<oy-y<45 and abs(ox-words[0]['x'])<60)):
                    candidates.append(v)
            if len(candidates)==1: return candidates[0]
        return None

    def protection_circles(self,image):
        key=hashlib.blake2b(image.tobytes(),digest_size=16).digest()
        if getattr(self,'_shield_key',None)==key:return self._shield_circles
        a=np.array(image).astype('float32');r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
        mask=((r-g>105)&(r-b>95)&(r>160)&(g<125)).astype('uint8')*255
        mask[:70]=0;mask[620:]=0;mask[:,:145]=0;mask[:,1150:]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        circles=cv2.HoughCircles(cv2.GaussianBlur(mask,(5,5),1),cv2.HOUGH_GRADIENT,
                                 1,30,param1=80,param2=23,minRadius=22,maxRadius=78)
        thick=cv2.dilate(mask,np.ones((3,3),np.uint8));result=[]
        angles=np.linspace(0,2*np.pi,96,endpoint=False)
        for x,y,radius in (() if circles is None else circles[0]):
            xx=np.clip(np.rint(x+radius*np.cos(angles)).astype(int),0,1279)
            yy=np.clip(np.rint(y+radius*np.sin(angles)).astype(int),0,719)
            covered=thick[yy,xx]>0
            if covered.mean()<.55 or sum(part.mean()>.3 for part in np.array_split(covered,4))<3:continue
            result.append((float(x),float(y),float(radius)))
        self._shield_key,self._shield_circles=key,result
        return result

    def is_shielded(self,image,target):
        # The candidate is anchored under a floating flag; near the edge the
        # actual castle and its circle sit toward the perspective center.
        projected=(target[0]+(640-target[0])*.15,target[1])
        return any(min(np.linalg.norm(np.array(target)-[x,y]),np.linalg.norm(np.array(projected)-[x,y]))<max(45,radius*.95)
                   for x,y,radius in self.protection_circles(image))

    def enemy_candidates(self, image,include_shielded=False):
        # Solid red/pink level badges, not town roofs or thin territory circles.
        a = np.array(image).astype('float32')
        r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
        mask=(((r>210)&(r<250)&(g>40)&(g<80)&(b<45))|
              ((r>195)&(r<235)&(g>100)&(g<145)&(b>105)&(b<150)&(np.abs(g-b)<15))).astype('uint8')*255
        mask[:110] = 0; mask[600:] = 0; mask[:,:180] = 0; mask[:,1120:] = 0
        candidates = []
        for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            x,y,w,h = cv2.boundingRect(contour)
            if 25<=w<=58 and 45<=h<=90 and cv2.contourArea(contour)/(w*h)>.3:
                # Revealed troop text extends to the right of the flag. It does
                # not move the underlying castle's interaction point.
                px,py=x+w//2,y+h+33
                if 190<px<1100 and 150<py<615:
                    if all((px-qx)**2+(py-qy)**2>45**2 for qx,qy in candidates):
                        candidates.append((px,py))
        safe=[p for p in candidates if include_shielded or not self.is_shielded(image,p)]
        if not include_shielded:self.last_shield_exclusions=len(candidates)-len(safe)
        return sorted(safe,key=lambda p:(p[0]-640)**2+(p[1]-330)**2)

    def map_enemy_owner(self,image,point):
        x,y=point
        if y<180:return ''
        return self.regions(image,{'attacker':(max(145,x-125),y-159,min(1140,x+125),y-119)})['attacker'].strip()

    def target_levels(self,image,candidates):
        boxes={str(i):(p[0]-25,p[1]-62,p[0]+25,p[1]-30) for i,p in enumerate(candidates)}
        fields=self.regions(image,boxes)
        result={}
        for i,p in enumerate(candidates):
            raw=fields[str(i)].strip()
            result[tuple(p)]=int(raw) if re.fullmatch(r'\d{1,3}',raw) and 1<=int(raw)<=200 else None
        return result

    def enemy_click_point(self,image,body):
        """Use the observed level badge; perspective offsets the actual castle
        from the flag by up to ~70px near the screen edge."""
        px,py=body
        q=(round(px),round(py-55))
        patch=np.asarray(image.crop((q[0]-12,q[1]-12,q[0]+12,q[1]+12))).astype('float32')
        r,g,b=patch[:,:,0],patch[:,:,1],patch[:,:,2]
        badge=((r>195)&(r<250)&(g>95)&(g<150)&(b>100)&(b<155))|((r>210)&(g>40)&(g<80)&(b<45))
        return q if badge.mean()>.25 else tuple(body)

    def own_flags(self,image):
        key=hashlib.blake2b(image.tobytes(),digest_size=16).digest()
        if getattr(self,'_own_flags_key',None)==key:return self._own_flags_result
        template=self.templates.get('own_flag')
        if template is None:return []
        a=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        found=[];rgb=np.array(image).astype('float32')
        for scale in (.85,.95,1,1.075,1.15,1.25,1.35,1.45):
            t=cv2.resize(template,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
            scores=cv2.matchTemplate(a,t,cv2.TM_CCOEFF_NORMED)
            for _ in range(8):
                _,score,_,(x,y)=cv2.minMaxLoc(scores)
                if score<.83:break
                scores[max(0,y-25):y+26,max(0,x-25):x+26]=0
                cx,cy=x+t.shape[1]//2,y+t.shape[0]//2
                if not (145<cx<1120 and 55<cy<590):continue
                body=rgb[max(0,cy-round(13*scale)):cy+round(17*scale),cx+round(25*scale):min(1280,cx+round(140*scale))]
                if body.size and np.mean((body[:,:,2]>120)&(body[:,:,2]>body[:,:,0]*1.3))>.35:
                    found.append((cx,cy,scale,score))
        result=[]
        for item in sorted(found,key=lambda v:-v[3]):
            if all(np.linalg.norm(np.array(item[:2])-other[:2])>25 for other in result):result.append(item)
        self._own_flags_key,self._own_flags_result=key,result[:12]
        return self._own_flags_result

    def own_sources(self,image,owner):
        if not owner:return []
        result=[]
        for cx,cy,scale,_ in self.own_flags(image):
            position=(cx+round((640-cx)*.07),cy+round(95*scale))
            if not (135<position[0]<1145 and 120<position[1]<605):continue
            def box(left,top,right,bottom):
                return (cx+round(left*scale),cy+round(top*scale),min(1280,cx+round(right*scale)),min(720,cy+round(bottom*scale)))
            fields=self.regions(image,{'troops':box(26,-16,85,15)})
            troops=numeric_amount(fields['troops'])
            if troops is None or ('.' in fields['troops'] and not re.search(r'[kKmMgGtT]',fields['troops'])):
                # The white troop icon can merge with the suffix and suppress
                # the whole OCR line. A narrower, higher crop keeps the value.
                alternate=self.regions(image,{'troops':box(26,-16,109,15)})['troops']
                troops=numeric_amount(alternate)
                if '.' in alternate and not re.search(r'[kKmMgGtT]',alternate): troops=None
            if troops is not None and troops>=0:
                # The perspective camera places the castle body slightly
                # toward the centre relative to its floating flag.
                result.append((position,troops))
        return result

    def own_sources_batch(self,image,owner):
        """Read visible garrisons in one OCR batch instead of one call per flag."""
        if not owner:return []
        rows=[];boxes={}
        for cx,cy,scale,_ in self.own_flags(image):
            position=(cx+round((640-cx)*.07),cy+round(95*scale))
            if not (135<position[0]<1145 and 120<position[1]<605):continue
            key='source'+str(len(rows))
            box=(cx+round(26*scale),cy-round(16*scale),min(1280,cx+round(85*scale)),min(720,cy+round(15*scale)))
            boxes[key]=box
            boxes[key+'_wide']=(box[0],box[1],min(1280,cx+round(109*scale)),box[3])
            boxes[key+'_tight']=(box[0]-1,box[1]-1,box[2]+1,box[3]-4)
            rows.append((key,position,box))
        values=self.regions(image,boxes) if boxes else {}
        result=[]
        for key,position,box in rows:
            n=None
            for variant in (key,key+'_tight',key+'_wide'):
                value=values.get(variant,'');n=numeric_amount(value)
                if '.' in value and not re.search(r'[kKmMgGtT]',value):n=None
                if n is not None:break
            if n is not None and n>=0:result.append((position,n))
        return result

    def selected_own_city(self,image,owner,require_troops=True):
        self.last_owned_read=None
        button=self.match(image,'move_troops',.84) or self.match(image,'move_troops_city88',.84)
        if not button or not owner: return None
        x,y,_=button
        if x<180 or y<215: return None
        crown=self.match(image,'own_city_crown',.80,area=(x-210,y-40,x-130,y+35))
        # Upgrade plus troop move controls and a blue banner identify an owned city;
        # terrain behind the floating owner name is unsuitable for reliable OCR.
        banner=np.array(image)[y-200:y-150,x-260:x-130].astype('float32')
        blue=np.mean((banner[:,:,2]>100)&(banner[:,:,2]>banner[:,:,0]*1.25))>.25
        if not crown or not blue: return None
        # At the upper-right edge, the notification/settings HUD covers part
        # of the city name. A readable suffix such as 'ley' is not its identity.
        if y-207<112 and x+60>940:return None
        fields=self.regions(image,{'city_name':(x-90,y-207,min(1280,x+60),y-176),
                                  'troops':(x-64,y-184,min(1280,x+60),y-149)})
        self.last_owned_read=fields.copy()
        troops=numeric_amount(fields['troops'])
        if (troops is None or re.search(r'\d+[.,]\d{3}[kKmMgG]',fields['troops']) or
                ('.' in fields['troops'] and not re.search(r'[kKmMgGtT]',fields['troops']))):
            # The first stroke of the troop icon sits just beyond the G/M.
            # Including it produced 2.01GV / 2.01GJU and hid valid armies.
            alternate=self.regions(image,{'troops':(x-64,y-180,min(1280,x+4),y-151)})['troops']
            self.last_owned_read['troops_retry']=alternate
            troops=numeric_amount(alternate)
            if '.' in alternate and not re.search(r'[kKmMgGtT]',alternate): troops=None
        name=fields['city_name'].strip()
        if (troops is None and require_troops) or (troops is not None and troops<0) or not name: return None
        return {'name':name,'troops':troops,'move_button':button[:2],'position':(x-87,y-40)}

    def own_flag_for_castle(self,image,position):
        candidates=[]
        for cx,cy,scale,_ in self.own_flags(image):
            body=(cx+round((640-cx)*.07),cy+round(95*scale))
            distance=np.linalg.norm(np.array(body)-position)
            if distance<125:candidates.append((distance,(cx,cy)))
        return min(candidates)[1] if candidates else None

    def troop_slider_handle(self,image):
        a=np.array(image)[540:582,360:795].astype('int16')
        white=(np.min(a,axis=2)>205)&((np.max(a,axis=2)-np.min(a,axis=2))<35)
        ys,xs=np.where(white)
        if len(xs)<15 or np.ptp(xs)>45 or np.ptp(ys)>38: return None
        return (int(xs.min()+xs.max())/2+360)

    def troop_bubble_box(self,image):
        a=np.asarray(image.convert('RGB'))[455:540,320:800].astype('int16')
        yellow=(a[:,:,0]>200)&(a[:,:,1]>130)&(a[:,:,2]<100)
        # Friendly support uses a blue bubble with DARK timer glyphs. Attack
        # uses a yellow bubble. Detect both, not just the attack variant.
        blue=(a[:,:,2]>140)&(a[:,:,1]>90)&(a[:,:,2]>a[:,:,0]*1.25)
        mask=cv2.morphologyEx((yellow|blue).astype('uint8')*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        boxes=[]
        for c in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            x,y,w,h=cv2.boundingRect(c)
            if 75<=w<=185 and 25<=h<=85:boxes.append((x,y,w,h))
        if not boxes:return None
        x,y,w,h=max(boxes,key=lambda r:r[2]*r[3]);return x+320,y+455,w,h

    def troop_selection_timer(self,image):
        bubble=self.troop_bubble_box(image)
        if not bubble:return None
        x,y,w,h=bubble;box=(x+4,y+31,x+w-4,min(y+h-2,540))
        if box[3]<=box[1]:return None
        raw=self.regions(image,{'troop_eta':box})['troop_eta'];value=march_countdown(raw)
        if value is not None:return value
        crop=np.asarray(image.crop(box).convert('RGB'))
        white=(crop.min(axis=2)>185)&((crop.max(axis=2)-crop.min(axis=2))<65)
        clean=Image.fromarray(np.where(white,0,255).astype('uint8'))
        value=march_countdown(self.regions(clean,{'troop_eta':(0,0,clean.width,clean.height)})['troop_eta'])
        if value is not None:return value
        # The blue friendly-transfer timer is dark navy, not white.
        dark=(crop[:,:,0]<65)&(crop[:,:,1]<120)&(crop[:,:,2]<155)
        clean=Image.fromarray(np.where(dark,0,255).astype('uint8'))
        return march_countdown(self.regions(clean,{'troop_eta_dark':(0,0,clean.width,clean.height)})['troop_eta_dark'])

    def troop_selection_amount(self,image):
        """Read just the numeric row of the yellow army/ETA bubble."""
        bubble=self.troop_bubble_box(image)
        if not bubble:return None
        x,y,w,h=bubble
        box=(x+9,y+4,x+w-26,min(y+30,540))
        raw=self.regions(image,{'selection_number':box})['selection_number']
        value=amount(raw)
        if value is not None:return value
        crop=np.asarray(image.crop(box).convert('RGB'))
        white=(crop.min(axis=2)>185)&((crop.max(axis=2)-crop.min(axis=2))<65)
        clean=Image.fromarray(np.where(white,0,255).astype('uint8'))
        return amount(self.regions(clean,{'selection_number':(0,0,clean.width,clean.height)})['selection_number'])

    def city_list_army(self,image,y):
        from .defense_logistics import explicit_army
        box=(491,y-16,569,y+16)
        crop=np.asarray(image.crop(box).convert('RGB'))
        white=(crop.min(axis=2)>175)&((crop.max(axis=2)-crop.min(axis=2))<65)
        clean=Image.fromarray(np.where(white,0,255).astype('uint8'))
        text=self.regions(clean,{'list_army_ink':(0,0,clean.width,clean.height)})['list_army_ink']
        return explicit_army(text)

    def outgoing_support_rows(self,image):
        from PIL import ImageDraw
        from .defense_logistics import explicit_army
        work=image.copy();result=[];observed=time.time()
        for _ in range(6):
            pin=self.match(work,'march_pin',.85,area=(620,100,685,650))
            if not pin:break
            x,y=pin[:2]
            fields=self.regions(image,{'support_city':(x-165,y-21,x-31,y+17),
                                       'support_troops':(x+35,y-21,x+129,y+30),
                                       'support_time':(x-430,y-2,x-350,y+31)})
            troops=explicit_army(fields['support_troops']);remaining=march_countdown(fields['support_time'])
            if troops is not None and remaining is not None and fields['support_city'].strip():
                result.append({'city':fields['support_city'].strip(),'troops':troops,'remaining':remaining,'pin':(x,y),'row':y,'observed':observed})
            ImageDraw.Draw(work).rectangle((620,y-36,685,y+36),fill=(0,0,0))
        return result

    def labeled_controls(self,image,box):
        """OCR-backed controls; coordinates come from observed text, not guesses."""
        path=self.data/'ocr-controls.png';image.crop(box).save(path)
        result=self.ocr.read(path,'ko');controls=[]
        for line in result.get('lines',[]):
            words=line.get('words',[])
            if not words:continue
            left=min(w['x'] for w in words);right=max(w['x']+w['w'] for w in words)
            top=min(w['y'] for w in words);bottom=max(w['y']+w['h'] for w in words)
            controls.append({'text':line['text'],'point':(round(box[0]+(left+right)/2),round(box[1]+(top+bottom)/2))})
        return controls

    def march_acceleration_button(self,image,row):
        marker=self.match(image,'march_speedup_free',.9,area=(275,max(95,row-35),360,min(660,row+40)))
        if marker:return marker
        marker=self.match(image,'march_speedup',.9,area=(200,max(95,row-35),1080,min(660,row+40)))
        if marker:return marker
        for control in self.labeled_controls(image,(200,max(95,row-35),1080,min(660,row+40))):
            if ''.join(control['text'].split()) in ('가속','Speedup','SpeedUp'):
                return (*control['point'],1.)
        return None

    def acceleration_option(self,image):
        controls=self.labeled_controls(image,(200,100,1080,650))
        texts=[c['text'] for c in controls];joined=' '.join(texts)
        if '가속' not in joined or any(t in joined for t in ('₩','KRW','구매','결제','다이아')):return None
        if not re.search(r'보유\s*[:：]?\s*[1-9]\d*',joined):return None
        after=None
        for text in texts:
            match=re.search(r'(?:가속\s*후|적용\s*후)\s*[:：]?\s*(.+)',text)
            if match:after=march_countdown(match[1])
        buttons=[c for c in controls if ''.join(c['text'].split()) in ('사용','적용')]
        if after is None or len(buttons)!=1:return None
        return {'remaining_after':after,'button':buttons[0]['point']}

    def same_viewport(self,image,reference):
        """Validate terrain alignment so old city coordinates cannot be used after a pan."""
        matrix=self.viewport_transform(reference,image)
        if matrix is None: return False
        points=((300,200),(1000,200),(300,500),(1000,500),(640,350))
        return all(np.linalg.norm(np.array(self.transform_point(matrix,p))-p)<8 for p in points)

    @staticmethod
    def sweep_frame_delta(before,after):
        """Coarse map change for camera travel only, never for combat coordinates."""
        from PIL import ImageFilter
        def terrain(image):
            crop=image.crop((180,150,1120,590)).resize((110,55),Image.Resampling.BILINEAR)
            return np.asarray(crop.filter(ImageFilter.GaussianBlur(1))).astype('float32')
        diff=np.abs(terrain(before)-terrain(after)).mean(axis=2)
        return float(np.mean(diff>12)),float(diff.mean())

    @staticmethod
    def sweep_motion_delta(before,after):
        """A sea-dominated view still moves when its islands move.
        Ignore gray water and white wave animation in this travel-only check.
        This evidence never supplies a castle's combat coordinates.
        """
        from PIL import ImageFilter
        def terrain(image):
            return np.asarray(image.crop((180,150,1120,590)).resize((110,55),Image.Resampling.BILINEAR)
                              .filter(ImageFilter.GaussianBlur(1))).astype('float32')
        a,b=terrain(before),terrain(after)
        land=((a.max(axis=2)-a.min(axis=2))>55)|((b.max(axis=2)-b.min(axis=2))>55)
        if np.mean(land)<.03:return Vision.sweep_frame_delta(before,after)
        diff=np.abs(a-b).mean(axis=2)[land]
        return float(np.mean(diff>12)),float(diff.mean())

    def deployment_restriction(self,image):
        """Read the actual gray restriction tooltip, never clan or keybind icons."""
        a=np.asarray(image.convert('RGB')).astype('int16')
        gray=((a.max(axis=2)-a.min(axis=2)<28)&(a.min(axis=2)>65)&(a.max(axis=2)<175)).astype('uint8')*255
        contours=cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        boxes={}
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if w>180 and 14<=h<=85:
                boxes['restriction'+str(len(boxes))]=(max(0,x-2),max(0,y-2),min(image.width,x+w+2),min(image.height,y+h+2))
        if not boxes:return None
        for text in self.regions(image,boxes).values():
            clean=re.sub(r'\s+','',text)
            if ('제한' in clean and ('병력' in clean or '배치' in clean)) or ('영웅점수' in clean and '적을경우' in clean):
                return text
        return None

    def terrain_features(self,image,kind):
        key=(kind,hashlib.blake2b(image.tobytes(),digest_size=16).digest())
        if key in self.feature_cache:
            self.feature_cache.move_to_end(key);return self.feature_cache[key]
        mask=np.zeros((720,1280),dtype=np.uint8)
        mask[150:590,160 if kind=='sift' else 200:1140 if kind=='sift' else 1120]=255
        # Water waves are a screen-space shader: they can match at identity
        # while islands have moved hundreds of pixels. Only colored terrain
        # supplies camera geometry; gray water and white wave flecks do not.
        rgb=np.asarray(image).astype('int16');r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
        water=((r>55)&(r<105)&(g>r+6)&(g<125)&(b>r+8)&(b<140)&
               ((rgb.max(axis=2)-rgb.min(axis=2))<40)).astype('uint8')*255
        water=cv2.morphologyEx(water,cv2.MORPH_CLOSE,np.ones((17,17),np.uint8))
        mask[water>0]=0
        detector=cv2.SIFT_create(nfeatures=2400) if kind=='sift' else cv2.ORB_create(nfeatures=int(kind))
        result=detector.detectAndCompute(cv2.cvtColor(np.array(image),cv2.COLOR_RGB2GRAY),mask)
        self.feature_cache[key]=result
        while len(self.feature_cache)>64:self.feature_cache.popitem(last=False)
        return result

    def viewport_transform(self,reference,image):
        """Map known terrain coordinates into a newly observed camera position."""
        ka,da=self.terrain_features(reference,'800')
        kb,db=self.terrain_features(image,'800')
        if da is None or db is None: return None
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
        good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
        matrix,inliers=None,None
        if len(good)>=14:
            a=np.float32([ka[m.queryIdx].pt for m in good]); b=np.float32([kb[m.trainIdx].pt for m in good])
            matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3)
        if matrix is not None and np.sum(inliers)>=12:
            scale=float(np.hypot(matrix[0,0],matrix[0,1]))
            span=np.ptp(a[inliers.ravel().astype(bool)],axis=0)
            # A fixed map-edge pointer can supply >12 keypoints on its own.
            # Its identity transform is not evidence that the terrain stayed put.
            if .99<scale<1.01 and abs(matrix[0,1])<.01 and span[0]>180 and span[1]>90:return matrix
        # The game's perspective camera changes apparent terrain scale on a pan.
        # A translation-only tolerance rejected the real observed ~6% change.
        ka,da=self.terrain_features(reference,'4000')
        kb,db=self.terrain_features(image,'4000')
        if da is None or db is None: return None
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
        good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance]
        if len(good)<35: return None
        a=np.float32([ka[m.queryIdx].pt for m in good]); b=np.float32([kb[m.trainIdx].pt for m in good])
        homography,inliers=cv2.findHomography(a,b,cv2.RANSAC,3)
        if homography is None or not np.all(np.isfinite(homography)) or inliers is None: return None
        inside=inliers.ravel().astype(bool)
        if np.sum(inside)<30 or np.mean(inside)<.2: return None
        # Reject a fit supported by one repeated flag/castle rather than the map.
        span=np.ptp(a[inside],axis=0)
        if span[0]<180 or span[1]<90: return None
        projected=cv2.perspectiveTransform(a[inside].reshape(-1,1,2),homography).reshape(-1,2)
        if np.median(np.linalg.norm(projected-b[inside],axis=1))>2: return None
        for x,y in ((300,200),(1000,200),(300,500),(1000,500),(640,350)):
            points=np.float32([[[x,y],[x+40,y],[x,y+40]]])
            mapped=cv2.perspectiveTransform(points,homography)[0]
            if not np.all(np.isfinite(mapped)) or np.linalg.norm(mapped[0]-[x,y])>650: return None
            u,v=(mapped[1]-mapped[0])/40,(mapped[2]-mapped[0])/40
            if not (.75<np.linalg.norm(u)<1.25 and .75<np.linalg.norm(v)<1.25): return None
            if u[0]<=0 or v[1]<=0 or abs(u[1])>.2 or abs(v[0])>.3: return None
        return homography

    @staticmethod
    def transform_point(matrix,point):
        mapped=matrix@np.array([point[0],point[1],1.])
        if len(mapped)==3: mapped=mapped[:2]/mapped[2]
        x,y=mapped
        return (round(float(x)),round(float(y)))

    def muster_transform(self,reference,image):
        """Track wider pans using distinctive terrain, not repeated castle crests."""
        ka,da=self.terrain_features(reference,'sift')
        kb,db=self.terrain_features(image,'sift')
        if da is not None and db is not None:
            pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
            good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.65*b.distance]
            if len(good)>=30:
                a=np.float32([ka[m.queryIdx].pt for m in good]);b=np.float32([kb[m.trainIdx].pt for m in good])
                h,inliers=cv2.findHomography(a,b,cv2.RANSAC,3)
                if h is not None and np.all(np.isfinite(h)) and inliers is not None:
                    inside=inliers.ravel().astype(bool);span=np.ptp(a[inside],axis=0)
                    if np.sum(inside)>=24 and np.mean(inside)>=.2 and span[0]>180 and span[1]>90:
                        valid=True
                        for x,y in ((300,200),(1000,200),(300,500),(1000,500),(640,350)):
                            mapped=cv2.perspectiveTransform(np.float32([[[x,y],[x+40,y],[x,y+40]]]),h)[0]
                            u,v=(mapped[1]-mapped[0])/40,(mapped[2]-mapped[0])/40
                            if (not np.all(np.isfinite(mapped)) or np.linalg.norm(mapped[0]-[x,y])>650 or
                                not .75<np.linalg.norm(u)<1.25 or not .75<np.linalg.norm(v)<1.25 or
                                u[0]<=0 or v[1]<=0 or abs(u[1])>.2 or abs(v[0])>.3):valid=False;break
                        if valid:return h
        matrix=self.viewport_transform(reference,image)
        if matrix is not None and all(np.linalg.norm(np.array(self.transform_point(matrix,p))-p)<15
            for p in ((300,200),(1000,500),(640,350))):return matrix
        return None

    def has_incoming_warning(self,image):
        # Requires both a red march banner AND a countdown; stationary enemy labels do not qualify.
        a=np.array(image)
        mask=((a[:,:,0]>155)&(a[:,:,1]<95)&(a[:,:,2]<100)).astype('uint8')*255
        contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c)
            if 45<w<150 and 28<h<68 and 150<x<1150 and 100<y<600:
                region=self.regions(image,{'warning':(x,y,x+w,y+h)})['warning']
                if re.search(r'\d+\s*[ms분초]',region,re.I): return region
        return None
