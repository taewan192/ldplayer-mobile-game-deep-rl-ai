from dataclasses import dataclass
import math


def attack_power(troops,bonus_percent):
    return troops*(1+bonus_percent/100)


def required_attack(defense,bonus_percent,margin=1.5):
    if defense<0 or bonus_percent<=-100 or margin<1: raise ValueError('Invalid combat inputs')
    return math.ceil(defense*margin/(1+bonus_percent/100))


def march_seconds(text):
    import re
    hours=re.search(r'(\d+)\s*(?:h|시간)',text,re.I)
    minutes=re.search(r'(\d+)\s*(?:m|분)',text,re.I)
    seconds=re.search(r'(\d+)\s*(?:s|초)',text,re.I)
    if not any((hours,minutes,seconds)): return None
    return ((int(hours[1])*3600 if hours else 0)+
            (int(minutes[1])*60 if minutes else 0)+(int(seconds[1]) if seconds else 0))


@dataclass(frozen=True)
class MusterDonor:
    position:tuple
    troops:int
    distance:float


@dataclass(frozen=True)
class MusterContribution:
    position:tuple
    troops:int
    available:int


@dataclass(frozen=True)
class MusterPlan:
    required:int
    staging_reserve:int
    contributions:tuple


def plan_muster(required,staging_troops,donors,max_fraction=.35,strategy='nearest',quantum=0,staging_reserve=None):
    """Collect only the shortfall, preserving the original reserve at every source."""
    if (not math.isfinite(required) or not math.isfinite(staging_troops) or
            required<=0 or staging_troops<0 or not 0<max_fraction<=.70):
        raise ValueError('Invalid muster inputs')
    if strategy not in ('nearest','fewest','single','single_nearest'): raise ValueError('Unknown muster strategy')
    reserve=math.ceil(staging_troops*(1-max_fraction)) if staging_reserve is None else staging_reserve
    if not math.isfinite(reserve) or reserve<0: raise ValueError('Invalid staging reserve')
    if not 0<=quantum<=.5: raise ValueError('Invalid slider quantum')
    goal=math.ceil(required)+reserve
    if quantum:
        totals=[max(required/(i*quantum),reserve/(1-i*quantum))
                for i in range(1,math.ceil(1/quantum)) if i*quantum<1]
        goal=math.ceil(min(totals))
    shortfall=max(0,goal-math.floor(staging_troops))
    if not shortfall: return MusterPlan(math.ceil(required),reserve,())
    unique={}
    for donor in donors:
        if donor.troops<0 or not math.isfinite(donor.troops) or not math.isfinite(donor.distance):
            raise ValueError('Invalid donor')
        # A duplicated screen detection cannot contribute twice.
        if donor.position not in unique: unique[donor.position]=donor
    ordered=sorted(unique.values(),key=(lambda d:d.distance) if strategy in ('nearest','single_nearest') else (lambda d:(-d.troops,d.distance)))
    fraction=math.floor((max_fraction+1e-9)/quantum)*quantum if quantum else max_fraction
    if strategy in ('single','single_nearest'):
        ordered=[donor for donor in ordered if math.floor(donor.troops*fraction)>=shortfall]
        if not ordered:return None
        ordered=ordered[:1]
    if sum(math.floor(d.troops*fraction) for d in ordered)<shortfall: return None
    result=[]
    for donor in ordered:
        if shortfall<=0:break
        available=math.floor(donor.troops*fraction)
        take=min(shortfall,available)
        if quantum and take:
            take=min(available,math.ceil((take/donor.troops-1e-9)/quantum)*quantum*donor.troops)
            take=math.floor(take)
        if take:
            result.append(MusterContribution(donor.position,take,available))
            shortfall-=take
        if shortfall<=0: break
    return MusterPlan(math.ceil(required),reserve,tuple(result))


@dataclass
class Reinforcement:
    source:str
    troops:int
    travel_seconds:float


def reinforcement_plan(required,remaining_seconds,candidates,buffer_seconds=12):
    """Only routes that can arrive before impact. UI execution requires observed route data."""
    result=[]
    for candidate in sorted(candidates,key=lambda c:c.travel_seconds):
        if candidate.travel_seconds+buffer_seconds>=remaining_seconds: continue
        take=min(required,candidate.troops)
        if take>0:
            result.append(Reinforcement(candidate.source,take,candidate.travel_seconds))
            required-=take
        if required<=0: return result
    return []


def defense_shortfall(attack,total_defense,city_bonus,guardian_bonus,margin=1.3):
    """Use observed city total (walls included) and additive troop defense bonuses."""
    values=(attack,total_defense,city_bonus,guardian_bonus,margin)
    if any(not math.isfinite(v) for v in values) or attack<=0 or total_defense<0 or margin<1:
        raise ValueError('Invalid observed defense inputs')
    multiplier=1+(city_bonus+guardian_bonus)/100
    if multiplier<=0: raise ValueError('Invalid defense multiplier')
    # Screen totals are rounded. Do not count the optimistic edge of the display.
    return max(0,math.ceil((attack*margin-total_defense*.99)/multiplier))


@dataclass(frozen=True)
class SupportRoute:
    source:str
    position:tuple
    total:int
    travel_seconds:float


def support_plan(required,impact,now,routes,max_fraction=.35,buffer=20,setup_seconds=12, strategy='fastest'):
    """All-or-nothing capacity check, including serial UI setup and 10% slider units."""
    if required<0 or not all(math.isfinite(v) for v in (required,impact,now,buffer,setup_seconds)):
        raise ValueError('Invalid defense deadline')
    if not 0<max_fraction<=.7 or buffer<0 or setup_seconds<0: raise ValueError('Invalid reserve')
    if strategy not in ('fastest','fewest'): raise ValueError('Invalid defense strategy')
    if required==0: return ()
    unique={r.source:r for r in routes}
    ordered=sorted(unique.values(),key=(lambda r:r.travel_seconds) if strategy=='fastest' else (lambda r:(-r.total,r.travel_seconds)))
    fraction=math.floor((max_fraction+1e-9)*10)/10
    elapsed=0; result=[]
    for route in ordered:
        if (not math.isfinite(route.travel_seconds) or not math.isfinite(route.total) or
            route.travel_seconds<0 or route.total<=0): continue
        if now+elapsed+setup_seconds+route.travel_seconds+buffer>=impact: continue
        cap=math.floor(route.total*fraction)
        take=min(cap,math.ceil((required/route.total-1e-9)*10)*route.total/10)
        take=math.floor(take)
        if take<=0: continue
        result.append((route,take)); required-=take; elapsed+=setup_seconds
        if required<=0: return tuple(result)
    return None
