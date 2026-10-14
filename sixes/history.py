"""Histories are confined to a player-role and strictly earlier dates."""
import numpy as np
import pandas as pd
from .core import C, MEMORIES


def snapshot(player, cutoff):
    x=player.loc[player.date.lt(cutoff)].sort_values(['date','match_id','innings','delivery_sequence'])
    y=x.is_six.to_numpy(float);n=len(y)
    result={'expanding':(float(y.sum()),float(n))}
    current=x.loc[x.season.eq(int(cutoff[:4])),'is_six']
    result['current']=(float(current.sum()),float(len(current)))
    for k in [*C['rolling_balls'],C['stress_rolling_balls']]:
        result[f'rolling_{k}']=(float(y[-k:].sum()),float(min(k,n)))
    for h in C['ew_balls_halflives']:
        w=2**(-np.arange(n-1,-1,-1)/h);result[f'ew_balls_{h}']=(float(w@y),float(w.sum()))
    ages=(pd.Timestamp(cutoff)-pd.to_datetime(x.date)).dt.days.to_numpy()
    for h in C['ew_days_halflives']:
        w=2**(-ages/h);result[f'ew_days_{h}']=(float(w@y),float(w.sum()))
    return result


def features(d):
    rows=[]
    for (pid,role),g in d.groupby(['batter_id','role'],sort=True):
        g=g.sort_values(['date','match_id','innings','delivery_sequence'])
        y=g.is_six.to_numpy(float);prefix=np.r_[0,np.cumsum(y)]
        days={h:np.zeros(2) for h in C['ew_days_halflives']}
        balls={h:np.zeros(2) for h in C['ew_balls_halflives']}
        offset=0;start=0;last_year=None;last_day=None
        for date,day in g.groupby('date',sort=True):
            year=int(day.season.iloc[0]);now=pd.Timestamp(date)
            if year!=last_year:start=offset
            if last_day is not None:
                for h in days:days[h]*=2**(-(now-last_day).days/h)
            row=dict(batter_id=pid,role=role,date=date,season=year,history_last_date=None if last_day is None else last_day.date().isoformat(),
                     expanding_B=prefix[offset],expanding_N=offset,current_B=prefix[offset]-prefix[start],current_N=offset-start)
            for k in [*C['rolling_balls'],C['stress_rolling_balls']]:
                i=max(0,offset-k);row[f'rolling_{k}_B']=prefix[offset]-prefix[i];row[f'rolling_{k}_N']=offset-i
            for h in days:row[f'ew_days_{h}_B'],row[f'ew_days_{h}_N']=days[h]
            for h in balls:row[f'ew_balls_{h}_B'],row[f'ew_balls_{h}_N']=balls[h]
            rows.append(row)
            n=len(day);values=y[offset:offset+n]
            for h in balls:
                decay=2**(-1/h);w=decay**np.arange(n-1,-1,-1)
                balls[h]=balls[h]*decay**n+np.array([w@values,w.sum()])
            for h in days:days[h]+=np.array([values.sum(),n])
            offset+=n;last_day=now;last_year=year
    return pd.DataFrame(rows)


def selected_weights(player, name, cutoff):
    assert player.date.lt(cutoff).all()
    n=len(player)
    if name=='expanding':return np.ones(n)
    if name=='current':return player.season.eq(int(cutoff[:4])).to_numpy(float)
    if name.startswith('rolling_'):return (np.arange(n)>=max(0,n-int(name.rsplit('_',1)[1]))).astype(float)
    if name.startswith('ew_balls_'):return 2**(-np.arange(n-1,-1,-1)/int(name.rsplit('_',1)[1]))
    if name.startswith('ew_days_'):return 2**(-(pd.Timestamp(cutoff)-pd.to_datetime(player.date)).dt.days.to_numpy()/int(name.rsplit('_',1)[1]))
    raise ValueError(name)
