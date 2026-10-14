from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from models import fit_prior, posterior

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT/'config/sixes_protocol.json').read_text())
OUT = ROOT/'results/sixes'
OUT.mkdir(exist_ok=True, parents=True)
KEYS = ['season', 'date', 'match_id', 'innings', 'batter_id', 'role']
CANDIDATES = ['expanding', 'current'] + [f'rolling_{k}' for k in C['rolling_balls']] + [f'ew_balls_{h}' for h in C['ew_balls_halflives']] + [f'ew_days_{h}' for h in C['ew_days_halflives']]
MEMORIES = list(dict.fromkeys(CANDIDATES + ['rolling_30']))


def save(x, name):
    x.to_csv(OUT/f'{name}.csv', index=False)


def read(name):
    return pd.read_parquet(ROOT/f'data/processed/{name}.parquet')


def protected_hashes():
    names = subprocess.check_output(['git','ls-tree','-r','--name-only',C['baseline_commit']],cwd=ROOT,text=True).splitlines()
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names if p not in ['README.md','run_all.py','.gitignore']}


def target(deliveries):
    x = deliveries.loc[deliveries.primary_eligible & ~deliveries.is_super_over & deliveries.batting_entry_order.isin(C['positions'])].copy()
    x['is_six'] = (x.batter_runs.eq(6) & ~x.non_boundary).astype(int)
    x['role'] = x.batting_entry_order.where(x.batting_entry_order.gt(2),0).astype(int)
    assert not x.delivery_id.duplicated().any()
    return x


def aggregate(d):
    return d.groupby(KEYS,as_index=False).agg(N=('is_six','size'),B=('is_six','sum'),standard_conditions=('standard_conditions','all'))


def fit_group(counts):
    prior = fit_prior(counts.B,counts.N)
    prior['pooled'] = float(counts.B.sum()/counts.N.sum())
    return prior


def annual_priors(d, years):
    rows=[]
    for year in years:
        train=d.loc[d.season.between(year-C['prior_lookback_seasons'],year-1)]
        for role in C['roles']:
            g=train.loc[train.role.eq(role)].groupby('batter_id',as_index=False).agg(N=('is_six','size'),B=('is_six','sum'))
            fallback=len(g)<2
            if fallback:g=train.groupby('batter_id',as_index=False).agg(N=('is_six','size'),B=('is_six','sum'))
            p=fit_group(g)
            rows.append(dict(season=year,role=role,**p,train_max_date=train.date.max(),fallback=fallback,
                             prior_sd=np.sqrt(p['mu']*(1-p['mu'])/(p['strength']+1))))
    return pd.DataFrame(rows)


def losses(B,N,p,clip=None):
    B,N,p=np.asarray(B,float),np.asarray(N,float),np.asarray(p,float)
    pc=np.clip(p,C['logloss_clip'] if clip is None else clip,1-(C['logloss_clip'] if clip is None else clip))
    return -B*np.log(pc)-(N-B)*np.log1p(-pc), B*(1-p)**2+(N-B)*p**2


def metrics(wide, models=None):
    rows=[]
    models=models or [c[2:] for c in wide if c.startswith('p_')]
    for model in models:
        ll,bs=losses(wide.B,wide.N,wide['p_'+model])
        pl=pd.DataFrame(dict(pid=wide.batter_id,N=wide.N,ll=ll,bs=bs)).groupby('pid').sum()
        rows.append(dict(model=model,N=int(wide.N.sum()),B=int(wide.B.sum()),players=wide.batter_id.nunique(),
                         log_loss=ll.sum()/wide.N.sum(),brier=bs.sum()/wide.N.sum(),
                         player_macro_log_loss=(pl.ll/pl.N).mean(),player_macro_brier=(pl.bs/pl.N).mean()))
    return pd.DataFrame(rows)


def paired(wide, a, b, unit='date'):
    la,ba=losses(wide.B,wide.N,wide['p_'+a]);lb,bb=losses(wide.B,wide.N,wide['p_'+b])
    frame=wide[['season','date','batter_id','N']].assign(ll=la-lb,bs=ba-bb,reference_ll=lb)
    cols=['ll','bs','reference_ll','N']
    rng=np.random.default_rng(C['seed']);total=np.zeros((C['bootstrap_replicates'],4))
    blocks=[g.groupby('date')[cols].sum().to_numpy() for _,g in frame.groupby('season')] if unit=='date' else [frame.groupby('batter_id')[cols].sum().to_numpy()]
    for block in blocks:
        total+=block[rng.integers(len(block),size=(len(total),len(block)))].sum(axis=1)
    delta=total[:,:2]/total[:,3:4];skill=-100*total[:,0]/total[:,2]
    return dict(model=a,reference=b,unit=unit,N=int(wide.N.sum()),date_blocks=wide[['season','date']].drop_duplicates().shape[0],
                log_loss_difference=float((la-lb).sum()/wide.N.sum()),log_loss_low=float(np.quantile(delta[:,0],.025)),log_loss_high=float(np.quantile(delta[:,0],.975)),
                brier_difference=float((ba-bb).sum()/wide.N.sum()),brier_low=float(np.quantile(delta[:,1],.025)),brier_high=float(np.quantile(delta[:,1],.975)),
                skill_pct=float(100*(1-la.sum()/lb.sum())),skill_low=float(np.quantile(skill,.025)),skill_high=float(np.quantile(skill,.975)))


def calibration(wide, models):
    rows=[]
    for model in models:
        x=wide[['B','N']].copy();x['p']=wide['p_'+model]
        x['bin']=pd.cut(x.p,[-.001,.025,.05,.075,.1,.125,.15,.2,.3,1.001],right=False)
        for label,g in x.groupby('bin',observed=True):
            rows.append(dict(model=model,bin=str(label),N=int(g.N.sum()),predicted=np.average(g.p,weights=g.N),observed=g.B.sum()/g.N.sum()))
    return pd.DataFrame(rows)
