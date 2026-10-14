"""Early-only spike flags, followed by explicitly retrospective case selection."""
import numpy as np
import pandas as pd
from .core import *


def run():
    d=pd.read_parquet(OUT/'deliveries.parquet');splits=read('match_splits')
    h=d.merge(splits[['match_id','half','cutoff_date']],on='match_id',validate='many_to_one')
    pri=pd.read_csv(OUT/'annual_priors.csv').set_index(['season','role'])
    rows=[]
    for (year,role,pid),early in h.loc[h.season.ge(2018)&h.half.eq('early')].groupby(['season','role','batter_id']):
        past=h.loc[h.batter_id.eq(pid)&h.role.eq(role)&h.season.lt(year)]
        if len(past)<C['case_min_prior_N'] or len(early)<C['case_min_early_N']:continue
        p=pri.loc[(year,role)];baseline=float(posterior(past.is_six.sum(),len(past),p));rate=float(early.is_six.mean())
        if rate-baseline<C['case_spike_probability_points']:continue
        late=h.loc[h.batter_id.eq(pid)&h.role.eq(role)&h.season.eq(year)&h.half.eq('late')]
        later=float(late.is_six.mean()) if len(late) else np.nan
        simple=float(posterior(early.is_six.sum(),len(early),p))
        retained=float(posterior(past.is_six.sum()+early.is_six.sum(),len(past)+len(early),p))
        rows.append(dict(player=early.batter_name.iloc[-1],batter_id=pid,season=year,role=role,cutoff_date=early.cutoff_date.iloc[0],
                         prior_N=len(past),early_N=len(early),early_B=int(early.is_six.sum()),late_N=len(late),late_B=int(late.is_six.sum()),
                         baseline=baseline,early_rate=rate,early_only_shrinkage=simple,history_retaining_shrinkage=retained,late_rate=later,
                         early_excess=rate-baseline,reversion_example=bool(len(late)>=C['case_min_later_N_for_reversion'] and later<=baseline+C['case_reversion_tolerance'])))
    flags=pd.DataFrame(rows).sort_values(['early_excess','season','batter_id'],ascending=[False,True,True])
    save(flags,'all_early_spike_flags')
    selected=flags.loc[flags.reversion_example].drop_duplicates('batter_id').head(3)
    save(selected,'temporary_form_cases')
    trajectories=[]
    for r in selected.itertuples():
        g=d.loc[d.batter_id.eq(r.batter_id)&d.role.eq(r.role)]
        s=g.groupby('season',as_index=False).agg(N=('is_six','size'),B=('is_six','sum'))
        s['rate']=s.B/s.N;s['player']=r.player;s['batter_id']=r.batter_id;s['role']=r.role
        trajectories.append(s)
    save(pd.concat(trajectories,ignore_index=True) if trajectories else pd.DataFrame(columns=['season','N','B','rate','player','batter_id','role']),'case_trajectories')
    print(f'{len(flags)} early spike flags; {len(selected)} retrospective examples',flush=True)


if __name__=='__main__':run()
