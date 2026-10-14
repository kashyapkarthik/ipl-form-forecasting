"""Qualify individual six-rate changes using shrinkage and matched memory rules."""
import json
import hashlib
import numpy as np
import pandas as pd
from .core import ROOT, C
from .history import selected_weights
from models import fit_prior

OUT=ROOT/'results/player_memory'
REPS=1000
SEED=20261016
PRIMARY=['rolling_120','rolling_300','rolling_600','rolling_1200','ew_balls_120','ew_balls_300','ew_balls_600','ew_balls_1200','expanding']
MODELS=PRIMARY+['rolling_30','season_only']


def aligned_weights(g,model,cutoff):
    """Weights on a common event array; future outcomes cannot enter earlier counts."""
    eligible=g.date.lt(cutoff).to_numpy()
    w=np.zeros(len(g));w[eligible]=selected_weights(g.loc[eligible],model,cutoff)
    return w


def primary_simultaneous_bands(point,draw):
    """Conditional max-standardised bootstrap bands over a fixed comparison family."""
    sd=np.std(draw,axis=0,ddof=1);valid=sd>1e-12
    deviations=np.zeros_like(draw)
    deviations[:,valid]=np.abs((draw[:,valid]-draw[:,valid].mean(axis=0))/sd[valid])
    critical=float(np.quantile(deviations.max(axis=1),.95))
    return point-critical*sd,point+critical*sd,critical


def prior_resamples(d):
    train=d.loc[d.season.between(2023,2025)].groupby(['role','batter_id'],as_index=False).agg(B=('is_six','sum'),N=('is_six','size'))
    ids=sorted(train.batter_id.unique());locations={pid:i for i,pid in enumerate(ids)}
    groups={r:g for r,g in train.groupby('role')};rng=np.random.default_rng(SEED)
    rows=[];draws={r:np.empty((REPS,2)) for r in C['roles']}
    for rep in range(REPS):
        w=rng.exponential(size=len(ids))
        for role,g in groups.items():
            p=fit_prior(g.B,g.N,w[g.batter_id.map(locations).to_numpy()])
            draws[role][rep]=[p['alpha'],p['strength']]
            rows.append(dict(replicate=rep,role=role,**p))
        if (rep+1)%100==0:print(f'Prior refits {rep+1}/{REPS}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'prior_resamples.csv',index=False)
    return draws,frame


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    d=pd.read_parquet(ROOT/'results/sixes/deliveries.parquet')
    candidates=pd.read_csv(ROOT/'results/player_improvement/player_changes.csv').sort_values('batter_id')
    prior=pd.read_csv(ROOT/'results/sixes/annual_priors.csv').query('season==2026').set_index('role')
    prior.to_csv(OUT/'fixed_priors.csv')
    before=d.loc[d.season.eq(2026),'date'].min();after='2026-06-01'
    draws,prior_frame=prior_resamples(d)
    rows=[];role_rows=[];change_draws=[]
    for c in candidates.itertuples():
        player=d.loc[d.batter_id.eq(c.batter_id)&d.date.lt(after)].sort_values(['date','match_id','innings','delivery_sequence']).copy()
        weights=player.loc[player.season.eq(2026)].groupby('role').size();weights=weights/weights.sum()
        base=np.zeros(3) # shrunk, raw, population
        base_draw=np.zeros(REPS)
        points=np.zeros((len(MODELS),6)) # before shrunk, after shrunk, raw before/after, data weight before/after
        samples=np.zeros((REPS,len(MODELS),2))
        rng=np.random.default_rng(SEED+int(c.batter_id,16))
        # Same season-level innings resample across roles and memory variants.
        keys=player[['season','match_id','innings']].drop_duplicates().sort_values(['season','match_id','innings']).reset_index(drop=True)
        counts=np.zeros((REPS,len(keys)))
        for _,block in keys.groupby('season'):
            counts[:,block.index]=rng.multinomial(len(block),np.full(len(block),1/len(block)),size=REPS)
        key_map={tuple(r):i for i,r in enumerate(keys.to_numpy())}
        for role,mix in weights.items():
            g=player.loc[player.role.eq(role)];y=g.is_six.to_numpy(float);p=prior.loc[role]
            block_ids=np.array([key_map[tuple(r)] for r in g[['season','match_id','innings']].to_numpy()])
            wbase=g.season.between(2023,2025).to_numpy(float)
            columns=[wbase*y,wbase]
            specs=[]
            for model in MODELS:
                if model=='season_only':wa=g.season.eq(2025).to_numpy(float);wb=g.season.eq(2026).to_numpy(float)
                else:wa=aligned_weights(g,model,before);wb=aligned_weights(g,model,after)
                specs.append((model,wa,wb));columns.extend([wa*y,wa,wb*y,wb])
            values=np.stack(columns,axis=1);blocks=np.zeros((len(keys),values.shape[1]));np.add.at(blocks,block_ids,values)
            boot=counts@blocks
            alpha=draws[role][:,0];strength=draws[role][:,1]
            base[0]+=mix*(p.alpha+wbase@y)/(p.strength+wbase.sum())
            base[1]+=mix*((wbase@y)/wbase.sum() if wbase.sum() else p.pooled)
            base[2]+=mix*p.pooled
            base_draw+=mix*(alpha+boot[:,0])/(strength+boot[:,1])
            for j,(model,wa,wb) in enumerate(specs):
                n0,n1=wa.sum(),wb.sum();b0,b1=wa@y,wb@y
                q0=(p.alpha+b0)/(p.strength+n0);q1=(p.alpha+b1)/(p.strength+n1)
                raw0=b0/n0 if n0 else p.pooled;raw1=b1/n1 if n1 else p.pooled
                points[j]+=mix*np.array([q0,q1,raw0,raw1,n0/(n0+p.strength),n1/(n1+p.strength)])
                offset=2+4*j
                samples[:,j,0]+=mix*(alpha+boot[:,offset])/(strength+boot[:,offset+1])
                samples[:,j,1]+=mix*(alpha+boot[:,offset+2])/(strength+boot[:,offset+3])
                role_rows.append(dict(batter_id=c.batter_id,player=c.player,role=int(role),role_weight=mix,model=model,
                                      before_B=b0,before_N=n0,after_B=b1,after_N=n1,before=q0,after=q1,
                                      before_last_date=g.loc[g.date.lt(before),'date'].max() if g.date.lt(before).any() else None,
                                      prior_2025_N=int(g.season.eq(2025).sum()),
                                      alpha=p.alpha,strength=p.strength,baseline_B=wbase@y,baseline_N=wbase.sum()))
        for j,model in enumerate(MODELS):
            q0,q1,raw0,raw1,weight0,weight1=points[j];delta=samples[:,j,1]-samples[:,j,0]
            low,high=np.quantile(delta,[.025,.975]);bl,bh=np.quantile(samples[:,j,1]-base_draw,[.025,.975])
            rows.append(dict(batter_id=c.batter_id,player=c.player,model=model,primary=model in PRIMARY,
                             before=q0,after=q1,change=q1-q0,change_low=low,change_high=high,
                             raw_before=raw0,raw_after=raw1,raw_change=raw1-raw0,
                             baseline_2023_25=base[0],baseline_raw=base[1],population=base[2],
                             above_baseline=q1-base[0],above_baseline_low=bl,above_baseline_high=bh,
                             personal_weight_before=weight0,personal_weight_after=weight1,
                             latest_N=c.latest_N,latest_B=c.latest_B))
            if model in PRIMARY:change_draws.append(delta)
    results=pd.DataFrame(rows);mask=results.primary
    all_draw=np.stack(change_draws,axis=1);lo,hi,critical=primary_simultaneous_bands(results.loc[mask,'change'].to_numpy(),all_draw)
    results.loc[mask,'simultaneous_low']=lo;results.loc[mask,'simultaneous_high']=hi
    results.to_csv(OUT/'memory_changes.csv',index=False);pd.DataFrame(role_rows).to_csv(OUT/'role_details.csv',index=False)
    summary=[]
    for pid,g in results.loc[mask].groupby('batter_id'):
        summary.append(dict(batter_id=pid,player=g.player.iloc[0],positive_methods=int(g.change.gt(0).sum()),methods=len(g),
                            smallest_change=g.change.min(),largest_change=g.change.max(),
                            pointwise_positive=int(g.change_low.gt(0).sum()),simultaneous_positive=int(g.simultaneous_low.gt(0).sum()),
                            min_personal_weight=g.personal_weight_after.min(),max_personal_weight=g.personal_weight_after.max()))
    pd.DataFrame(summary).to_csv(OUT/'player_summary.csv',index=False)
    role=pd.DataFrame(role_rows)
    audit={'latest_season':2026,'before_cutoff':before,'after_cutoff':after,'prior_training_seasons':[2023,2024,2025],
           'role_mix':'Fixed actual 2026 exposure at both endpoints; retrospective standardisation',
           'primary_models':PRIMARY,'stress_and_diagnostic':['rolling_30','season_only'],
           'players':len(candidates),'replicates':REPS,'seed':SEED,'simultaneous_comparisons':int(mask.sum()),
           'simultaneous_critical_value':critical,'prior_fits_at_bound':int(prior_frame.at_strength_bound.sum()),
           'prior_fits_total':len(prior_frame),'positive_simultaneous_findings':results.loc[mask&results.simultaneous_low.gt(0),['player','model']].to_dict('records'),
           'protocol_sha256':hashlib.sha256((ROOT/'docs/PLAYER_MEMORY_PROTOCOL.md').read_bytes()).hexdigest(),
           'checks':{'all_players_and_models':len(results)==len(candidates)*len(MODELS),
                     'valid_probabilities':bool(results[['before','after']].ge(0).all().all() and results[['before','after']].le(1).all().all()),
                     'all_preseason_prior_dates':bool(prior.train_max_date.lt(before).all()),
                     'role_weights_sum_one':bool(np.allclose(role.groupby(['batter_id','model']).role_weight.sum(),1)),
                     'nonnegative_effective_counts':bool(role[['before_B','before_N','after_B','after_N']].ge(0).all().all()),
                     'successes_not_above_opportunities':bool((role.before_B<=role.before_N+1e-10).all() and (role.after_B<=role.after_N+1e-10).all())},
           'limits':['Different memory windows are correlated checks, not independent confirmations.',
                     'Bootstrap stability bands condition on fixed role mix and event memory weights; prior and personal resampling are an approximation, not a fully joint Bayesian posterior.',
                     'The simultaneous band covers the predefined 40x9 family conditionally; stress/season-only and older-baseline contrasts are pointwise only.',
                     'Role adjustment does not remove venue, phase, opposition, health or league effects. Previous context-adjusted uncertainty remains applicable.',
                     'A model estimate rising during 2026 does not prove permanent ability improvement.']}
    assert all(v for v in audit['checks'].values());(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2));print(results.loc[results.player.isin(['KL Rahul','Ishan Kishan','RG Sharma','B Sai Sudharsan']),['player','model','before','after','change','change_low','change_high']].round(4).to_string(index=False))


if __name__=='__main__':main()
