"""Within-season, sequential and deployment-matched preseason comparisons."""
import json
import hashlib
import numpy as np
import pandas as pd
from .core import *
from .history import features, snapshot
from openers_experiment import match_count_splits


def brown(d, splits):
    x=aggregate(d).merge(splits[['match_id','half']],on='match_id',validate='many_to_one')
    rows=[];priors=[]
    for (year,role),g in x.groupby(['season','role']):
        early=g.loc[g.half.eq('early')].groupby('batter_id',as_index=False)[['N','B']].sum()
        p=fit_group(early);priors.append(dict(season=year,role=role,**p))
        late=g.loc[g.half.eq('late')].copy()
        late=late.merge(early.rename(columns={'N':'early_N','B':'early_B'}),on='batter_id',how='left',validate='many_to_one')
        late[['early_N','early_B']]=late[['early_N','early_B']].fillna(0)
        n,b=late.early_N,late.early_B
        late['p_population']=p['pooled'];late['p_raw']=np.divide(b,n,out=np.full(len(n),p['pooled']),where=n.gt(0))
        late['p_shrinkage']=posterior(b,n,p)
        sums=late.groupby('batter_id').N.sum();late['late_N']=late.batter_id.map(sums)
        rows.append(late)
    return pd.concat(rows,ignore_index=True),pd.DataFrame(priors)


def memory_forecasts(counts, prior):
    result={f'p_{name}':float(posterior(*counts[name],prior)) for name in MEMORIES}
    for name,mem in [('raw_expanding','expanding'),('raw_recent','rolling_120')]:
        b,n=counts[mem];result[f'p_{name}']=b/n if n else prior['pooled']
    result['p_population']=prior['pooled']
    return result


def sequential(d, priors):
    hist=features(d);hist.to_parquet(OUT/'history_features.parquet',index=False)
    x=aggregate(d).merge(hist,on=['season','date','batter_id','role'],validate='many_to_one')
    rows=[]
    for (year,role),g in x.loc[x.season.ge(C['evaluation_start'])].groupby(['season','role']):
        p=priors.loc[(year,role)]
        g=g.copy()
        for name in MEMORIES:g['p_'+name]=posterior(g[name+'_B'],g[name+'_N'],p)
        for name,mem in [('raw_expanding','expanding'),('raw_recent','rolling_120')]:
            n,b=g[mem+'_N'],g[mem+'_B']
            g['p_'+name]=np.divide(b,n,out=np.full(len(n),p.pooled),where=n.gt(0))
        g['p_population']=p.pooled;rows.append(g)
    return pd.concat(rows,ignore_index=True)


def role_weights(d, pid, year):
    past=d.loc[d.batter_id.eq(pid)&d.season.lt(year)]
    if len(past):
        last=int(past.season.max());g=past.loc[past.season.eq(last)];basis=f'player roles in {last}'
    else:
        g=d.loc[d.season.between(year-3,year-1)];basis='past population role mix; no player history'
    w=g.groupby('role').size().reindex(C['roles'],fill_value=0)
    return w/w.sum(),basis


def preseason(d, priors):
    groups={key:g for key,g in d.groupby(['batter_id','role'])}
    empty=d.iloc[:0];rows=[];mix_rows=[];weight_rows=[]
    for year in range(C['evaluation_start'],max(C['test_seasons'])+1):
        cutoff=d.loc[d.season.eq(year),'date'].min()
        players=sorted(d.loc[d.season.eq(year),'batter_id'].unique())
        for pid in players:
            weights,basis=role_weights(d,pid,year);per_role={};history_N=0
            for role in C['roles']:
                counts=snapshot(groups.get((pid,role),empty),cutoff)
                per_role[role]=memory_forecasts(counts,priors.loc[(year,role)])
                history_N+=counts['expanding'][1]
                rows.append(dict(season=year,batter_id=pid,role=role,forecast_date=cutoff,prior_N=counts['expanding'][1],**per_role[role]))
                weight_rows.append(dict(season=year,batter_id=pid,role=role,weight=weights[role],basis=basis))
            probabilities={c:sum(weights[r]*per_role[r][c] for r in C['roles']) for c in per_role[0]}
            mix_rows.append(dict(season=year,batter_id=pid,forecast_date=cutoff,prior_N=history_N,**probabilities))
        print(f'Preseason snapshots: {year}',flush=True)
    forecasts=pd.DataFrame(rows);mixed=pd.DataFrame(mix_rows)
    inn=aggregate(d).query('season>=@C["evaluation_start"]')
    conditional=inn.merge(forecasts,on=['season','batter_id','role'],validate='many_to_one')
    deployed=inn.merge(mixed,on=['season','batter_id'],validate='many_to_one')
    save(forecasts,'preseason_role_forecasts');save(mixed,'preseason_player_forecasts');save(pd.DataFrame(weight_rows),'preseason_role_weights')
    return conditional,deployed


def choose(wide, label):
    val=metrics(wide.loc[wide.season.eq(C['selection_season'])]).sort_values(['log_loss','model'])
    candidates=[x for x in CANDIDATES if x!='current' or label=='sequential']
    eligible=val.loc[val.model.isin(candidates)]
    rec=eligible.loc[eligible.model.ne('expanding')]
    rolling=eligible.loc[eligible.model.str.startswith('rolling_')].iloc[0].model
    exponential=eligible.loc[eligible.model.str.startswith('ew_')].iloc[0].model
    names=list(dict.fromkeys(['population','raw_expanding','expanding',rec.iloc[0].model,rolling,exponential,'raw_recent','rolling_30']+(['current'] if label=='sequential' else [])))
    save(val,f'{label}_validation_grid')
    return dict(winner=eligible.iloc[0].model,recent_winner=rec.iloc[0].model,rolling_winner=rolling,exponential_winner=exponential,
                display_models=names,validation_log_loss=float(eligible.iloc[0].log_loss),selection_year=2024,
                selection_target='past-role-mixture player forecast' if label=='preseason' else 'observed role conditional next-innings forecast')


def evaluate(wide, label, models):
    periods=[]
    for period,years in [('development',range(C['evaluation_start'],2024)),('validation',[2024]),('evaluation',C['test_seasons'])]:
        periods.append(metrics(wide.loc[wide.season.isin(years)],models).assign(period=period))
    save(pd.concat(periods),label+'_metrics')
    save(pd.concat([metrics(g,models).assign(season=y) for y,g in wide.groupby('season')]),label+'_annual_metrics')
    evaluation=wide.loc[wide.season.isin(C['test_seasons'])]
    save(calibration(evaluation,models),label+'_calibration')
    keep=[c for c in evaluation if not c.startswith('p_') or c[2:] in models]
    save(evaluation[keep],label+'_evaluation_predictions')


def research():
    before=protected_hashes();raw=read('deliveries');d=target(raw)
    d.to_parquet(OUT/'deliveries.parquet',index=False)
    coverage=d.groupby('season',as_index=False).agg(N=('is_six','size'),sixes=('is_six','sum'),players=('batter_id','nunique'),matches=('match_id','nunique'))
    save(coverage,'coverage');save(aggregate(d),'innings')
    audit=dict(raw_events=len(raw),all_position_legal_balls=int(raw.primary_eligible.sum()),top7_legal_balls=len(d),sixes=int(d.is_six.sum()),
               min_date=d.date.min(),max_date=d.date.max(),matches=d.match_id.nunique(),players=d.batter_id.nunique(),
               source_sha256=hashlib.sha256((ROOT/'data/raw/ipl_json.zip').read_bytes()).hexdigest())
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    b,bp=brown(d,read('match_splits'));b.to_parquet(OUT/'brown_predictions.parquet',index=False);save(bp,'brown_priors')
    evaluate(b,'brown',['population','raw','shrinkage'])
    ev=b.loc[b.season.isin(C['test_seasons'])]
    save(pd.DataFrame([paired(ev,'shrinkage',r) for r in ['population','raw']]),'brown_uncertainty')
    sub=ev.loc[ev.early_N.ge(11)&ev.late_N.ge(11)]
    save(metrics(sub),'brown_min11_metrics');save(pd.DataFrame([paired(sub,'shrinkage',r) for r in ['population','raw']]),'brown_min11_uncertainty')
    ms=match_count_splits(read('matches'));mb,_=brown(d,ms);me=mb.loc[mb.season.isin(C['test_seasons'])]
    save(metrics(me),'brown_matchcount_metrics');save(pd.DataFrame([paired(me,'shrinkage',r) for r in ['population','raw']]),'brown_matchcount_uncertainty')
    prior=annual_priors(d,range(C['evaluation_start'],C['forecast_season']+1));save(prior,'annual_priors')
    prior=prior.set_index(['season','role'])
    print('Constructing sequential forecasts',flush=True)
    seq=sequential(d,prior);seq.to_parquet(OUT/'sequential_predictions.parquet',index=False)
    conditional,pre=preseason(d,prior)
    pre.to_parquet(OUT/'preseason_predictions.parquet',index=False);conditional.to_parquet(OUT/'preseason_conditional_predictions.parquet',index=False)
    # No evaluation-period scoring participates in these choices.
    selection={label:choose(frame.loc[frame.season.le(2024)],label) for label,frame in [('sequential',seq),('preseason',pre)]}
    selection['protocol_sha256']=hashlib.sha256((ROOT/'config/sixes_protocol.json').read_bytes()).hexdigest()
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    for label,wide in [('sequential',seq),('preseason',pre),('preseason_conditional',conditional)]:
        spec=selection['sequential' if label=='sequential' else 'preseason'];models=spec['display_models']
        evaluate(wide,label,models);x=wide.loc[wide.season.isin(C['test_seasons'])]
        selected=spec['recent_winner'];pairs={(selected,'expanding'),(spec['winner'],'population'),(spec['winner'],'raw_expanding'),
                  (selected,'raw_recent'),(selected,'rolling_30'),(spec['rolling_winner'],'expanding'),(spec['exponential_winner'],'expanding')}
        if label=='sequential':pairs.add((selected,'current'))
        intervals=[paired(x,a,b) for a,b in sorted(pairs) if a!=b]
        intervals.append(paired(x,selected,'expanding',unit='player'))
        save(pd.DataFrame(intervals),label+'_uncertainty')
        standard=x.loc[x.standard_conditions]
        save(metrics(standard,models),label+'_standard_metrics')
        save(pd.DataFrame([paired(standard,selected,'expanding')]),label+'_standard_uncertainty')
    clips=[]
    for label,wide,rawmodels in [('brown',b,['raw']),('sequential',seq,['raw_expanding','raw_recent']),('preseason',pre,['raw_expanding','raw_recent'])]:
        x=wide.loc[wide.season.isin(C['test_seasons'])]
        for model in rawmodels:
            for clip in [1e-6,1e-4,1e-3]:
                ll,bs=losses(x.B,x.N,x['p_'+model],clip=clip)
                clips.append(dict(experiment=label,model=model,clip=clip,log_loss=ll.sum()/x.N.sum(),brier=bs.sum()/x.N.sum()))
    save(pd.DataFrame(clips),'raw_clipping_sensitivity')
    # Export annual all-candidate scores for transparent historical stability.
    for label,x in [('sequential',seq),('preseason',pre)]:
        save(pd.concat([metrics(g).assign(season=y) for y,g in x.groupby('season')]),label+'_all_annual_metrics')
    assert before==protected_hashes()
    (OUT/'protected_files.json').write_text(json.dumps(before,indent=2)+'\n')
    print(json.dumps(selection,indent=2),flush=True)


if __name__=='__main__':research()
