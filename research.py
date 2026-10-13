"""Run frozen experiments. Outputs are machine-readable and figure-ready."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from models import fit_prior, posterior, history_counts, snapshot_counts, summarise, paired_bootstrap

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config/protocol.json').read_text())
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)


def read(name): return pd.read_parquet(ROOT/f'data/processed/{name}.parquet')

def save(df,name): df.to_csv(OUT/f'{name}.csv',index=False)

def prior_for(seasons, year):
    x=seasons.loc[seasons.season.between(year-CONFIG['prior_lookback_seasons'],year-1)]
    return fit_prior(x.B,x.N)


def brown():
    half=read('season_halves'); matches=read('match_splits'); model=read('model_deliveries')
    rows=[]; parameters=[]
    for year,g in half.groupby('season'):
        early=g.loc[g.half=='early'].set_index('batter_id'); late=g.loc[g.half=='late']
        prior=fit_prior(early.B,early.N); parameters.append(dict(season=int(year),**prior))
        for r in late.itertuples():
            n=int(early.loc[r.batter_id,'N']) if r.batter_id in early.index else 0
            b=int(early.loc[r.batter_id,'B']) if n else 0
            ps={'population':prior['pooled'],'raw':b/n if n else prior['pooled'],'beta_binomial':float(posterior(b,n,prior))}
            for name,p in ps.items():
                rows.append(dict(season=year,batter_id=r.batter_id,N=r.N,B=r.B,early_N=n,early_B=b,model=name,p=p))
    p=pd.DataFrame(rows); save(p,'brown_predictions'); save(pd.DataFrame(parameters),'brown_priors')
    metrics=[]
    for label, subset in [('all_late_players',p),('min11_both_halves',p.loc[(p.early_N>=11)&(p.N>=11)])]:
        for period, years in [('development',range(2008,2024)),('validation',[2024]),('test',[2025,2026])]:
            q=subset.loc[subset.season.isin(years)]
            s=summarise(q); s['subset']=label; s['period']=period
            for ix,row in s.iterrows():
                v=q.loc[q.model==row.model]
                s.loc[ix,'player_rate_mse']=np.mean((v.p-v.B/v.N)**2)
                transformed=np.arcsin(np.sqrt((v.B+.25)/(v.N+.5)))
                error=(np.arcsin(np.sqrt(v.p))-transformed)**2-1/(4*v.N)
                s.loc[ix,'noise_adjusted_arcsine_mse']=error.mean()
            metrics.append(s)
    save(pd.concat(metrics),'brown_metrics')
    # Return per-innings loss pairs for a cluster-aware held-out comparison.
    late_ids=set(matches.loc[matches.half=='late','match_id'])
    inn=read('batter_innings'); inn=inn.loc[inn.match_id.isin(late_ids)&inn.season.isin([2025,2026])]
    test=inn.merge(p[['season','batter_id','model','p']],on=['season','batter_id'],validate='many_to_many')
    comparisons=[paired_bootstrap(test,a,b,repetitions=CONFIG['bootstrap_replicates'])
        for a,b in [('beta_binomial','raw'),('beta_binomial','population')]]
    save(pd.DataFrame(comparisons),'brown_uncertainty')
    return p


def sequential():
    d=read('model_deliveries'); innings=read('batter_innings'); seasons=read('batter_seasons')
    hist=history_counts(d,CONFIG); hist.to_parquet(OUT/'history_features.parquet',index=False)
    base=innings.merge(hist,on=['batter_id','date','season'],validate='many_to_one')
    rows=[]; priors=[]
    candidates=['expanding','current']+[f'rolling_{k}' for k in CONFIG['rolling_balls']]+[f'ew_balls_{h}' for h in CONFIG['ew_balls_halflives']]+[f'ew_days_{h}' for h in CONFIG['ew_days_halflives']]
    for year,g in base.groupby('season'):
        if year<CONFIG['evaluation_start']: continue
        prior=prior_for(seasons,year); priors.append(dict(season=int(year),**prior))
        common=g[['match_id','date','season','innings','batter_id','N','B','standard_conditions']]
        for name in candidates+['population','raw_expanding']:
            x=common.copy(); x['model']=name
            if name=='population': x['p']=prior['pooled']
            elif name=='raw_expanding': x['p']=np.where(g.expanding_N>0,g.expanding_B/g.expanding_N,prior['pooled'])
            else: x['p']=posterior(g[f'{name}_B'],g[f'{name}_N'],prior)
            rows.append(x)
    preds=pd.concat(rows,ignore_index=True); preds.to_parquet(OUT/'sequential_predictions.parquet',index=False)
    save(pd.DataFrame(priors),'annual_priors')
    return preds,candidates


def preseason(candidates):
    d=read('model_deliveries'); seasons=read('batter_seasons'); innings=read('batter_innings')
    groups={pid:g for pid,g in d.groupby('batter_id')}
    rows=[]
    for year in range(CONFIG['evaluation_start'],2027):
        cutoff=d.loc[d.season==year,'date'].min(); prior=prior_for(seasons,year)
        for r in seasons.loc[seasons.season==year].itertuples():
            counts=snapshot_counts(groups[r.batter_id],cutoff,CONFIG)
            for name in candidates+['population','raw_expanding']:
                if name=='population': p=prior['pooled']
                elif name=='raw_expanding':
                    b,n=counts['expanding']; p=b/n if n else prior['pooled']
                else: p=float(posterior(*counts[name],prior))
                rows.append(dict(season=year,batter_id=r.batter_id,model=name,p=p,N=r.N,B=r.B,forecast_date=cutoff,
                    prior_N=counts['expanding'][1]))
    p=pd.DataFrame(rows); save(p,'preseason_predictions')
    # Score identical fixed forecasts on innings, preserving date clusters for inference.
    full=innings.loc[innings.season>=CONFIG['evaluation_start']].merge(
        p[['season','batter_id','model','p']],on=['season','batter_id'],validate='many_to_many')
    full.to_parquet(OUT/'preseason_innings_predictions.parquet',index=False)
    return full


def select_and_evaluate(preds,candidates,label):
    val=summarise(preds.loc[preds.season==2024]).sort_values(['log_loss','model'])
    allowed=val.loc[val.model.isin(candidates)]
    winner=allowed.iloc[0].model
    family={prefix:val.loc[val.model.str.startswith(prefix)].iloc[0].model for prefix in ['rolling_','ew_balls_','ew_days_']}
    selected=['population','raw_expanding','current','expanding']+list(family.values())
    selected=list(dict.fromkeys(selected))
    records=[]
    for period,years in [('development',range(2012,2024)),('validation',[2024]),('test',[2025,2026])]:
        x=preds.loc[preds.season.isin(years)&preds.model.isin(selected)]
        s=summarise(x); s['period']=period; records.append(s)
    save(val,f'{label}_validation_grid'); save(pd.concat(records),f'{label}_metrics')
    save(summarise(preds.loc[preds.model.isin(selected)],('season','model')),f'{label}_by_season')
    tests=preds.loc[preds.season.isin([2025,2026])]
    pairs=[(m,'expanding') for m in selected if m!='expanding']
    save(pd.DataFrame([paired_bootstrap(tests,a,b,repetitions=CONFIG['bootstrap_replicates']) for a,b in pairs]),f'{label}_uncertainty')
    return dict(winner=winner,family_winners=family,display_models=selected,
        selection_season=2024,selection_log_loss=float(allowed.iloc[0].log_loss))


def main():
    print('Brown-inspired experiment',flush=True); brown()
    print('Constructing past-only histories',flush=True); seq,candidates=sequential()
    selection={'sequential':select_and_evaluate(seq,candidates,'sequential')}
    print('Evaluating fixed preseason forecasts',flush=True); pre=preseason(candidates)
    selection['preseason']=select_and_evaluate(pre,candidates,'preseason')
    selection['protocol_sha256']=hashlib.sha256((ROOT/'config/protocol.json').read_bytes()).hexdigest()
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    print(json.dumps(selection,indent=2))

if __name__=='__main__': main()
