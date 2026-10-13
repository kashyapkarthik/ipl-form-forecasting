"""Past-trained context checks; no test-period fitting or feature selection."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from models import losses, summarise, paired_bootstrap
from research import ROOT, CONFIG, OUT, read, save


def canonical_venue(name):
    # Conservative spelling/suffix merges. Major renamed/rebuilt venues remain separate era labels.
    base=name.split(',')[0].replace('M.Chinnaswamy','M Chinnaswamy')
    if base=='Punjab Cricket Association IS Bindra Stadium': base='Punjab Cricket Association Stadium'
    if base=='Zayed Cricket Stadium': base='Sheikh Zayed Stadium'
    return base


def main():
    select=json.loads((OUT/'selection.json').read_text()); selected=select['sequential']['winner']
    seq=pd.read_parquet(OUT/'sequential_predictions.parquet')
    wide=seq.pivot(index=['match_id','innings','batter_id'],columns='model',values='p').reset_index()
    d=read('model_deliveries'); d=d.loc[d.season>=2012].merge(wide,on=['match_id','innings','batter_id'],validate='many_to_one')
    d['venue_group']=d.venue.map(canonical_venue)
    d['opposition']=d.bowling_team.replace({'Delhi Daredevils':'Delhi Capitals','Kings XI Punjab':'Punjab Kings',
        'Royal Challengers Bangalore':'Royal Challengers Bengaluru','Rising Pune Supergiants':'Rising Pune Supergiant'})
    d['ability_logit']=logit(np.clip(d["expanding"],.001,.999))
    d['recency_logit_difference']=logit(np.clip(d[selected],.001,.999))-d.ability_logit
    d['year_trend']=d.season-2012
    cats=['venue_group','opposition','phase','innings']
    nums=['ability_logit','batting_entry_order','score_before','wickets_before','legal_balls_before','year_trend']
    rows=[]; coefficients=[]
    for year in [2025,2026]:
        train=d.loc[d.season.between(year-CONFIG['context_training_seasons'],year-1)]
        test=d.loc[d.season==year]
        for add_recent in [False,True]:
            numeric=nums+(['recency_logit_difference'] if add_recent else [])
            transformer=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num',StandardScaler(),numeric)])
            model=make_pipeline(transformer,LogisticRegression(C=CONFIG['context_logistic_C'],max_iter=1000,solver='lbfgs'))
            model.fit(train[cats+numeric],train.is_boundary.astype(int))
            out=test[['delivery_id','season','date','match_id','innings','batter_id','venue_group','opposition','phase','standard_conditions']].copy()
            out['B']=test.is_boundary.astype(int).to_numpy(); out['N']=1
            out['model']='context_plus_recent' if add_recent else 'context_expanding'
            out['p']=model.predict_proba(test[cats+numeric])[:,1]; rows.append(out)
            coefficients.append(dict(season=year,model=out.model.iloc[0],training_last_date=train.date.max(),
                test_first_date=test.date.min(),iterations=int(model[-1].n_iter_[0]),
                recency_coefficient=float(model[-1].coef_[0][-1]) if add_recent else None))
    p=pd.concat(rows,ignore_index=True); p.to_parquet(OUT/'context_predictions.parquet',index=False)
    save(summarise(p),'context_metrics'); save(pd.DataFrame(coefficients),'context_fits')
    # Aggregate each delivery forecast's actual loss to the innings for a paired date bootstrap.
    ll,bs=losses(p.B,p.N,p.p); p['ll']=ll; p['bs']=bs
    x=p.groupby(['season','date','model'],as_index=False)[['N','ll','bs']].sum()
    rng=np.random.default_rng(CONFIG['seed']); by={}
    for m,g in x.groupby('model'): by[m]=g.set_index(['season','date'])
    a=by['context_plus_recent']; b=by['context_expanding'].reindex(a.index)
    z=pd.DataFrame({'N':a.N,'ll_diff':a.ll-b.ll,'bs_diff':a.bs-b.bs}).reset_index()
    draws=[]; blocks=[g[['ll_diff','bs_diff','N']].to_numpy() for _,g in z.groupby('season')]
    for _ in range(CONFIG['bootstrap_replicates']):
        t=sum((g[rng.integers(len(g),size=len(g))].sum(axis=0) for g in blocks),start=np.zeros(3)); draws.append(t[:2]/t[2])
    draws=np.array(draws)
    result=dict(model='context_plus_recent',reference='context_expanding',
        log_loss_difference=float(z.ll_diff.sum()/z.N.sum()),log_loss_low=float(np.quantile(draws[:,0],.025)),log_loss_high=float(np.quantile(draws[:,0],.975)),
        brier_difference=float(z.bs_diff.sum()/z.N.sum()),brier_low=float(np.quantile(draws[:,1],.025)),brier_high=float(np.quantile(draws[:,1],.975)))
    save(pd.DataFrame([result]),'context_uncertainty')
    # Descriptive subgroup diagnostics for fixed main models, not model selection.
    sub=d.loc[d.season.isin([2025,2026])].copy()
    for col in ['phase','venue_group','opposition']:
        rec=[]
        for level,g in sub.groupby(col):
            la,ba=losses(g.is_boundary,np.ones(len(g)),g[selected]); lb,bb=losses(g.is_boundary,np.ones(len(g)),g["expanding"])
            rec.append(dict(group=level,N=len(g),boundary_rate=float(g.is_boundary.mean()),
                log_loss_difference=float(np.mean(la-lb)),brier_difference=float(np.mean(ba-bb))))
        save(pd.DataFrame(rec),f'subgroups_{col}')
    simple=seq.loc[seq.season.isin([2025,2026])&seq.standard_conditions&seq.model.isin(['expanding',selected])]
    save(summarise(simple),'standard_conditions_metrics')
    save(pd.DataFrame([paired_bootstrap(simple,selected,'expanding',repetitions=CONFIG['bootstrap_replicates'])]),'standard_conditions_uncertainty')
    save(d[['venue','venue_group']].drop_duplicates(),'venue_mapping')
    print(result)

if __name__=='__main__': main()
