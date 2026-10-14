"""Context diagnostics, with past-selected historical recency features."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from .core import *
from context_analysis import canonical_venue


def run():
    selection=json.loads((OUT/'selection.json').read_text())['sequential']
    chosen=selection['recent_winner']
    seq=pd.read_parquet(OUT/'sequential_predictions.parquet')
    d=pd.read_parquet(OUT/'deliveries.parquet')
    pcols=[c for c in seq if c.startswith('p_')]
    d=d.loc[d.season.ge(C['evaluation_start'])].merge(seq[KEYS+pcols],on=KEYS,validate='many_to_one')
    d['venue_group']=d.venue.map(canonical_venue)
    d['opposition']=d.bowling_team.replace({'Delhi Daredevils':'Delhi Capitals','Kings XI Punjab':'Punjab Kings',
         'Royal Challengers Bangalore':'Royal Challengers Bengaluru','Rising Pune Supergiants':'Rising Pune Supergiant'})
    d['ability_logit']=logit(np.clip(d.p_expanding,.001,.999));d['year_trend']=d.season-C['evaluation_start']
    # Every historical training row uses a recency choice selected before its season.
    mapping=[]
    for year,g in d.groupby('season'):
        past=seq.loc[seq.season.eq(year-1)]
        candidates=[x for x in CANDIDATES if x!='expanding']
        if len(past):
            selected=metrics(past,candidates).sort_values(['log_loss','model']).iloc[0].model
            last=past.date.max()
        else:selected='ew_days_365';last=None
        mask=d.season.eq(year)
        d.loc[mask,'historical_recency']=logit(np.clip(d.loc[mask,'p_'+selected],.001,.999))-d.loc[mask,'ability_logit']
        mapping.append(dict(feature_season=year,selected_memory=selected,selection_last_date=last))
    save(pd.DataFrame(mapping),'context_historical_selection')
    cats=['venue_group','opposition','phase','innings','role'];nums=['ability_logit','score_before','wickets_before','legal_balls_before','year_trend']
    for c in cats:d[c]=d[c].astype(str)
    predictions=[];fits=[]
    for year in C['test_seasons']:
        train=d.loc[d.season.between(year-C['context_training_seasons'],year-1)].copy()
        test=d.loc[d.season.eq(year)].copy()
        train['recent_signal']=train.historical_recency
        test['recent_signal']=logit(np.clip(test['p_'+chosen],.001,.999))-test.ability_logit
        out=test[['delivery_id','season','date','match_id','innings','batter_id','role','venue_group','opposition','phase','standard_conditions']].copy()
        out['B']=test.is_six.to_numpy();out['N']=1
        for recent in [False,True]:
            numeric=nums+(['recent_signal'] if recent else [])
            model=make_pipeline(ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num',StandardScaler(),numeric)]),
                                LogisticRegression(C=C['context_logistic_C'],max_iter=1500,solver='lbfgs'))
            model.fit(train[cats+numeric],train.is_six)
            label='context_recent' if recent else 'context'
            out['p_'+label]=model.predict_proba(test[cats+numeric])[:,1]
            fits.append(dict(season=year,model=label,training_last_date=train.date.max(),test_first_date=test.date.min(),
                             training_N=len(train),iterations=int(model[-1].n_iter_[0]),recent_coefficient=float(model[-1].coef_[0][-1]) if recent else None))
        predictions.append(out)
        print(f'Context models: {year}',flush=True)
    out=pd.concat(predictions,ignore_index=True);out.to_parquet(OUT/'context_predictions.parquet',index=False)
    save(metrics(out),'context_metrics');save(pd.DataFrame(fits),'context_fits')
    save(pd.DataFrame([paired(out,'context_recent','context')]),'context_uncertainty')
    save(calibration(out,['context','context_recent']),'context_calibration')
    simple=d.loc[d.season.isin(C['test_seasons'])].copy();simple['B']=simple.is_six;simple['N']=1
    for column in ['phase','venue_group','opposition']:
        rows=[]
        for label,g in simple.groupby(column):
            rows.append(dict(group=label,sixes=int(g.B.sum()),observed_rate=float(g.B.mean()),**paired(g,chosen,'expanding')))
        save(pd.DataFrame(rows).sort_values('N',ascending=False),'subgroups_'+column)
    save(d[['venue','venue_group']].drop_duplicates(),'venue_mapping')
    save(d[['bowling_team','opposition']].drop_duplicates(),'opposition_mapping')
    # Daily totals permit independent recalculation without publishing a large duplicate ball table.
    rows=[]
    for model in ['context','context_recent']:
        ll,bs=losses(out.B,out.N,out['p_'+model]);x=out[['season','date','N']].assign(ll=ll,bs=bs,model=model)
        rows.append(x.groupby(['season','date','model'],as_index=False)[['N','ll','bs']].sum())
    save(pd.concat(rows),'context_date_losses')


if __name__=='__main__':run()
