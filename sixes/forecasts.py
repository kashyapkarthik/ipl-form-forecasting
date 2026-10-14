"""Dated role-mixture forecasts with shared prior and innings-resampling uncertainty."""
import json
import numpy as np
import pandas as pd
from .core import *
from .history import selected_weights,snapshot
from .study import role_weights


def run():
    d=pd.read_parquet(OUT/'deliveries.parquet')
    chosen=json.loads((OUT/'selection.json').read_text())['preseason']['winner']
    year=C['forecast_season'];latest_year=year-1;cutoff=C['forecast_reference_date']
    prior=pd.read_csv(OUT/'annual_priors.csv').query('season==@year').set_index('role')
    train=d.loc[d.season.between(year-3,year-1)].groupby(['role','batter_id'],as_index=False).agg(N=('is_six','size'),B=('is_six','sum'))
    ids=sorted(train.batter_id.unique());locations={pid:i for i,pid in enumerate(ids)}
    reps=C['forecast_bootstrap_replicates'];rng=np.random.default_rng(C['seed'])
    prior_draws={role:[] for role in C['roles']};prior_rows=[]
    groups={role:g for role,g in train.groupby('role')}
    for rep in range(reps):
        weights=rng.exponential(size=len(ids))
        for role in C['roles']:
            g=groups[role];w=weights[g.batter_id.map(locations).to_numpy()]
            p=fit_prior(g.B,g.N,w);prior_draws[role].append(p)
            prior_rows.append(dict(replicate=rep,role=role,**p))
        if (rep+1)%50==0:print(f'Forecast prior resamples: {rep+1}/{reps}',flush=True)
    save(pd.DataFrame(prior_rows),'forecast_prior_resamples')
    active=sorted(d.loc[d.season.eq(latest_year),'batter_id'].unique())
    rows=[];role_rows=[]
    for pid in active:
        player=d.loc[d.batter_id.eq(pid)].sort_values(['date','match_id','innings','delivery_sequence']).copy()
        latest=player.loc[player.season.eq(latest_year)];pre=player.loc[player.season.lt(latest_year)]
        weights,basis=role_weights(d,pid,year)
        forecast_draw=np.zeros(reps);baseline_draw=np.zeros(reps);full_draw=np.zeros(reps)
        means=np.zeros(3);effective=0
        for role in C['roles']:
            mix=float(weights[role])
            if mix==0:continue
            g=player.loc[player.role.eq(role)].copy();p=prior.loc[role]
            w=selected_weights(g,chosen,cutoff);y=g.is_six.to_numpy(float);old=g.season.lt(latest_year).to_numpy(float)
            selected=float(posterior(w@y,w.sum(),p));baseline=float(posterior(old@y,old.sum(),p));full=float(posterior(y.sum(),len(y),p))
            means+=mix*np.array([selected,baseline,full]);effective+=mix*w.sum()
            role_rows.append(dict(batter_id=pid,player=player.batter_name.iloc[-1],role=role,mix_weight=mix,model=chosen,
                                  history_N=len(g),effective_N=w.sum(),forecast=selected,baseline=baseline,full_history=full))
            g['selected_B']=w*y;g['selected_N']=w;g['baseline_B']=old*y;g['baseline_N']=old;g['full_B']=y;g['full_N']=1
            blocks=g.groupby(['season','match_id','innings'],as_index=False)[['selected_B','selected_N','baseline_B','baseline_N','full_B','full_N']].sum()
            sample=np.zeros((reps,6))
            for _,s in blocks.groupby('season'):
                values=s.iloc[:,3:].to_numpy();frequency=rng.multinomial(len(s),np.full(len(s),1/len(s)),size=reps)
                sample+=frequency@values
            alpha=np.array([r['alpha'] for r in prior_draws[role]]);strength=np.array([r['strength'] for r in prior_draws[role]])
            forecast_draw+=mix*(alpha+sample[:,0])/(strength+sample[:,1])
            baseline_draw+=mix*(alpha+sample[:,2])/(strength+sample[:,3])
            full_draw+=mix*(alpha+sample[:,4])/(strength+sample[:,5])
        f,b,full=means;latest_rate=float(latest.is_six.mean());eligible=len(latest)>=C['watchlist_min_latest_N'] and len(pre)>=C['watchlist_min_pre_latest_N']
        flo,fhi=np.quantile(forecast_draw,[.05,.95]);dlo,dhi=np.quantile(forecast_draw-baseline_draw,[.05,.95])
        observed=rng.binomial(100,np.clip(forecast_draw,0,1));olo,ohi=np.quantile(observed,[.05,.95])
        rows.append(dict(batter_id=pid,player=player.batter_name.iloc[-1],forecast_season=year,forecast_as_of=C['research_date'],reference_date=cutoff,
                         model=chosen,role_basis=basis,latest_N=len(latest),latest_sixes=int(latest.is_six.sum()),pre_latest_N=len(pre),career_N=len(player),
                         latest_rate=latest_rate,baseline=b,forecast=f,full_history_forecast=full,forecast_low=flo,forecast_high=fhi,
                         change_from_baseline=f-b,change_low=dlo,change_high=dhi,memory_adjustment=f-full,
                         bootstrap_above_baseline_fraction=float(np.mean(forecast_draw>baseline_draw)),expected_sixes_per100=100*f,
                         expected_per100_low=100*flo,expected_per100_high=100*fhi,illustrative_100ball_sixes_low=int(olo),illustrative_100ball_sixes_high=int(ohi),
                         effective_role_weighted_N=effective,eligible=eligible,
                         in_form_candidate=bool(eligible and latest_rate>b and f-b>=C['watchlist_increase_probability_points']),
                         bounce_back_candidate=bool(eligible and latest_rate<b and f>latest_rate),
                         change_range_above_zero=bool(dlo>0)))
    result=pd.DataFrame(rows).sort_values(['forecast','player'],ascending=[False,True]);save(result,'forecast_2027_all');save(pd.DataFrame(role_rows),'forecast_2027_roles')
    save(result.loc[result.in_form_candidate].sort_values('change_from_baseline',ascending=False),'watchlist_in_form')
    save(result.loc[result.bounce_back_candidate].sort_values('forecast',ascending=False),'watchlist_bounce_back')
    meta=dict(model=chosen,as_of=C['research_date'],source_through=d.date.max(),reference_date=cutoff,reference_note=C['forecast_reference_note'],
              interval='90% stability range: shared population-player reweighting and within-player/season/role innings resampling; fixed model and role weights',
              predictive_example='100 independent future Bernoulli trials mixed over bootstrap rates; illustrative, not a calibrated season-total prediction interval',
              baseline='Pre-2026 player history, evaluated with the same 2027 priors and 2026 role mix as the forecast; retrospective comparison, not an issued historical forecast',
              limits=['Conditional on playing in the IPL and having the same role mix; no current roster, injury or availability model.',
                      'No future venue/opposition mix prediction; six rate is a power-hitting measure, not all-round batting quality.',
                      'Stability ranges exclude model-selection uncertainty and unseen regime changes; not probabilities of true ability improvement.',
                      'Player ranges are marginal, without adjustment for screening many candidates.',
                      'In-form/rebound tags are point-estimate screens, not statistically confirmed claims.'],replicates=reps)
    (OUT/'forecast_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'{len(result)} forecasts; {result.in_form_candidate.sum()} in-form; {result.bounce_back_candidate.sum()} bounce-back screens',flush=True)


if __name__=='__main__':run()
