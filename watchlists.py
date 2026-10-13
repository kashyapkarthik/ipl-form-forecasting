"""Dated 2027 forecasts and block-resampling stability intervals."""
import json
import numpy as np
import pandas as pd
from models import fit_prior, posterior, snapshot_counts
from research import ROOT, CONFIG, OUT, read, save, prior_for


def selected_weights(player, name, cutoff):
    n=len(player)
    if name=='expanding': return np.ones(n)
    if name=='current': return (player.season==int(cutoff[:4])).to_numpy(float)
    if name.startswith('rolling_'): return (np.arange(n)>=max(0,n-int(name.rsplit('_',1)[1]))).astype(float)
    if name.startswith('ew_balls_'): return 2**(-np.arange(n-1,-1,-1)/int(name.rsplit('_',1)[1]))
    if name.startswith('ew_days_'):
        age=(pd.Timestamp(cutoff)-pd.to_datetime(player.date)).dt.days.to_numpy()
        return 2**(-age/int(name.rsplit('_',1)[1]))
    raise ValueError(name)


def main():
    d=read('model_deliveries'); seasons=read('batter_seasons')
    selected=json.loads((OUT/'selection.json').read_text())['preseason']['winner']
    cutoff=CONFIG['forecast_2027_reference_date']; prior=prior_for(seasons,2027); baseline_prior=prior_for(seasons,2026)
    reps=CONFIG['watchlist_interval_bootstraps']; rng=np.random.default_rng(CONFIG['seed'])
    # Shared player-level Bayesian bootstrap weights propagate empirical-prior uncertainty.
    a=seasons.loc[seasons.season.between(2024,2026)]; b=seasons.loc[seasons.season.between(2023,2025)]
    ids=sorted(set(a.batter_id)|set(b.batter_id)); pos={p:i for i,p in enumerate(ids)}
    ai=np.array([pos[p] for p in a.batter_id]); bi=np.array([pos[p] for p in b.batter_id])
    prior_draws=[]; base_draws=[]
    for i in range(reps):
        w=rng.exponential(size=len(ids))
        prior_draws.append(fit_prior(a.B,a.N,w[ai])); base_draws.append(fit_prior(b.B,b.N,w[bi]))
        if (i+1)%200==0: print(f'Population-prior resamples: {i+1}/{reps}',flush=True)
    alpha=np.array([p['alpha'] for p in prior_draws]); strength=np.array([p['strength'] for p in prior_draws])
    base_alpha=np.array([p['alpha'] for p in base_draws]); base_strength=np.array([p['strength'] for p in base_draws])
    current=set(seasons.loc[seasons.season==2026,'batter_id']); rows=[]
    for pid, player in d.loc[d.batter_id.isin(current)].groupby('batter_id'):
        player=player.sort_values(['date','match_id','innings','delivery_sequence']).copy()
        y=player.is_boundary.to_numpy(float); weights=selected_weights(player,selected,cutoff)
        pre=(player.season<2026).to_numpy(float)
        player['sel_B']=weights*y; player['sel_N']=weights
        player['base_B']=pre*y; player['base_N']=pre
        blocks=player.groupby(['season','match_id','innings'],sort=True)[['sel_B','sel_N','base_B','base_N']].sum().reset_index()
        samples=np.zeros((reps,4))
        for _,g in blocks.groupby('season'):
            arr=g[['sel_B','sel_N','base_B','base_N']].to_numpy()
            # Resample innings clusters within season, preserving their original window weights.
            frequency=rng.multinomial(len(g),np.full(len(g),1/len(g)),size=reps)
            samples += frequency@arr
        pdra=(alpha+samples[:,0])/(strength+samples[:,1])
        bdra=(base_alpha+samples[:,2])/(base_strength+samples[:,3])
        forecast=float(posterior(weights@y,weights.sum(),prior))
        baseline=float(posterior(pre@y,pre.sum(),baseline_prior))
        latest=player.loc[player.season==2026]; rate=float(latest.is_boundary.mean())
        recent=float(latest.tail(120).is_boundary.mean()); delta=forecast-baseline
        elo,ehi=np.quantile(pdra,[.025,.975]); dlo,dhi=np.quantile(pdra-bdra,[.025,.975])
        eligible=len(latest)>=CONFIG['watchlist_min_latest_N'] and pre.sum()>=CONFIG['watchlist_min_pre_latest_N']
        rows.append(dict(batter_id=pid,player=player.batter_name.iloc[-1],forecast_season=2027,
            forecast_as_of=CONFIG['research_date'],reference_date=cutoff,model=selected,
            career_N=len(player),pre_2026_N=int(pre.sum()),latest_N=len(latest),latest_B=int(latest.is_boundary.sum()),
            baseline=baseline,latest_rate=rate,last_120_rate=recent,forecast=forecast,
            forecast_low=elo,forecast_high=ehi,change_from_baseline=delta,change_low=dlo,change_high=dhi,
            bootstrap_positive_fraction=float(np.mean(pdra>bdra)),effective_weighted_N=float(weights.sum()),
            eligible=eligible,potential_improver=bool(eligible and delta>.01),
            rebound_candidate=bool(eligible and rate<baseline and forecast>rate),
            interval_excludes_zero=bool(dlo>0),
            population_mean_change=prior['mu']-baseline_prior['mu']))
    forecasts=pd.DataFrame(rows).sort_values('forecast',ascending=False)
    save(forecasts,'forecast_2027_all')
    save(forecasts.loc[forecasts.potential_improver].sort_values('change_from_baseline',ascending=False),'watchlist_improvers')
    save(forecasts.loc[forecasts.rebound_candidate].sort_values('forecast',ascending=False),'watchlist_rebounds')
    (OUT/'watchlist_metadata.json').write_text(json.dumps(dict(selected_model=selected,prior=prior,baseline_prior=baseline_prior,
        forecast_as_of=CONFIG['research_date'],assumed_reference_date=cutoff,bootstrap_replicates=reps,
        interval='Percentile stability interval from season-stratified innings resampling and player-level empirical-prior reweighting; not a prediction interval for realised 2027 rate',
        limitations=['Conditional on fixed model and selected history window; no model-selection uncertainty.',
            'Cannot account for unseen injury, role, squad selection or league changes.',
            'Intervals are marginal; multiple-testing adjustments are not applied to player lists.',
            'Bootstrap positive fraction is not a posterior probability of true skill improvement.',
            'Baseline comparison also reflects a change in the estimated league population prior.']),indent=2))
    print(f'Created forecasts for {len(forecasts)} batters; {int(forecasts.eligible.sum())} meet watchlist history thresholds')

if __name__=='__main__': main()
