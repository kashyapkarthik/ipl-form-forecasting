"""Probability estimation, past-only history summaries and scoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit


def fit_prior(B, N, weights=None):
    B, N = np.asarray(B, float), np.asarray(N, float)
    keep = N > 0
    B, N = B[keep], N[keep]
    w = np.ones(len(N)) if weights is None else np.asarray(weights, float)[keep]
    if len(N) < 2 or not np.all((B >= 0) & (B <= N)):
        raise ValueError('Need at least two valid player records')
    pooled = np.clip(np.dot(w, B) / np.dot(w, N), .001, .999)
    def objective(z):
        mu, k = expit(z[0]), np.exp(z[1])
        a, b = mu * k, (1-mu) * k
        return -np.sum(w * (betaln(a+B, b+N-B) - betaln(a, b)))
    fits = [minimize(objective, [logit(pooled), np.log(k)], method='L-BFGS-B',
        bounds=[(-7, 3), (-3, 12)]) for k in [20, 100, 500]]
    valid = [x for x in fits if x.success and np.isfinite(x.fun)]
    if not valid:
        raise RuntimeError('Empirical Bayes prior optimisation failed')
    fit = min(valid, key=lambda x: x.fun)
    mu, k = expit(fit.x[0]), np.exp(fit.x[1])
    return dict(alpha=float(mu*k), beta=float((1-mu)*k), mu=float(mu), strength=float(k),
        pooled=float(pooled), records=len(N), converged=True,
        at_strength_bound=bool(fit.x[1] < -2.99 or fit.x[1] > 11.99))


def posterior(B, N, prior):
    return (prior['alpha'] + np.asarray(B)) / (prior['strength'] + np.asarray(N))


def losses(B, N, p):
    B, N = np.asarray(B, float), np.asarray(N, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return -B*np.log(p)-(N-B)*np.log1p(-p), B*(1-p)**2+(N-B)*p**2


def summarise(predictions, groupby=('model',)):
    rows=[]
    for key, g in predictions.groupby(list(groupby), observed=True):
        if not isinstance(key, tuple): key=(key,)
        ll, bs=losses(g.B,g.N,g.p)
        players=pd.DataFrame({'player':g.batter_id,'N':g.N,'ll':ll,'bs':bs}).groupby('player').sum(numeric_only=True)
        rows.append(dict(zip(groupby,key), N=int(g.N.sum()), players=len(players),
            log_loss=float(ll.sum()/g.N.sum()), brier=float(bs.sum()/g.N.sum()),
            player_macro_log_loss=float((players.ll/players.N).mean()),
            player_macro_brier=float((players.bs/players.N).mean())))
    return pd.DataFrame(rows)


def history_counts(deliveries, config):
    """One row/player/day, computed before any outcome on that date is observed."""
    rows=[]
    rb=config['rolling_balls']; eb=config['ew_balls_halflives']; ed=config['ew_days_halflives']
    for pid, player in deliveries.groupby('batter_id', sort=True):
        player=player.sort_values(['date','match_id','innings','delivery_sequence'])
        y=player.is_boundary.to_numpy(float)
        prefix=np.r_[0,np.cumsum(y)]
        day_arrays={h:np.zeros(2) for h in ed}
        ball_arrays={h:np.zeros(2) for h in eb}
        offset=0; season_start=0; last_season=None; last_day=None
        for date, g in player.groupby('date',sort=True):
            season=int(g.season.iloc[0]); day=pd.Timestamp(date)
            if season != last_season: season_start=offset
            if last_day is not None:
                for h in ed: day_arrays[h] *= 2**(-(day-last_day).days/h)
            row=dict(batter_id=pid,date=date,season=season)
            row.update(expanding_N=float(offset),expanding_B=float(prefix[offset]),
                current_N=float(offset-season_start),current_B=float(prefix[offset]-prefix[season_start]))
            for k in rb:
                start=max(0,offset-k)
                row[f'rolling_{k}_N']=float(offset-start); row[f'rolling_{k}_B']=float(prefix[offset]-prefix[start])
            for h in eb:
                row[f'ew_balls_{h}_B'],row[f'ew_balls_{h}_N']=ball_arrays[h]
            for h in ed:
                row[f'ew_days_{h}_B'],row[f'ew_days_{h}_N']=day_arrays[h]
            rows.append(row)
            n=len(g); vals=y[offset:offset+n]
            for h in eb:
                decay=2**(-1/h); weights=decay**np.arange(n-1,-1,-1)
                ball_arrays[h]=ball_arrays[h]*decay**n+np.array([weights@vals,weights.sum()])
            for h in ed: day_arrays[h]+=np.array([vals.sum(),n])
            offset+=n; last_day=day; last_season=season
    return pd.DataFrame(rows)


def snapshot_counts(player, cutoff, config):
    """All candidates at a common forecast cutoff, using strictly earlier days."""
    x=player.loc[player.date < cutoff].sort_values(['date','match_id','innings','delivery_sequence'])
    y=x.is_boundary.to_numpy(float); n=len(y)
    result={'expanding':(y.sum(),float(n))}
    current=x.loc[x.season==int(cutoff[:4]),'is_boundary'].to_numpy(float)
    result['current']=(current.sum(),float(len(current)))
    for k in config['rolling_balls']: result[f'rolling_{k}']=(y[-k:].sum(),float(min(k,n)))
    for h in config['ew_balls_halflives']:
        w=2**(-np.arange(n-1,-1,-1)/h)
        result[f'ew_balls_{h}']=(w@y,w.sum())
    ages=(pd.Timestamp(cutoff)-pd.to_datetime(x.date)).dt.days.to_numpy()
    for h in config['ew_days_halflives']:
        w=2**(-ages/h); result[f'ew_days_{h}']=(w@y,w.sum())
    return result


def paired_bootstrap(predictions, a, b, seed=20261012, repetitions=2000):
    """Negative difference favours model a; resample dates within season."""
    keys=['season','date','match_id','innings','batter_id']
    # Preseason aggregates use synthetic season-level match/innings placeholders.
    x=predictions.loc[predictions.model==a, keys+['B','N','p']].merge(
        predictions.loc[predictions.model==b, keys+['p']],on=keys,validate='one_to_one',suffixes=('_a','_b'))
    la,ba=losses(x.B,x.N,x.p_a); lb,bb=losses(x.B,x.N,x.p_b)
    x['ll_diff']=la-lb; x['bs_diff']=ba-bb
    grouped=x.groupby(['season','date'],as_index=False)[['ll_diff','bs_diff','N']].sum()
    rng=np.random.default_rng(seed); draws=[]
    blocks=[g[['ll_diff','bs_diff','N']].to_numpy() for _,g in grouped.groupby('season')]
    for _ in range(repetitions):
        totals=sum((g[rng.integers(len(g),size=len(g))].sum(axis=0) for g in blocks),start=np.zeros(3))
        draws.append(totals[:2]/totals[2])
    draws=np.array(draws)
    return dict(model=a,reference=b,observations=len(x),date_blocks=len(grouped),
        log_loss_difference=float((la-lb).sum()/x.N.sum()),
        log_loss_low=float(np.quantile(draws[:,0],.025)),log_loss_high=float(np.quantile(draws[:,0],.975)),
        brier_difference=float((ba-bb).sum()/x.N.sum()),
        brier_low=float(np.quantile(draws[:,1],.025)),brier_high=float(np.quantile(draws[:,1],.975)))
