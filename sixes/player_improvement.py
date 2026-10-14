"""Retrospective individual-player changes; separate from the forecasting experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import t
from .core import ROOT, read, C
from context_analysis import canonical_venue

OUT=ROOT/'results/player_improvement'
SEED=20261015
REPS=5000


def bh(p):
    p=np.asarray(p,float); order=np.argsort(p); ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    result=np.empty(len(p));result[order]=np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1]);return result


def draw_rate(x,rng):
    out=np.zeros((REPS,2))
    for _,g in x.groupby('season'):
        v=g[['B','N']].to_numpy(float)
        frequency=rng.multinomial(len(v),np.full(len(v),1/len(v)),size=REPS)
        out+=frequency@v
    return out[:,0]/out[:,1]


def adjusted_fit(d,ids):
    """OLS change contrasts with CR1 match-cluster covariance; no forecast claims."""
    cats=pd.DataFrame({'role_phase':d.role.astype(str)+'_'+d.phase,'venue':d.venue.map(canonical_venue),
                      'opposition':d.bowling_team.replace({'Delhi Daredevils':'Delhi Capitals','Kings XI Punjab':'Punjab Kings',
                      'Royal Challengers Bangalore':'Royal Challengers Bengaluru'}),'innings':d.innings.astype(str)},index=d.index)
    players=pd.get_dummies(pd.Categorical(d.batter_id,categories=ids),dtype=float)
    players.index=d.index;players.columns=['base_'+p for p in ids]
    change=players.mul(d.season.eq(2026).to_numpy(),axis=0);change.columns=['change_'+p for p in ids]
    controls=pd.get_dummies(cats,drop_first=True,dtype=float)
    frame=pd.concat([players,change,controls],axis=1)
    full=frame.to_numpy(float);_,r,piv=qr(full,mode='economic',pivoting=True)
    rank=int(np.sum(np.abs(np.diag(r))>1e-9));keep=np.sort(piv[:rank]);names=frame.columns[keep].tolist();X=full[:,keep]
    assert all('change_'+pid in names for pid in ids),'A player change is unidentified'
    y=d.is_six.to_numpy(float);bread=np.linalg.inv(X.T@X);coef=bread@(X.T@y);residual=y-X@coef
    codes,unique=pd.factorize(d.match_id,sort=True);scores=np.zeros((len(unique),X.shape[1]));np.add.at(scores,codes,X*residual[:,None])
    correction=len(unique)/(len(unique)-1)*(len(X)-1)/(len(X)-X.shape[1])
    cov=correction*bread@(scores.T@scores)@bread
    n=d.loc[d.season.eq(2026)].groupby('batter_id').size().reindex(ids).to_numpy(float)
    indices=np.array([names.index('change_'+pid) for pid in ids]);rows=[]
    for j,pid in enumerate(ids):
        for label in ['absolute','relative']:
            contrast=np.zeros(len(coef));contrast[indices[j]]=1
            if label=='relative':
                weights=n.copy();weights[j]=0;weights/=weights.sum();contrast[indices]-=weights
            value=float(contrast@coef);se=float(np.sqrt(max(contrast@cov@contrast,0)));critical=t.ppf(.975,len(unique)-1)
            rows.append(dict(batter_id=pid,comparison=label,change=value,se=se,low=value-critical*se,high=value+critical*se,
                             p=float(2*t.sf(abs(value/se),len(unique)-1))))
    out=pd.DataFrame(rows)
    out['q']=out.groupby('comparison').p.transform(lambda p:bh(p.to_numpy()))
    out.to_csv(OUT/'context_adjusted_changes.csv',index=False)
    meta={'method':'Linear probability regression; player fixed effects and player-by-2026 interactions; role-phase, venue, opposition, innings controls; CR1 match-cluster covariance and t intervals',
          'balls':len(d),'players':len(ids),'matches':len(unique),'design_columns':len(frame.columns),'design_rank':rank,
          'dropped_columns':[c for c in frame if c not in names],
          'relative_reference':'2026-ball-weighted average adjusted change among all OTHER eligible players',
          'limits':['Retrospective associations, not causal ability changes or probability forecasts.',
                    'Context effects are common across players and periods; no bowler quality, health, intent, or within-season dynamic effects.',
                    'Historical comparator pools 2023–2025; separate descriptive 2025 comparison checks rebound versus new level.',
                    'Match clustering does not capture every possible dependence across a player career.',
                    'Screening adjustment assumes suitable dependence of p-values; a q-value is not the probability a player has improved.']}
    (OUT/'model_audit.json').write_text(json.dumps(meta,indent=2)+'\n')
    return out


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    all_data=pd.read_parquet(ROOT/'results/sixes/deliveries.parquet')
    d=all_data.loc[all_data.season.between(2023,2026)].copy()
    latest=d.loc[d.season.eq(2026)].groupby('batter_id').agg(latest_N=('is_six','size'),latest_B=('is_six','sum'),player=('batter_name','last'))
    past=d.loc[d.season.lt(2026)].groupby('batter_id').agg(past_N=('is_six','size'),past_B=('is_six','sum'))
    coverage=latest.join(past).fillna({'past_N':0,'past_B':0});coverage['eligible']=coverage.latest_N.ge(120)&coverage.past_N.ge(300)
    coverage.to_csv(OUT/'coverage.csv');ids=sorted(coverage.index[coverage.eligible]);x=d.loc[d.batter_id.isin(ids)].copy()
    x=x.merge(read('match_splits')[['match_id','half']],on='match_id',validate='many_to_one')
    innings=x.groupby(['batter_id','season','match_id','date','innings','half'],as_index=False).agg(N=('is_six','size'),B=('is_six','sum'))
    annual=innings.groupby(['batter_id','season'],as_index=False)[['N','B']].sum();annual['rate']=annual.B/annual.N
    annual=annual.merge(coverage[['player']],left_on='batter_id',right_index=True);annual.to_csv(OUT/'annual_rates.csv',index=False)
    rows=[]
    for pid in ids:
        g=innings.loc[innings.batter_id.eq(pid)];recent=g.loc[g.season.eq(2026)];prior=g.loc[g.season.lt(2026)];last=g.loc[g.season.eq(2025)]
        rng=np.random.default_rng(SEED+int(pid,16));b=prior.B.sum()/prior.N.sum();f=recent.B.sum()/recent.N.sum()
        draws=draw_rate(recent,rng)-draw_rate(prior,rng);lo,hi=np.quantile(draws,[.025,.975])
        row=dict(batter_id=pid,player=coverage.loc[pid,'player'],prior_B=int(prior.B.sum()),prior_N=int(prior.N.sum()),prior_rate=b,
                 latest_B=int(recent.B.sum()),latest_N=int(recent.N.sum()),latest_rate=f,latest_innings=len(recent),raw_change=f-b,raw_low=lo,raw_high=hi)
        if len(last):
            v=draw_rate(recent,rng)-draw_rate(last,rng);a,z=np.quantile(v,[.025,.975]);row.update(rate_2025=last.B.sum()/last.N.sum(),N_2025=int(last.N.sum()),change_2025=f-last.B.sum()/last.N.sum(),change_2025_low=a,change_2025_high=z)
        for half in ['early','late']:
            h=recent.loc[recent.half.eq(half)];row[half+'_N']=int(h.N.sum());row[half+'_B']=int(h.B.sum());row[half+'_rate']=h.B.sum()/h.N.sum() if h.N.sum() else np.nan
        peak=recent.sort_values(['B','date','match_id'],ascending=[False,True,True]).iloc[0];rest=recent.drop(peak.name)
        row.update(peak_innings_B=int(peak.B),peak_innings_N=int(peak.N),without_peak_rate=rest.B.sum()/rest.N.sum())
        balls=x.loc[x.batter_id.eq(pid)];observed=set(zip(balls.loc[balls.season.lt(2026),'role'],balls.loc[balls.season.lt(2026),'phase']))
        now=balls.loc[balls.season.eq(2026)];row['prior_role_phase_coverage']=np.mean([(r,p) in observed for r,p in zip(now.role,now.phase)])
        rows.append(row)
    out=pd.DataFrame(rows)
    adjusted=adjusted_fit(x,ids)
    for comparison in ['absolute','relative']:
        a=adjusted.loc[adjusted.comparison.eq(comparison)].drop(columns='comparison').rename(columns={c:comparison+'_'+c for c in ['change','se','low','high','p','q']})
        out=out.merge(a,on='batter_id',validate='one_to_one')
    out['both_halves_above_prior']=out.early_N.ge(60)&out.late_N.ge(60)&out.early_rate.gt(out.prior_rate)&out.late_rate.gt(out.prior_rate)
    out['adjusted_positive_q05']=out.absolute_change.gt(0)&out.absolute_q.lt(.05)
    out['relative_positive_q05']=out.relative_change.gt(0)&out.relative_q.lt(.05)
    out=out.sort_values(['relative_q','relative_change'],ascending=[True,False]);out.to_csv(OUT/'player_changes.csv',index=False)
    exposure=x.assign(period=np.where(x.season.eq(2026),'2026','2023–2025')).groupby(['batter_id','period','role','phase'],as_index=False).agg(N=('is_six','size'),B=('is_six','sum'))
    exposure.to_csv(OUT/'role_phase_exposure.csv',index=False)
    audit={'eligible_players':len(ids),'active_players':len(coverage),'latest_season':2026,'historical_seasons':[2023,2024,2025],
           'bootstrap_replicates':REPS,'seed':SEED,'absolute_positive_q05':out.loc[out.adjusted_positive_q05,'player'].tolist(),
           'relative_positive_q05':out.loc[out.relative_positive_q05,'player'].tolist(),
           'checks':{'counts_reconcile':int(innings.N.sum())==len(x),'valid_outcomes':bool((innings.B<=innings.N).all()),'all_eligible_reported':len(out)==len(ids),'no_duplicate_player_results':not out.batter_id.duplicated().any()}}
    assert all(audit['checks'].values());(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2));print(out[['player','prior_rate','rate_2025','latest_rate','early_rate','late_rate','absolute_change','absolute_q','relative_change','relative_q']].head(15).round(4).to_string(index=False))


if __name__=='__main__':main()
