"""Independent source audit and integration checks; no expected-win assertions."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from .core import *
from .history import snapshot,selected_weights
from .study import role_weights


def run():
    checks=[];counts=defaultdict(lambda:[0,0]);raw_events=0;all_legal=0
    with zipfile.ZipFile(ROOT/'data/raw/ipl_json.zip') as z:
        for name in z.namelist():
            if not name.endswith('.json'):continue
            match=json.loads(z.read(name));mid=Path(name).stem;date=min(match['info']['dates']);registry=match['info']['registry']['people']
            for inn_num,inn in enumerate(match.get('innings',[]),1):
                seen={}
                for over in inn['overs']:
                    for ball in over['deliveries']:
                        raw_events+=1
                        for name in [ball['batter'],ball['non_striker']]:
                            if name not in seen:seen[name]=len(seen)+1
                        extras=ball.get('extras',{})
                        if inn.get('super_over',False) or extras.get('noballs',0) or extras.get('wides',0):continue
                        all_legal+=1;pos=seen[ball['batter']]
                        if pos>7:continue
                        role=0 if pos<=2 else pos;key=(int(date[:4]),date,mid,inn_num,registry[ball['batter']],role)
                        counts[key][0]+=1;counts[key][1]+=int(ball['runs']['batter']==6 and not ball['runs'].get('non_boundary',False))
    raw=pd.DataFrame([dict(zip(KEYS,k),N=v[0],B=v[1]) for k,v in counts.items()])
    inn=pd.read_csv(OUT/'innings.csv',dtype={'match_id':str})
    x=raw.merge(inn,on=KEYS,how='outer',indicator=True,suffixes=('_raw','_saved'),validate='one_to_one')
    assert x._merge.eq('both').all();np.testing.assert_array_equal(x[['N_raw','B_raw']],x[['N_saved','B_saved']])
    audit=json.loads((OUT/'data_audit.json').read_text())
    assert raw_events==audit['raw_events'] and all_legal==audit['all_position_legal_balls']
    assert raw.N.sum()==audit['top7_legal_balls'] and raw.B.sum()==audit['sixes']
    assert audit['source_sha256']==hashlib.sha256((ROOT/'data/raw/ipl_json.zip').read_bytes()).hexdigest()
    checks.append('Every six and legal opportunity independently reconstructed from original JSON, including opening roles and source non-boundary flags')
    assert json.loads((OUT/'protected_files.json').read_text())==protected_hashes()
    checks.append('All earlier research outputs and source files remain byte-identical')
    d=pd.read_parquet(OUT/'deliveries.parquet');hist=pd.read_parquet(OUT/'history_features.parquet')
    assert (hist.history_last_date.isna() | hist.history_last_date.lt(hist.date)).all()
    # Distinct algorithm: direct weighted snapshots, compared with online recurrence.
    sample=hist.sample(n=min(150,len(hist)),random_state=C['seed'])
    for row in sample.itertuples():
        player=d.loc[d.batter_id.eq(row.batter_id)&d.role.eq(row.role)]
        expected=snapshot(player,row.date)
        for name,(b,n) in expected.items():
            np.testing.assert_allclose([getattr(row,name+'_B'),getattr(row,name+'_N')],[b,n],atol=1e-9,rtol=1e-10)
    checks.append('All history timestamps strictly precede prediction dates; 150 full memory vectors agree with independent direct snapshots')
    pri=pd.read_csv(OUT/'annual_priors.csv').set_index(['season','role'])
    for (year,role),p in pri.iterrows():
        train=raw.loc[raw.season.between(year-3,year-1)&raw.role.eq(role)].groupby('batter_id')[['N','B']].sum()
        assert p.records==len(train) and p.train_max_date<f'{year}-01-01'
        np.testing.assert_allclose(p.pooled,train.B.sum()/train.N.sum(),atol=1e-12)
    checks.append('Annual population priors use only preceding three complete seasons and correct player-role pooled counts')
    selections=json.loads((OUT/'selection.json').read_text())
    assert selections['protocol_sha256']==hashlib.sha256((ROOT/'config/sixes_protocol.json').read_bytes()).hexdigest()
    for label in ['sequential','preseason']:
        grid=pd.read_csv(OUT/f'{label}_validation_grid.csv')
        candidates=[m for m in CANDIDATES if m!='current' or label=='sequential']
        win=grid.loc[grid.model.isin(candidates)].sort_values(['log_loss','model']).iloc[0].model
        assert win==selections[label]['winner']
    for label in ['brown','sequential','preseason','preseason_conditional']:
        pred=pd.read_parquet(OUT/f'{label}_predictions.parquet');ev=pred.loc[pred.season.isin(C['test_seasons'])]
        assert not pred.duplicated(KEYS).any()
        pc=[c for c in pred if c.startswith('p_')];assert np.isfinite(pred[pc]).all().all() and ((pred[pc]>=0)&(pred[pc]<=1)).all().all()
        expected=raw.loc[raw.season.isin(C['test_seasons'])]
        if label=='brown':
            sp=read('match_splits');expected=expected.merge(sp[['match_id','half']],on='match_id').query('half=="late"')
        np.testing.assert_allclose(ev[['N','B']].sum(),expected[['N','B']].sum(),atol=0)
        score=pd.read_csv(OUT/f'{label}_metrics.csv').query('period=="evaluation"').set_index('model')
        for m,r in score.iterrows():
            p=ev['p_'+m];clip=np.clip(p,C['logloss_clip'],1-C['logloss_clip'])
            ll=-(ev.B*np.log(clip)+(ev.N-ev.B)*np.log1p(-clip)).sum()/ev.N.sum()
            bs=(ev.B-2*ev.B*p+ev.N*p*p).sum()/ev.N.sum()
            np.testing.assert_allclose([ll,bs],[r.log_loss,r.brier],atol=1e-12)
        if label=='sequential':
            for row in ev.sample(100,random_state=C['seed']).itertuples():
                p=pri.loc[(row.season,row.role)]
                for m in MEMORIES:
                    np.testing.assert_allclose(getattr(row,'p_'+m),posterior(getattr(row,m+'_B'),getattr(row,m+'_N'),p),atol=1e-12)
        uncertainty=pd.read_csv(OUT/f'{label}_uncertainty.csv')
        for r in uncertainty.itertuples():
            np.testing.assert_allclose(r.log_loss_difference,score.loc[r.model,'log_loss']-score.loc[r.reference,'log_loss'],atol=1e-12)
    checks.append('Selection uses 2024 only; identical evaluation targets, all published scores and paired point estimates independently verified')
    # Bridge to earlier exact-position six analysis, excluding only the now-combined opening pair.
    old=pd.read_csv(ROOT/'results/outcomes/evaluation_predictions.csv',dtype={'match_id':str})
    old=old.loc[old.outcome.eq('six')&old.window.eq('current')&old.position_group.gt(2)].rename(columns={'position_group':'role'})
    brown=pd.read_parquet(OUT/'brown_predictions.parquet');bridge=brown.loc[brown.season.isin(C['test_seasons'])&brown.role.gt(2)].merge(old[KEYS+['p_beta']],on=KEYS,validate='one_to_one')
    assert len(bridge)==len(old);np.testing.assert_allclose(bridge.p_shrinkage,bridge.p_beta,atol=1e-10,rtol=0)
    checks.append('Within-season six forecasts at positions 3–7 reproduce the previously committed outcome study to 1e-10')
    mixes=pd.read_csv(OUT/'preseason_role_weights.csv')
    for (year,pid),g in mixes.groupby(['season','batter_id']):
        expected,_=role_weights(d,pid,int(year));np.testing.assert_allclose(g.set_index('role').weight.reindex(C['roles']),expected,atol=1e-12)
    checks.append('Every preseason role mixture is built from completed past seasons only, including debutant fallback')
    cf=pd.read_csv(OUT/'context_fits.csv');assert cf.training_last_date.lt(cf.test_first_date).all();assert cf.iterations.lt(1500).all()
    cm=pd.read_csv(OUT/'context_historical_selection.csv').dropna(subset=['selection_last_date'])
    assert cm.apply(lambda r:r.selection_last_date<f'{r.feature_season}-01-01',axis=1).all()
    checks.append('Context encoders, coefficients and historical recency-feature choices are past-only and converge')
    f=pd.read_csv(OUT/'forecast_2027_all.csv');roles=pd.read_csv(OUT/'forecast_2027_roles.csv');meta=json.loads((OUT/'forecast_metadata.json').read_text())
    assert set(f.batter_id)==set(d.loc[d.season.eq(2026),'batter_id'])
    for r in roles.itertuples():
        g=d.loc[d.batter_id.eq(r.batter_id)&d.role.eq(r.role)]
        counts=snapshot(g,C['forecast_reference_date']);p=pri.loc[(2027,r.role)]
        np.testing.assert_allclose(r.forecast,posterior(*counts[meta['model']],p),atol=1e-12)
        weights,_=role_weights(d,r.batter_id,2027);np.testing.assert_allclose(r.mix_weight,weights[r.role],atol=1e-12)
    for r in f.itertuples():
        g=roles.loc[roles.batter_id.eq(r.batter_id)];np.testing.assert_allclose(r.forecast,(g.mix_weight*g.forecast).sum(),atol=1e-12)
        assert 0<=r.forecast_low<=r.forecast_high<=1 and r.latest_N>0
    checks.append('Every dated 2027 role forecast and player mixture independently reconstructed; uncertainty ranges and candidate coverage valid')
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*')) if p.is_file() and p.suffix!='.parquet'}
    report=dict(passed=True,checks=checks,raw_events=raw_events,target_balls=int(raw.N.sum()),sixes=int(raw.B.sum()),
                fallback_fits=int(pri.fallback.sum()),bound_fits=int(pri.at_strength_bound.sum()),source_sha256=audit['source_sha256'],result_sha256=files)
    (ROOT/'reports/sixes_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='result_sha256'},indent=2))


if __name__=='__main__':run()
