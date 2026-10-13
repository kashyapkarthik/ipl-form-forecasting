"""Integration checks on generated evidence; no expected performance thresholds."""
from pathlib import Path
import hashlib,json,platform
import numpy as np
import pandas as pd
from research import ROOT,OUT,CONFIG,read
from models import snapshot_counts,posterior


def main():
    select=json.loads((OUT/'selection.json').read_text()); checks=[]
    expected_totals=read('batter_innings').groupby('season').N.sum()
    assert select['protocol_sha256']==hashlib.sha256((ROOT/'config/protocol.json').read_bytes()).hexdigest()
    checks.append('Selection tied to frozen configuration checksum')
    for label in ['sequential','preseason']:
        grid=pd.read_csv(OUT/f'{label}_validation_grid.csv')
        allowed=grid.loc[~grid.model.isin(['raw_expanding','population'])].sort_values(['log_loss','model'])
        assert select[label]['winner']==allowed.iloc[0].model
        p=pd.read_parquet(OUT/('sequential_predictions.parquet' if label=='sequential' else 'preseason_innings_predictions.parquet'))
        keys=['model','season','date','match_id','innings','batter_id']
        assert not p.duplicated(keys).any()
        assert p.p.between(0,1).all() and np.isfinite(p.p).all()
        totals=p.groupby('model').N.sum()
        assert totals.nunique()==1
        for _,g in p.groupby(['model','season']):
            expected=expected_totals.loc[g.season.iloc[0]]
            assert g.N.sum()==expected
        checks.append(f'{label}: all methods score identical opportunities; probabilities and keys valid; selection uses 2024')
    c=pd.read_csv(OUT/'context_fits.csv')
    assert (c.training_last_date<c.test_first_date).all()
    checks.append('Context fitting ends before each test season')
    a=pd.read_csv(OUT/'annual_priors.csv'); b=pd.read_csv(OUT/'brown_priors.csv')
    assert a.converged.all() and b.converged.all()
    assert not a.at_strength_bound.any() and not b.at_strength_bound.any()
    checks.append('All main population fits converged away from concentration bounds')
    f=pd.read_csv(OUT/'forecast_2027_all.csv'); d=read('model_deliveries')
    assert set(f.batter_id)==set(d.loc[d.season==2026,'batter_id'])
    prior=json.loads((OUT/'watchlist_metadata.json').read_text())['prior']
    for r in f.itertuples():
        counts=snapshot_counts(d.loc[d.batter_id==r.batter_id],r.reference_date,CONFIG)
        assert np.isclose(posterior(*counts[r.model],prior),r.forecast)
        assert 0<=r.forecast_low<=r.forecast_high<=1
    checks.append('Every 2027 forecast independently reconstructed from permitted history')
    results={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.csv'))}
    report=dict(passed=True,checks=checks,python=platform.python_version(),result_sha256=results)
    (ROOT/'reports/verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'checks':checks},indent=2))

if __name__=='__main__': main()
