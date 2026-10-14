"""Independently reconstruct every published memory estimate from delivery outcomes."""
import json
import numpy as np
import pandas as pd
from .core import ROOT


def count(g,model,cutoff):
    x=g.loc[g.date.lt(cutoff)].sort_values(['date','match_id','innings','delivery_sequence'])
    y=x.is_six.to_numpy(float)
    if model=='season_only':
        y=x.loc[x.season.eq(2025 if cutoff=='2026-03-28' else 2026),'is_six'].to_numpy(float)
        return float(y.sum()),float(len(y))
    if model=='expanding':return float(y.sum()),float(len(y))
    if model.startswith('rolling_'):
        n=int(model.split('_')[-1]);return float(y[-n:].sum()),float(min(len(y),n))
    h=int(model.split('_')[-1]);w=np.exp2(-np.arange(len(y)-1,-1,-1)/h)
    return float(np.sum(w*y)),float(np.sum(w))


def main():
    out=ROOT/'results/player_memory';a=json.loads((out/'audit.json').read_text())
    data=pd.read_parquet(ROOT/'results/sixes/deliveries.parquet');details=pd.read_csv(out/'role_details.csv');main=pd.read_csv(out/'memory_changes.csv')
    original=pd.read_csv(ROOT/'results/player_improvement/player_changes.csv')
    pri=pd.read_csv(out/'fixed_priors.csv').set_index('role');draws=pd.read_csv(out/'prior_resamples.csv')
    groups={k:g for k,g in data.groupby(['batter_id','role'])}
    latest=data.loc[data.season.eq(2026)].groupby(['batter_id','role']).size();totals=latest.groupby('batter_id').sum()
    max_difference=0
    for r in details.itertuples():
        g=groups[(r.batter_id,r.role)]
        for prefix,cutoff in [('before',a['before_cutoff']),('after',a['after_cutoff'])]:
            b,n=count(g,r.model,cutoff)
            np.testing.assert_allclose([b,n],[getattr(r,prefix+'_B'),getattr(r,prefix+'_N')],atol=1e-9,rtol=1e-12)
            p=pri.loc[r.role];expected=(p.alpha+b)/(p.strength+n)
            max_difference=max(max_difference,abs(expected-getattr(r,prefix)))
            np.testing.assert_allclose(expected,getattr(r,prefix),atol=1e-12)
        np.testing.assert_allclose(r.role_weight,latest[(r.batter_id,r.role)]/totals[r.batter_id],atol=1e-12)
        past=g.loc[g.season.between(2023,2025)]
        assert (len(past),int(past.is_six.sum()))==(r.baseline_N,r.baseline_B)
        assert r.prior_2025_N==int(g.season.eq(2025).sum())
    dgroups={k:g for k,g in details.groupby(['batter_id','model'])}
    for r in main.itertuples():
        g=dgroups[(r.batter_id,r.model)]
        for prefix in ['before','after']:
            expected=np.sum(g.role_weight*g[prefix]);np.testing.assert_allclose(expected,getattr(r,prefix),atol=1e-12)
        np.testing.assert_allclose(r.change,r.after-r.before,atol=1e-12)
        base=np.sum(g.role_weight*(g.alpha+g.baseline_B)/(g.strength+g.baseline_N))
        np.testing.assert_allclose(base,r.baseline_2023_25,atol=1e-12)
        np.testing.assert_allclose(r.above_baseline,r.after-base,atol=1e-12)
    checks={'all_40_original_players':set(main.batter_id)==set(original.batter_id),'all_440_comparisons':len(main)==440,
            'no_duplicate_comparisons':not main.duplicated(['batter_id','model']).any(),
            'all_role_counts_reconstructed':True,'all_role_and_player_shrunk_estimates_reconstructed':True,'same_fixed_role_mix_reconstructed':True,
            'all_6000_prior_refits_present':len(draws)==6000 and not draws.duplicated(['replicate','role']).any(),
            'all_prior_refits_converged':bool(draws.converged.all()),'positive_prior_parameters':bool((draws.alpha>0).all() and (draws.beta>0).all()),
            'all_simultaneous_intervals_contain_zero':bool((main.loc[main.primary,'simultaneous_low']<=0).all() and (main.loc[main.primary,'simultaneous_high']>=0).all())}
    assert all(checks.values());report={'passed':True,'checks':checks,'largest_reconstruction_difference':max_difference,'role_rows_checked':len(details),
                                      'interval_scope':'Pointwise/resampling algorithm tested separately; this audit reconstructs point estimates, counts and reported simultaneous no-finding result.'}
    (ROOT/'reports/player_memory_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
