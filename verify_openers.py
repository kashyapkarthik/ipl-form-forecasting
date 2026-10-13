"""Independent reconstruction checks for saved opener evidence; no win thresholds."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from openers_experiment import ROOT, OUT, CONFIG, protected_hashes, binary_losses


def main():
    checks = []
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['protocol_sha256'] == hashlib.sha256((ROOT/'config/openers_protocol.json').read_bytes()).hexdigest()
    assert manifest['source_sha256'] == hashlib.sha256((ROOT/'data/raw/ipl_json.zip').read_bytes()).hexdigest()
    assert manifest['protected_artifact_hashes'] == protected_hashes()
    checks.append('Raw data and follow-up protocol hashes match; original analysis artifacts remain unchanged')
    # Reconstruct opening identities from the pinned JSON, independently of entry ranks.
    pairs = []
    with zipfile.ZipFile(ROOT/'data/raw/ipl_json.zip') as archive:
        for name in archive.namelist():
            if not name.endswith('.json'):
                continue
            source = json.loads(archive.read(name))
            registry = source['info']['registry']['people']
            for innings, inn in enumerate(source.get('innings', []), 1):
                if inn.get('super_over', False):
                    continue
                events = [ball for over in inn['overs'] for ball in over['deliveries']]
                if not events:
                    continue
                for key in ['batter', 'non_striker']:
                    pairs.append((Path(name).stem, innings, registry[events[0][key]]))
    pairs = pd.DataFrame(pairs, columns=['match_id', 'innings', 'batter_id'])
    d = pd.read_parquet(ROOT/'data/processed/model_deliveries.parquet')
    opening = d.merge(pairs, on=['match_id', 'innings', 'batter_id'], validate='many_to_one')
    assert len(opening) == manifest['opening_audit']['opening_opportunities']
    assert opening.is_boundary.sum() == manifest['opening_audit']['opening_boundaries']
    splits = pd.read_parquet(ROOT/'data/processed/match_splits.parquet')
    opening = opening.merge(splits[['match_id', 'half', 'cutoff_date']], on='match_id', validate='many_to_one')
    assert (opening.loc[opening.half=='early', 'date'] <= opening.loc[opening.half=='early', 'cutoff_date']).all()
    assert (opening.loc[opening.half=='late', 'date'] > opening.loc[opening.half=='late', 'cutoff_date']).all()
    checks.append('Opening identities reconstructed directly from raw JSON, including illegal first events; chronology valid')
    p = pd.read_csv(OUT/'predictions.csv')
    assert not p.duplicated(['scenario', 'season', 'batter_id', 'model']).any()
    assert p.p.between(0, 1).all() and np.isfinite(p.p).all()
    for _, g in p.groupby(['scenario', 'season']):
        assert g.groupby('model').N.sum().nunique() == 1
        assert g.groupby('batter_id').N.nunique().eq(1).all()
        assert g.groupby('batter_id').B.nunique().eq(1).all()
        assert g.groupby('batter_id').model.nunique().nunique() == 1
    primary = p.loc[p.scenario=='primary']
    priors = pd.read_csv(OUT/'priors.csv')
    for year, g in primary.groupby('season'):
        early = opening.loc[(opening.season==year)&(opening.half=='early')].groupby('batter_id').is_boundary.agg(['sum', 'size'])
        late = opening.loc[(opening.season==year)&(opening.half=='late')].groupby('batter_id').is_boundary.agg(['sum', 'size'])
        prior = priors.loc[(priors.scenario=='primary')&(priors.population=='openers')&(priors.season==year)].iloc[0]
        assert prior.records == len(early) and prior.early_N == early['size'].sum()
        np.testing.assert_allclose(prior.pooled, early['sum'].sum()/early['size'].sum())
        for r in g.itertuples():
            n, b = (early.loc[r.batter_id, 'size'], early.loc[r.batter_id, 'sum']) if r.batter_id in early.index else (0, 0)
            assert r.early_N == n and r.early_B == b
            assert r.N == late.loc[r.batter_id, 'size'] and r.B == late.loc[r.batter_id, 'sum']
            if r.model=='beta_binomial':
                np.testing.assert_allclose(r.p, (prior.alpha+b)/(prior.strength+n))
            if r.model=='population':
                np.testing.assert_allclose(r.p, prior.pooled)
            if r.model=='raw':
                np.testing.assert_allclose(r.p, b/n if n else prior.pooled)
        np.testing.assert_allclose(prior.prior_sd, np.sqrt(prior.mu*(1-prior.mu)/(prior.strength+1)))
    checks.append('Every primary forecast reconstructed from only early opening outcomes; all models share late targets')
    # Independently recalculate validation selection and all primary all-player losses.
    summary = pd.read_csv(OUT/'metrics.csv')
    for period, years in [('development', range(2008, 2024)), ('validation', [2024]), ('test', [2025, 2026])]:
        for model, g in primary.loc[primary.season.isin(years)].groupby('model'):
            r = summary.loc[(summary.scenario=='primary')&(summary.subset=='all_late_players')&
                            (summary.period==period)&(summary.model==model)].iloc[0]
            probability = np.clip(g.p, 1e-6, 1-1e-6)
            ll = -(g.B*np.log(probability)+(g.N-g.B)*np.log(1-probability)).sum()/g.N.sum()
            bs = (g.B-2*g.B*g.p+g.N*g.p**2).sum()/g.N.sum()
            np.testing.assert_allclose([r.log_loss, r.brier], [ll, bs], atol=1e-12)
    grid = pd.read_csv(OUT/'strength_validation_grid.csv')
    ranked = grid.assign(distance=np.abs(np.log(grid.multiplier))).sort_values(['log_loss', 'distance', 'multiplier'])
    assert ranked.iloc[0].model == manifest['selected_strength_model']
    selected = primary.loc[primary.model=='selected_strength'].sort_values(['season', 'batter_id'])
    fixed = primary.loc[primary.model==manifest['selected_strength_model']].sort_values(['season', 'batter_id'])
    np.testing.assert_array_equal(selected.p, fixed.p)
    checks.append('Primary metrics independently recalculated; selected width matches 2024 ranking and stays fixed')
    innings = pd.read_csv(OUT/'evaluation_innings_predictions.csv')
    keys = ['season', 'date', 'match_id', 'innings', 'batter_id']
    assert not innings.duplicated(keys+['model']).any()
    for model, g in innings.groupby('model'):
        expected = primary.loc[primary.season.isin([2025, 2026])&(primary.model==model)].sort_values(['season', 'batter_id'])
        counts = g.groupby(['season', 'batter_id'], as_index=False)[['N', 'B']].sum()
        np.testing.assert_array_equal(counts[['N', 'B']], expected[['N', 'B']])
    u = pd.read_csv(OUT/'uncertainty.csv')
    assert (u.log_loss_low <= u.log_loss_high).all() and (u.brier_low <= u.brier_high).all()
    for r in u.loc[(u.scenario=='primary')&(u.subset=='all_late_players')].itertuples():
        a = summary.loc[(summary.scenario=='primary')&(summary.subset=='all_late_players')&(summary.period=='test')].set_index('model')
        np.testing.assert_allclose(r.log_loss_difference, a.loc[r.model, 'log_loss']-a.loc[r.reference, 'log_loss'], atol=1e-12)
    checks.append('Bootstrap innings match published player-season totals; interval estimates match score differences')
    assert priors.converged.all()
    assert not priors.loc[priors.scenario=='primary', 'at_strength_bound'].any()
    assert not priors.loc[priors.season.isin([2025, 2026]), 'at_strength_bound'].any()
    bounds = priors.loc[priors.at_strength_bound, ['scenario', 'season', 'population']].to_dict('records')
    checks.append('Primary and evaluation prior fits converge away from bounds; development sensitivity bounds disclosed')
    report = dict(passed=True, checks=checks, development_sensitivity_bound_fits=bounds,
                  original_artifacts_unchanged=True,
                  result_sha256={x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(OUT.glob('*.csv'))})
    (ROOT/'reports/openers_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
