"""Reconstruct positions and counts directly from raw JSON; verify published forecasts."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from positions_experiment import ROOT, OUT, CONFIG, protected_hashes


def main():
    checks = []
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['protected_artifact_hashes']==protected_hashes()
    for field, file in [('source_sha256', 'data/raw/ipl_json.zip'), ('protocol_sha256', 'config/positions_protocol.json'),
                        ('protocol_text_sha256', 'docs/POSITIONS_PROTOCOL.md')]:
        assert manifest[field]==hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    checks.append('Source/protocol hashes match; all earlier tracked research artifacts unchanged')
    counts = defaultdict(lambda: [0, 0])
    with zipfile.ZipFile(ROOT/'data/raw/ipl_json.zip') as archive:
        for path in archive.namelist():
            if not path.endswith('.json'): continue
            source = json.loads(archive.read(path))
            match_id, date = Path(path).stem, min(source['info']['dates'])
            registry = source['info']['registry']['people']
            for inn_num, inn in enumerate(source.get('innings', []), 1):
                if inn.get('super_over', False): continue
                seen = {}
                for over in inn['overs']:
                    for ball in over['deliveries']:
                        for name in [ball['batter'], ball['non_striker']]:
                            if name not in seen: seen[name] = len(seen)+1
                        extras = ball.get('extras', {})
                        if extras.get('wides', 0) or extras.get('noballs', 0): continue
                        pos = seen[ball['batter']]
                        key = (int(date[:4]), date, match_id, inn_num, registry[ball['batter']], pos)
                        counts[key][0] += 1
                        counts[key][1] += int(ball['runs']['batter'] in (4, 6) and not ball['runs'].get('non_boundary', False))
    columns = ['season', 'date', 'match_id', 'innings', 'batter_id', 'position_group']
    raw = pd.DataFrame([dict(zip(columns, key), N=value[0], B=value[1]) for key, value in counts.items()])
    processed = pd.read_parquet(ROOT/'data/processed/batter_innings.parquet').rename(columns={'batting_entry_order': 'position_group'})
    joined = raw.merge(processed[columns+['N', 'B']], on=columns, how='outer', indicator=True, validate='one_to_one', suffixes=('_raw', '_processed'))
    assert joined._merge.eq('both').all()
    assert joined.N_raw.eq(joined.N_processed).all() and joined.B_raw.eq(joined.B_processed).all()
    checks.append('All eligible batter-innings counts and arrival ranks independently reconstructed from raw JSON')
    splits = pd.read_parquet(ROOT/'data/processed/match_splits.parquet')
    raw = raw.merge(splits[['match_id', 'half', 'cutoff_date']], on='match_id', validate='many_to_one')
    assert (raw.loc[raw.half=='early', 'date'] <= raw.loc[raw.half=='early', 'cutoff_date']).all()
    assert (raw.loc[raw.half=='late', 'date'] > raw.loc[raw.half=='late', 'cutoff_date']).all()
    target = raw.loc[raw.position_group.isin(CONFIG['positions'])]
    half = target.groupby(['season', 'half', 'position_group', 'batter_id'])[['N', 'B']].sum()
    early = half.xs('early', level='half')
    late = half.xs('late', level='half')
    early_all = raw.loc[raw.half.eq('early')].groupby(['season', 'batter_id'])[['N', 'B']].sum()
    early_top7 = target.loc[target.half.eq('early')].groupby(['season', 'batter_id'])[['N', 'B']].sum()
    p = pd.read_csv(OUT/'predictions.csv')
    prior = pd.read_csv(OUT/'priors.csv')
    primary = p.loc[p.scenario=='separate_positions']
    pri = prior.loc[(prior.scenario=='separate_positions')&prior.position_group.gt(0)].set_index(['season', 'position_group'])
    assert not p.duplicated(['scenario', 'season', 'position_group', 'batter_id']).any()
    for r in primary.itertuples():
        key = (r.season, r.position_group, r.batter_id)
        n, b = (early.loc[key, 'N'], early.loc[key, 'B']) if key in early.index else (0, 0)
        assert r.early_N==n and r.early_B==b
        assert r.N==late.loc[key, 'N'] and r.B==late.loc[key, 'B']
        pr = pri.loc[(r.season, r.position_group)]
        np.testing.assert_allclose(r.p_beta_binomial, (b+pr.alpha)/(n+pr.strength), atol=1e-12)
        np.testing.assert_allclose(r.p_raw, b/n if n else pr.pooled, atol=1e-12)
        np.testing.assert_allclose(r.p_population, pr.pooled, atol=1e-12)
        player_key = (r.season, r.batter_id)
        for source, prefix, label in [(early_all, 'all', -1), (early_top7, 'top7', -2)]:
            en, eb = (source.loc[player_key, 'N'], source.loc[player_key, 'B']) if player_key in source.index else (0, 0)
            assert getattr(r, prefix+'_early_N')==en and getattr(r, prefix+'_early_B')==eb
            reference = prior.loc[prior.scenario.eq('separate_positions')&prior.season.eq(r.season)&prior.position_group.eq(label)].iloc[0]
            probability = r.p_original_beta if prefix=='all' else r.p_top7_beta
            np.testing.assert_allclose(probability, (eb+reference.alpha)/(en+reference.strength), atol=1e-12)
    assert len(primary)==len(late)
    for (year, pos), r in pri.iterrows():
        group = early.loc[(year, pos)]
        assert r.position_records==len(group) and r.early_N==group.N.sum()
        np.testing.assert_allclose(r.pooled, group.B.sum()/group.N.sum(), atol=1e-12)
    checks.append('Every primary posterior, raw rate, cold-start fallback and target count reconstructed from same-position early data')
    old = pd.read_csv(ROOT/'results/brown_predictions.csv').query("model=='beta_binomial'")
    paired = primary.merge(old[['season', 'batter_id', 'p']], on=['season', 'batter_id'], validate='many_to_one')
    np.testing.assert_allclose(paired.p_original_beta, paired.p, atol=1e-10, rtol=0)
    op = pd.read_csv(ROOT/'results/openers/predictions.csv').query("scenario=='primary' and model=='beta_binomial'")
    combined = p.loc[(p.scenario=='combined_openers')&p.position_group.eq(0)].merge(
        op[['season', 'batter_id', 'N', 'B', 'p']], on=['season', 'batter_id'], validate='one_to_one', suffixes=('', '_old'))
    assert len(combined)==len(op)
    np.testing.assert_allclose(combined.p_beta_binomial, combined.p, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(combined[['N', 'B']].to_numpy(), combined[['N_old', 'B_old']].to_numpy())
    checks.append('Original all-batter reference and combined-opening forecasts reproduce committed studies to 1e-10')
    summary = pd.read_csv(OUT/'metrics.csv', dtype={'position_group': str})
    for scenario, g in p.groupby('scenario'):
        for position in ['all', *map(str, sorted(g.position_group.unique()))]:
            q = g.loc[g.season.isin([2025, 2026])]
            if position!='all': q = q.loc[q.position_group.eq(int(position))]
            scores = summary.loc[(summary.scenario==scenario)&summary.position_group.eq(position)&
                                 summary.subset.eq('all_late')&summary.period.eq('evaluation')].set_index('model')
            for model, row in scores.iterrows():
                prob = q[f'p_{model}']
                assert np.isfinite(prob).all() and prob.between(0, 1).all()
                pc = np.clip(prob, 1e-6, 1-1e-6)
                ll = -(q.B*np.log(pc)+(q.N-q.B)*np.log1p(-pc)).sum()/q.N.sum()
                bs = (q.B-2*q.B*prob+q.N*prob**2).sum()/q.N.sum()
                np.testing.assert_allclose([ll, bs], [row.log_loss, row.brier], atol=1e-12)
                assert row.N==q.N.sum()
    ip = pd.read_csv(OUT/'evaluation_innings_predictions.csv')
    for scenario, g in ip.groupby('scenario'):
        assert not g.duplicated(columns).any()
        expected = p.loc[p.scenario.eq(scenario)&p.season.isin([2025, 2026])].sort_values(['season', 'position_group', 'batter_id'])
        sums = g.groupby(['season', 'position_group', 'batter_id'], as_index=False)[['N', 'B']].sum()
        np.testing.assert_array_equal(sums[['N', 'B']].to_numpy(), expected[['N', 'B']].to_numpy())
        # Same row contains every model prediction: target equality is structural.
        assert g.loc[:, [c for c in g if c.startswith('p_') and c!='p_leave_one_out']].notna().all().all()
    intervals = pd.read_csv(OUT/'uncertainty.csv', dtype={'position_group': str})
    for r in intervals.itertuples():
        s = summary.loc[(summary.scenario==r.scenario)&summary.position_group.eq(r.position_group)&
                        summary.subset.eq(r.subset)&summary.period.eq('evaluation')].set_index('model')
        np.testing.assert_allclose(r.log_loss_difference, s.loc[r.model, 'log_loss']-s.loc[r.reference, 'log_loss'], atol=1e-12)
    checks.append('All evaluation scores independently recalculated; identical targets and paired interval point estimates verified')
    convention = pd.read_csv(OUT/'opener_convention_comparison.csv', dtype={'position_group': str})
    total = summary.loc[summary.position_group.eq('all')&summary.subset.eq('all_late')&summary.period.eq('evaluation')&summary.model.eq('beta_binomial')].set_index('scenario')
    np.testing.assert_allclose(convention.loc[convention.position_group.eq('all'), 'log_loss_difference'].item(),
                               total.loc['combined_openers', 'log_loss']-total.loc['separate_positions', 'log_loss'], atol=1e-12)
    checks.append('Shared 1–7 history/prior forecasts and paired opening-convention aggregate difference reconstructed')
    report = dict(passed=True, checks=checks, test_suite_command='python -m pytest -q',
                  fallback_fits=int(prior.fallback_to_all.sum()),
                  evaluation_bound_fits=prior.loc[prior.season.isin([2025, 2026])&prior.at_strength_bound,
                                                   ['scenario', 'season', 'position_group']].to_dict('records'),
                  result_sha256={x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(OUT.glob('*.csv'))})
    (ROOT/'reports/positions_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key!='result_sha256'}, indent=2))


if __name__=='__main__':
    main()
