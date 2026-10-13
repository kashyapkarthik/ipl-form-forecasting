"""Independent raw-JSON audit of alternative outcomes and published forecasts."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from outcomes_experiment import ROOT, OUT, CONFIG, KEYS, MODELS, protected_hashes


def main():
    checks = []
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert manifest['protected_files'] == protected_hashes()
    assert manifest['raw_sha256'] == hashlib.sha256((ROOT/'data/raw/ipl_json.zip').read_bytes()).hexdigest()
    assert manifest['protocol_sha256'] == hashlib.sha256((ROOT/'config/outcomes_protocol.json').read_bytes()).hexdigest()
    assert manifest['selection_sha256'] == hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    for name, expected in manifest['results_sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == expected
    checks.append('Raw archive, protocol, selection, outputs and every protected earlier-study file match their hashes')
    records, illegal_runs = [], 0
    with zipfile.ZipFile(ROOT/'data/raw/ipl_json.zip') as archive:
        for path in archive.namelist():
            if not path.endswith('.json'): continue
            match = json.loads(archive.read(path))
            mid, date = Path(path).stem, min(match['info']['dates'])
            registry = match['info']['registry']['people']
            for inn_num, inn in enumerate(match.get('innings', []), 1):
                if inn.get('super_over', False): continue
                seen, counts = {}, defaultdict(lambda: [0, 0, 0, 0, 0])
                for over in inn['overs']:
                    for ball in over['deliveries']:
                        for name in [ball['batter'], ball['non_striker']]:
                            if name not in seen: seen[name] = len(seen)+1
                        bat, runs = ball['batter'], ball['runs']['batter']
                        counts[bat][0] += runs
                        extras = ball.get('extras', {})
                        if extras.get('wides', 0) or extras.get('noballs', 0):
                            illegal_runs += runs
                            continue
                        counts[bat][1] += 1
                        counts[bat][2] += int(runs in (4, 6) and not ball['runs'].get('non_boundary', False))
                        counts[bat][3] += int(runs == 6 and not ball['runs'].get('non_boundary', False))
                        counts[bat][4] += int(runs == 0)
                for name, rank in seen.items():
                    r, n, b, six, dot = counts[name]
                    records.append(dict(season=int(date[:4]), date=date, match_id=mid, innings=inn_num,
                                        batter_id=registry[name], position_group=rank, total_runs=r,
                                        legal_balls=n, boundary=b, six=six, dot=dot))
    raw = pd.DataFrame(records)
    app = pd.read_csv(OUT/'batting_appearances.csv', dtype={'match_id':str})
    pair = raw.merge(app, on=KEYS, how='outer', indicator=True, validate='one_to_one', suffixes=('_raw', '_saved'))
    assert pair._merge.eq('both').all()
    np.testing.assert_array_equal(pair[['total_runs_raw', 'legal_balls_raw']], pair[['total_runs_saved', 'legal_balls_saved']])
    checks.append('Every ordinary batting appearance, complete innings score and legal-ball count rebuilt independently from raw match JSON')
    split = pd.read_parquet(ROOT/'data/processed/match_splits.parquet')
    raw = raw.loc[raw.position_group.le(7)].merge(split[['match_id', 'half', 'cutoff_date']], on='match_id', validate='many_to_one')
    assert (raw.loc[raw.half.eq('early'), 'date'] <= raw.loc[raw.half.eq('early'), 'cutoff_date']).all()
    assert (raw.loc[raw.half.eq('late'), 'date'] > raw.loc[raw.half.eq('late'), 'cutoff_date']).all()
    pred = pd.read_csv(OUT/'evaluation_predictions.csv', dtype={'match_id':str})
    pri = pd.read_csv(OUT/'priors.csv').set_index(['outcome', 'window', 'season', 'position_group'])
    for (outcome, window, year), g in pred.groupby(['outcome', 'window', 'season']):
        x = raw.copy()
        x['N'] = 1 if outcome.startswith('score') else x.legal_balls
        x['B'] = x.total_runs.ge(int(outcome[5:])).astype(int) if outcome.startswith('score') else x[outcome]
        x = x.loc[x.N.gt(0)]
        train_mask = x.season.eq(year)&x.half.eq('early')
        if window != 'current': train_mask |= x.season.between(year-3, year-1)
        train = x.loc[train_mask].groupby(['position_group', 'batter_id'])[['N', 'B']].sum()
        target = x.loc[x.season.eq(year)&x.half.eq('late')]
        paired = g.merge(target[KEYS+['N', 'B']], on=KEYS, how='outer', indicator=True,
                         validate='one_to_one', suffixes=('', '_raw'))
        assert paired._merge.eq('both').all()
        np.testing.assert_array_equal(paired[['N', 'B']], paired[['N_raw', 'B_raw']])
        for r in g.itertuples():
            key = (r.position_group, r.batter_id)
            n, b = (train.loc[key, 'N'], train.loc[key, 'B']) if key in train.index else (0, 0)
            assert r.early_N == n and r.early_B == b
            prior = pri.loc[(outcome, window, year, r.position_group)]
            np.testing.assert_allclose(r.p_beta, (b+prior.alpha)/(n+prior.strength), atol=1e-12)
            pooled = train.loc[r.position_group].B.sum()/train.loc[r.position_group].N.sum()
            np.testing.assert_allclose(r.p_population, pooled, atol=1e-12)
            np.testing.assert_allclose(r.p_raw, b/n if n else pooled, atol=1e-12)
            np.testing.assert_allclose(r.p_jeffreys, (b+.5)/(n+1) if n else pooled, atol=1e-12)
    checks.append('All ten evaluation target sets, same-position past-only histories, posterior formulas and comparator forecasts match raw reconstruction')
    score = pd.read_csv(OUT/'evaluation_metrics.csv').set_index(['outcome', 'window', 'model'])
    for (outcome, window), g in pred.groupby(['outcome', 'window']):
        assert not g.duplicated(KEYS).any()
        for model in MODELS:
            p = g[f'p_{model}']
            assert np.isfinite(p).all() and p.between(0, 1).all()
            pc = np.clip(p, CONFIG['logloss_clip'], 1-CONFIG['logloss_clip'])
            ll = -(g.B*np.log(pc)+(g.N-g.B)*np.log1p(-pc)).sum()/g.N.sum()
            bs = (g.B-2*g.B*p+g.N*p*p).sum()/g.N.sum()
            s = score.loc[(outcome, window, model)]
            np.testing.assert_allclose([ll, bs], [s.log_loss, s.brier], atol=1e-12)
    old = pd.read_csv(ROOT/'results/positions/metrics.csv', dtype={'position_group':str})
    old = old.loc[old.scenario.eq('separate_positions')&old.position_group.eq('all')&
                  old.period.eq('evaluation')&old.subset.eq('all_late')].set_index('model')
    for new, prev in [('beta','beta_binomial'), ('population','population'), ('raw','raw'), ('jeffreys','jeffreys')]:
        s = score.loc[('boundary', 'current', new)]
        np.testing.assert_allclose([s.log_loss, s.brier], old.loc[prev, ['log_loss', 'brier']].to_numpy(float), atol=1e-10, rtol=0)
    checks.append('All evaluation scores recalculated independently; boundary control reproduces the earlier seven-position study to 1e-10')
    intervals = pd.read_csv(OUT/'uncertainty.csv')
    for r in intervals.itertuples():
        sb, sr = score.loc[(r.outcome, r.window, 'beta')], score.loc[(r.outcome, r.window, r.reference)]
        np.testing.assert_allclose(r.log_loss_difference, sb.log_loss-sr.log_loss, atol=1e-12)
        np.testing.assert_allclose(r.log_skill_pct, 100*(1-sb.log_loss/sr.log_loss), atol=1e-10)
    chosen = json.loads((OUT/'selection.json').read_text())['selected']
    validation = pd.read_csv(OUT/'validation_metrics.csv')
    development = pd.read_csv(OUT/'development_metrics.csv')
    for pick in chosen:
        v = validation.loc[validation.window.eq(pick['window'])].pivot(index='outcome', columns='model', values='log_loss')
        dv = development.loc[development.window.eq(pick['window'])].pivot(index='outcome', columns='model', values='log_loss')
        eligible = (v.beta < v.jeffreys) & (dv.beta < dv.population)
        skill = 1-v.beta/v.population
        expected = skill.loc[eligible].idxmax() if eligible.any() else skill.idxmax()
        assert pick['outcome'] == expected
    checks.append('Paired uncertainty point estimates and validation-only outcome selections independently verified')
    report = dict(passed=True, checks=checks, ordinary_batting_appearances=len(app),
                  batter_runs_on_illegal_deliveries=illegal_runs, zero_legal_ball_appearances=int(app.legal_balls.eq(0).sum()),
                  evaluation_dates=pred[['season', 'date']].drop_duplicates().shape[0],
                  evaluation_players=pred.batter_id.nunique(), fallback_fits=int(pri.fallback.sum()),
                  bounded_evaluation_fits=int(pri.reset_index().query('season in [2025, 2026]').at_strength_bound.sum()))
    (ROOT/'reports/outcomes_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
