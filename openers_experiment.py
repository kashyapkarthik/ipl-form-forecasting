"""Isolated, exploratory repeat of experiment 1; never runs later experiments."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from models import fit_prior, posterior

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config/openers_protocol.json').read_text())
OUT = ROOT / 'results/openers'
KEYS = ['season', 'date', 'match_id', 'innings', 'batter_id']
CORE = ['population', 'raw', 'beta_binomial']


def read(name):
    return pd.read_parquet(ROOT / f'data/processed/{name}.parquet')


def save(frame, name):
    frame.to_csv(OUT / f'{name}.csv', index=False)


def opening_deliveries(raw, eligible):
    """Use both identities on the first raw event, before eligibility filtering."""
    first = (raw.loc[~raw.is_super_over]
             .sort_values(['match_id', 'innings', 'delivery_sequence'])
             .drop_duplicates(['match_id', 'innings']))
    pair = pd.concat([
        first[['match_id', 'innings', 'batter_id']],
        first[['match_id', 'innings', 'non_striker_id']].rename(columns={'non_striker_id': 'batter_id'})
    ], ignore_index=True)
    assert not pair.duplicated().any()
    result = eligible.merge(pair, on=['match_id', 'innings', 'batter_id'], validate='many_to_one')
    assert set(result.delivery_id) == set(eligible.loc[eligible.batting_entry_order <= 2, 'delivery_id'])
    assert not result.delivery_id.duplicated().any()
    assert result.primary_eligible.all() and not result.is_super_over.any()
    audit = dict(ordinary_innings=len(first), opening_pair_slots=len(pair),
                 eligible_opening_innings=len(result[['match_id', 'innings', 'batter_id']].drop_duplicates()),
                 opening_opportunities=len(result), opening_boundaries=int(result.is_boundary.sum()),
                 first_event_illegal_innings=int((~first.is_legal).sum()),
                 first_event_nonzero_over_innings=int((first.over != 0).sum()),
                 entry_order_agrees=True)
    return result, audit


def match_count_splits(matches):
    rows = []
    for year, g in matches.groupby('season'):
        counts = g.groupby('date').size().sort_index()
        # At least one complete date in each half; pandas idxmin chooses earliest tie.
        cutoff = (counts.cumsum().iloc[:-1] - len(g) / 2).abs().idxmin()
        for r in g.itertuples():
            rows.append(dict(match_id=r.match_id, season=year, cutoff_date=cutoff,
                             half='early' if r.date <= cutoff else 'late'))
    return pd.DataFrame(rows)


def halves(d, splits):
    x = d.merge(splits[['match_id', 'half']], on='match_id', validate='many_to_one')
    return x.groupby(['season', 'half', 'batter_id'], as_index=False).agg(
        N=('is_boundary', 'size'), B=('is_boundary', 'sum'))


def scaled_prior(prior, scale):
    return {**prior, 'alpha': prior['alpha'] * scale, 'beta': prior['beta'] * scale,
            'strength': prior['strength'] * scale}


def fit_season(early, late_ids, all_early=None, diagnostics=True):
    """Only early outcomes are accepted. Late IDs supply a scoring roster, no outcomes."""
    early = early.set_index('batter_id')
    prior = fit_prior(early.B, early.N)
    original = fit_prior(all_early.B, all_early.N) if all_early is not None else None
    all_index = all_early.set_index('batter_id') if all_early is not None else None
    rows = []
    for pid in sorted(late_ids):
        n, b = (int(early.loc[pid, 'N']), int(early.loc[pid, 'B'])) if pid in early.index else (0, 0)
        forecasts = {'population': prior['pooled'], 'raw': b/n if n else prior['pooled'],
                     'beta_binomial': float(posterior(b, n, prior))}
        if diagnostics:
            forecasts.update(prior_mean=prior['mu'], jeffreys=(b+.5)/(n+1) if n else prior['pooled'])
            for scale in CONFIG['strength_multipliers']:
                forecasts[f'strength_x{scale:g}'] = float(posterior(b, n, scaled_prior(prior, scale)))
            other = early.drop(index=pid, errors='ignore')
            loo = fit_prior(other.B, other.N) if pid in early.index else prior
            forecasts['leave_one_out'] = float(posterior(b, n, loo))
            assert original is not None
            an, ab = (int(all_index.loc[pid, 'N']), int(all_index.loc[pid, 'B'])) if pid in all_index.index else (0, 0)
            forecasts.update(original_population=original['pooled'],
                             original_beta=float(posterior(ab, an, original)),
                             all_prior_opening_history=float(posterior(b, n, original)))
            # Post-protocol descriptive ablation: separate prior centre from strength.
            for name, mu, k in [('opening_mean_all_strength', prior['mu'], original['strength']),
                                ('all_mean_opening_strength', original['mu'], prior['strength'])]:
                forecasts[name] = (mu*k+b)/(k+n)
        for model, probability in forecasts.items():
            rows.append(dict(batter_id=pid, model=model, p=probability, early_N=n, early_B=b,
                             opening_fitted_data_weight=n/(n+prior['strength']),
                             all_role_early_N=an if diagnostics else np.nan,
                             loo_strength=loo['strength'] if diagnostics and model=='leave_one_out' else np.nan,
                             loo_at_bound=loo['at_strength_bound'] if diagnostics and model=='leave_one_out' else False))
    return pd.DataFrame(rows), prior, original


def binary_losses(B, N, probability):
    """Log loss clips only to avoid log(0); Brier retains genuine raw 0/1."""
    B, N, p = np.asarray(B), np.asarray(N), np.asarray(probability)
    clipped = np.clip(p, CONFIG['raw_logloss_clip'], 1-CONFIG['raw_logloss_clip'])
    return -B*np.log(clipped)-(N-B)*np.log1p(-clipped), B*(1-p)**2+(N-B)*p**2


def metrics(predictions):
    rows = []
    for model, g in predictions.groupby('model'):
        ll, bs = binary_losses(g.B, g.N, g.p)
        player = pd.DataFrame(dict(batter_id=g.batter_id, N=g.N, ll=ll, bs=bs)).groupby('batter_id').sum()
        transformed = np.arcsin(np.sqrt((g.B+.25)/(g.N+.5)))
        rows.append(dict(model=model, N=int(g.N.sum()), B=int(g.B.sum()),
                         players=g.batter_id.nunique(), player_seasons=len(g),
                         log_loss=ll.sum()/g.N.sum(), brier=bs.sum()/g.N.sum(),
                         player_macro_log_loss=(player.ll/player.N).mean(),
                         player_macro_brier=(player.bs/player.N).mean(),
                         player_rate_mse=np.mean((g.p-g.B/g.N)**2),
                         noise_adjusted_arcsine_mse=np.mean((np.arcsin(np.sqrt(g.p))-transformed)**2-1/(4*g.N))
                         if (g.N.ge(11)&g.early_N.ge(11)).all() else np.nan))
    return pd.DataFrame(rows)


def subset(frame, label):
    if label == 'min11_both_halves':
        return frame.loc[(frame.early_N >= 11) & (frame.N >= 11)]
    if label == 'early_history':
        return frame.loc[frame.early_N > 0]
    return frame


def paired_interval(predictions, model, reference, repetitions=None):
    a = predictions.loc[predictions.model == model, KEYS+['N', 'B', 'p']]
    b = predictions.loc[predictions.model == reference, KEYS+['N', 'B', 'p']]
    pair = a.merge(b, on=KEYS, how='outer', validate='one_to_one', suffixes=('_a', '_b'), indicator=True)
    assert pair._merge.eq('both').all(), 'Models must score identical innings'
    assert pair.N_a.eq(pair.N_b).all() and pair.B_a.eq(pair.B_b).all()
    al, ab = binary_losses(pair.B_a, pair.N_a, pair.p_a)
    bl, bb = binary_losses(pair.B_b, pair.N_b, pair.p_b)
    blocks = pair[['season', 'date']].assign(ll=al-bl, bs=ab-bb, N=pair.N_a).groupby(['season', 'date']).sum().reset_index()
    groups = [g[['ll', 'bs', 'N']].to_numpy() for _, g in blocks.groupby('season')]
    rng = np.random.default_rng(CONFIG['seed'])
    draws = []
    for _ in range(repetitions or CONFIG['bootstrap_replicates']):
        totals = sum((g[rng.integers(len(g), size=len(g))].sum(axis=0) for g in groups), start=np.zeros(3))
        draws.append(totals[:2]/totals[2])
    draws = np.asarray(draws)
    return dict(model=model, reference=reference, N=int(pair.N_a.sum()), innings=len(pair),
                date_blocks=len(blocks), log_loss_difference=(al-bl).sum()/pair.N_a.sum(),
                log_loss_low=np.quantile(draws[:, 0], .025), log_loss_high=np.quantile(draws[:, 0], .975),
                brier_difference=(ab-bb).sum()/pair.N_a.sum(),
                brier_low=np.quantile(draws[:, 1], .025), brier_high=np.quantile(draws[:, 1], .975))


def run_scenario(d, all_d, splits, name, diagnostics=False):
    h, all_h = halves(d, splits), halves(all_d, splits)
    predictions, priors, coverage = [], [], []
    for year, g in h.groupby('season'):
        early, late = g.loc[g.half == 'early'], g.loc[g.half == 'late']
        ae = all_h.loc[(all_h.season == year) & (all_h.half == 'early')]
        p, prior, original = fit_season(early, late.batter_id, ae if diagnostics else None, diagnostics)
        p = p.merge(late[['batter_id', 'N', 'B']], on='batter_id', validate='many_to_one')
        p['season'], p['scenario'] = year, name
        predictions.append(p)
        for population, pr, source in [('openers', prior, early), ('all_batters', original, ae)]:
            if pr is None:
                continue
            weights = source.N/(source.N+pr['strength'])
            priors.append(dict(season=year, scenario=name, population=population, **pr,
                               prior_sd=np.sqrt(pr['mu']*(1-pr['mu'])/(pr['strength']+1)),
                               early_N=int(source.N.sum()), early_B=int(source.B.sum()),
                               median_early_N=float(source.N.median()), median_data_weight=float(weights.median()),
                               min_data_weight=float(weights.min()), max_data_weight=float(weights.max())))
        unique = p.loc[p.model == 'population']
        coverage.append(dict(season=year, scenario=name, early_players=len(early),
                             late_players=len(late), early_N=int(early.N.sum()), late_N=int(late.N.sum()),
                             late_B=int(late.B.sum()), new_opening_players=int(unique.early_N.eq(0).sum()),
                             new_opening_N=int(unique.loc[unique.early_N.eq(0), 'N'].sum()),
                             new_openers_with_other_role_history=int(((unique.early_N==0)&(unique.all_role_early_N>0)).sum()) if diagnostics else np.nan,
                             min11_late_N=int(unique.loc[(unique.early_N>=11)&(unique.N>=11), 'N'].sum())))
    p = pd.concat(predictions, ignore_index=True)
    late_ids = set(splits.loc[splits.half == 'late', 'match_id'])
    innings = d.loc[d.match_id.isin(late_ids)].groupby(KEYS, as_index=False).agg(N=('is_boundary', 'size'), B=('is_boundary', 'sum'))
    # Separate cutoff forecasts from the late outcomes used solely for scoring.
    forecasts = p.drop(columns=['N', 'B'])
    ip = innings.merge(forecasts, on=['season', 'batter_id'], validate='many_to_many')
    return p, pd.DataFrame(priors), pd.DataFrame(coverage), ip


def evaluate(p, innings):
    summaries, per_season, intervals = [], [], []
    for scenario, sp in p.groupby('scenario'):
        si = innings.loc[innings.scenario == scenario]
        for label in ['all_late_players', 'min11_both_halves', 'early_history']:
            q = subset(sp, label)
            for period, years in [('development', range(2008, 2024)), ('validation', [2024]), ('test', [2025, 2026])]:
                m = metrics(q.loc[q.season.isin(years)])
                summaries.append(m.assign(scenario=scenario, subset=label, period=period))
            for year, g in q.groupby('season'):
                per_season.append(metrics(g).assign(scenario=scenario, subset=label, season=year))
            eligible_ids = q[['season', 'batter_id']].drop_duplicates()
            test = si.loc[si.season.isin([2025, 2026])].merge(eligible_ids, on=['season', 'batter_id'], validate='many_to_one')
            pairs = [('beta_binomial', 'population'), ('beta_binomial', 'raw')]
            if scenario == 'primary':
                pairs += [('beta_binomial', x) for x in ['jeffreys', 'original_beta', 'original_population', 'all_prior_opening_history', 'prior_mean']]
                pairs += [('leave_one_out', 'population'), ('selected_strength', 'beta_binomial'), ('selected_strength', 'population')]
            for a, b in pairs:
                intervals.append(dict(scenario=scenario, subset=label, period='test', **paired_interval(test, a, b)))
    return pd.concat(summaries), pd.concat(per_season), pd.DataFrame(intervals)


def protected_hashes():
    """Guard the original experiment outputs and implementation byte-for-byte."""
    paths = [*ROOT.glob('results/*.*'), *ROOT.glob('figures/*.*')]
    paths += [ROOT / p for p in ['models.py', 'research.py', 'pipeline.py', 'context_analysis.py', 'watchlists.py',
                                  'robustness.py', 'run_all.py', 'config/protocol.json', 'docs/PROTOCOL.md',
                                  'docs/RESEARCH_REPORT.md', 'docs/research.html', '02_research_results.ipynb']]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    raw, d, matches, splits = read('deliveries'), read('model_deliveries'), read('matches'), read('match_splits')
    opening, audit = opening_deliveries(raw, d)
    mc = match_count_splits(matches)
    save(mc, 'match_count_splits')
    league_ids = set(matches.loc[matches.event_stage.isna(), 'match_id'])
    scenarios = [
        ('primary', opening, d, splits),
        ('match_count_half', opening, d, mc),
        ('league_stage_only', opening.loc[opening.match_id.isin(league_ids)], d.loc[d.match_id.isin(league_ids)], splits),
        ('standard_conditions', opening.loc[opening.standard_conditions], d.loc[d.standard_conditions], splits),
        ('powerplay_only', opening.loc[opening.phase == 'powerplay'], d.loc[d.phase == 'powerplay'], splits)]
    predictions, priors, coverage, innings = [], [], [], []
    for name, sample, all_sample, cut in scenarios:
        print(f'Fit {name}: {len(sample):,} eligible opening deliveries', flush=True)
        p, pr, co, ip = run_scenario(sample, all_sample, cut, name, diagnostics=name == 'primary')
        predictions.append(p); priors.append(pr); coverage.append(co); innings.append(ip)
    p, pr, co, ip = map(lambda xs: pd.concat(xs, ignore_index=True), [predictions, priors, coverage, innings])
    grid = metrics(p.loc[(p.scenario == 'primary') & (p.season == 2024) & p.model.str.startswith('strength_x')])
    grid['multiplier'] = grid.model.str.replace('strength_x', '').astype(float)
    grid['tie_distance'] = np.abs(np.log(grid.multiplier))
    grid = grid.sort_values(['log_loss', 'tie_distance', 'multiplier']).drop(columns='tie_distance')
    winner = grid.iloc[0].model
    for frame, destination in [(p, predictions), (ip, innings)]:
        selected = frame.loc[(frame.scenario == 'primary') & (frame.model == winner)].copy()
        selected['model'] = 'selected_strength'
        destination.append(selected)
    p, ip = pd.concat(predictions, ignore_index=True), pd.concat(innings, ignore_index=True)
    print(f'2024-only strength choice: {winner}; scoring all fixed scenarios', flush=True)
    summary, annual, uncertainty = evaluate(p, ip)
    for frame, name in [(p, 'predictions'), (pr, 'priors'), (co, 'coverage'), (summary, 'metrics'),
                        (annual, 'by_season'), (uncertainty, 'uncertainty'), (grid, 'strength_validation_grid')]:
        save(frame, name)
    save(ip.loc[(ip.scenario=='primary') & ip.season.isin([2025, 2026])], 'evaluation_innings_predictions')
    calibration = []
    for model in CORE + ['original_beta', 'selected_strength']:
        q = p.loc[(p.scenario == 'primary') & p.season.isin([2025, 2026]) & (p.model == model)].copy()
        q['bin'] = pd.cut(q.p, bins=np.linspace(0, 1, 21), include_lowest=True).astype(str)
        for label, g in q.groupby('bin'):
            calibration.append(dict(model=model, bin=label, N=int(g.N.sum()), player_seasons=len(g),
                                    forecast=np.average(g.p, weights=g.N), observed=g.B.sum()/g.N.sum()))
    save(pd.DataFrame(calibration), 'calibration')
    phase = opening.groupby(['season', 'phase'], as_index=False).agg(N=('is_boundary', 'size'), B=('is_boundary', 'sum'))
    save(phase, 'phase_coverage')
    # Compare regenerated original forecasts to their committed reference before publishing.
    old = pd.read_csv(ROOT / 'results/brown_predictions.csv')
    check = p.loc[(p.scenario == 'primary') & (p.model == 'original_beta')].merge(
        old.loc[old.model == 'beta_binomial', ['season', 'batter_id', 'p']], on=['season', 'batter_id'],
        validate='one_to_one', suffixes=('_new', '_old'))
    difference = float(np.max(np.abs(check.p_new-check.p_old)))
    assert difference < 1e-10
    assert before == protected_hashes(), 'An original experiment artifact changed'
    manifest = dict(research_date=CONFIG['research_date'], study_status=CONFIG['status'],
                    seed=CONFIG['seed'], python=platform.python_version(),
                    source_sha256=hashlib.sha256((ROOT/'data/raw/ipl_json.zip').read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256((ROOT/'config/openers_protocol.json').read_bytes()).hexdigest(),
                    original_reference_commit='d9776f43dd9b1cb78a75b213a311d1e306841b72',
                    original_predictions_max_absolute_difference=difference,
                    selected_strength_model=winner, selected_multiplier=float(grid.iloc[0].multiplier),
                    original_artifacts_unchanged=True, protected_artifact_hashes=before, opening_audit=audit,
                    league_stage_rule='matches.event_stage is missing; matches with a named stage are excluded',
                    excluded_named_stages=sorted(matches.event_stage.dropna().unique().tolist()))
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary.loc[(summary.scenario=='primary') & (summary.period=='test') &
                      (summary.subset=='all_late_players'), ['model', 'N', 'log_loss', 'brier']].to_string(index=False))


if __name__ == '__main__':
    main()
