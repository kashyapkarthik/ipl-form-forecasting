"""Position-specific experiment 1 only. Existing studies are never regenerated."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from models import fit_prior, posterior
from openers_experiment import match_count_splits

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config/positions_protocol.json').read_text())
OUT = ROOT/'results/positions'
KEYS = ['season', 'date', 'match_id', 'innings', 'batter_id', 'position_group']
PRIMARY = 'separate_positions'


def read(name):
    return pd.read_parquet(ROOT/f'data/processed/{name}.parquet')


def save(df, name):
    df.to_csv(OUT/f'{name}.csv', index=False)


def protected_hashes():
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', CONFIG['baseline_commit']], cwd=ROOT, text=True).splitlines()
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths if p != 'README.md'}


def position_sample(d, combine_openers=False):
    result = d.loc[d.batting_entry_order.isin(CONFIG['positions'])].copy()
    result['position_group'] = result.batting_entry_order.astype(int)
    if combine_openers:
        result.loc[result.position_group.le(2), 'position_group'] = 0
    assert result.primary_eligible.all() and not result.delivery_id.duplicated().any()
    return result


def counts(d, splits, by_position=True):
    x = d.merge(splits[['match_id', 'half']], on='match_id', validate='many_to_one')
    group = ['season', 'half'] + (['position_group'] if by_position else []) + ['batter_id']
    return x.groupby(group, as_index=False).agg(N=('is_boundary', 'size'), B=('is_boundary', 'sum'))


def position_forecasts(early, late_ids, all_early, all_prior, leave_one_out=True, top7_early=None, top7_prior=None):
    """Late outcomes are intentionally absent from this function's interface."""
    ep = early.set_index('batter_id')
    ap = all_early.set_index('batter_id')
    fallback = len(ep) < 2
    prior = all_prior.copy() if fallback else fit_prior(ep.B, ep.N)
    rows = []
    for pid in sorted(set(late_ids)):
        b, n = (int(ep.loc[pid, 'B']), int(ep.loc[pid, 'N'])) if pid in ep.index else (0, 0)
        ab, an = (int(ap.loc[pid, 'B']), int(ap.loc[pid, 'N'])) if pid in ap.index else (0, 0)
        mu, k = prior['mu'], prior['strength']
        am, ak = all_prior['mu'], all_prior['strength']
        row = dict(batter_id=pid, early_B=b, early_N=n, all_early_B=ab, all_early_N=an,
                   position_data_weight=n/(n+k),
                   p_population=prior['pooled'], p_raw=b/n if n else prior['pooled'],
                   p_beta_binomial=float(posterior(b, n, prior)),
                   p_jeffreys=(b+.5)/(n+1) if n else prior['pooled'],
                   p_prior_mean=mu, p_original_population=all_prior['pooled'],
                   p_original_beta=float(posterior(ab, an, all_prior)),
                   p_all_prior_position_history=float(posterior(b, n, all_prior)),
                   p_position_mean_all_strength=(b+mu*ak)/(n+ak),
                   p_all_mean_position_strength=(b+am*k)/(n+k))
        if top7_prior is not None:
            tp = top7_early.set_index('batter_id')
            tb, tn = (int(tp.loc[pid, 'B']), int(tp.loc[pid, 'N'])) if pid in tp.index else (0, 0)
            row.update(top7_early_B=tb, top7_early_N=tn, p_top7_population=top7_prior['pooled'],
                       p_top7_beta=float(posterior(tb, tn, top7_prior)))
        if leave_one_out:
            other = ep.drop(index=pid, errors='ignore')
            loo_fallback = len(other) < 2
            loo = (all_prior if loo_fallback else fit_prior(other.B, other.N)) if pid in ep.index else prior
            row.update(p_leave_one_out=float(posterior(b, n, loo)),
                       loo_strength=loo['strength'], loo_at_bound=loo['at_strength_bound'],
                       loo_fallback=bool(loo_fallback if pid in ep.index else fallback))
        rows.append(row)
    return pd.DataFrame(rows), dict(**prior, position_records=len(ep), fallback_to_all=fallback)


def run_scenario(d, all_d, splits, name):
    h, ah = counts(d, splits), counts(all_d, splits, by_position=False)
    rows, prior_rows, coverage = [], [], []
    for year, year_group in h.groupby('season'):
        ae = ah.loc[(ah.season==year)&(ah.half=='early')]
        all_prior = fit_prior(ae.B, ae.N)
        te = year_group.loc[year_group.half=='early'].groupby('batter_id', as_index=False)[['N', 'B']].sum()
        top7_prior = fit_prior(te.B, te.N)
        prior_rows.append(dict(scenario=name, season=year, position_group=-2, **top7_prior,
                               prior_sd=np.sqrt(top7_prior['mu']*(1-top7_prior['mu'])/(top7_prior['strength']+1)),
                               early_N=int(te.N.sum()), early_B=int(te.B.sum()), position_records=len(te), fallback_to_all=False))
        prior_rows.append(dict(scenario=name, season=year, position_group=-1, **all_prior,
                               prior_sd=np.sqrt(all_prior['mu']*(1-all_prior['mu'])/(all_prior['strength']+1)),
                               early_N=int(ae.N.sum()), early_B=int(ae.B.sum()), position_records=len(ae), fallback_to_all=False))
        for pos, group in year_group.groupby('position_group'):
            early = group.loc[group.half=='early']
            late = group.loc[group.half=='late']
            f, prior = position_forecasts(early, late.batter_id, ae, all_prior, leave_one_out=name==PRIMARY,
                                         top7_early=te, top7_prior=top7_prior)
            f = f.merge(late[['batter_id', 'N', 'B']], on='batter_id', validate='one_to_one')
            f['scenario'], f['season'], f['position_group'] = name, year, pos
            rows.append(f)
            prior_rows.append(dict(scenario=name, season=year, position_group=pos, **prior,
                                   prior_sd=np.sqrt(prior['mu']*(1-prior['mu'])/(prior['strength']+1)),
                                   early_N=int(early.N.sum()), early_B=int(early.B.sum()),
                                   median_early_N=float(early.N.median()),
                                   median_data_weight=float((early.N/(early.N+prior['strength'])).median())))
            coverage.append(dict(scenario=name, season=year, position_group=pos, early_players=len(early),
                                 late_players=len(late), early_N=int(early.N.sum()), late_N=int(late.N.sum()),
                                 late_B=int(late.B.sum()), no_position_history_players=int(f.early_N.eq(0).sum()),
                                 no_position_history_N=int(f.loc[f.early_N.eq(0), 'N'].sum()),
                                 other_position_history_players=int((f.early_N.eq(0)&f.all_early_N.gt(0)).sum()),
                                 other_position_history_N=int(f.loc[f.early_N.eq(0)&f.all_early_N.gt(0), 'N'].sum()),
                                 min11_N=int(f.loc[f.early_N.ge(11)&f.N.ge(11), 'N'].sum())))
    forecasts = pd.concat(rows, ignore_index=True)
    late = d.merge(splits[['match_id', 'half']], on='match_id', validate='many_to_one').query("half=='late'")
    innings = late.groupby(KEYS, as_index=False).agg(N=('is_boundary', 'size'), B=('is_boundary', 'sum'))
    # Keep the half-level eligibility flag when scoring individual innings.
    payload = forecasts.rename(columns={'N': 'late_N', 'B': 'late_B'})
    ip = innings.merge(payload, on=['season', 'position_group', 'batter_id'], validate='many_to_one')
    return forecasts, pd.DataFrame(prior_rows), pd.DataFrame(coverage), ip


def long_format(wide):
    pcols = [c for c in wide if c.startswith('p_')]
    ids = [c for c in wide if c not in pcols]
    result = wide.melt(id_vars=ids, value_vars=pcols, var_name='model', value_name='p').dropna(subset=['p'])
    result['model'] = result.model.str.removeprefix('p_')
    return result


def losses(B, N, p):
    B, N, p = np.asarray(B), np.asarray(N), np.asarray(p)
    pc = np.clip(p, CONFIG['raw_logloss_clip'], 1-CONFIG['raw_logloss_clip'])
    return -B*np.log(pc)-(N-B)*np.log1p(-pc), B*(1-p)**2+(N-B)*p**2


def metrics(predictions, restricted=False):
    rows = []
    for model, g in predictions.groupby('model'):
        ll, bs = losses(g.B, g.N, g.p)
        player = pd.DataFrame(dict(batter_id=g.batter_id, N=g.N, ll=ll, bs=bs)).groupby('batter_id').sum()
        trans = np.arcsin(np.sqrt((g.B+.25)/(g.N+.5)))
        rows.append(dict(model=model, N=int(g.N.sum()), B=int(g.B.sum()), players=g.batter_id.nunique(),
                         player_season_positions=len(g), log_loss=ll.sum()/g.N.sum(), brier=bs.sum()/g.N.sum(),
                         player_macro_log_loss=(player.ll/player.N).mean(), player_macro_brier=(player.bs/player.N).mean(),
                         player_position_rate_mse=np.mean((g.p-g.B/g.N)**2),
                         noise_adjusted_arcsine_mse=np.mean((np.arcsin(np.sqrt(g.p))-trans)**2-1/(4*g.N)) if restricted else np.nan))
    return pd.DataFrame(rows)


def filter_subset(data, label, innings=False):
    if label=='min11_both_halves':
        return data.loc[data.early_N.ge(11) & data['late_N' if innings else 'N'].ge(11)]
    if label=='early_history':
        return data.loc[data.early_N.gt(0)]
    return data


def block_comparisons(wide, comparisons):
    """One common date draw per season across all positions, including zero-exposure cells."""
    dates = wide[['season', 'date']].drop_duplicates().sort_values(['season', 'date'])
    positions = sorted(wide.position_group.unique())
    rng = np.random.default_rng(CONFIG['seed'])
    index_draws = {int(year): rng.integers(len(g), size=(CONFIG['bootstrap_replicates'], len(g)))
                   for year, g in dates.groupby('season')}
    rows = []
    for model, reference in comparisons:
        if f'p_{model}' not in wide or wide[f'p_{model}'].isna().any():
            continue
        ll_a, bs_a = losses(wide.B, wide.N, wide[f'p_{model}'])
        ll_b, bs_b = losses(wide.B, wide.N, wide[f'p_{reference}'])
        frame = wide[KEYS+['N']].assign(ll=ll_a-ll_b, bs=bs_a-bs_b)
        samples = []
        point = []
        for pos in positions:
            group = frame.loc[frame.position_group==pos]
            point.append(group[['ll', 'bs', 'N']].sum().to_numpy())
            totals = np.zeros((CONFIG['bootstrap_replicates'], 3))
            for year, year_dates in dates.groupby('season'):
                block = group.loc[group.season==year].groupby('date')[['ll', 'bs', 'N']].sum()
                block = block.reindex(year_dates.date, fill_value=0).to_numpy()
                totals += block[index_draws[int(year)]].sum(axis=1)
            samples.append(totals)
        samples, point = np.stack(samples, axis=1), np.stack(point)
        diffs = np.divide(samples[:, :, :2], samples[:, :, 2:3], out=np.full_like(samples[:, :, :2], np.nan), where=samples[:, :, 2:3]>0)
        observed = point[:, :2]/point[:, 2:3]
        # Conservative unstudentized common-width band for the seven position effects.
        max_deviation = np.nanmax(np.abs(diffs[:, :, 0]-observed[None, :, 0]), axis=1)
        halfwidth = np.quantile(max_deviation, .95)
        total_sample = samples.sum(axis=1)
        total_point = point.sum(axis=0)
        for j, label in enumerate([*map(str, positions), 'all']):
            if label=='all':
                distribution = total_sample[:, :2]/total_sample[:, 2:3]
                estimate, n = total_point[:2]/total_point[2], total_point[2]
            else:
                distribution, estimate, n = diffs[:, j], observed[j], point[j, 2]
            rows.append(dict(position_group=label, model=model, reference=reference, N=int(n),
                             log_loss_difference=estimate[0], log_loss_low=np.nanquantile(distribution[:, 0], .025),
                             log_loss_high=np.nanquantile(distribution[:, 0], .975),
                             brier_difference=estimate[1], brier_low=np.nanquantile(distribution[:, 1], .025),
                             brier_high=np.nanquantile(distribution[:, 1], .975),
                             simultaneous_log_loss_low=estimate[0]-halfwidth if label!='all' else np.nan,
                             simultaneous_log_loss_high=estimate[0]+halfwidth if label!='all' else np.nan,
                             common_date_blocks=len(dates),
                             observed_date_blocks=frame.loc[frame.position_group==int(label), ['season', 'date']].drop_duplicates().shape[0] if label!='all' else len(dates)))
    return pd.DataFrame(rows)


def evaluate(wide, innings):
    summary, annual, intervals = [], [], []
    for scenario, g in wide.groupby('scenario'):
        ip = innings.loc[innings.scenario==scenario]
        for subset in ['all_late', 'min11_both_halves', 'early_history']:
            sub = filter_subset(g, subset)
            for position in ['all', *map(str, sorted(g.position_group.unique()))]:
                target = sub if position=='all' else sub.loc[sub.position_group==int(position)]
                target = long_format(target)
                for period, years in [('development', range(2008, 2024)), ('validation', [2024]), ('evaluation', [2025, 2026])]:
                    s = metrics(target.loc[target.season.isin(years)], restricted=subset=='min11_both_halves')
                    summary.append(s.assign(scenario=scenario, position_group=position, subset=subset, period=period))
                for year, ys in target.groupby('season'):
                    annual.append(metrics(ys, restricted=subset=='min11_both_halves').assign(
                        scenario=scenario, position_group=position, subset=subset, season=year))
            if subset=='early_history' or (scenario!=PRIMARY and subset!='all_late'):
                continue
            eval_innings = filter_subset(ip.loc[ip.season.isin([2025, 2026])], subset, innings=True)
            pairs = [('beta_binomial', 'population'), ('beta_binomial', 'raw'), ('beta_binomial', 'original_beta'),
                     ('population', 'original_population'), ('beta_binomial', 'jeffreys'), ('beta_binomial', 'top7_beta')]
            if scenario==PRIMARY:
                pairs += [('leave_one_out', 'population'), ('beta_binomial', 'all_prior_position_history'),
                          ('beta_binomial', 'prior_mean')]
            intervals.append(block_comparisons(eval_innings, pairs).assign(scenario=scenario, subset=subset))
    return pd.concat(summary), pd.concat(annual), pd.concat(intervals)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    d, splits, matches = read('model_deliveries'), read('match_splits'), read('matches')
    split_mc = match_count_splits(matches)
    all_predictions, all_priors, all_coverage, all_innings = [], [], [], []
    for scenario in CONFIG['scenarios']:
        sample = position_sample(d, combine_openers=scenario=='combined_openers')
        cut = split_mc if scenario=='match_count_half' else splits
        print(f'Fitting {scenario}: {len(sample):,} eligible deliveries', flush=True)
        p, priors, coverage, innings = run_scenario(sample, d, cut, scenario)
        all_predictions.append(p); all_priors.append(priors); all_coverage.append(coverage); all_innings.append(innings)
    p, priors, coverage, innings = [pd.concat(xs, ignore_index=True) for xs in [all_predictions, all_priors, all_coverage, all_innings]]
    print('Scoring fixed forecasts, by position and jointly; paired date uncertainty', flush=True)
    summary, annual, intervals = evaluate(p, innings)
    for frame, name in [(p, 'predictions'), (priors, 'priors'), (coverage, 'coverage'), (summary, 'metrics'),
                        (annual, 'by_season'), (intervals, 'uncertainty'), (split_mc, 'match_count_splits')]:
        save(frame, name)
    save(innings.loc[innings.season.isin([2025, 2026])], 'evaluation_innings_predictions')
    sample = position_sample(d)
    totals = d.groupby('season', as_index=False).agg(all_N=('is_boundary', 'size'), all_B=('is_boundary', 'sum'))
    selected = sample.groupby('season', as_index=False).agg(position_1_to_7_N=('is_boundary', 'size'), position_1_to_7_B=('is_boundary', 'sum'))
    save(totals.merge(selected).assign(excluded_N=lambda x: x.all_N-x.position_1_to_7_N), 'target_coverage')
    cal = []
    for pos in ['all', *map(str, CONFIG['positions'])]:
        target = p.loc[(p.scenario==PRIMARY)&p.season.isin([2025, 2026])]
        if pos!='all': target = target.loc[target.position_group==int(pos)]
        for model in ['population', 'raw', 'beta_binomial', 'original_beta']:
            groups = pd.cut(target[f'p_{model}'], np.linspace(0, 1, 21), include_lowest=True)
            for label, group in target.groupby(groups, observed=True):
                cal.append(dict(position_group=pos, model=model, bin=str(label), N=int(group.N.sum()),
                                forecast=np.average(group[f'p_{model}'], weights=group.N), observed=group.B.sum()/group.N.sum()))
    save(pd.DataFrame(cal), 'calibration')
    assert before==protected_hashes(), 'Existing research changed'
    manifest = dict(**CONFIG, protocol_sha256=hashlib.sha256((ROOT/'config/positions_protocol.json').read_bytes()).hexdigest(),
                    protocol_text_sha256=hashlib.sha256((ROOT/'docs/POSITIONS_PROTOCOL.md').read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256((ROOT/'data/raw/ipl_json.zip').read_bytes()).hexdigest(),
                    protected_artifact_hashes=before, existing_research_unchanged=True,
                    main_prior_bound_fits=priors.loc[(priors.scenario==PRIMARY)&priors.at_strength_bound, ['season', 'position_group']].to_dict('records'),
                    fallback_fits=int(priors.fallback_to_all.sum()))
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary.loc[(summary.scenario==PRIMARY)&(summary.position_group=='all')&
                      (summary.period=='evaluation')&(summary.subset=='all_late'), ['model', 'N', 'log_loss', 'brier']].to_string(index=False))


if __name__=='__main__':
    main()
