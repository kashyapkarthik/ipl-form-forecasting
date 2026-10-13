"""Frozen alternative-outcome extension. Never regenerates the earlier studies."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from models import fit_prior, posterior

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config/outcomes_protocol.json').read_text())
OUT = ROOT/'results/outcomes'
KEYS = ['season', 'date', 'match_id', 'innings', 'batter_id', 'position_group']
MODELS = ['population', 'beta', 'raw', 'jeffreys', 'prior_mean', 'current_population']


def save(df, name):
    df.to_csv(OUT/f'{name}.csv', index=False)


def protected_hashes():
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', CONFIG['baseline_commit']], cwd=ROOT, text=True).splitlines()
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths if p != 'README.md'}


def appearances(deliveries):
    """First seen striker/non-striker defines entry rank; totals use ALL events."""
    d = deliveries.loc[~deliveries.is_super_over].sort_values(['match_id', 'innings', 'delivery_sequence'])
    rows = []
    for (mid, inn), g in d.groupby(['match_id', 'innings'], sort=False):
        people = {}
        for ball in g.itertuples():
            for pid in (ball.batter_id, ball.non_striker_id):
                if pid not in people:
                    people[pid] = len(people)+1
        totals = g.groupby('batter_id').batter_runs.sum()
        legal = g.loc[g.primary_eligible].groupby('batter_id').size()
        for pid, pos in people.items():
            rows.append(dict(season=int(g.season.iloc[0]), date=g.date.iloc[0], match_id=mid,
                             innings=inn, batter_id=pid, position_group=pos,
                             total_runs=int(totals.get(pid, 0)), legal_balls=int(legal.get(pid, 0))))
    return pd.DataFrame(rows)


def samples(d, a, splits):
    legal = d.loc[d.primary_eligible & ~d.is_super_over & d.batting_entry_order.isin(CONFIG['positions'])].copy()
    legal['position_group'] = legal.batting_entry_order.astype(int)
    legal['six'] = (legal.batter_runs.eq(6) & ~legal.non_boundary).astype(int)
    legal['dot'] = legal.batter_runs.eq(0).astype(int)
    a = a.loc[a.position_group.isin(CONFIG['positions'])].copy()
    result = {}
    for outcome in CONFIG['outcomes']:
        if outcome.startswith('score'):
            x = a[KEYS].copy()
            x['N'], x['B'] = 1, a.total_runs.ge(int(outcome[5:])).astype(int)
        else:
            col = 'is_boundary' if outcome == 'boundary' else outcome
            x = legal.groupby(KEYS, as_index=False).agg(N=(col, 'size'), B=(col, 'sum'))
        x = x.merge(splits[['match_id', 'half', 'cutoff_date']], on='match_id', validate='many_to_one')
        result[outcome] = x
    return result


def history_sample(x, year, window):
    current = x.season.eq(year) & x.half.eq('early')
    previous = x.season.between(year-3, year-1) if window == 'previous3_plus_current' else False
    train = x.loc[current | previous]
    cutoff = x.loc[x.season.eq(year), 'cutoff_date'].iloc[0]
    # The original midpoint includes dates <= cutoff in early.
    assert train.date.max() <= cutoff
    assert set(train.match_id).isdisjoint(x.loc[x.season.eq(year)&x.half.eq('late'), 'match_id'])
    return train


def forecast(sample, outcome, window, combine_openers=False, years=None):
    x = sample.copy()
    if combine_openers:
        x.loc[x.position_group.le(2), 'position_group'] = 0
    records, priors = [], []
    years = years or [*CONFIG['development_seasons'], CONFIG['validation_season'], *CONFIG['evaluation_seasons']]
    for year in years:
        train = history_sample(x, year, window)
        early = train.groupby(['position_group', 'batter_id'], as_index=False)[['N', 'B']].sum()
        current = x.loc[x.season.eq(year)&x.half.eq('early')].groupby('position_group')[['N', 'B']].sum()
        late = x.loc[x.season.eq(year)&x.half.eq('late')].copy()
        for pos, target in late.groupby('position_group'):
            e = early.loc[early.position_group.eq(pos)]
            fallback = len(e) < 2
            pe = train.groupby('batter_id', as_index=False)[['N', 'B']].sum() if fallback else e
            prior = fit_prior(pe.B, pe.N)
            pop = pe.B.sum()/pe.N.sum()
            p_current = current.loc[pos, 'B']/current.loc[pos, 'N']
            out = target.merge(e[['batter_id', 'N', 'B']].rename(columns={'N':'early_N', 'B':'early_B'}),
                               on='batter_id', how='left', validate='many_to_one')
            out[['early_N', 'early_B']] = out[['early_N', 'early_B']].fillna(0).astype(int)
            n, b = out.early_N, out.early_B
            out['p_population'] = pop
            out['p_current_population'] = p_current
            out['p_beta'] = posterior(b, n, prior)
            out['p_raw'] = np.divide(b, n, out=np.full(len(n), pop), where=n.gt(0))
            out['p_jeffreys'] = np.where(n.gt(0), (b+.5)/(n+1), pop)
            out['p_prior_mean'] = prior['mu']
            out['data_weight'] = n/(n+prior['strength'])
            out['outcome'], out['window'] = outcome, window
            records.append(out)
            priors.append(dict(outcome=outcome, window=window, season=year, position_group=pos, **prior,
                               population=pop, fallback=fallback, train_min_date=train.date.min(), train_max_date=train.date.max(),
                               prior_sd=np.sqrt(prior['mu']*(1-prior['mu'])/(prior['strength']+1)),
                               median_training_N=float(e.N.median()), train_N=int(e.N.sum()), train_B=int(e.B.sum()),
                               target_N=int(out.N.sum()), cold_N=int(out.loc[n.eq(0), 'N'].sum()),
                               target_weight=float(np.average(out.data_weight, weights=out.N))))
    return pd.concat(records, ignore_index=True), pd.DataFrame(priors)


def losses(B, N, p):
    B, N, p = np.asarray(B), np.asarray(N), np.asarray(p)
    pc = np.clip(p, CONFIG['logloss_clip'], 1-CONFIG['logloss_clip'])
    return -B*np.log(pc)-(N-B)*np.log1p(-pc), B*(1-p)**2+(N-B)*p**2


def summary(wide):
    rows = []
    for (outcome, window), g in wide.groupby(['outcome', 'window']):
        for model in MODELS:
            ll, bs = losses(g.B, g.N, g[f'p_{model}'])
            players = pd.DataFrame(dict(player=g.batter_id, N=g.N, ll=ll, bs=bs)).groupby('player').sum()
            rows.append(dict(outcome=outcome, window=window, model=model, N=int(g.N.sum()), B=int(g.B.sum()),
                             players=g.batter_id.nunique(), appearances=len(g), log_loss=ll.sum()/g.N.sum(),
                             brier=bs.sum()/g.N.sum(), player_macro_log_loss=(players.ll/players.N).mean(),
                             player_macro_brier=(players.bs/players.N).mean()))
    result = pd.DataFrame(rows)
    refs = result.loc[result.model.eq('population'), ['outcome', 'window', 'log_loss', 'brier']].rename(
        columns={'log_loss':'population_ll', 'brier':'population_bs'})
    result = result.merge(refs, on=['outcome', 'window'], validate='many_to_one')
    result['log_skill_pct'] = 100*(1-result.log_loss/result.population_ll)
    result['brier_skill_pct'] = 100*(1-result.brier/result.population_bs)
    return result


def selection(pre_evaluation):
    development = summary(pre_evaluation.loc[pre_evaluation.season.isin(CONFIG['development_seasons'])])
    validation = summary(pre_evaluation.loc[pre_evaluation.season.eq(CONFIG['validation_season'])])
    save(development, 'development_metrics')
    save(validation, 'validation_metrics')
    decisions = []
    for window, g in validation.groupby('window'):
        beta = g.loc[g.model.eq('beta')].copy()
        jeff = g.loc[g.model.eq('jeffreys')].set_index('outcome').log_loss
        dev = development.loc[development.model.eq('beta')&development.window.eq(window)].set_index('outcome').log_skill_pct
        beta['eligible'] = (beta.log_loss < beta.outcome.map(jeff)) & beta.outcome.map(dev).gt(0)
        beta = beta.sort_values(['log_skill_pct', 'outcome'], ascending=[False, True])
        chosen = beta.loc[beta.eligible].iloc[0] if beta.eligible.any() else beta.iloc[0]
        decisions.append(dict(window=window, outcome=chosen.outcome, eligible=bool(chosen.eligible),
                              validation_skill_pct=float(chosen.log_skill_pct),
                              development_skill_pct=float(dev[chosen.outcome])))
    decision = dict(rule=CONFIG['selection'], selected=decisions,
                    stage='Written using development and validation only, before evaluation scoring')
    (OUT/'selection.json').write_text(json.dumps(decision, indent=2)+'\n')
    return decisions


def uncertainty(wide, cluster='date', family=True):
    """Paired resampling, common draws across candidates; fixed fitted forecasts."""
    rng = np.random.default_rng(CONFIG['seed'])
    reps = CONFIG['bootstrap_replicates']
    if cluster == 'date':
        units = {int(y): sorted(g.date.unique()) for y, g in wide.groupby('season')}
        draws = {y: rng.integers(len(ds), size=(reps, len(ds))) for y, ds in units.items()}
    else:
        units = sorted(wide.batter_id.unique())
        draws = rng.integers(len(units), size=(reps, len(units)))
    rows, family_draws, family_points = [], [], []
    for (outcome, window), g in wide.groupby(['outcome', 'window']):
        ll, bs = losses(g.B, g.N, g.p_beta)
        for ref in ['population', 'raw', 'jeffreys', 'prior_mean', 'current_population']:
            lr, br = losses(g.B, g.N, g[f'p_{ref}'])
            frame = g[['season', 'date', 'batter_id', 'N']].assign(ll=ll, bs=bs, lr=lr, br=br)
            cols = ['ll', 'bs', 'lr', 'br', 'N']
            if cluster == 'date':
                totals = np.zeros((reps, 5))
                for y, ds in units.items():
                    block = frame.loc[frame.season.eq(y)].groupby('date')[cols].sum().reindex(ds, fill_value=0).to_numpy()
                    totals += block[draws[y]].sum(axis=1)
            else:
                block = frame.groupby('batter_id')[cols].sum().reindex(units, fill_value=0).to_numpy()
                totals = block[draws].sum(axis=1)
            delta = (totals[:, 0]-totals[:, 2])/totals[:, 4]
            bdelta = (totals[:, 1]-totals[:, 3])/totals[:, 4]
            skill = 100*(1-totals[:, 0]/totals[:, 2])
            point = 100*(1-ll.sum()/lr.sum())
            row = dict(outcome=outcome, window=window, reference=ref, cluster=cluster, N=int(g.N.sum()),
                       log_loss_difference=(ll-lr).sum()/g.N.sum(), log_loss_low=np.quantile(delta, .025),
                       log_loss_high=np.quantile(delta, .975), brier_difference=(bs-br).sum()/g.N.sum(),
                       brier_low=np.quantile(bdelta, .025), brier_high=np.quantile(bdelta, .975),
                       log_skill_pct=point, skill_low=np.quantile(skill, .025), skill_high=np.quantile(skill, .975))
            rows.append(row)
            if ref == 'population':
                family_draws.append(skill)
                family_points.append(point)
    if family:
        deviation = np.abs(np.array(family_draws).T-np.array(family_points))
        width = np.quantile(deviation.max(axis=1), .95)
        for row in rows:
            if row['reference'] == 'population':
                row['family_skill_low'] = row['log_skill_pct']-width
                row['family_skill_high'] = row['log_skill_pct']+width
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = protected_hashes()
    d = pd.read_parquet(ROOT/'data/processed/deliveries.parquet')
    splits = pd.read_parquet(ROOT/'data/processed/match_splits.parquet')
    a = appearances(d)
    save(a, 'batting_appearances')
    w = pd.read_parquet(ROOT/'data/processed/wickets.parquet')
    ordinary_ids = d.loc[~d.is_super_over, 'delivery_id']
    w = w.loc[w.delivery_id.isin(ordinary_ids)]
    absent = w.merge(a[['match_id', 'innings', 'batter_id']], left_on=['match_id', 'innings', 'player_out_id'],
                     right_on=['match_id', 'innings', 'batter_id'], how='left', indicator=True).query("_merge=='left_only'")
    save(absent, 'wicket_only_appearance_audit')
    assert absent.empty, 'Unobserved wicket-only appearance needs a protocol amendment before fitting'
    assert a.total_runs.sum() == d.loc[~d.is_super_over, 'batter_runs'].sum()
    sample = samples(d, a, splits)
    forecasts, priors = [], []
    for outcome in CONFIG['outcomes']:
        for window in CONFIG['history_windows']:
            f, p = forecast(sample[outcome], outcome, window)
            forecasts.append(f); priors.append(p)
            print(f'Fitted {outcome}, {window}', flush=True)
    wide, prior = pd.concat(forecasts, ignore_index=True), pd.concat(priors, ignore_index=True)
    selected = selection(wide.loc[wide.season.le(CONFIG['validation_season'])])
    selection_sha = hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    save(prior, 'priors')
    evaluation = wide.loc[wide.season.isin(CONFIG['evaluation_seasons'])].copy()
    save(evaluation, 'evaluation_predictions')
    save(summary(evaluation), 'evaluation_metrics')
    save(pd.concat([summary(g).assign(season=y) for y, g in wide.groupby('season')]), 'annual_metrics')
    save(uncertainty(evaluation), 'uncertainty')
    sensitivities, sensitivity_intervals = [], []
    for pick in selected:
        outcome, window = pick['outcome'], pick['window']
        g = evaluation.loc[evaluation.outcome.eq(outcome)&evaluation.window.eq(window)]
        sensitivity_intervals.append(uncertainty(g, cluster='player', family=False).assign(sensitivity='player_cluster'))
        f, _ = forecast(sample[outcome], outcome, window, combine_openers=True, years=CONFIG['evaluation_seasons'])
        sensitivities.append(summary(f).assign(sensitivity='combined_openers'))
        sensitivity_intervals.append(uncertainty(f, family=False).assign(sensitivity='combined_openers'))
        sensitivities.append(summary(g.loc[g.early_N.gt(0)]).assign(sensitivity='has_history'))
    save(pd.concat(sensitivities), 'sensitivity_metrics')
    save(pd.concat(sensitivity_intervals), 'sensitivity_uncertainty')
    # Added after first inspecting evaluation results, explicitly NOT a new selection.
    # Six rate was the largest realised improvement, so audit that recommendation too.
    six = evaluation.loc[evaluation.outcome.eq('six')&evaluation.window.eq('previous3_plus_current')]
    posthoc = [uncertainty(six, cluster='player', family=False).assign(sensitivity='posthoc_six_player_cluster')]
    combined, _ = forecast(sample['six'], 'six', 'previous3_plus_current', combine_openers=True,
                            years=CONFIG['evaluation_seasons'])
    posthoc.append(uncertainty(combined, family=False).assign(sensitivity='posthoc_six_combined_openers'))
    save(pd.concat(posthoc), 'posthoc_six_uncertainty')
    save(summary(combined), 'posthoc_six_combined_metrics')
    near_pool = evaluation.merge(prior[['outcome', 'window', 'season', 'position_group', 'strength']],
                                 on=['outcome', 'window', 'season', 'position_group'], validate='many_to_one')
    near_pool.loc[near_pool.strength.ge(10000), 'p_beta'] = near_pool.loc[near_pool.strength.ge(10000), 'p_population']
    save(summary(near_pool), 'near_pooling_metrics')
    coverage = a.loc[a.position_group.le(7)].merge(splits[['match_id', 'half']], on='match_id', validate='many_to_one')
    coverage['score30'], coverage['score50'] = coverage.total_runs.ge(30), coverage.total_runs.ge(50)
    save(coverage.groupby(['season', 'half'], as_index=False).agg(appearances=('batter_id','size'),
         players=('batter_id','nunique'), score30=('score30','sum'), score50=('score50','sum'),
         zero_legal_balls=('legal_balls',lambda s: int(s.eq(0).sum()))), 'coverage')
    assert protected == protected_hashes(), 'An earlier study changed'
    manifest = dict(baseline_commit=CONFIG['baseline_commit'], selection_sha256=selection_sha,
                    raw_sha256=hashlib.sha256((ROOT/'data/raw/ipl_json.zip').read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256((ROOT/'config/outcomes_protocol.json').read_bytes()).hexdigest(),
                    protected_files=protected, ordinary_appearances=len(a), wicket_only_appearances=len(absent),
                    results_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*')) if p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == '__main__':
    run()
