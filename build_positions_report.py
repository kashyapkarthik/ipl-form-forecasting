"""Publish the isolated position experiment's report and static scientific figures."""
from pathlib import Path
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from positions_experiment import ROOT, OUT, PRIMARY, block_comparisons, long_format, metrics
from build_openers_report import table

FIG = ROOT/'figures/positions'
LABELS = {'beta_binomial': 'Position-specific shrinkage', 'population': 'Position population',
          'raw': 'Raw player-position rate', 'jeffreys': 'Smoothed player-position rate',
          'original_beta': 'Original all-batter shrinkage', 'original_population': 'Original all-batter population',
          'top7_beta': 'Shared positions 1–7 shrinkage', 'top7_population': 'Shared positions 1–7 population',
          'leave_one_out': 'Leave-one-player-out shrinkage', 'prior_mean': 'Position prior mean, no update',
          'all_prior_position_history': 'All-batter prior + position history',
          'position_mean_all_strength': 'Position mean + all-batter strength',
          'all_mean_position_strength': 'All-batter mean + position strength'}


def main():
    FIG.mkdir(exist_ok=True)
    m = pd.read_csv(OUT/'metrics.csv', dtype={'position_group': str})
    a = pd.read_csv(OUT/'by_season.csv', dtype={'position_group': str})
    u = pd.read_csv(OUT/'uncertainty.csv', dtype={'position_group': str})
    priors = pd.read_csv(OUT/'priors.csv')
    p = pd.read_csv(OUT/'predictions.csv')
    ip = pd.read_csv(OUT/'evaluation_innings_predictions.csv')
    c = pd.read_csv(OUT/'coverage.csv')
    eval = m.loc[m.scenario.eq(PRIMARY)&m.subset.eq('all_late')&m.period.eq('evaluation')]
    total = eval.loc[eval.position_group.eq('all')].set_index('model')
    position = eval.loc[eval.position_group.ne('all')].pivot(index='position_group', columns='model', values='log_loss')
    position['N'] = eval.loc[eval.model.eq('beta_binomial')].set_index('position_group').N
    position = position.reset_index()
    total_unc = u.loc[u.scenario.eq(PRIMARY)&u.subset.eq('all_late')&u.position_group.eq('all')]
    main_priors = priors.loc[priors.scenario.eq(PRIMARY)&priors.position_group.gt(0)&priors.season.isin([2025, 2026])]
    coverage = c.loc[c.scenario.eq(PRIMARY)&c.season.isin([2025, 2026])]
    history = coverage.groupby('position_group', as_index=False)[['late_N', 'no_position_history_N', 'other_position_history_N', 'no_position_history_players']].sum()
    history['no_position_history_fraction'] = history.no_position_history_N/history.late_N
    relative = 100*(total.loc['original_beta', 'log_loss']-total.loc['beta_binomial', 'log_loss'])/total.loc['original_beta', 'log_loss']
    pop_relative = 100*(total.loc['population', 'log_loss']-total.loc['beta_binomial', 'log_loss'])/total.loc['population', 'log_loss']
    # Paired comparison of the two pre-specified opener conventions on identical innings.
    keys = ['season', 'date', 'match_id', 'innings', 'batter_id']
    left = ip.loc[ip.scenario.eq(PRIMARY)].copy()
    right = ip.loc[ip.scenario.eq('combined_openers'), keys+['B', 'N', 'p_beta_binomial']]
    compare = left.merge(right, on=keys, how='outer', validate='one_to_one', indicator=True, suffixes=('', '_combined'))
    assert compare._merge.eq('both').all()
    assert compare.N.eq(compare.N_combined).all() and compare.B.eq(compare.B_combined).all()
    compare['p_combined'] = compare.p_beta_binomial_combined
    pair = block_comparisons(compare, [('combined', 'beta_binomial')])
    pair.to_csv(OUT/'opener_convention_comparison.csv', index=False)
    convention = pair.loc[pair.position_group.eq('all')].iloc[0]
    # Numerical diagnostic, not a selected model: replace near-pooled fits by their limit.
    near = p.loc[p.scenario.eq(PRIMARY)&p.season.isin([2025, 2026])].merge(
        main_priors[['season', 'position_group', 'strength']], on=['season', 'position_group'], validate='many_to_one')
    near['p_pooling_limit_check'] = np.where(near.strength.ge(10000), near.p_population, near.p_beta_binomial)
    num = metrics(long_format(near).query("model in ['beta_binomial','pooling_limit_check']"))
    num.to_csv(OUT/'numerical_pooling_check.csv', index=False)
    limit_loss = num.loc[num.model.eq('pooling_limit_check'), 'log_loss'].item()
    names = pd.read_parquet(ROOT/'data/processed/model_deliveries.parquet')[['batter_id', 'batter_name']].drop_duplicates('batter_id', keep='last')
    cases = p.loc[p.scenario.eq(PRIMARY)&p.season.eq(2026)].merge(names, on='batter_id', validate='many_to_one')
    cases = cases.loc[cases.batter_name.isin(['HH Pandya', 'KL Rahul', 'SV Samson'])].copy().sort_values(['batter_name', 'position_group'])
    cases['raw_early_rate'] = np.where(cases.early_N.gt(0), cases.early_B/cases.early_N, np.nan)
    cases.to_csv(OUT/'player_examples.csv', index=False)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'positions-20261013'})
    teal, blue, orange = '#137d75', '#35618a', '#b86d31'
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6), layout='constrained')
    for ax, ref, title in zip(axs, ['original_beta', 'population'], ['Versus original all-batter shrinkage', 'Versus each position’s population']):
        data = u.loc[u.scenario.eq(PRIMARY)&u.subset.eq('all_late')&u.model.eq('beta_binomial')&u.reference.eq(ref)].copy()
        order = ['1', '2', '3', '4', '5', '6', '7', 'all']
        data = data.set_index('position_group').loc[order]
        colors = [teal if x < 0 else orange for x in data.log_loss_difference]
        for j, (_, row) in enumerate(data.iterrows()):
            ax.errorbar(row.log_loss_difference*1000, j,
                        xerr=np.array([[row.log_loss_difference-row.log_loss_low], [row.log_loss_high-row.log_loss_difference]])*1000,
                        color=colors[j], fmt='o', capsize=4)
        ax.axvline(0, ls='--', color='#778a92')
        ax.set(yticks=range(8), yticklabels=['1*', '2*', '3', '4', '5', '6', '7', 'All 1–7'],
               xlabel='Position shrinkage − reference log loss × 1,000', title=title)
        ax.invert_yaxis(); ax.xaxis.set_major_locator(MaxNLocator(5))
    axs[0].set_ylabel('Observed batting position')
    fig.suptitle('Position matching helps overall, but the gain is not universal', fontsize=15, fontweight='bold')
    fig.supxlabel('Negative favours position shrinkage. Bars: individual 95% date-bootstrap intervals.\n*1/2 are initial striker/non-striker, not independently verified scorecard slots.', fontsize=9)
    for ext in ['png', 'svg']:
        fig.savefig(FIG/f'comparison.{ext}', dpi=170, metadata={'Date': None} if ext=='svg' else {})
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    for year, color in [(2025, blue), (2026, teal)]:
        q = main_priors.loc[main_priors.season.eq(year)].sort_values('position_group')
        axs[0].plot(q.position_group, q.mu*100, 'o-', color=color, label=str(year))
        axs[1].plot(q.position_group, q.strength, 'o-', color=color, label=str(year))
    axs[0].set(title='Position-specific prior centres', xlabel='Position', ylabel='Prior mean boundary probability (%)', xticks=range(1, 8))
    axs[1].set(title='Some positions are effectively pooled', xlabel='Position', ylabel='Fitted concentration k (log scale)', yscale='log', xticks=range(1, 8))
    axs[0].legend(); axs[1].legend()
    axs[2].bar(history.position_group, history.no_position_history_fraction*100, color=orange)
    axs[2].set(title='Personal history is fragmented', xlabel='Position', ylabel='Scored balls with no early same-position history (%)', xticks=range(1, 8), ylim=(0, 45))
    fig.supxlabel('2025–2026 evaluation; positions 1/2 use initial striker/non-striker convention. Very large k is not precisely identified.', fontsize=9)
    for ext in ['png', 'svg']:
        fig.savefig(FIG/f'prior_and_coverage.{ext}', dpi=170, metadata={'Date': None} if ext=='svg' else {})
    plt.close(fig)
    for svg in FIG.glob('*.svg'):
        svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    # All numerical tables come from saved outputs; text interprets the frozen run.
    display_order = ['beta_binomial', 'population', 'original_beta', 'top7_beta', 'raw', 'jeffreys', 'leave_one_out']
    cols = {'model': 'Model', 'log_loss': 'Log loss ↓', 'brier': 'Brier ↓', 'player_macro_log_loss': 'Equal-player log loss ↓'}
    labels = {'model': lambda x: LABELS.get(x, x), 'reference': lambda x: LABELS.get(x, x)}
    all_table = table(total.loc[display_order].reset_index(), cols, labels)
    pos_table = table(position, {'position_group': 'Position', 'N': 'Late balls', 'population': 'Position population',
                                'raw': 'Raw', 'beta_binomial': 'Position shrinkage', 'original_beta': 'Original shrinkage',
                                'top7_beta': 'Shared 1–7 shrinkage'}, {'N': lambda x: f'{int(x):,}'})
    key_unc = total_unc.loc[total_unc.model.eq('beta_binomial')&total_unc.reference.isin(['original_beta', 'top7_beta', 'population', 'raw', 'jeffreys'])]
    interval_table = table(key_unc, {'reference': 'Reference', 'log_loss_difference': 'Difference', 'log_loss_low': '95% low', 'log_loss_high': '95% high'}, labels)
    pos_unc = u.loc[u.scenario.eq(PRIMARY)&u.subset.eq('all_late')&u.model.eq('beta_binomial')&u.reference.eq('population')&u.position_group.ne('all')]
    simultaneous_table = table(pos_unc, {'position_group': 'Position', 'log_loss_difference': 'Shrinkage minus population',
                                        'log_loss_low': 'Individual low', 'log_loss_high': 'Individual high',
                                        'simultaneous_log_loss_low': 'Simultaneous low', 'simultaneous_log_loss_high': 'Simultaneous high'})
    prior_table = table(main_priors, {'season': 'Season', 'position_group': 'Position', 'position_records': 'Early players', 'early_N': 'Early balls',
                                    'mu': 'Prior mean', 'strength': 'k', 'prior_sd': 'Prior SD', 'at_strength_bound': 'At bound'},
                        {'mu': lambda x: f'{x:.2%}', 'strength': lambda x: f'{x:,.1f}', 'prior_sd': lambda x: f'{x*100:.2f} pp'})
    coverage_table = table(history, {'position_group': 'Position', 'late_N': 'Late balls', 'no_position_history_N': 'No same-position history',
                                    'no_position_history_fraction': 'Share', 'other_position_history_N': 'History at other positions only'},
                          {'no_position_history_fraction': lambda x: f'{x:.1%}'})
    ann = a.loc[a.scenario.eq(PRIMARY)&a.subset.eq('all_late')&a.position_group.eq('all')&a.season.ge(2024)&a.model.isin(['beta_binomial', 'population', 'original_beta', 'top7_beta'])]
    annual_table = table(ann, {'season': 'Season', 'model': 'Model', 'log_loss': 'Log loss', 'brier': 'Brier'}, labels)
    restricted = m.loc[m.scenario.eq(PRIMARY)&m.subset.eq('min11_both_halves')&m.position_group.eq('all')&m.period.eq('evaluation')].set_index('model')
    diagnosis = table(total.loc[['all_prior_position_history', 'all_mean_position_strength', 'position_mean_all_strength', 'beta_binomial', 'prior_mean']].reset_index(),
                      {'model': 'Diagnostic', 'log_loss': 'Log loss', 'brier': 'Brier'}, labels)
    cases_table = table(cases, {'batter_name': 'Player', 'position_group': 'Position', 'early_N': 'Early balls', 'raw_early_rate': 'Raw early rate',
                              'p_population': 'Position population', 'p_beta_binomial': 'Position shrinkage', 'p_original_beta': 'Original shrinkage', 'N': 'Late balls'},
                        {key: lambda x: 'No history' if pd.isna(x) else f'{x:.1%}' for key in ['raw_early_rate', 'p_population', 'p_beta_binomial', 'p_original_beta']})
    no_history = int(history.no_position_history_N.sum())
    other_history = int(history.other_position_history_N.sum())
    dates = ip.loc[ip.scenario.eq(PRIMARY), ['season', 'date']].drop_duplicates().shape[0]
    innings = int(ip.scenario.eq(PRIMARY).sum())
    report = f'''# Position-specific shrinkage: batting positions 1–7

**Experiment 1 extension · 13 October 2026 · exploratory**

## Answer

**Separate batting-position priors improve delivery-weighted log loss by {relative:.2f}% over the original all-batter shrinkage model on identical 2025–2026 late-half deliveries.** The effect is uneven: positions 1, 2, 3 and 7 have better point scores; 4, 5 and 6 have worse ones. Position 2's improvement over the original is tiny and it does not beat the shared 1–7 shrinkage benchmark.

**There is still little evidence that individual shrinkage beats simply using each position's population rate.** Its aggregate point improvement is only {pop_relative:.3f}%, with a bootstrap interval crossing zero. None of the seven position-specific shrinkage-versus-population intervals excludes zero, even before simultaneous adjustment. Better role centring is useful, but seven completely separate personal histories are not a universally superior replacement.

![Position comparisons](../figures/positions/comparison.png)

## What each separate database means

There is one early-half player record for each **season × position × player ID**. A player batting at 3 and 5 has distinct records; only their early position-3 outcomes update their position-3 prior. Each prior uses all early players at that position, regardless of later appearances. Its strength is estimated by the same beta-binomial marginal likelihood as the original study. No strength is forced wider and no new model is chosen using the evaluation results.

The target is every eligible delivery faced at positions 1–7, not every batter assigned a permanent career label. Positions 8+ are excluded, and the original models are rescored on this exact 1–7 target. A separate shared-1–7 benchmark excludes the same tail positions but fits one prior and combines a player's history across ranks 1–7. It controls for the simpler effect of excluding lower-order records.

**Positions 1 and 2 need a qualification.** Cricsheet identifies the initial batter and non-striker; the pipeline calls these 1 and 2. It does not supply an independently validated scorecard opening order. Consequently these two results concern initial striker/non-striker, not different specialist opening roles. Ranks 3–7 are subsequent observed arrivals. Positions are reconstructed from every raw event before excluding illegal deliveries, including players first appearing as non-strikers. The two openers are also combined in a prespecified sensitivity analysis.

Predictions use the original calendar-half cutoffs, shared across all players and positions. They stay fixed for the late half. The outcome remains a batter boundary on a legal delivery outside Super Overs; legal byes/leg-byes remain zero outcomes and non-boundary flags are respected. Evaluation is **conditional on the observed later position**. We do not claim to know batting order at the midpoint or to forecast a player's probability of appearing at that position.

## Sample and aggregate results

The full positions-1–7 sample contains **267,994 eligible deliveries** across 2008–2026, versus 284,465 in the original all-position sample. The main evaluation has **12,881 late-half deliveries, 124 distinct players, 318 player-season-position records, {innings} batter-innings and {dates} match dates** across 2025–2026. Every model in each comparison scores the same outcomes.

{all_table}

Lower is better. Relative loss reductions are not percentage-point changes in boundary probability or accuracy. The equal-player metric aggregates each player's losses over positions/seasons before giving each player equal weight. It **slightly favours the original and shared-1–7 shrinkage models**, unlike the delivery-weighted result. Do not claim improvement for every weighting convention.

The shrinkage-minus-reference paired log-loss differences are:

{interval_table}

The shared-1–7 comparison is especially useful: the point gain over it is smaller and its interval crosses zero. Therefore the 0.35% gain over the original model cannot be attributed wholly and confidently to seven-way separation rather than the changed training population.

Paired uncertainty uses 2,000 whole-date resamples within evaluation season. Each resample uses common dates across positions so the aggregate retains same-day dependence. These intervals condition on fitted priors and observed exposures, omit refitting and some player/time dependence, and do not establish causal or permanent skill differences.

## Results at each position

These are log losses on each position's own identical target. Different positions have different boundary frequencies, so compare methods within a row, not absolute losses across rows.

{pos_table}

The largest point gain over the original shrinkage model is at position 3. Positions 4–6 show why an aggregate gain should not be used to impose the same modelling choice everywhere. The raw-rate failures are more severe when small histories are fragmented across positions.

For the individualisation question, uncertainty is:

{simultaneous_table}

The simultaneous columns use the 95th percentile of the maximum absolute centred bootstrap deviation across the seven effects, producing an approximate common-width family band. This is not a multiplicity correction for every exploratory model/scenario. None of the seven differences establishes a win over its own position population under either displayed interval.

## Prior centres, strengths and lost personal history

{prior_table}

The prior mean differs materially by position. For example, in 2026 it is about 25–27% for the two opening slots and position 3, versus about 17–21% for positions 4–7. A common mean misses some of these contextual differences.

Concentration k is **not** the number of early balls in the group. The individual-data weight is N/(N+k). Several groups have extremely large fitted k and forecasts close to their position population. In the primary fit, 2026 position 7 reaches the upper bound; 2025 positions 5/6 and 2026 position 4 are also effectively pooled. These values should be read as little detectable between-player heterogeneity under this model, not as precisely measured concentrations or proof of identical abilities.

As a numerical diagnostic, replacing all evaluation fits with k >= 10,000 by their exact pooled-probability limit changes aggregate log loss from {total.loc['beta_binomial','log_loss']:.6f} to **{limit_loss:.6f}**. This is a post-fit numerical check, not a tuned replacement model. The headline interpretation is unchanged. Bounds for all scenarios and all original fit flags remain in the published tables; no all-batter fallback was needed for insufficient position cohorts.

![Priors and history coverage](../figures/positions/prior_and_coverage.png)

{coverage_table}

Overall, **{no_history:,} of 12,881 deliveries ({no_history/12881:.1%})** are faced by players with no early-half history at that exact position. **{other_history:,}** of those deliveries belong to players with early batting history somewhere else. Their main position forecast discards that personal evidence and uses the position prior. Late-only population fallback is correct for the requested independent-database design, but the lost evidence is a reason to consider partial pooling later.

## Is the effect the prior centre, concentration or personal history?

The following calculations all keep the player's same-position early counts fixed:

{diagnosis}

Both centre and concentration affect the result; the improvement is not evidence that weaker shrinkage is desirable. Updating an all-batter prior with position-only history already changes performance relative to the original all-role personal history. The position prior then adds a smaller gain whose paired interval marginally crosses zero. A leave-one-player-out prior scores {total.loc['leave_one_out','log_loss']:.6f}, close to the primary {total.loc['beta_binomial','log_loss']:.6f}; reusing a player's early record in the empirical prior is not driving the headline result.

## Robustness and limits

**Combining the opening pair:** keeping ranks 3–7 separate but merging 1/2 gives aggregate log loss 0.541847, slightly better than the requested seven-way split's 0.541938. The paired combined-minus-separate difference is {convention.log_loss_difference:.6f}, interval [{convention.log_loss_low:.6f}, {convention.log_loss_high:.6f}]. There is no convincing additional benefit from the source's 1/2 convention. The merged-opening component reproduces the previous opener experiment exactly.

**Match-count halves:** a shared date cutoff nearest half the season's matches gives 15,952 evaluation deliveries. Position shrinkage scores 0.535477 versus 0.536679 for the correspondingly refitted all-batter reference, and 0.535903 for position pooling. Direction is similar, but the respective difference intervals [-0.002842, 0.000263] and [-0.001525, 0.000544] cross zero. Absolute scores cannot be compared to the calendar split because the scored balls differ.

**Minimum 11 early and late opportunities at the same position:** 9,059 balls remain. This is a retrospective evaluation subset, never a training filter.

{table(restricted.loc[['beta_binomial', 'population', 'original_beta', 'top7_beta', 'raw', 'jeffreys']].reset_index(), cols, labels)}

On this subset, shrinkage still clearly beats raw rates, but its advantage over original shrinkage is only -0.000281, interval [-0.001339, 0.000789]; shared-1–7 shrinkage has a slightly better point score. The very large all-sample raw log-loss penalty depends on raw zero/one probabilities and the fixed 10^-6 clip. The smoothed-rate and restricted checks show that shrinkage's advantage over naive personal rates is not solely this numerical effect. Brier is always computed without clipping.

**Season stability:**

{annual_table}

The seven-way split beats original shrinkage in both 2025 and 2026, but loses in 2024 (0.525172 versus 0.523451) and across development years (0.454032 versus 0.453601). These earlier losses, equal-player reversal and match-count uncertainty prevent a universal improvement claim.

## Real-player examples

The following 2026 cases were chosen after analysis to illustrate role changes, not as extra validation. A player can have several simultaneous position forecasts. “No history” means no eligible early-half ball at that position, not no IPL experience.

{cases_table}

## What this supports next

Keep position-matched population rates as strong benchmarks. Test **role-centred priors with partial sharing of personal history**, rather than automatically discarding all other-position records. Combine the opening pair unless independent position labels or prospective evidence justify separating them. Investigate why positions 4–6 do not benefit before replacing their existing forecasts. Continue to distinguish role/phase/environment effects from individual form.

This report does not implement those suggestions. The original sequential, preseason, context and 2027 watchlist experiments, previous opener results and earlier future-work plan are all unchanged. This extension was motivated by previously inspected outcomes: 2025–2026 remain exploratory, not newly untouched confirmation. No model is selected separately for each position from these results.

## Reproduction and verification

```bash
python pipeline.py  # only if processed data are missing
python positions_experiment.py
python build_positions_report.py
python verify_positions.py
python -m pytest -q
```

Use the repository's existing dependency environment. The new commands never invoke later experiments. Machine-readable outputs are in [results/positions](../results/positions), including wide forecasts (one probability column per model), priors, per-position/per-season metrics, raw coverage, cold starts, calibration, individual/simultaneous uncertainty and numerical checks. In priors, -1 denotes the original all-batter prior, -2 the shared-1–7 prior, and 0 the combined-opening group in its sensitivity. Prediction groups contain no negative ranks.

[The protocol](POSITIONS_PROTOCOL.md) and [configuration](../config/positions_protocol.json) record scope and timing. [Verification](../reports/positions_verification.json) independently rebuilds eligible innings counts and positions from the pinned raw JSON, reconstructs forecasts and losses, checks prior references against earlier committed studies, and checks hashes for every previously tracked research artifact except the additive README index. The initial local protocol commit preceded fitting; a shared-1–7 control was added before inspecting new performance, and the near-pooling numerical approximation was added after fitting and labelled accordingly.

The outcome is boundary probability, not overall batting quality, dismissal risk or selection probability. Later batting positions and innings survival have different context mixes; their fitted distributions do not isolate causal ability. Source data are the same retrospective [Cricsheet](https://cricsheet.org/) archive through 31 May 2026, with [the repository's attribution](../DATA_LICENSE.md).
'''
    (ROOT/'docs/POSITIONS_REPORT.md').write_text(report)
    images = ''.join('<figure><img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((FIG/f'{name}.png').read_bytes()).decode()+'"></figure>' for name in ['comparison', 'prior_and_coverage'])
    rendered = position.rename(columns={'position_group': 'Position', 'N': 'Late balls', **LABELS})
    rendered = rendered[['Position', 'Late balls', LABELS['population'], LABELS['beta_binomial'], LABELS['original_beta']]]
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Batting positions 1–7 — shrinkage study</title><style>body{font:17px/1.65 system-ui,sans-serif;background:#f3f7f8;color:#173740;margin:0}main{max-width:1200px;margin:auto;padding:40px 24px}h1{font-size:42px;line-height:1.15}a{color:#137d75}.tag{font-size:13px;letter-spacing:.09em;color:#137d75;font-weight:700}.note{padding:18px;border-left:4px solid #b86d31;background:#fff6e8}figure{margin:30px 0;background:white;padding:10px;border-radius:10px}img{width:100%}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:12px;border-bottom:1px solid #dce5e8;text-align:left}th{color:white;background:#173740}.scroll{overflow:auto}p{max-width:960px}</style><main><div class="tag">EXPERIMENT 1 · POSITIONS 1–7 · 13 OCTOBER 2026</div><h1>Position matching helps overall.<br>Seven separate histories do not help everywhere.</h1>'''
    html += f'<p>Position-specific shrinkage improves delivery log loss by <strong>{relative:.2f}%</strong> against original all-batter shrinkage on the same 12,881 late-half deliveries. The gain over each position’s population forecast is small and uncertain.</p><p class="note">Exploratory follow-up. Positions 1/2 mean initial striker/non-striker, not independently verified scorecard slots. Positions 4–6 and equal-player scoring do not favour the new model. Later experiments remain unchanged.</p><p><a href="POSITIONS_REPORT.md">Full results and methods</a> · <a href="POSITIONS_PROTOCOL.md">Protocol</a> · <a href="../results/positions/metrics.csv">All metrics</a></p><div class="scroll">'
    html += rendered.to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}')+'</div>'+images+'</main></html>'
    (ROOT/'docs/positions.html').write_text(html)
    print('Built position report, paired opener-convention check, numerical check and two figure sets')


if __name__=='__main__':
    main()
