"""Build only the opener follow-up figures and report from saved experiment outputs."""
from pathlib import Path
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/openers'
FIG = ROOT/'figures/openers'
LABELS = {
    'population': 'Opener population', 'raw': 'Raw opening rate', 'beta_binomial': 'Opener shrinkage',
    'original_beta': 'Original all-batter shrinkage', 'original_population': 'Original all-batter population',
    'selected_strength': 'Wider prior selected on 2024', 'jeffreys': 'Jeffreys-smoothed opening rate',
    'leave_one_out': 'Leave-one-player-out shrinkage', 'prior_mean': 'Opener prior mean, no update',
    'all_prior_opening_history': 'All-batter prior + opening history',
    'opening_mean_all_strength': 'Opener mean + all-batter strength',
    'all_mean_opening_strength': 'All-batter mean + opener strength'}


def table(df, columns, formats=None):
    formats = formats or {}
    rows = ['| '+' | '.join(columns.values())+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            if c in formats:
                v = formats[c](v)
            elif isinstance(v, (float, np.floating)):
                v = f'{v:.6f}'
            cells.append(str(v))
        rows.append('| '+' | '.join(cells)+' |')
    return '\n'.join(rows)


def main():
    FIG.mkdir(exist_ok=True)
    m = pd.read_csv(OUT/'metrics.csv')
    a = pd.read_csv(OUT/'by_season.csv')
    u = pd.read_csv(OUT/'uncertainty.csv')
    p = pd.read_csv(OUT/'priors.csv')
    c = pd.read_csv(OUT/'coverage.csv')
    grid = pd.read_csv(OUT/'strength_validation_grid.csv')
    manifest = json.loads((OUT/'manifest.json').read_text())
    test = m.query("scenario=='primary' and subset=='all_late_players' and period=='test'").set_index('model')
    restricted = m.query("scenario=='primary' and subset=='min11_both_halves' and period=='test'").set_index('model')
    uncertainty = u.query("scenario=='primary' and subset=='all_late_players'")
    priors = p.query("scenario=='primary'")
    year_priors = priors.loc[priors.season.isin([2025, 2026])]
    model_order = ['beta_binomial', 'population', 'raw', 'original_beta', 'original_population', 'selected_strength']
    cols = {'model': 'Model', 'log_loss': 'Log loss ↓', 'brier': 'Brier ↓', 'player_macro_log_loss': 'Equal-player log loss ↓'}
    fmt = {'model': lambda x: LABELS.get(x, x)}
    main_table = table(test.loc[model_order].reset_index(), cols, fmt)
    relative = 100*(test.loc['original_beta', 'log_loss']-test.loc['beta_binomial', 'log_loss'])/test.loc['original_beta', 'log_loss']
    relative_pop = 100*(test.loc['population', 'log_loss']-test.loc['beta_binomial', 'log_loss'])/test.loc['population', 'log_loss']
    relative_raw = 100*(test.loc['raw', 'log_loss']-test.loc['beta_binomial', 'log_loss'])/test.loc['raw', 'log_loss']

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'svg.hashsalt': 'openers-20261013'})
    teal, blue, orange = '#137d75', '#36618a', '#bc672d'
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), layout='constrained')
    order = ['raw', 'original_population', 'original_beta', 'population', 'selected_strength', 'beta_binomial']
    axes[0].barh([LABELS[x] for x in order], test.loc[order, 'log_loss'], color=[orange, blue, blue, '#889ba9', '#b3c9c5', teal])
    for i, x in enumerate(order):
        axes[0].text(test.loc[x, 'log_loss']+.0005, i, f"{test.loc[x, 'log_loss']:.6f}", va='center', fontsize=9)
    axes[0].set(xlim=(.565, .612), xlabel='Log loss per eligible delivery (lower is better)', title='Same 4,975 late-opening deliveries')
    axes[0].text(0, -.17, 'Axis starts at 0.565 to show small differences.', transform=axes[0].transAxes, fontsize=9, color='#52616a')
    refs = ['raw', 'jeffreys', 'original_beta', 'population']
    # Show ordinary-scale, useful-comparison intervals; raw's clip-sensitive tail is in the report.
    refs = refs[1:]
    q = uncertainty.loc[(uncertainty.model=='beta_binomial') & uncertainty.reference.isin(refs)].set_index('reference').loc[refs]
    axes[1].errorbar(q.log_loss_difference*1000, range(len(q)),
                     xerr=np.vstack([q.log_loss_difference-q.log_loss_low, q.log_loss_high-q.log_loss_difference])*1000,
                     fmt='o', capsize=5, color=teal)
    axes[1].axvline(0, color='#87959c', ls='--')
    axes[1].set(yticks=range(len(q)), yticklabels=[LABELS[x] for x in refs],
                xlabel='Log-loss difference × 1,000', title='Paired date-bootstrap 95% intervals')
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].text(0, -.17, 'Negative favours opener shrinkage.\nIntervals condition on fitted models.', transform=axes[1].transAxes, fontsize=9, color='#52616a')
    fig.suptitle('Opener-specific forecasts improve; the gain over opener pooling remains uncertain', fontsize=14)
    for ext in ['png', 'svg']:
        fig.savefig(FIG/f'comparison.{ext}', dpi=180, metadata={'Date': None} if ext=='svg' else {})
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), layout='constrained')
    for ax, year in zip(axes[0], [2025, 2026]):
        for pop, color, label in [('all_batters', blue, 'All batters'), ('openers', teal, 'Openers')]:
            r = year_priors.loc[(year_priors.season==year)&(year_priors.population==pop)].iloc[0]
            x = np.linspace(.01, .5, 500)
            ax.plot(x*100, beta.pdf(x, r.alpha, r.beta)/100, color=color,
                    label=f'{label}: mean {r.mu:.1%}, k={r.strength:.1f}')
        ax.set(title=f'{year}: fitted latent-player prior', xlabel='Boundary probability (%)', ylabel='Density per percentage point')
        ax.legend(fontsize=9)
    annual = priors.pivot(index='season', columns='population', values='prior_sd')*100
    axes[1, 0].plot(annual.index, annual.all_batters, 'o-', color=blue, label='All batters')
    axes[1, 0].plot(annual.index, annual.openers, 'o-', color=teal, label='Openers')
    axes[1, 0].set(title='A smaller cohort is usually not a wider prior', ylabel='Prior standard deviation (percentage points)', xlabel='Season')
    axes[1, 0].set_xticks([2008, 2012, 2016, 2020, 2024, 2026])
    axes[1, 0].legend()
    for period, color, label in [('validation', blue, '2024 selection'), ('test', teal, '2025–2026 evaluation')]:
        q = m.loc[(m.scenario=='primary') & (m.subset=='all_late_players') & (m.period==period) & m.model.str.startswith('strength_x')].copy()
        q['scale'] = q.model.str.replace('strength_x', '').astype(float)
        q = q.sort_values('scale')
        reference = q.loc[q.scale==1, 'log_loss'].item()
        axes[1, 1].plot(q.scale, (q.log_loss-reference)*1000, 'o-', color=color, label=label)
    axes[1, 1].axhline(0, color='#87959c', ls='--')
    axes[1, 1].set(xscale='log', xticks=[.1, .25, .5, 1, 2, 4], xticklabels=['0.1', '0.25', '0.5', '1', '2', '4'],
                   xlabel='Multiply fitted concentration k (<1 widens prior)', ylabel='Log-loss change vs fitted prior × 1,000',
                   title='Forcing a wider prior does not transfer')
    axes[1, 1].legend()
    for ext in ['png', 'svg']:
        fig.savefig(FIG/f'prior_diagnostics.{ext}', dpi=180, metadata={'Date': None} if ext=='svg' else {})
    plt.close(fig)

    # Matplotlib emits trailing spaces in SVG path data; normalise generated text.
    for vector in FIG.glob('*.svg'):
        vector.write_text('\n'.join(line.rstrip() for line in vector.read_text().splitlines())+'\n')

    prior_table = table(year_priors, {'season': 'Season', 'population': 'Prior population', 'records': 'Early players',
                                     'early_N': 'Early balls', 'mu': 'Mean', 'strength': 'k', 'prior_sd': 'SD'},
                        {'mu': lambda x: f'{x:.2%}', 'prior_sd': lambda x: f'{100*x:.2f} pp', 'strength': lambda x: f'{x:.2f}'})
    key_u = uncertainty.loc[(uncertainty.model=='beta_binomial') & uncertainty.reference.isin(['raw', 'jeffreys', 'original_beta', 'population'])]
    interval_table = table(key_u, {'reference': 'Reference', 'log_loss_difference': 'Shrinkage minus reference',
                                  'log_loss_low': '95% low', 'log_loss_high': '95% high'}, {'reference': lambda x: LABELS[x]})
    annual_test = a.query("scenario=='primary' and subset=='all_late_players' and season>=2025")
    annual_test = annual_test.loc[annual_test.model.isin(['population', 'beta_binomial', 'original_beta', 'selected_strength'])]
    annual_table = table(annual_test, {'season': 'Season', 'model': 'Model', 'log_loss': 'Log loss', 'brier': 'Brier'}, fmt)
    width = grid[['model', 'multiplier', 'log_loss']].rename(columns={'log_loss': 'validation_log_loss'}).merge(
        test.reset_index()[['model', 'log_loss']], on='model').sort_values('multiplier')
    width_table = table(width, {'multiplier': 'k multiplier', 'validation_log_loss': '2024 log loss', 'log_loss': '2025–2026 log loss'}, {'multiplier': lambda x: f'{x:g}'})
    sensitivity = u.query("subset=='all_late_players' and model=='beta_binomial' and reference=='population'")
    sensitivity_table = table(sensitivity, {'scenario': 'Scenario', 'N': 'Scored balls', 'log_loss_difference': 'Shrinkage minus population',
                                          'log_loss_low': '95% low', 'log_loss_high': '95% high'})
    decomposition = ['all_prior_opening_history', 'all_mean_opening_strength', 'opening_mean_all_strength', 'beta_binomial']
    ablation_table = table(test.loc[decomposition].reset_index(), {'model': 'Prior + opening-only history', 'log_loss': 'Log loss', 'brier': 'Brier'}, fmt)
    narrower = int((annual.openers < annual.all_batters).sum())
    coverage_table = table(c.query("scenario=='primary' and season>=2025"), {'season': 'Season', 'early_players': 'Early openers',
                              'late_players': 'Late openers', 'early_N': 'Early balls', 'late_N': 'Late balls',
                              'new_opening_players': 'No early-opening history', 'new_opening_N': 'Their late balls'})
    names = pd.read_parquet(ROOT/'data/processed/model_deliveries.parquet')[['batter_id', 'batter_name']].drop_duplicates('batter_id', keep='last')
    predictions = pd.read_csv(OUT/'predictions.csv')
    selected_names = ['V Kohli', 'RG Sharma', 'YBK Jaiswal', 'Abhishek Sharma']
    examples = predictions.loc[(predictions.scenario=='primary') & (predictions.season==2026) &
                               (predictions.model=='beta_binomial')].merge(names, on='batter_id', validate='one_to_one')
    examples = examples.loc[examples.batter_name.isin(selected_names)].copy()
    examples['early_rate'] = examples.early_B/examples.early_N
    examples['late_rate'] = examples.B/examples.N
    examples = examples.merge(predictions.loc[(predictions.scenario=='primary') & (predictions.season==2026) &
                                              (predictions.model=='original_beta'), ['batter_id', 'p']].rename(columns={'p': 'original_p'}),
                              on='batter_id', validate='one_to_one').sort_values('batter_name')
    examples.to_csv(OUT/'player_examples.csv', index=False)
    examples_table = table(examples, {'batter_name': 'Player', 'early_N': 'Early opening balls', 'early_rate': 'Raw early rate',
                                      'original_p': 'Original shrinkage', 'p': 'Opener shrinkage', 'late_rate': 'Observed late rate',
                                      'N': 'Late opening balls'}, {key: lambda v: f'{v:.1%}' for key in ['early_rate', 'original_p', 'p', 'late_rate']})
    report = f'''# Opening batters: does shrinkage improve?

**Experiment 1 follow-up · 13 October 2026 · exploratory analysis**

## Finding

**Yes, using opener-specific training improves forecasts for openers relative to the original all-batter shrinkage model: {relative:.2f}% lower log loss on the same 4,975 late-opening deliveries in 2025–2026.** The paired date-bootstrap interval for the difference excludes zero, conditional on the fitted models. Shrinkage also beats raw rates. Its {relative_pop:.2f}% point improvement over an **opener-only population** remains uncertain. This is evidence for a better role-matched forecast, not a conclusive win for individualisation over role-matched pooling.

**The narrower-prior explanation is not supported.** The opener prior is narrower in 2025 and nearly the same width in 2026. Forcing a wider prior worsens evaluation forecasts. The main benefit is associated with a more appropriate, higher prior centre for opening batters.

![Forecast comparison](../figures/openers/comparison.png)

## What was repeated, and what was preserved

Only experiment 1 was rerun. The original sequential, preseason, contextual and 2027 watchlist code/results were preserved byte-for-byte, along with the original experiment-1 outputs. This extension has its own configuration, results folder, figures and report. [The future-work plan](OPENERS_NEXT_STEPS.md) proposes subsequent changes; none were executed.

The original outcomes were already known when this question was proposed. Therefore 2025–2026 are an **exploratory evaluation period**, not a newly untouched confirmatory test. The opener protocol was committed before its first fit. Development summaries cover 2008–2023; 2024 alone chooses the optional width multiplier. No 2025–2026 result changes that selection. The added parameter-swap diagnostics below are explicitly post-protocol.

## Opening-batter definition and coverage

The opening pair is the batter and non-striker on the first recorded event of each ordinary innings, even if that event is illegal. Membership agrees exactly with the pipeline's entry positions 1–2. There are 151 innings with an illegal first event; identifying the first two players to face legal balls would be unsafe. All recorded innings start in over zero. Nine of the 4,960 opening-pair slots have no eligible batting delivery; they are not invented zero-rate records.

An eligible outcome is the same legal-delivery batter boundary as before, excluding no-balls, wides and Super Overs, preserving legal byes/leg-byes as zeroes and respecting non-boundary flags. All of an opener's eligible deliveries in that innings count, including later overs. Their non-opening innings do not enter the opener fit or target. Players can change role; this is an innings role, not a permanent career label. Outcomes are conditional on actually opening, not a prediction of selection or opening probability.

The complete opener sample contains **105,948 opportunities, 20,717 boundaries and 4,951 observed opening batter-innings** over 2008–2026. The primary comparison scores **4,975 opportunities, 1,293 boundaries, 240 batter-innings, 53 player-seasons, 38 distinct players and 54 match dates** in the late halves of 2025–2026.

{coverage_table}

Eight late player-season records have no early-opening opportunities, accounting for 345 scored balls (6.93%). Three of those records have earlier non-opening history; that history is excluded from the opener model but available to the original-model comparator. Raw and pooled forecasts fall back to the pooled opener rate; empirical Bayes falls back to its fitted prior mean, exactly as the original implementation does. These two population centres are not identical.

Shared calendar cutoffs, with complete dates together, match the original experiment. Forecasts use early outcomes only and remain fixed for the late half. The >=11 subset is a retrospective evaluation restriction, never a training or player-selection rule.

## Results on identical opening deliveries

Lower scores are better. Equal-player scores first aggregate all evaluation opportunities for each player and then average across the 38 players; player-rate MSE in the CSV instead averages player-season rates. The original all-batter models are rescored on the **same opening deliveries**, so this comparison does not confuse a changed target population with a better model.

{main_table}

Opener shrinkage improves log loss by {relative_raw:.2f}% against raw rates, {relative:.2f}% against the original shrinkage forecast, and {relative_pop:.2f}% against the opener population forecast. These are relative loss reductions, not percentage-point increases in boundary probability or prediction accuracy.

Paired loss differences use 2,000 bootstrap resamples of whole match dates within each evaluation season. Negative favours opener shrinkage:

{interval_table}

The Brier difference versus opener pooling is -0.000757, interval [-0.001865, 0.000217], also crossing zero. Against original shrinkage it is -0.000813, interval [-0.001269, -0.000333]. Both log loss and Brier therefore support the same qualified interpretation.

### Are raw rates making the gain look too large?

Raw 0/1 forecasts require clipping at 10^-6 for log loss. Their Brier/MSE scores remain unclipped in this extension; the original common scoring helper clipped Brier too, a negligible discrepancy corrected locally without modifying the earlier studies. A Jeffreys-smoothed individual rate scores **{test.loc['jeffreys','log_loss']:.6f}**, still worse than fitted opener shrinkage, but much closer than raw log loss.

The >=11-in-both-opening-halves subset contains 4,543 late balls and 40 player-seasons:

{table(restricted.loc[['beta_binomial','population','raw','original_beta','selected_strength']].reset_index(), cols, fmt)}

Here shrinkage minus raw log loss is -0.002177, interval [-0.003747, -0.000670]; versus opener pooling it is -0.002143, interval [-0.005259, 0.000697]. The gain over raw is not solely a zero-clipping artefact. Equal-player log loss on this restricted subset slightly favours original shrinkage over opener shrinkage; do not claim superiority for every weighting choice.

### Does it hold in both evaluation seasons?

{annual_table}

Both years favour opener shrinkage over original shrinkage and opener pooling in delivery-weighted point estimates. The broader historical gain over opener pooling is much smaller: development log losses are 0.481835 versus 0.481908. In 2024, opener shrinkage (0.546566) actually loses to original shrinkage (0.545298) on opening deliveries. The result is not a uniform historical improvement.

## Why the prior is not everyone's accumulated balls

The model fits a distribution of latent player probabilities: p_i ~ Beta(alpha, beta). Player outcomes then follow B_i | p_i, N_i ~ Binomial(N_i, p_i). Its predictive mean is:

`p_hat_i = (B_i + mu*k)/(N_i + k)`, where `mu = alpha/(alpha+beta)` and `k = alpha+beta`.

The weight on a player's own rate is `N_i/(N_i+k)`. The effective prior strength k is learned from **between-player variation after accounting for sampling noise**. It is not the number of balls faced by the population. A precise estimate of the population mean is also different from a narrow distribution of individual player probabilities.

Shrinkage theory does not guarantee a win on every realised sample or against every baseline. Under suitable assumptions it trades sampling variance against bias; if persistent differences between players are small, population pooling can be difficult to beat. Changing conditions and mismatch between the model and real deliveries further limit a theoretical guarantee here.

{prior_table}

For example, the all-batter prior uses 10,330 early balls in 2025 but has strength only **70.89**, not 10,330. Restricting to 3,643 opening balls produces strength **111.30**, meaning *more* shrinkage at a fixed individual N. With N=100, the player-data weight changes from 58.5% to 47.3%. In 2026 the weights are 65.4% and 62.6%, respectively.

Prior standard deviation, sqrt(mu*(1-mu)/(k+1)), measures actual width. It falls from 4.69 to 3.94 probability points in 2025. In 2026 it changes from 5.56 to 5.62 points, almost unchanged despite the slightly larger k because the mean changes too. The opener prior is narrower by this measure in **{narrower} of 19 seasons**. Fewer records increase uncertainty about the fitted prior; they do not mechanically widen the estimated distribution of player abilities.

![Prior diagnostics](../figures/openers/prior_diagnostics.png)

The 2014 opener fit has k around 13,153 and is effectively pooled; its exact large value should not be overinterpreted. One development-only fit, the 2023 powerplay sensitivity, hits the concentration upper bound. All primary and evaluation-period fits converge away from that bound. Small opener samples make prior-parameter uncertainty important even when the optimiser reports convergence. These are conditional empirical-Bayes fits, not full hierarchical posterior uncertainty estimates.

## What explains the improvement?

The prior centre shifts from 19.67% to 22.53% in 2025 and from 21.15% to 26.00% in 2026. This supplies a more relevant comparison group. The original all-batter prior updated with opening-only history scores 0.574700, compared with 0.572397 for the opener prior updated with the same history. Removing non-opening individual history alone yields only a small part of the change from 0.574907.

To separate centre and strength, the following **post-protocol descriptive ablation** swaps one fitted parameter at a time while holding opening-only individual counts fixed:

{ablation_table}

Changing the mean accounts for most of the reduction along either swap path. Changing strength alone at the all-batter mean does not help. This supports role-matched centring as the explanation within this forecasting model; it does not identify a causal cricket mechanism.

The no-individual-update opener prior mean scores 0.574316, and its paired comparison with individual shrinkage remains uncertain. Leaving each player's early record out of its prior fit gives 0.572375, essentially unchanged; this diagnostic does not remove shared match effects or quantify all prior-fitting uncertainty.

### Real-player illustrations

These named 2026 cases explain the forecasts; they were chosen after analysis and are not extra validation. Later observed rates are noisy outcomes, not true abilities. The population forecast for all these opening innings is the same early-opening pooled rate; the fitted prior centre is 26.00%.

{examples_table}

## Does deliberately widening the prior help?

The mean stays fixed while k is multiplied by the values below. A smaller multiplier widens the prior and gives the player more weight. The entire grid was fixed before the opener results were inspected.

{width_table}

2024 selects **k × 0.1**. That fixed choice scores **0.574362** on 2025–2026, versus **0.572397** for the fitted width: a loss increase of **0.001966**, with interval **[0.000549, 0.003552]**. It is worse in each evaluation year. Every tested widening factor has worse aggregate evaluation log loss than the fitted prior. Although k × 2 has the best evaluation point score within the displayed grid, selecting it now would use evaluation outcomes; it is not adopted. A single selection year gives an unstable width choice.

## Other experiment-1 checks

Each sensitivity refits the prior and individual early records on its specified training subset, then compares models on a common late subset. Negative differences favour shrinkage over that subset's own population forecast:

{sensitivity_table}

Match-count halves keep whole dates together and address calendar interruptions. League-only excludes matches with named event stages while preserving original calendar cutoffs. Standard-conditions excludes documented shortened/miscounted innings. Powerplay-only uses nominal overs 1–6 faced by the opening pair; it changes the target and is not the primary analysis.

The powerplay-only result reverses the population comparison: **population pooling has the better point score**, although the interval again spans zero. Sharing the new-ball phase does not make deliveries independent, nor does it establish stronger individual predictability. Openers face varying venues, opponents and chase conditions; those surviving longer contribute more middle/death balls. The full-opening-innings result cannot be interpreted as skill differences under identical conditions. Calibration bins are supplied without independent-ball error bars.

## Limits and decision

- These paired intervals condition on estimated priors, the observed opportunity mix and two seasons. They omit refitting, model-selection, some cross-date/player dependence and future regime uncertainty. Multiple diagnostic comparisons are not adjusted for multiplicity.
- The current archive is a retrospective data vintage, with coverage through 31 May 2026. Outcome-dependent exposure and survival remain; there is no selection/availability forecast.
- No claim of exact Brown estimator replication, permanent improvement, full batting quality or profitable trading follows from this boundary-rate study.
- Keep the learned opener prior as the primary experiment-1 specification for this cohort. Keep opener pooling as a serious benchmark. **Do not force wider priors or propagate the 2024-selected multiplier into the other experiments.**

## Reproduce and inspect

From the repository, with its Python dependencies installed:

```bash
python pipeline.py
python openers_experiment.py
python build_openers_report.py
python verify_openers.py
python -m pytest -q
```

The first command is needed only if processed tables are absent; it uses the existing pinned archive. The opener commands never invoke the sequential, preseason, contextual or watchlist stages. Original baseline reference: commit `d9776f43dd9b1cb78a75b213a311d1e306841b72`. Reconstructed original shrinkage forecasts match the committed values to <10^-10. Protocol, raw archive and original artifact hashes are in `results/openers/manifest.json`; numerical tables, all player-season predictions and evaluation innings predictions are adjacent. The width selection is reconstructed from 2024 only during verification. CSV probabilities are unrounded; displayed figures round for readability.

See [the follow-up protocol](OPENERS_PROTOCOL.md), [configuration](../config/openers_protocol.json), [results folder](../results/openers), [verification](../reports/openers_verification.json) and [detailed plan for the remaining project](OPENERS_NEXT_STEPS.md).

Sources: the existing pinned [Cricsheet IPL data](https://cricsheet.org/downloads/) and [JSON field definitions](https://cricsheet.org/format/json/), with the repository's [data attribution](../DATA_LICENSE.md); the original [Brown-inspired protocol](PROTOCOL.md). All numerical results above come from this repository's experiment, not from the external documentation.
'''
    (ROOT/'docs/OPENERS_REPORT.md').write_text(report)
    images = ''.join('<figure><img alt="'+label+'" src="data:image/png;base64,'+base64.b64encode((FIG/name).read_bytes()).decode()+'"><figcaption>'+label+'</figcaption></figure>' for name, label in [('comparison.png', 'Identical opening deliveries; conditional paired date-bootstrap intervals'), ('prior_diagnostics.png', 'Prior centre, width and the effect of forcing less shrinkage')])
    display = test.loc[model_order, ['log_loss', 'brier', 'player_macro_log_loss']].rename(index=LABELS).reset_index().rename(columns={'model': 'Model', 'log_loss': 'Log loss', 'brier': 'Brier', 'player_macro_log_loss': 'Equal-player log loss'})
    view = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Opening batters — experiment 1 follow-up</title><style>body{font:17px/1.65 system-ui,sans-serif;background:#f3f7f8;color:#163740;margin:0}main{max-width:1200px;margin:auto;padding:40px 24px}h1{font-size:42px;line-height:1.15}h2{margin-top:36px}.tag{color:#137d75;font-size:13px;font-weight:700;letter-spacing:.08em}a{color:#137d75}figure{margin:30px 0;padding:12px;background:white;border-radius:12px}img{width:100%}figcaption{font-size:13px;color:#52616a}table{border-collapse:collapse;background:white;width:100%;font-size:15px}th,td{padding:12px;text-align:left;border-bottom:1px solid #dce5e8}th{background:#163740;color:white}.scroll{overflow:auto}.note{padding:18px;border-left:4px solid #bc672d;background:#fff6e9}p{max-width:960px}</style><main><div class="tag">EXPERIMENT 1 · EXPLORATORY FOLLOW-UP · 13 OCTOBER 2026</div><h1>A better comparison group helps.<br>A wider prior does not.</h1>'''
    view += f'<p>Opener-specific shrinkage reduces log loss by <strong>{relative:.2f}%</strong> versus original all-batter shrinkage on the same 4,975 late-opening deliveries in 2025–2026. Its advantage over an opener-only population forecast remains uncertain.</p><p class="note">This is a follow-up motivated by known original results. It is not a fresh confirmatory test. The 2024-selected wider prior worsens evaluation forecasts. Later experiments and 2027 watchlists are unchanged.</p><p><a href="OPENERS_REPORT.md">Full results and methods</a> · <a href="OPENERS_NEXT_STEPS.md">Detailed next-stage plan</a> · <a href="../results/openers/metrics.csv">Download scores</a></p><div class="scroll">'
    view += display.to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}')+'</div>'+images
    view += '<h2>Recommended direction</h2><p>Test role-matched prior centres and retain learned shrinkage strengths. Keep population pooling as a benchmark; separate role and league-environment shifts from player improvement. Do not change the published later experiments until a new comparison and evaluation protocol is agreed.</p><p>Data: <a href="https://cricsheet.org/">Cricsheet</a>, transformed under the repository attribution. See the full report for uncertainty, coverage and source limitations.</p></main></html>'
    (ROOT/'docs/openers.html').write_text(view)
    print('Built opener report, HTML viewer and two PNG/SVG figure sets')


if __name__ == '__main__':
    main()
