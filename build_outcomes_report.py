"""Build the alternative-outcome report and static research figures from saved evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from outcomes_experiment import ROOT, OUT

LABEL = {'boundary':'Boundary', 'six':'Six', 'dot':'Zero batter runs', 'score30':'Score 30+', 'score50':'Score 50+'}
ORDER = ['boundary', 'six', 'dot', 'score30', 'score50']
FIG = ROOT/'figures/outcomes'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                       *['| '+' | '.join(map(str, r))+' |' for r in rows]])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(OUT/'evaluation_metrics.csv').set_index(['outcome','window','model'])
    u = pd.read_csv(OUT/'uncertainty.csv')
    pop = u.loc[u.reference.eq('population')].set_index(['outcome','window'])
    prior = pd.read_csv(OUT/'priors.csv').query('season in [2025,2026]').copy()
    prior['cv'] = prior.prior_sd/prior.mu
    prediction = pd.read_csv(OUT/'evaluation_predictions.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.hashsalt':'outcomes-20261013'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharey=True)
    for ax, window, title, color in zip(axes, ['current','previous3_plus_current'],
                                      ['Current early season only', 'Three prior seasons + current early season'],
                                      ['#718096','#007F83']):
        values = pop.xs(window, level='window').reindex(ORDER)
        y = np.arange(len(ORDER))
        ax.axvline(0, color='#a1a1aa', lw=1)
        ax.errorbar(values.log_skill_pct, y, xerr=np.array([values.log_skill_pct-values.skill_low,
                    values.skill_high-values.log_skill_pct]), fmt='o', color=color, capsize=4, lw=2, markersize=7)
        ax.set_yticks(y, [LABEL[o] for o in ORDER]); ax.set_ylim(4.6, -.6)
        ax.set_xlim(-3.5, 3.5); ax.set_title(title, fontsize=12, pad=14)
        ax.grid(axis='x', alpha=.15); ax.set_xlabel('Log-loss improvement over matching population (%)')
        for i, value in enumerate(values.log_skill_pct):
            ax.text(3.35, i, f'{value:+.2f}%', ha='right', va='center', fontsize=10)
    fig.suptitle('Six rate is the strongest observed illustration', x=.08, ha='left', fontsize=19, fontweight='bold')
    fig.text(.08,.89,'IPL 2025–26 late halves · each player compared with the same batting position',fontsize=11,color='#4b5563')
    fig.text(.08,.035,'Positive = better forecasts. Bars: individual 95% date-bootstrap intervals.\nExploratory comparison of ten candidates; no effect clears the conservative simultaneous family band.',fontsize=10,color='#4b5563')
    fig.subplots_adjust(top=.78,bottom=.19,left=.16,right=.97,wspace=.13)
    for extension in ['png','svg']:
        fig.savefig(FIG/f'comparison.{extension}', dpi=180, metadata={'Date':None} if extension=='svg' else {})
    plt.close(fig)
    svg=FIG/'comparison.svg'; svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')

    comparisons=[]
    for outcome in ORDER:
        values=[]
        for window in ['current','previous3_plus_current']:
            r=pop.loc[(outcome,window)]
            values.append(f'{r.log_skill_pct:+.2f}% [{r.skill_low:+.2f}, {r.skill_high:+.2f}]')
        b=scores.loc[(outcome,'current','beta')]
        comparisons.append([LABEL[outcome],f'{int(b.N):,}',f'{int(b.B):,}',*values])
    comparison_table=table(['Outcome','Opportunities','Events','Current history: skill [95% interval]','Longer history: skill [95% interval]'], comparisons)
    detail=[]
    for outcome in ['six','score50']:
        for model in ['population','current_population','raw','jeffreys','prior_mean','beta']:
            r=scores.loc[(outcome,'previous3_plus_current',model)]
            detail.append([LABEL[outcome],model,f'{r.log_loss:.6f}',f'{r.brier:.6f}',f'{r.player_macro_log_loss:.6f}'])
    model_table=table(['Outcome','Model','Log loss','Brier','Equal-player log loss'],detail)
    diagnostics=[]
    for outcome in ORDER:
        for window in ['current','previous3_plus_current']:
            p=prior.loc[prior.outcome.eq(outcome)&prior.window.eq(window)]
            g=prediction.loc[prediction.outcome.eq(outcome)&prediction.window.eq(window)]
            diagnostics.append([LABEL[outcome],'Current' if window=='current' else 'Longer',f'{p.strength.median():.1f}',
                                f'{p.prior_sd.median():.4f}',f'{p.cv.median():.3f}',
                                f'{np.average(g.data_weight,weights=g.N):.1%}',
                                f'{g.loc[g.early_N.eq(0),"N"].sum()/g.N.sum():.1%}'])
    diagnostic_table=table(['Outcome','History','Median k','Median prior SD','Median SD / mean','Exposure-weighted personal weight','No matching history'],diagnostics)
    annual=pd.read_csv(OUT/'annual_metrics.csv')
    annual_rows=[]
    for outcome in ['boundary','six','score50']:
        for window in ['current','previous3_plus_current']:
            g=annual.loc[annual.outcome.eq(outcome)&annual.window.eq(window)&annual.model.eq('beta')].set_index('season')
            annual_rows.append([LABEL[outcome], 'Current' if window=='current' else 'Longer',
                                *[f'{g.loc[y,"log_skill_pct"]:+.2f}%' for y in [2024,2025,2026]]])
    yearly_table=table(['Outcome','History','2024 validation','2025','2026'],annual_rows)
    report=f'''# Which batting outcome best illustrates shrinkage?

Research date: 13 October 2026. This is an exploratory extension of experiment 1 on the existing pinned archive, not a new confirmatory holdout. Positions 1–7 remain separate. No later experiment or prospective watchlist has been changed.

**Recommendation: use six probability, with three previous seasons plus the current early season, as the clearest observed illustration of individual information and shrinkage. Keep 50+ probability as the innings-level companion example.** Six rate improves log loss by 1.22% over the matching population and 0.81% over a current-season population; 50+ improves by 1.04% over its matching population, with uncertainty spanning no gain. These are modest improvements, not a dramatic forecasting breakthrough. Six rate's advantage over Jeffreys smoothing is also uncertain.

![Comparison](../figures/outcomes/comparison.png)

## First clarify the earlier “0.3%”

The seven-position boundary experiment improved delivery-weighted log loss by **0.35% versus the original all-batter shrinkage model**, not versus its own population. Its absolute loss difference was −0.001917, with a conditional date-bootstrap 95% interval [−0.003464, −0.000426]. That particular comparison excluded zero, although the equal-player metric and alternative split were less supportive.

Against the **position-specific population**, boundary shrinkage improved only **0.074%**: loss 0.541938 versus 0.542338. The earlier interval for the difference was [−0.001555, +0.000672], which includes zero. Repeating the same comparison here with 4,000 draws gives relative skill +0.074%, interval [−0.132%, +0.289%]. It is not persuasive evidence of superiority over that population benchmark. The earlier opener-only advantage over opener population was about 0.32%, also uncertain. The original project's separate 0.30% recency result compares different history models and does not answer this question.

These percentages are relative reductions in log loss, not percentage-point changes in boundary probability or percentages of correctly predicted balls. Statistical detectability and practical value are separate questions.

## Why the boundary gain is small

Shrinkage does not add one batter's balls to the entire database. The fitted population is represented by a mean μ and estimated strength k. The posterior is:

`posterior = [n / (n + k)] × player rate + [k / (n + k)] × fitted population mean`

For example, with 50 observations and k=100, the player receives one-third of the weight, regardless of whether the database contains thousands or millions of balls. This is a mathematical illustration, not a fitted player example. More population data estimates the population distribution more precisely; it does not inherently force the distribution of player abilities to become narrower.

Several effects reduce the incremental benefit here:

1. **The population comparator is already role-aware.** Some apparent player variation is a difference between roles and batting circumstances. Position matching already captures part of that signal.
2. **Personal history is noisy and fragmented.** The median evaluated player-season-position record has only 15.5 early legal balls, and 20.7% of evaluation balls have no early history at that exact position. A rate near 20% observed over 50 independent balls already has a binomial standard error of about 5.7 percentage points. Real deliveries also share innings conditions.
3. **The fitted model often finds little reliably identifiable residual variation.** Current boundary priors have median k≈132 over the 14 evaluated season/position fits; some cells are effectively pooled. Overall, only 30.0% of forecast weight goes to personal early data. This is evidence about learnable variation with this sample and model, not proof that every batter has the same true rate.
4. **An extra boundary is a coarse measure.** Combining fours and sixes may hide differences in power-hitting style. Six rate is more informative here even though its absolute prior SD is smaller. Stability, relative separation and sample size matter more than simply maximising variance.
5. **The assumptions are approximate.** Player form, bowling, venues, phases and selection change; balls are not independent repetitions of a permanent ability. Empirical estimation of the prior adds error. Shrinkage is not guaranteed to beat a strong population forecast on every finite test sample. The motivating [Brown study](https://arxiv.org/abs/0803.3697) is a forecasting comparison, not a promise of improvement for every outcome and cohort.

The relative prior spread illustrates the distinction: using longer histories, the median fitted SD/mean across season/position cells is 0.341 for sixes versus 0.175 for boundaries. These are descriptive fitted quantities with their own estimation uncertainty, not a significance test of latent variance.

## What was tested

Five outcomes, two fixed history windows, the same observed positions 1–7 and the same calendar-midpoint early/late split. Priors and personal counts use only past observations at the same position. No late outcomes tune forecasts. In 2026, 2025 outcomes legitimately enter the fixed three-year history because they are then in the past.

- **Boundary / six / dot:** per eligible legal delivery; no wides, no-balls or Super Overs. “Dot” means zero batter runs, including legal byes and leg-byes, not necessarily zero team runs.
- **30+ / 50+:** per actual observed batting appearance, including zero-legal-ball appearances; no did-not-bat roster entries. A hundred counts as 50+. Full scores include batter runs on illegal deliveries. Definitions follow the source's [Cricsheet JSON fields](https://cricsheet.org/format/json/).
- **Current history:** current early half. **Longer history:** three previous complete seasons plus current early half; freeze all forecasts at the split.
- Every model for an outcome is scored on identical target rows. The population is the pooled rate for the position and history window. Long-history models also face a current-season position population control.

The evaluation comprises **12,881 legal balls**, or **760 batting appearances** for innings outcomes, over **54 match dates and 124 players**. Different outcomes have different base rates, so absolute log losses cannot rank them. We compare each model's percentage reduction relative to its own outcome's population.

{comparison_table}

Positive skill means shrinkage beats population. All intervals above are individual paired date-bootstrap intervals, conditional on the fitted forecasts. The same date draws are shared across candidates. The conservative simultaneous band over all ten outcome/history effects includes zero for every candidate; for long-history sixes it is [−0.754%, +3.197%]. Searching several outcomes therefore does not establish a confirmatory winner.

## Selection honesty and the best illustration

The [protocol](OUTCOMES_PROTOCOL.md) was committed before fitting these outcomes. Selection used 2024, requiring an aggregated development-period population advantage and a 2024 advantage over Jeffreys, then ranking eligible outcomes by 2024 population skill. The outcomes and histories were fixed in advance; no width tuning was performed.

- **Longer history selected 50+.** Validation skill was +1.53%, and development skill +0.21%. Evaluation skill was +1.04%, but the date interval [−0.03%, +2.16%] crosses zero.
- **Current history selected dot rate mechanically**, because it alone met both eligibility checks. Its validation population skill was already negative (−0.061%) and its evaluation skill was −0.182%. The frozen eligibility rule did not require positive validation population skill; this is a limitation of that rule. This selection supplies no successful current-half demonstration. We do not change the rule after seeing results.
- **Long-history six rate is the largest realised evaluation improvement**, not the validation-selected outcome. It had positive aggregated development (+0.54%), validation (+0.94%) and both evaluation-year skills. It is the best explanatory case in this dataset, but recommending it now is explicitly exploratory. A new season is required for a fresh test.

{yearly_table}

## Six rate: what improves, and what remains uncertain

There were **1,153 sixes** among the 12,881 evaluation balls. Longer-history empirical Bayes has log loss **0.299679**, compared with **0.303385** for its matching population: **1.22% improvement**, individual 95% interval **[0.52%, 1.96%]**. Using the stronger current-season population comparator, the improvement is **0.81%**, interval **[0.22%, 1.45%]**. Thus the gain is not solely caused by comparing against a stale multi-season population.

Brier score improves **0.78%** over the matching population, and equal-player log loss also improves (0.287457 versus 0.289846). The log-loss gain is +0.95% in 2025 and +1.44% in 2026.

Shrinkage reduces log loss by **10.80% relative to raw rates**, partly because raw zero rates are punished heavily by log loss. The more informative comparison is Jeffreys smoothing: empirical Bayes is **0.59% better**, but its interval **[−0.09%, +1.29%]** crosses zero. We therefore have better evidence that personal information helps than that this particular fitted prior is decisively better than simple smoothing.

Because six rate became the recommended illustration after looking at evaluation results, two extra diagnostics were added and labelled **post hoc**. Player-cluster resampling gives a population-skill interval [0.31%, 2.40%]. Combining opening slots 1/2 yields +1.20%, date interval [0.48%, 1.98%]. These support descriptive stability, but do not remove selection uncertainty or replace the simultaneous family band.

## Your 50+ idea

It is a useful alternative, but **50+ in half a season is too sparse for a strong demonstration**. Evaluated player-season-position records have a median of only two early appearances; many players have zero fifties. Current-season 50+ priors frequently approach complete pooling, giving personal data only 5.1% of exposure-weighted forecast weight. Current-history shrinkage improves by only 0.13%, with an interval spanning a loss.

Three prior seasons increase median available appearances to six for evaluated records and personal weight to 13.8%. Evaluation contains **140 scores of 50+ in 760 appearances**. The longer-history model improves by **1.04% over matching population** and **5.18% over Jeffreys smoothing**; the latter interval is [1.15%, 9.01%]. Equal-player log loss improves too. This is a good sparse-data illustration of why an unsmoothed personal success rate is unreliable, but not firm evidence of beating population.

The population comparison is borderline under dependence choices: its date interval crosses zero; the player-cluster interval is [0.02%, 2.24%]. Combining openers still improves by +1.06%, with a date interval crossing zero. Do not privilege the interval that happens to exclude zero.

A fifty is also not pure batting quality. Openers have more opportunity; a finisher may complete a valuable short chase without any possibility of reaching 50. A rain-shortened or not-out innings below 50 is an observed failure for this target. This experiment predicts milestones in actual appearances, not milestones under equal or unlimited opportunity. Position controls help but do not fully solve that problem. Do not condition on a future realised balls-faced threshold, which would change the forecast question using future information.

## Full model comparison for the two useful cases

{model_table}

Lower scores are better. All rows here use longer histories except the explicitly labelled current-population comparator. `prior_mean` is the fitted mean without the player update; it differs from a ball/innings-weighted pooled rate. Six and 50+ posteriors both beat their fitted-mean-only comparator, supporting an actual contribution from personal history.

## Prior and exposure diagnostics

{diagnostic_table}

k is in units of the outcome's trials: balls for delivery rates, innings for milestones. It cannot be compared across those denominators as if it were a common sample size. SD and SD/mean are medians across 14 fitted season/position cells; personal weights are weighted by evaluation opportunities. Near-pooling fits often make k poorly determined, so large concentrations should not be read as precise quantities. There were no fallback fits. Replacing every fitted k≥10,000 by its exact pooled forecast changes any aggregate log loss by at most 0.000020 and does not change the recommendation.

## Plan for the rest of the project — not implemented

1. **Keep the existing boundary study as the baseline and negative-result lesson.** Correct any presentation that labels its 0.35% comparison as an advantage over position population. Explain why a role-aware population is difficult to beat and why narrower priors can be appropriate.
2. **Make six rate the proposed primary outcome for a fresh forecasting study; keep 50+ secondary.** Freeze the two targets, history choices, split dates, comparison models and practical success criterion before obtaining a new season. Record that six rate was nominated from this exploratory analysis. Evaluate both and report failures. A positive point estimate alone should not count as success.
3. **Repeat chronological selection across earlier seasons before choosing a history method.** Use an outer future-season evaluation and inner past-only selection for history length/decay. Begin with the fixed three-season window and current-history reference here. Do not choose a window by the 2025–2026 score. Include leave-one-player-out prior estimation and refit uncertainty as checks of the empirical prior.
4. **Improve role modelling while retaining useful player history.** Combine the arbitrary first-striker/non-striker opening labels. Compare the seven separate groups against broad roles and a hierarchical model with position effects plus a shared player effect. The latter lets evidence follow a batter who changes position. Fit all role and player effects from past data only and compare with position-only forecasts on identical targets. This is a future model comparison, not an assumption that broader pooling must win.
5. **Only then extend the sequential experiment to six rate.** Produce pre-match forecasts using all prior dates; compare pooled position rates, Jeffreys, the fixed empirical-Bayes window and a small preregistered decay grid. Preserve date blocking, equal-player scores, cold-start reporting, calibration and 2025/2026 year-by-year summaries as historical diagnostics. Separate the value of personal history from the value of recent weighting.
6. **Extend preseason prediction separately.** Freeze information before the season, not at its midpoint. Refit the prior from available completed seasons and test whether cross-season six-hitting differences persist. Do not reuse within-season gains as evidence for preseason forecasts; the information sets differ.
7. **Model innings opportunity before making 50+ a performance headline.** Forecast an actual appearance using information available at entry (position, innings, chase requirement and balls remaining where known). A joint scoring-and-dismissal/survival model is a candidate for handling not-outs and limited opportunity. Possible additional targets are dismissal risk per eligible ball, runs per innings with explicit censoring/termination assumptions, and strike-rotation rate (1–3 batter runs per ball). These are proposals, not tested findings here. Powerplay/death splits may explain styles but reduce samples and require their own past-only role/context benchmarks.
8. **Context and watchlists come last.** Ask whether a shrunk six signal improves over phase, venue, opponent and match-state predictions, rather than simply identifying an aggressive role. Promote a watchlist only after calibration and a fresh-season test. Describe six rate as power-hitting tendency; combine scoring, survival and opportunity measures before calling it overall batting quality. Keep the existing dated 2027 boundary watchlists intact until such evidence exists.

## Reproduction and audit

Run `python outcomes_experiment.py`, `python verify_outcomes.py`, `python build_outcomes_report.py`, and `python -m pytest -q` in the existing environment. The original pipeline must already have produced its processed tables. The study retains the pinned raw-data SHA-256 and protected-file hashes in [the manifest](../results/outcomes/manifest.json).

The raw-JSON audit reconstructs **19,111 ordinary batting appearances**, including **351 zero-legal-ball appearances**, and accounts for **1,696 batter runs on illegal deliveries** across the full archive. There were no wicket-only appearances missing from the striker/non-striker reconstruction. It verifies all evaluation denominators and successes, same-position past-only histories, forecast formulas and score calculations. Boundary controls reproduce the earlier seven-position experiment to 1e-10. The full suite has **32 passing tests**.

[Frozen protocol](OUTCOMES_PROTOCOL.md) · [Saved selection](../results/outcomes/selection.json) · [All numerical evidence](../results/outcomes) · [Independent verification](../reports/outcomes_verification.json)
'''
    (ROOT/'docs/OUTCOMES_REPORT.md').write_text(report)
    print('Built docs/OUTCOMES_REPORT.md and figures/outcomes/comparison.png/.svg')


if __name__ == '__main__':
    main()
