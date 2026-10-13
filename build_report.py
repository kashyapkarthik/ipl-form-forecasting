"""Build the GitHub research report, README, and portable HTML from checked results."""
from pathlib import Path
import json, base64, html
import pandas as pd
from research import ROOT, OUT, CONFIG
from visualise import LABELS


def table(df, columns, formats=None):
    formats=formats or {}
    headers=list(columns.values()); lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for _,r in df.iterrows():
        vals=[]
        for col in columns:
            value=r[col]
            if col in formats: value=formats[col](value)
            elif isinstance(value,float): value=f'{value:.6f}'
            vals.append(str(value))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def main():
    seq=pd.read_csv(OUT/'sequential_metrics.csv').query("period=='test'")
    pre=pd.read_csv(OUT/'preseason_metrics.csv').query("period=='test'")
    bm=pd.read_csv(OUT/'brown_metrics.csv').query("period=='test' and subset=='min11_both_halves'")
    b_all=pd.read_csv(OUT/'brown_metrics.csv').query("period=='test' and subset=='all_late_players'")
    ctx=pd.read_csv(OUT/'context_metrics.csv'); unc=pd.read_csv(OUT/'sequential_uncertainty.csv').set_index('model').loc['ew_days_365']
    pun=pd.read_csv(OUT/'preseason_uncertainty.csv').set_index('model').loc['rolling_300']
    cun=pd.read_csv(OUT/'context_uncertainty.csv').iloc[0]
    smooth=pd.read_csv(OUT/'brown_smoothed_sensitivity.csv').set_index('subset')
    improve=pd.read_csv(OUT/'watchlist_improvers.csv'); rebound=pd.read_csv(OUT/'watchlist_rebounds.csv')
    allf=pd.read_csv(OUT/'forecast_2027_all.csv'); audit=json.loads((ROOT/'reports/data_audit.json').read_text())
    sb=seq.set_index('model'); pb=pre.set_index('model')
    reduction=100*(sb.loc['expanding','log_loss']-sb.loc['ew_days_365','log_loss'])/sb.loc['expanding','log_loss']
    pred_reduction=100*(pb.loc['expanding','log_loss']-pb.loc['rolling_300','log_loss'])/pb.loc['expanding','log_loss']
    cols={'model':'Model','log_loss':'Log loss ↓','brier':'Brier score ↓','player_macro_log_loss':'Equal-player log loss ↓'}
    fmt={'model':lambda x:LABELS.get(x,x)}
    watchcols={'player':'Player','baseline':'Pre-2026 baseline','latest_rate':'Observed 2026','forecast':'2027 forecast','forecast_low':'Interval low','forecast_high':'Interval high','change_from_baseline':'Change vs baseline'}
    wf={c:(lambda v:f'{100*v:.1f}%') for c in watchcols if c!='player'}
    wf['change_from_baseline']=lambda v:f'{100*v:+.1f} pp'
    report=f'''# How quickly should we believe a batter has improved?

**A reproducible IPL boundary-forecasting study · Kashyap Karthik · 12 October 2026**

## Findings

Remembering less history improves simple forecasts, but this is not sufficient evidence of genuine skill improvement. A 365-day exponential half-life, chosen on 2024, reduces delivery-weighted log loss by **{reduction:.2f}%** relative to all-history shrinkage on 2025–2026. Its extra value after contextual adjustment is small and statistically indistinguishable from zero under our bootstrap. The project therefore supports a forecasting improvement, with a qualified interpretation of “form”.

![Main comparison](../figures/model_comparison.png)

- **Shrinkage:** beta-binomial forecasts improve on raw early-season player rates. The gain over a pooled population rate is uncertain.
- **Memory during a season:** recent weighting helps in the reserved seasons, but not uniformly in earlier seasons or for every player-weighting scheme.
- **Memory across an off-season:** 2024 selects a 300-ball rolling window for forecasts frozen before a season. It reduces test log loss by {pred_reduction:.2f}% versus expanding history. We retain that selected model even though another candidate scores slightly better on the test set.
- **Context:** adding recent-history information to a past-trained contextual model yields a log-loss difference of {cun.log_loss_difference:.6f}, with a 95% interval [{cun.log_loss_low:.6f}, {cun.log_loss_high:.6f}]. This does not establish extra predictive value after adjustment.
- **Application:** dated 2027 forecasts for {len(allf)} batters who faced an eligible ball in 2026, with two candidate watchlists. They forecast one aspect of batting, not overall quality or likelihood of selection.

## 1. Question and relation to Brown

Brown (2008) used early-season baseball records to forecast later batting averages and compared shrinkage estimators with naïve estimates. This study adapts that prediction question to a binary cricket outcome. It uses a beta-binomial estimator rather than reproducing Brown's complete transformed-normal estimator suite. Brown also examined streakiness; this project does not claim to introduce that question. Its particular extension compares history lengths and decay rules across IPL seasons.

An exact reproduction of Brown's numerical baseball results is outside this study. An arcsine noise-adjusted score on the ≥11-opportunity subset provides a methodological connection, not an estimator replication.

## 2. Data and outcome

The pinned Cricsheet archive contains **1,243 matches**, **295,732 delivery events**, and seasons **2008–2026**. Source coverage ends **31 May 2026**. The primary sample has **284,465 eligible deliveries**, including **49,905 boundaries**, aggregated into **18,760 batter-innings**. This is a current retrospective data vintage, not a reconstruction of each past publication date. Coverage has not been independently matched to every official scorecard.

A success is a four or six credited to the batter, unless marked `non_boundary`. Eligible opportunities are legal deliveries outside Super Overs. We retain ordinary legal byes/leg-byes as zero outcomes and preserve all excluded events in the raw layer. The exclusion accounting is 175 Super Over events + 9,871 ordinary wides + 1,221 ordinary no-balls. There are 29 raw fours/sixes marked non-boundary. We call the denominator eligible deliveries, not official scorecard balls faced.

Player identities come from Cricsheet registry IDs. Match/innings/event order prevents illegal-ball numbering errors. Dates normalise IPL season labels while preserving source labels. Wickets, rosters, substitutions, venue, opposition, nominal innings phase, and pre-delivery state remain available. Venue groups conservatively merge punctuation and city suffixes; major renamed/rebuilt venue labels remain distinct. The mapping is published in `results/venue_mapping.csv`.

Shortened and no-result innings with observed balls remain in the main sample. A standard-conditions sensitivity check removes documented shortened/miscounted cases. Nominal phases are overs 1–6, 7–15 and 16–20; they do not claim exact fielding restrictions in rain-shortened matches. Final chase targets and match winners are not forecast features. Batting-entry order is observed order, not an externally validated specialist role.

## 3. Model

For each player, B boundaries in N opportunities are modelled as binomial with unknown probability p, with a Beta(alpha, beta) population prior. The posterior mean is:

`forecast = (alpha + B) / (alpha + beta + N)`

Equivalently, the forecast averages the player's rate and the population mean. The weight on the player's rate is `N / (N + alpha + beta)`. Small samples receive stronger shrinkage. SciPy maximises the beta-binomial marginal likelihood over population mean and concentration, using multiple starting values and bounded optimisation.

In the within-season adaptation, population parameters use early-half players only. In the multi-season experiments, the prior for year t is fitted to player-season counts in years t−3 through t−1. The same annual prior is used for every history method. Repeated players mean those population records are not fully independent; the model is an approximation. No current-year outcomes enter that year's prior fit.

The expanding model uses all earlier IPL balls equally. Rolling models use the latest 120, 300, 600 or 1,200 opportunities. Exponential models use ball half-lives of 120/300/600/1,200 or calendar half-lives of 90/180/365/730 days. Weighted counts enter the same forecast equation. These are discounted-count forecasting rules, not automatically exact conjugate posteriors. Calendar decay reduces evidence during an off-season and increases shrinkage towards the population prior; ball decay does not decay through inactivity.

## 4. Experimental design and leakage controls

The protocol and grid were committed before model results were inspected. Development years run through 2023; 2024 selects candidates by delivery-weighted log loss; 2025–2026 evaluate the frozen choices. The snapshot counts were already inspected during data preparation. This is an internal analysis freeze, not external preregistration.

**Within season:** one calendar midpoint per season separates early and late halves. All forecasts freeze at the cutoff. New late-half batters receive a population forecast. Tables show both all late-half players and the subset with at least 11 opportunities in each half. The latter is a retrospectively defined comparable evaluation subset, not a deployable selection rule. Calendar halves are unusual in the interrupted 2021 season; our numerical claims focus on 2025–2026. IPL playoffs are included, unlike Brown's baseball regular-season restriction.

**Sequential:** form one forecast per batter before each match date and keep it fixed through that innings. All same-date outcomes are excluded from its historical features. Once outcomes occur, they can enter later forecasts under the frozen rule, including during the test seasons.

**Preseason:** freeze probabilities before the first observed match of the target season. Evaluate these unchanged forecasts on every later opportunity. This directly tests persistence over the off-season. New players have no invented history and receive the prior. Metrics are conditional on actual batting exposure; selection and opportunity volume are not forecast.

**Context:** fit regularised logistic models using the preceding five seasons, then predict the next season's deliveries. Features include historical expanding ability, venue group, opposition team, nominal phase, innings, observed batting-entry order, score/wickets/balls before delivery, and a season trend. Compare adding the pre-date recent-minus-expanding logit signal. Encoders and scalers use training data only. Context forecasts have a richer pre-delivery information set than the pre-innings main forecasts; their absolute scores are not a like-for-like model contest with the main table. Opposition is team-level, not an estimated bowler-strength model.

Log loss and Brier score reward well-calibrated probabilities. Scores are reported both per delivery and with equal weight per player. Raw zero/one rates are clipped to 10^-6 only for numerical log loss; their poor scores can be sensitive to this choice. We therefore show Brier score, sample-restricted results and population comparisons too.

Paired loss differences use 2,000 date-block bootstrap resamples within each test season. These intervals condition on fitted models and two observed seasons. They do not capture every serial dependence, retraining uncertainty, model-selection uncertainty or future regime shift. Subgroup results are descriptive, without multiple-comparison claims.

## 5. Within-season adaptation

Held-out 2025–2026, at least 11 opportunities in each half (12,270 late-half deliveries):

{table(bm,cols,fmt)}

Across all late-half observations, the raw versus shrinkage comparison is more dramatic because tiny early samples can generate zero/one forecasts. The all-player beta-binomial versus population log-loss difference is −0.000388, with date-bootstrap interval [−0.002326, 0.001445]. **The data do not clearly establish that individualised shrinkage beats the population baseline at this horizon.**

A post-protocol sensitivity check replaces raw rates with a Jeffreys-smoothed individual estimate `(B + 0.5)/(N + 1)`, retaining the population fallback for no early history. Its all-player test log loss is {smooth.loc['all_late_players','log_loss']:.6f}, versus 0.536012 for empirical Bayes; on the ≥11 subset it is {smooth.loc['min11_both_halves','log_loss']:.6f}, versus 0.541251. Shrinkage's advantage is smaller than against clipped raw rates but remains in the point estimates. This additional diagnostic did not select the forecasting model.

The arcsine noise-adjusted mean squared error on the ≥11 subset is 0.002349 for beta-binomial versus 0.007329 for raw rates; this transformed score is approximate and only reported on the restricted subset. It is not a probability or a literal percentage of variance explained.

## 6. Which history is useful?

Sequential forecasts on 33,171 held-out deliveries:

{table(seq,cols,fmt)}

2024 selects **365-day exponential decay**. Relative to expanding shrinkage, its test log-loss difference is **{unc.log_loss_difference:.6f}**, interval **[{unc.log_loss_low:.6f}, {unc.log_loss_high:.6f}]**. Its Brier improvement is consistent in sign. However, equal-player log loss is worse than expanding in this test: weighting every batter equally changes the answer. The chosen criterion emphasises players with more opportunities.

Preseason forecasts, scored on the same held-out opportunities:

{table(pre,cols,fmt)}

2024 selects **the last 300 balls**. Its test difference from expanding is **{pun.log_loss_difference:.6f}**, interval **[{pun.log_loss_low:.6f}, {pun.log_loss_high:.6f}]**. The 300-ball exponential model has a slightly better realised test score, but switching to it now would use test outcomes to choose the method.

![Uncertainty](../figures/uncertainty.png)

![Calibration](../figures/calibration.png)

The reliability plot exposes calibration imperfections; recent weighting does not make the forecasts perfectly calibrated. Points use fixed probability bins and at least 200 opportunities; they are descriptive, without independent-ball confidence bars.

## 7. Does this identify genuine improvement?

![Environment](../figures/changing_environment.png)

The league's scoring environment changes materially. A whole-career average may lag a league-wide shift even if the player's relative skill has not changed. Context-adjusted log loss is 0.517997 with expanding ability and 0.517950 when the recency signal is added. The interval for that incremental change includes zero.

Recency's simple-model advantage varies by phase: log-loss differences versus expanding are −0.003938 in the powerplay, +0.000270 in the middle overs and −0.002069 at the death. Standard-conditions-only results retain the main advantage (−0.001633, interval [−0.002301, −0.000996]). Venue and opposition breakdowns are supplied as CSV, without treating small subgroups as definitive findings.

The supported conclusion is **better simple forecasts from recent weighting in these test seasons**. The study does **not** establish a general causal or permanent increase in individual batting skill. Context, role, opposition quality, league evolution and remaining unobserved factors can all matter. A null incremental result is part of the research finding, not a reason to suppress it.

## 8. Prospective IPL 2027 application

Forecasts were generated on **12 October 2026** using observations through the available 2026 season. The selected preseason model is rolling 300 balls with a prior refitted through 2026. For time-decay comparisons, 28 March 2027 is an explicit assumed reference date, not an announced schedule; the selected rolling forecast itself does not depend on that date.

Watchlist eligibility requires ≥120 eligible balls in 2026 and ≥300 before 2026. {int(allf.eligible.sum())} players meet those history thresholds. An improver candidate has forecast boundary probability more than one percentage point above their pre-2026 shrunk historical baseline. A rebound candidate has an observed 2026 rate below baseline and a forecast above the observed 2026 rate. The labels can overlap because they answer different comparisons. A strong last-120-ball rate is provided as a separate descriptive column; candidate status does not require a strong finish.

**Potential boundary-rate improvers (largest projected changes):**

{table(improve.head(8),watchcols,wf)}

**Projected rebound candidates (ranked by forecast rate):**

{table(rebound.head(8),watchcols,wf)}

![2027 watchlists](../figures/watchlists_2027.png)

Intervals use 1,000 season-stratified innings-block resamples, preserving original window membership/weights, and shared player-level Bayesian bootstrap weights to refit population priors. They describe conditional resampling stability of the estimate, not a prediction interval for realised 2027 outcomes. Baseline and forecast are resampled together, preserving their overlapping evidence. The population prior changes between baseline and forecast; the prior mean rises by 1.04 percentage points. Do not interpret the entire forecast-minus-baseline difference as player-specific improvement.

No injury, age, other-league form, expected team, availability or squad-selection model is included. These are marginal boundary-rate forecasts under an unspecified future context mix. Intervals are marginal across players, and no multiple-testing correction is applied. Bootstrap positive fractions are stability diagnostics, not posterior probabilities of true improvement. Realised 2027 outcomes remain unknown and must be evaluated later without rewriting the dated forecast.

In the complete forecast CSV, a player with no pre-2026 IPL observations has a population-only baseline, explicitly marked in `baseline_basis`; it is not an observed personal history. Such players cannot enter the history-qualified watchlists. Stability intervals can be artificially narrow when very few innings are available, another reason not to read them as full player-ability uncertainty.

## 9. Real-player illustrations

![Player trajectories](../figures/player_trajectories.png)

Cases are selected after analysis to illustrate contrasting histories. Lines are pre-match forecasts; dots are realised innings rates, with larger dots for more opportunities. A spectacular innings can be noisy, and a forecast responds only after it has happened. These charts explain the mechanism and are not additional independent validation.

## 10. Reproduce and inspect

See the repository README for one-command reproduction. `config/protocol.json` defines all grids, thresholds and seeds. Raw archive SHA-256: `{audit['archive_sha256']}`. CSVs contain priors, candidate selection, metrics, loss differences, subgroups, calibration and every prospective forecast. Generated Parquet prediction tables are local intermediates. Tests check event accounting, leakage, exact history calculations, scoring and model recovery on simulated data.

Useful libraries: pandas and NumPy for chronological tables, SciPy for empirical Bayes estimation, scikit-learn for contextual models, Matplotlib for figures, and PyArrow for typed data storage. Computation was assisted by Codex; code, assumptions, limitations and generated results are exposed for scrutiny.

The connection to quantitative research is noisy signal estimation, shrinkage, choosing adaptation speeds, avoiding look-ahead bias and judging held-out evidence. No claim about profitable trading follows from cricket prediction scores.

## Sources

- [Brown (2008), In-season prediction of batting averages](https://www.stat.berkeley.edu/users/aldous/157/Papers/brown_batting.pdf), DOI 10.1214/07-AOAS138.
- [Cricsheet IPL data](https://cricsheet.org/downloads/), [JSON field definitions](https://cricsheet.org/format/json/), [licensing](https://cricsheet.org/licensing/).
- This study's [frozen protocol](PROTOCOL.md), [machine-readable configuration](../config/protocol.json), and [data attribution](../DATA_LICENSE.md).
'''
    (ROOT/'docs/RESEARCH_REPORT.md').write_text(report)
    summary=f'''# IPL form forecasting

### How quickly should we believe a batter has improved?

A Python research project adapting Brown's baseball forecasting question to IPL boundary probabilities, then testing whether forecasts improve when older observations receive less weight.

**Finding:** a 365-day decay model selected on 2024 improves delivery-weighted log loss by **{reduction:.2f}%** over all-history shrinkage on 2025–2026. The incremental gain after context adjustment is uncertain. Recent weighting improves simple predictions here; that is not proof of permanent player improvement.

![Model comparison](figures/model_comparison.png)

**[Read the full research report](docs/RESEARCH_REPORT.md)** · [Results notebook](02_research_results.ipynb) · [2027 improver candidates](results/watchlist_improvers.csv) · [2027 rebound candidates](results/watchlist_rebounds.csv) · [All 2027 forecasts](results/forecast_2027_all.csv)

## What is here

- A checksum-pinned Cricsheet archive and tested cleaning pipeline: **1,243 matches, 295,732 recorded deliveries, 284,465 modelling opportunities**, 2008–2026.
- An early-to-late-season beta-binomial shrinkage experiment inspired by Brown (2008).
- Past-only expanding, current-season, rolling and exponential histories spanning multiple seasons.
- Separate within-season and preseason model selection on 2024, followed by evaluation on 2025–2026.
- Log loss, Brier score, calibration, equal-player metrics, paired date-bootstrap uncertainty and context/phase/venue/opposition checks.
- Dated 2027 boundary-rate forecasts with block-resampling stability intervals; six figure sets and two executed notebooks.

## Key results

| Experiment | Selected model | Test log loss | Expanding reference | What it supports |
|---|---|---:|---:|---|
| Updated before each match date | 365-day exponential decay | {sb.loc['ew_days_365','log_loss']:.6f} | {sb.loc['expanding','log_loss']:.6f} | Modest improvement for opportunity-weighted forecasts |
| Frozen before the season | Last 300 balls + shrinkage | {pb.loc['rolling_300','log_loss']:.6f} | {pb.loc['expanding','log_loss']:.6f} | Some recent information persists across the off-season |
| Recency added to contextual model | Context + recent-history signal | {ctx.set_index('model').loc['context_plus_recent','log_loss']:.6f} | {ctx.set_index('model').loc['context_expanding','log_loss']:.6f} | Difference is uncertain; not evidence of isolated skill change |

These horizons have different information sets. Compare models within each experiment. Equal-player results do not always favour the selected method. Brown's exact baseball numbers and full estimator suite are not replicated.

![Uncertainty](figures/uncertainty.png)

## Reproduce

Python 3.13 is the tested interpreter. Create a local environment, install the portable dependencies, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
python -m pytest -q
```

On Windows activate with `.venv\\Scripts\\activate`. The raw 5 MB archive is included under its data license, so analysis does not need a live download. The run rebuilds cleaned tables, experiments, uncertainty, figures, reports and executed notebooks. It can take several minutes, particularly the 1,000 watchlist prior resamples. `requirements.lock.txt` records the tested macOS environment; use `requirements.txt` across platforms. No API key is required for research reproduction.

Open `docs/research.html` locally for the self-contained report viewer and filterable forecast table. GitHub renders the Markdown report and notebooks directly; it does not run the HTML viewer in a repository file preview.

## Project map

| Path | Purpose |
|---|---|
| `pipeline.py` | Validate and flatten Cricsheet; clean outcomes and build counts |
| `models.py` | Prior estimation, posterior means, historical features and scoring |
| `research.py` | Brown-inspired, sequential and preseason experiments |
| `context_analysis.py` | Past-trained contextual models and subgroup checks |
| `watchlists.py` | Dated forecasts and resampling intervals |
| `visualise.py`, `build_report.py` | Figures and research write-up |
| `01_data_walkthrough.ipynb` | Explain the data and shrinkage |
| `02_research_results.ipynb` | Inspect actual model choices, results and watchlists |
| `config/protocol.json`, `docs/PROTOCOL.md` | Frozen study choices |
| `results/`, `figures/`, `tests/` | Evidence, visualisations and verification |
| `docs/INTERVIEW_GUIDE.md` | Explain the question, findings and limitations |

## Outcome and interpretation

A boundary is a batter four/six unless explicitly marked non-boundary. Primary opportunities exclude wides, no-balls and Super Overs; byes and leg-byes on legal balls remain zero outcomes. Player IDs prevent name-matching errors. Historical features exclude all same-date outcomes. Match winners and final revised chase targets are never forecast features.

The 2027 forecasts are dated **12 October 2026**, use data through **31 May 2026**, and are conditional on a player batting. They do not predict injuries, selection, overall batting quality or guaranteed improvement. Watchlist intervals measure resampling stability, not the full uncertainty of realised next-season performance. See the full report for source coverage, dependence, calibration, era changes and selection limitations.

## References and attribution

[Brown (2008)](https://www.stat.berkeley.edu/users/aldous/157/Papers/brown_batting.pdf) supplies the forecasting/shrinkage motivation. [Cricsheet](https://cricsheet.org/) supplies the data, licensed under the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/); see [DATA_LICENSE.md](DATA_LICENSE.md). Code is MIT licensed. Data cleaning, aggregation and analysis are this project's transformations; Cricsheet does not endorse the results. Built with Codex assistance, with reproducible code and checks provided.
'''
    (ROOT/'README.md').write_text(summary)
    # Offline viewer embeds every image and the forecast table; no external services or scripts.
    images=''.join(f'<figure><img src="data:image/png;base64,{base64.b64encode((ROOT/"figures"/f"{name}.png").read_bytes()).decode()}" alt="{title}"><figcaption>{title}</figcaption></figure>' for name,title in [('model_comparison','Model comparison'),('uncertainty','Paired loss differences'),('calibration','Calibration'),('changing_environment','Changing league conditions'),('player_trajectories','Player case studies'),('watchlists_2027','2027 watchlists')])
    table_rows=''.join('<tr data-kind="'+('improver ' if r.potential_improver else '')+('rebound' if r.rebound_candidate else '')+'"><td>'+html.escape(r.player)+f'</td><td>{r.latest_N}</td><td>{r.baseline:.1%}</td><td>{r.latest_rate:.1%}</td><td>{r.forecast:.1%}</td><td>{r.forecast_low:.1%}–{r.forecast_high:.1%}</td><td>{100*r.change_from_baseline:+.1f} pp</td><td>'+('Yes' if r.eligible else 'Limited history')+'</td></tr>' for r in allf.itertuples())
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IPL form forecasting — research results</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#173142;background:#f3f7f8;margin:0}main{max-width:1180px;margin:auto;padding:40px 24px}h1{font-size:clamp(30px,4vw,52px);line-height:1.15}p{max-width:900px}figure{margin:36px 0;background:#fff;padding:16px;border-radius:12px}img{width:100%;height:auto}figcaption{color:#526b76;font-size:14px}.tag{color:#14776c;font-weight:650;letter-spacing:.1em;font-size:13px}input,select{padding:12px;border:1px solid #b6c9ce;border-radius:6px;font:inherit;margin:8px 12px 12px 0}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #dae5e8}th{background:#164b52;color:white;position:sticky;top:0}.scroll{overflow:auto;max-height:640px}a{color:#10776c}.note{border-left:4px solid #cc8538;padding:16px;background:#fff6e7}</style><main><div class="tag">CRICKET × STATISTICAL FORECASTING · 12 OCTOBER 2026</div><h1>How much history should a forecast remember?</h1><p>Recent weighting improves simple boundary forecasts in the held-out 2025–2026 seasons. Once venue, opposition, phase, pre-delivery state and trend are included, its incremental value is uncertain. This is evidence about prediction, not proof of permanent player improvement.</p><p><a href="RESEARCH_REPORT.md">Full methods and findings</a> · <a href="../02_research_results.ipynb">Results notebook</a> · <a href="../results/forecast_2027_all.csv">Download forecasts</a></p>'''+images+'''<h2>Explore the dated 2027 forecasts</h2><p class="note">Boundary rate only. Intervals show conditional resampling stability, not a realised-season prediction range. Squad selection, injuries and future conditions are not forecast. A potential improver is a forecast more than 1 percentage point above the pre-2026 baseline; rebound means forecast above an underperforming 2026 rate. Labels may overlap.</p><label>Player <input id="search" placeholder="Search a player" aria-label="Search a player"></label><label>Watchlist <select id="kind" aria-label="Watchlist"><option value="">All 2026 batters</option value="improver">Potential improvers</option><option value="rebound">Rebound candidates</option></select></label><p id="count" aria-live="polite"></p><div class="scroll"><table><thead><tr><th>Player</th><th>2026 opportunities</th><th>Baseline</th><th>Observed 2026</th><th>2027 forecast</th><th>95% stability interval</th><th>Change</th><th>Watchlist history</th></tr></thead><tbody>'''+table_rows+'''</tbody></table></div><p>Data: <a href="https://cricsheet.org/">Cricsheet</a>, Open Data Commons Attribution License 1.0. Modified through cleaning and statistical analysis. See DATA_LICENSE.md in the repository. No external scripts or trackers.</p></main><script>const search=document.querySelector('#search'),kind=document.querySelector('#kind');function filter(){let n=0;document.querySelectorAll('tbody tr').forEach(r=>{const show=r.cells[0].textContent.toLowerCase().includes(search.value.toLowerCase())&&(!kind.value||r.dataset.kind.includes(kind.value));r.hidden=!show;if(show)n++});document.querySelector('#count').textContent=n+(n===1?' player shown':' players shown')}search.addEventListener('input',filter);kind.addEventListener('change',filter);filter();</script></html>'''
    (ROOT/'docs/research.html').write_text(doc)
    print('Built README, full report and offline interactive report viewer')

if __name__=='__main__': main()
