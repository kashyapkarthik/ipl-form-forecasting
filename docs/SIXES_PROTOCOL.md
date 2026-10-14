# Six-rate forecasting project: analysis freeze

14 October 2026. Frozen before fitting this full six-rate project. This follows exploratory outcome selection on the same archive, so 2025–2026 are chronologically later evaluation seasons, not pristine confirmatory data. Preserve the original studies as historical evidence. Do not force a positive recency effect or describe a player as proven to have improved or been lucky.

## Data and target

Use the checksum-pinned Cricsheet archive already in the repository. Reconstruct six = batter runs equal six and non_boundary false. Denominator: legal deliveries outside Super Overs, including legal byes/leg-byes as zero outcomes. Do not discard zeroes or classify extras/overthrows as sixes. Audit counts directly against raw JSON. Primary batting cohort: observed innings entry positions 1–7. Combine initial striker/non-striker as opening group 0 because their separate numbering is arbitrary; retain 3–7 separately. Positions 8+ remain in the coverage audit but outside this batting-focused target. A player's role is innings-specific, not a future permanent label. Use registry IDs.

## Experiment 1: early predicts late within each season

Reuse the shared calendar-date midpoint. Separately fit each season/role Beta-binomial prior from early counts, one record per player. Compare only population, raw early player rate, and posterior mean. No-history raw falls back to population; posterior uses the fitted prior mean. Evaluate all late opportunities and the separately labelled minimum 11 balls in both halves subset. Include yearly scores, Brier, log loss, calibration and date-cluster uncertainty; check a date-preserving match-count split. Fit all seasons; headline 2025–2026. Retain a bridge check against the previous combined-opener six experiment.

## Experiment 2: how much past history predicts the next innings?

Freeze each forecast before its match date; exclude all same-date outcomes. History is from the same player and role. For each forecast year/role, fit a common prior from the previous three complete seasons, aggregated once per player. Hold that prior fixed across all memory methods within the year. A cell with fewer than two training players falls back to the top-seven pooled prior and is flagged.

Compare expanding-history shrinkage; current-season shrinkage; rolling last 120/300/600/1200 balls; exponential half-lives 120/300/600/1200 balls and 90/180/365/730 days. Population and raw expanding rates are comparators. Raw last-120-ball and shrunk last-30-ball forecasts are explicit aggressive-memory diagnostics, not eligible selection candidates. Raw probabilities remain unsmoothed; clip only for log scoring and show clipping sensitivity. No additional smoothing estimators.

Select the best eligible shrunk model on 2024 delivery-weighted log loss, allowing expanding history to win. Separately record the best rolling and exponential models and the best recency candidate. Freeze all choices before scoring 2025–2026. Test selected recency versus expanding, population, raw expanding and aggressive-memory controls on identical targets. Report whether recency helps, whether retaining long history helps, and whether both propositions hold. A shorter window may win; report that honestly. Report the full validation grid, annual results and equal-player metrics.

## Preseason forecasts require a separate test

Freeze history before each historical season and select the preseason memory rule independently on 2024. Eligible preseason models exclude current-season history because there are no observations yet. Prior and model weights use only completed prior seasons. Scores are conditional on an observed batting role; do not claim to forecast team selection or batting order. Also backtest one player-level forecast using the player's previous-season role-exposure mix (latest earlier season if absent), fixed before the season; debutants use the historical population exposure mix. This is the deployment-style control for next-season player forecasts.

## Context, uncertainty and robustness

Use paired date-block bootstrap, stratified by evaluation season, with 2,000 replicates; report 95% intervals on score differences and percentage improvements. Add player-cluster sensitivity for the main recency contrast. These are conditional on fitted models; they do not capture all dependence, selection or future-regime uncertainty. Raw clipping checks use 1e-6, 1e-4 and 1e-3, leaving Brier unchanged. Show standard-conditions and equal-player sensitivity. Repeated historical evaluation does not create independent replications.

Show descriptive recency-versus-expanding differences by venue, nominal innings phase and opposition, with sample sizes and date-bootstrap uncertainty. Fit two regularised logistic contextual models on the previous five seasons: long-history ability plus venue/opposition/phase/innings/entry-role and pre-ball state; then add the selected recent-minus-long-history signal. Coefficients/encoders fit only past seasons. This has a pre-delivery information set, distinct from main pre-innings forecasts. Historical recency features used to train the 2025/2026 context model follow a separate past-only rule: select their memory from data through the preceding year. The evaluation feature uses the fixed 2024-selected recency candidate. Avoid using 2024 outcome-driven model choice to construct earlier training signals. Venue/opposition mappings are reviewed and exported; no causal claims about surfaces or teams.

## Temporary-form cases

Flag player/role/season spells in 2018–2026 using early data only: at least 300 prior-role balls, at least 60 early-role balls, early six rate at least four percentage points above the pre-season shrunk career baseline. Save all flagged spells. Then identify descriptive reversion examples with at least 60 later balls and a later rate no more than two points above that baseline. Present up to three distinct players, ordered by initial excess, and compare early raw, early-only shrinkage, history-retaining shrinkage and later rate. The reversion condition uses future outcomes solely to choose retrospective illustrations; these cases are not unbiased validation and do not prove the increase was luck. Save a calendar-season trajectory for each example.

## Dated next-season output

As of 14 October 2026, forecast 2027 six rates for every player with a top-seven batting opportunity in 2026, using the independently selected preseason model. The assumed 28 March 2027 date is for calendar decay, not a schedule claim. Aggregate role-specific forecasts with fixed 2026 role-exposure weights. Return all players, plus sufficiently observed in-form and bounce-back candidates (2026 N>=120 and pre-2026 N>=300). In-form requires latest rate above the pre-latest-season career baseline and forecast at least one percentage point above it. Bounce-back requires latest rate below the baseline and forecast above latest rate. If no candidates qualify or intervals are inconclusive, say so.

Use the same 2027 role priors and same role weights for each player's forecast and retrospective pre-2026 baseline to avoid mechanically attributing a league-prior/role-mix change to the player. This baseline is a retrospective common-prior reference, not a forecast issued in 2026. Show the current full-history forecast too, to separate memory effects from new data. Resample innings within player/season/role and reweight population players to refit priors, with 400 shared prior draws. Report 90% stability ranges for expected sixes per 100 legal balls and paired changes; these are not intervals for realised season totals or probabilities of true improvement. Also show an illustrative 100-ball outcome range adding binomial event variation. No future squad, injury or role predictions.

## Delivery and CV alignment

Publish protocol first, then implementation/evidence, then the completed report to GitHub as work progresses. Make the six-rate project the README's main story and default reproducible workflow; retain boundary studies under an explicitly historical section. Produce an executed six-rate notebook, digestible Markdown/HTML report, figures, complete dated forecasts, verification and a CV-claim mapping. Do not edit the user's CV itself. The CV framing should say cricket adaptation of Brown, not replication of every estimator or proof of permanent ability changes.
