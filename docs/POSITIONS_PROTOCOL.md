# Separate batting positions 1–7: experiment-1 extension

13 October 2026. Recorded before the new fits. This is exploratory: original and opener evaluation results already motivated this request. No later experiment or watchlist will be rerun. Prior results, including the opener study, remain intact.

## Target and timing

Treat each observed batting position 1–7 as a separate early-to-late-season dataset. Group by season, position and registry player ID. Fit each prior on all positive-exposure early records in that position; never filter training players by late appearances. Update each player's prior with their early record at that position only. A player appearing in multiple positions has multiple position-specific records. Late-only entrants receive the appropriate population/prior forecast, not a fabricated personal rate. Positions 8+ are outside the target; report excluded exposure.

Use the original legal-delivery boundary definition and shared calendar half cutoffs. Forecasts stay fixed during the late half. Actual later position determines the conditional prediction evaluated, not a prediction of batting order or selection made at the season midpoint. No late outcomes enter priors or personal histories.

The source's first observed striker is rank 1 and the initial non-striker rank 2; these labels are **not independently verified scorecard positions**. Later ranks are first observed arrival order from all raw events, before filtering illegal balls. Keep 1 and 2 separate in the requested main analysis, disclose the convention, and repeat with the two openers combined while leaving 3–7 separate. Also repeat the seven-position fit using common match-count halves that never split a date.

## Models

For every season and position compare position population (pooled B/N), raw player-position rate (population fallback), and Beta-binomial empirical-Bayes posterior mean (fitted prior-mean fallback). Concentration is learned, never forced wider. The marginal-likelihood fit includes the target's early record in the primary empirical prior, matching the prior experiments. Separately fit leave-one-player-out priors for the main scenario to check sensitivity to that reuse.

Rescore the original all-batter population and shrinkage forecasts on exactly the same position 1–7 deliveries. Include Jeffreys-smoothed raw rates and four explanation checks: fitted position mean without individual updates; all-batter prior with position-only history; position mean with all-batter concentration; all-batter mean with position concentration. These diagnose population centre, strength and history fragmentation without choosing a winner after seeing evaluation scores. Reference priors always use their own early sample under the applicable split; match-count sensitivity is a refit reference, not the original calendar forecast.

Before inspecting the new performance results, add a shared positions-1–7 pooled/shrinkage benchmark. It fits one prior and aggregates each player's early observations across ranks 1–7. This separates the effect of excluding ranks 8+ from the effect of fitting each of the seven positions separately.

## Reporting

Report aggregate and each-position results, plus each evaluation season. Primary scores: delivery-weighted log loss and unclipped Brier. Supplement with equal-player scores, player-season-position rate MSE and the original >=11-in-each-half subset; the latter is retrospective evaluation, never training eligibility. Keep raw log-loss clipping at 10^-6. Publish the entire model table, not just favourable positions. Add early-history-only and new-position-history coverage.

Report learned means, alpha, beta, concentration, prior SD, cohort sizes, personal sample sizes, boundary flags and exact fallback usage. Keep near-pooled/boundary solutions visible rather than forcing weaker shrinkage. A missing or fewer-than-two-player fit would use the all-batter prior as an explicit flagged fallback; the pre-fit coverage audit found at least 13 early players per main position-season, so this is not expected in the primary fits.

Use 2,000 paired date-block resamples within evaluation season for the aggregate and each-position comparisons. Resample common dates jointly across positions when combining them, preserving same-date dependence. Include an approximate simultaneous interval across the seven main position-specific shrinkage-versus-population differences using the maximum absolute centred bootstrap deviation; distinguish it from individual 95% intervals. These conditional intervals omit model refitting, some serial/player dependence and future regime uncertainty. No multiplicity-adjusted claims across all exploratory models or scenarios.

## Checks and outputs

Independently reconstruct raw-event positions, verify no excluded balls or ranks enter counts, check position switches and cold starts, perturb late outcomes to test forecast invariance, and assert identical scored keys across models. Verify primary original-reference predictions and the combined-opening prior/forecast against committed studies. Publish wide player-season-position prediction CSVs, priors, coverage, metrics, uncertainty, figures and a readable report, with a standalone reproduction command. Preserve all pre-existing tracked research files byte-for-byte except an additive README index entry. Update GitHub after verification; no unrelated research changes.
