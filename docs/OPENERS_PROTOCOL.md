# Opening-batter follow-up: analysis decisions

13 October 2026. These decisions are recorded before fitting the new opener experiment. The original all-batter results, including 2025–2026, have already been seen and motivated this question. This is an **exploratory follow-up**, not a new untouched test set or an amendment to the original frozen protocol.

## Question and scope

Repeat experiment 1, early-to-late-season population versus raw versus beta-binomial forecasts, using opening-batter innings only. Test whether this changes the fitted prior width and predictive performance. Leave sequential, preseason, context and watchlist experiments and their outputs untouched. Write a separate evidence-based plan for those stages.

## Data and timing

- Use the same checksum-pinned archive, legal-delivery boundary outcome, player registry IDs, season dates and calendar midpoints as the original experiment. Include playoffs in the primary analysis.
- An opener is either player present on the **first recorded delivery event**, including an illegal first ball, of each ordinary innings. Check this directly against `batting_entry_order <= 2`. Opening-pair order does not matter. Do not define openers from who eventually faces the first two legal balls, or from future career roles.
- Include every eligible delivery faced by that player **in that opening innings**, including middle/death overs. Exclude their non-opening innings. A player can change role between innings. Do not filter early training records using later appearances or minimum later exposure.
- Estimate each season's prior using all early-half opener records with N > 0. No late outcomes enter the fit, including those of new late-half openers. Freeze predictions at the cutoff. Evaluation is conditional on actually opening, not a forecast of selection or role.
- Report all late opening opportunities, the original-style N >= 11 in both opening halves subset, and an early-history-only subset. The N >= 11 subset is retrospective evaluation, never a training filter. Report entrants and role-switch coverage explicitly.

## Models and fair comparisons

Primary: pooled early-opener boundary probability; raw early-opening B/N (pooled opener fallback when N=0); empirical-Bayes Beta posterior from early-opening records (fitted Beta mean when N=0, matching original code). Distinguish pooled ball-weighted probability from the fitted latent-player prior mean.

On **the identical late-opening observations**, also score:

1. Original all-batter population and empirical-Bayes forecasts, using their original all-role early records.
2. The all-batter prior updated with opening-only individual records, isolating the prior change from history filtering.
3. The opener prior mean with no individual update, isolating individualisation from the population baseline's different centre.
4. Jeffreys-smoothed raw rates, to check raw-zero clipping dependence.
5. Leave-one-player-out opener priors: exclude the target's early record from its prior fit, then update with its early-opening counts. This tests sensitivity to empirical Bayes reusing an individual's training record when the cohort is small.

The original all-batter scores remain a historical reference; differences between scores on different target samples are not claimed as improvements.

## Prior-width hypothesis and sensitivity checks

Report alpha, beta, mean, concentration k, latent-probability standard deviation sqrt(mu*(1-mu)/(k+1)), early cohort size, opportunities, optimisation boundary flags and actual player-data weights N/(N+k). Population sample size is **not** the prior's pseudo-count k. Fewer players need not imply a wider latent-player prior, and openers need not be independent or equally situated.

For a direct width check, keep each fitted opener mean fixed and multiply k by 0.1, 0.25, 0.5, 1, 2 or 4. Report the entire grid. Choose a single multiplier using 2024 log loss only and carry it unchanged into 2025–2026. No selection using the latter results; all are exploratory because the original outcomes motivated the follow-up.

Refit the three primary models in four separately labelled sensitivities: a shared match-count half (date cutoff nearest half the matches, earliest tie); league-stage-only matches (same original calendar cutoff); standard-conditions-only innings (same cutoff); nominal powerplay-only deliveries of openers (same cutoff). These change training and scoring samples together, so compare models within each scenario. They are diagnostics, not opportunities to choose a winning headline.

## Scores, uncertainty and reproducibility

Report development (2008–2023), 2024 selection and 2025–2026 follow-up evaluation separately, plus per-season metrics. Include delivery log loss/Brier, equal-player scores, player-season rate MSE, and the original approximate arcsine noise-adjusted score only on the >=11 subset. Clip raw forecasts to 10^-6 for log loss; use **unclipped** forecasts for Brier/MSE. This corrects a tiny shared-scoring discrepancy locally without altering other experiments.

Use 2,000 paired date-block bootstrap resamples within each evaluation season for log-loss and Brier differences, matching observation keys exactly. Intervals condition on fitted priors and the observed seasons, omit refitting/model-selection/regime uncertainty, and are not multiple-comparison adjusted. Report season-specific direction and calibration bins; do not treat deliveries as independent evidence.

Publish isolated code, meaningful leakage/role/scoring tests, machine-readable predictions/priors/coverage/scores, figures, a report and a staged future-work plan. Preserve original outputs byte-for-byte. Record source and protocol hashes, seed, reproduction command and verification results.
