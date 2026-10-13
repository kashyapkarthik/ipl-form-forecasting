# Alternative batting outcomes: protocol frozen before fitting

Date: 13 October 2026. Baseline: `69bfaacec2951bc99822ef70bc5ea5db3907b1a4`.

## Question

Does individual early batting history add more predictive information for sixes, dot balls, or innings milestones than for boundaries? Does a longer history help with rare innings outcomes? This extends experiment 1 only; the existing experiments, data, reports and forecasts remain untouched.

## Fixed candidates and denominators

- Boundary: four or six off the bat, excluding the source's non-boundary flag, per eligible legal delivery, as before.
- Six: six off the bat, excluding non-boundaries, per eligible legal delivery.
- Dot: zero batter runs per eligible legal delivery. Legal byes and leg-byes count as zero batter runs; this is not necessarily zero team runs.
- Score 30+: at least 30 total batter runs in an observed ordinary innings.
- Score 50+: at least 50 total batter runs in an observed ordinary innings, including hundreds.

Milestone totals include batter runs on no-balls, unlike the existing legal-delivery run aggregate. Observed batting appearances are players recorded as striker or non-striker on any delivery, including zero-legal-ball appearances; players who never bat are excluded. A player dismissed before ever appearing in either role would not be captured: the reconstruction will audit wicket records for that exception. Super Overs are excluded throughout. Innings ended by a completed chase, rain or a not-out finish remain observed milestone failures if below the threshold; the estimand is the chance of a milestone in an actual appearance, not an unlimited-opportunity ability parameter.

## Cohort, history and forecasts

Use observed innings positions 1–7, with separate priors for every position. Initial striker/non-striker are the source convention for slots 1/2. No permanent player-role assignment or future outcomes determine membership. Restrict both training and evaluation to these roles. Reuse the existing within-season calendar-midpoint early/late split.

Primary history: the same player's same-position early observations in the current season. Secondary fixed history: the three previous complete seasons plus current early observations, at the same position. Fit a Beta-binomial population distribution to one aggregated record per player, separately for outcome, forecast season, position and history window. All forecasts freeze at the early/late cutoff. Compare fitted empirical Bayes, pooled population, unsmoothed raw and Jeffreys `(B+0.5)/(N+1)`; no-history raw/Jeffreys use pooled population. Also score the fitted prior mean without personal data to isolate the contribution of updating. For long histories, show current-season population as an additional comparator to expose stale-prior effects. A prior fitting cell with fewer than two training players falls back to its same-outcome, same-window top-seven prior and is flagged.

Prior strength is estimated using the existing fitter, without forcing wider priors. Report latent prior SD, concentration, exposure, personal-data weights and bound hits. Population rates use observed B/N, clipped only for log scoring. Score 2011 onward so both histories have three seasons available. Development: 2011–2023; validation: 2024; already-seen exploratory evaluation: 2025–2026.

## Selection and uncertainty

Report every candidate, including failures. Absolute losses across different events are not comparable: selection uses relative log-loss reduction against each outcome's matching population benchmark. Select one outcome per history window on 2024, requiring positive development skill and a 2024 advantage over Jeffreys. If no outcome qualifies, disclose that and label the leading choice weak evidence. Record selections before scoring the 2025–2026 forecasts. Brier skill, yearly effects and equal-player log loss are complementary checks, not alternative selection rules.

Use 4,000 paired date-block bootstrap draws, stratified by evaluation season, and the same date draws across all ten outcome/history combinations. Individual intervals and a simultaneous family band across those ten empirical-Bayes versus population relative log-loss effects will be reported. Resampling dates preserves dependence within a match/day but does not account for fitted-prior uncertainty, dependence across dates for the same player, dataset-driven study design or future season shifts. Add a player-cluster sensitivity for the two validation-selected candidates (resample whole player trajectories) and report a combined-openers sensitivity for the same candidates. These do not supply fresh confirmation.

## Checks and scope

Audit milestone construction against raw match JSON totals and wicket-only appearances; verify no future leakage, identical comparison targets, position-history isolation, no-ball runs, legal eligibility, milestone nesting and zero-ball appearances. Reproduce the existing position study's boundary scores exactly. Record input/output hashes and protect all pre-existing tracked files except additive README navigation. Update GitHub with code, protocol, evidence and a readable report. Any recommendation for later experiments is a plan only, not an implementation.
