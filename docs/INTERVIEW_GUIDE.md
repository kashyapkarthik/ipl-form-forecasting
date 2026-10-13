# Explaining this project in an interview

## A 45-second explanation

I investigated how much of a cricket batter's history should influence their next forecast. I used Cricsheet IPL deliveries to model boundary probability, adapted Brown's empirical-Bayes forecasting idea, and compared full-history estimates with rolling and exponentially weighted histories. I selected settings on 2024 and evaluated them on 2025–2026. Recent weighting modestly improved opportunity-weighted forecasts, but its extra value was uncertain after contextual adjustment. That distinction matters: useful prediction does not prove a permanent change in skill. I also produced dated 2027 boundary-rate watchlists with uncertainty estimates.

## What I must be able to explain

- **Target:** a batter boundary on a legal delivery outside Super Overs. A total of four may be byes; a batter four may be explicitly non-boundary. Keep all eligible zeroes in the denominator.
- **Shrinkage:** combine a player's observed rate with the population rate. Small N gets less trust. The prior's concentration determines how much evidence is needed to move away from the population.
- **Empirical Bayes:** fit the population prior from training records by maximising the marginal likelihood. It is not a handpicked league mean or a fully Bayesian treatment of every parameter.
- **History:** expanding uses all past IPL balls; rolling uses only the most recent K; exponential weighting decays smoothly. Calendar decay also acts during inactivity. All methods share the same annual prior.
- **Leakage:** forecasts exclude the current date. Prior fitting, contextual encoders and model selection use past data only. Outcome fields and final revised chase targets are not predictors.
- **Two horizons:** a forecast updated after matches answers a different question from a forecast frozen before a season. Each has separate validation-based selection.
- **Evaluation:** log loss penalises confident errors; Brier score is mean squared probability error. Calibration compares predicted probabilities with observed frequencies. Equal-player and opportunity-weighted scoring answer different questions.
- **Uncertainty:** resample match dates for paired method comparisons, preserving shared conditions within dates. Player forecast intervals resample innings and refit population priors. They do not capture every source of future uncertainty.

## Findings to quote accurately

1. The dataset contains 295,732 recorded delivery events and 284,465 primary modelling opportunities, from 1,243 matches. Do not repeat an earlier unverified count.
2. A 365-day half-life chosen on 2024 reduces sequential test log loss from 0.525423 to 0.523852: about 0.30%, a modest gain.
3. A 300-ball window chosen on preseason validation reduces test log loss from 0.527164 to 0.525908: about 0.24%.
4. Shrinkage beats raw within-season rates, but its gain over a population-only forecast is uncertain.
5. The context model's recent-history increment has a confidence interval crossing zero. I cannot claim to have isolated genuine skill improvement.
6. The simple model's recency advantage varies by phase and does not persist under every weighting of players. This limits generalisation.

## Questions worth anticipating

**Why boundary probability rather than cricket batting average?** It creates a binary trial comparable to baseball hit probability. It measures one aspect of batting; dismissals, rotation of strike and innings value need additional models.

**Did you replicate Brown?** I adapted his earlier-to-later forecasting and shrinkage question using beta-binomial estimation. I did not reproduce his baseball numerical results or full transformed-normal estimator suite. He also studied streakiness.

**Does exponential weighting prove form exists?** No. It can follow league-wide scoring changes or changes in role/context. Predictive gains show useful recent information under the tested setup.

**Why not choose the best model on the test set?** That would use the answers to choose the forecast. The preseason exponential model happened to score slightly better than the selected rolling model, but the published choice remains the one selected on 2024.

**What about no history or no appearance next season?** No history receives the population prior. The study scores conditional on actual batting opportunities; it does not predict whether a player is selected. Watchlists require adequate observed history but are not confirmed rosters.

**What does a watchlist interval mean?** Stability of the estimated forecast under a specified resampling scheme. It is not a guarantee or a prediction interval for the player's eventual season rate. Overlapping histories are resampled together when assessing changes.

**What would you improve next?** More seasons of untouched evaluation, role and individual-bowler controls, other T20 leagues, injury/age data, better calibration, and a hierarchical time-varying model with fuller uncertainty. None is necessary to obscure a clear baseline.

**Where is the finance connection?** Estimating noisy signals, shrinking unstable estimates, selecting how fast to update, avoiding look-ahead bias and judging out-of-sample evidence. Cricket forecast gains do not establish a tradable financial edge.

## Authorship and chronology

The implementation was developed with Codex assistance. Review the code and notebook until you can explain and modify the core calculations yourself. Be accurate about when the project was completed and your contribution. Do not present a generated result as something you had already established before running the study.

## Evidence-backed CV wording

- Built a reproducible Python/pandas pipeline for 1,243 Cricsheet IPL matches, processing 295,732 delivery events into 284,465 eligible boundary outcomes.
- Adapted Brown's empirical-Bayes forecasting question using beta-binomial shrinkage; compared early-season estimates with held-out later-season performance.
- Compared expanding, rolling and exponentially weighted histories across seasons, with settings selected on 2024 and tested on 2025–2026; selected calendar decay reduced opportunity-weighted log loss by approximately 0.30% versus full-history shrinkage.
- Assessed log loss, Brier score, calibration, bootstrap uncertainty and contextual variation; produced player visualisations and dated 2027 boundary-rate watchlists, while reporting that incremental recency value after context adjustment was uncertain.
