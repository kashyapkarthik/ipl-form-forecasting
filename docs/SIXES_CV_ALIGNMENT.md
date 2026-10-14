# CV alignment and interview notes

The user's CV has not been edited. These statements describe the completed six-rate project and replace the earlier boundary-focused future-use wording when the user next revises it.

## Four evidence-backed project bullets

- Built a Python/pandas pipeline for Cricsheet IPL data, validating player identities and six definitions; analysed 267,994 eligible top-seven batting deliveries and constructed strictly past-only player-role features.
- Adapted Brown's early-to-late-season baseball forecasting question to IPL six rates, comparing empirical-Bayes shrinkage, raw player rates and position-population forecasts.
- Investigated changing player performance using expanding, rolling and exponential histories with chronological validation; a selected 300-ball shrinkage model reduced next-innings log loss by 0.24% versus full-history shrinkage in 2025–2026 historical evaluation.
- Evaluated log loss, Brier score, calibration and clustered uncertainty; examined venue, phase and opposition, visualised temporary-form cases, and produced dated 2027 forecasts while reporting weak contextual and preseason evidence.

## Explain the result without overselling it

The predictive question is how quickly to revise a forecast after a hot spell. Shrinkage protects against noisy personal estimates. Giving recent observations more influence modestly helped the simple in-season model, but the main advantage weakened under player-cluster uncertainty and disappeared after context adjustment. A different preseason model selected on 2024 did not beat full history in later evaluation. These results distinguish useful predictive information from an unsupported claim of permanent improvement.

The 0.24% is a relative reduction in log loss, not a 0.24 percentage-point increase in six probability or a classification-accuracy gain. Six rate describes power-hitting frequency, not total batting quality. 2025–2026 were already examined during exploratory outcome selection, so call them historical later-season evaluation, not a pristine final holdout. The implementation is a beta-binomial adaptation, not all Brown estimators or a numerical replication of his baseball findings.

## Be ready to demonstrate

- How a ball becomes a valid six opportunity and why zeroes must remain in the denominator.
- Why registry IDs, illegal-ball sequencing, role definitions and same-date exclusion matter.
- How estimated prior strength determines the weight on a player's own observations.
- How rolling, ball-decay and calendar-decay histories differ, including off-season inactivity.
- Why the population is role-specific and why separate roles can fragment useful player evidence.
- Why chronological model selection, raw-probability clipping and paired cluster uncertainty matter.
- Why retrospective reversion cases do not prove luck, and why the 2027 watchlists remain conditional.
- Why expected-rate uncertainty differs from the range of realised six counts in a future sample.

Start with the digestible report and executed notebook; the full CSVs and independent raw-JSON audit support the details.
