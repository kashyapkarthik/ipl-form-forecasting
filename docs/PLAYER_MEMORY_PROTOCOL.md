# Shrinkage and memory sensitivity of individual-player changes

14 October 2026. Follow-up specification written before fitting the new comparisons. This is a retrospective sensitivity analysis on already examined data, not a fresh confirmation or a search for a favourable window.

Use the same 40 players eligible for the preceding individual-player audit. Reuse the exact six/ball target and the role-specific empirical-Bayes model. Hold the 2026 prior (fitted on 2023–2025 only) fixed at both endpoints. Hold each player's 2026 role-exposure mixture fixed at both endpoints; this is retrospective standardisation, not a forecast that knew the 2026 batting order.

Primary comparison: for each memory rule, subtract its pre-2026 estimate from its end-of-2026 estimate. The memory rule, population prior and role weights are identical on both sides. This avoids comparing a recent-window endpoint to a different full-career starting model and calling the difference a new change. The early endpoint excludes all 2026 outcomes. The late endpoint includes all 2026 data through 31 May, with cutoff 1 June.

Nine primary memory rules: expanding all past same-role balls; rolling last 120, 300, 600 and 1,200; exponential ball half-lives 120, 300, 600 and 1,200. Last 30 balls is a separately labelled stress check. A separate season-record comparison shrinks 2025-only and 2026-only records with the same prior/role weights. No window is selected using the desired player result.

Also compare each endpoint with a common shrunk 2023–2025 reference, and export corresponding unshrunk rates and personal-data weights. Distinguish change during 2026 from being above an older baseline. Different windows are correlated sensitivity checks, not independent replications.

Uncertainty: 1,000 shared population-player reweightings refit the six pre-2026 role priors; independently resample player innings within each season using multinomial counts. Use the same innings resample across every endpoint and memory for a player, and the same prior draw across players. Rolling/decay weights and role mixtures remain fixed under resampling. Report paired 95% pointwise stability intervals, plus an approximate 95% simultaneous stability band using the maximum standardised centred bootstrap deviation over all 40 × 9 primary player-memory changes. These conditional bands do not establish persistent talent or account for every uncertainty in model choice/context. Keep the original context-adjusted audit as an independent caveat; no new context model is fitted here.

Report every primary method for every eligible player. Focus the explanation on the previously discussed Rahul, Kishan, Rohit and Sai Sudharsan, with Patidar and Nitish Kumar Reddy as rebound checks. Report any contradictions in that earlier shortlist. Do not widen priors or choose memory lengths to force an improvement.
