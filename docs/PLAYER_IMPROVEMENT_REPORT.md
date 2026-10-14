# Which individual players improved in 2026?

Exploratory player audit, 14 October 2026. The latest season in the pinned IPL archive is 2026, ending 31 May. Target: sixes per eligible legal delivery, top-seven batting positions; the definition is unchanged from the main project.

## What can be said confidently

**No player has a clearly positive context-adjusted 95% change interval in this audit. None passes the 5% false-discovery screen either.** That does not mean no one improved; it means this sample and these checks cannot identify a player as a statistically established, persistent ability improver.

**KL Rahul and Ishan Kishan are the clearest descriptive candidates for sustained increases; Rohit Sharma is promising on a smaller sample. Sai Sudharsan shows a more modest rise that weakened later in the season.** This is a qualitative shortlist based on magnitude, persistence, sample size and recent history, not a new statistical ranking or a list of confirmed improvers. The complete 40-player results include other increases, declines and inconclusive records.

We screened all 110 active players and retained the 40 with at least 120 eligible 2026 balls and 300 eligible balls in 2023–2025. Exclusions are based only on opportunity counts, never on whether their rates rose. There are 36,561 balls in the eligible comparison, from 293 matches. All eligible players are published, avoiding a search limited to the earlier in-form watchlist.

## Sustained-increase candidates

| Player | 2023–25 | 2025 | 2026 sixes / balls | 2026 rate | Early 2026 | Late 2026 | Without highest-six innings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KL Rahul | 4.4% | 5.8% | 31/339 | 9.1% | 9.9% | 8.2% | 8.1% |
| Ishan Kishan | 6.4% | 6.5% | 32/330 | 9.7% | 10.1% | 9.3% | 9.1% |
| Rohit Sharma | 7.6% | 7.9% | 21/179 | 11.7% | 9.8% | 13.4% | 10.4% |
| Sai Sudharsan | 4.3% | 4.3% | 30/454 | 6.6% | 8.7% | 5.0% | 5.6% |

The 2023–25 column pools observed balls in those years. It does not require playing in all three seasons. Early/late use the existing shared calendar midpoint, not a favourable player-specific cut. Removing the highest-six innings is a descriptive fragility check, not an additional significance test.

**KL Rahul:** the annual sequence is 1.7%, 4.7%, 5.8%, 9.1% in 2023–2026. His 2026 record is 31 sixes in 339 balls over 14 innings. Both halves exceed his prior level, and removing his highest-six innings still leaves 8.1%. This is a more persuasive sustained-change pattern than a single hot innings. His old record includes other roles, but the same-role opening comparison also rises: 30/762 = 3.9% in 2023–25 versus 31/339 = 9.1% in 2026. The raw innings-bootstrap rise versus the pooled three-year reference has a 95% range of +1.50 to +7.35 percentage points, before adjustment for searching players. The comparison against 2025 alone is less decisive, and the context-adjusted interval crosses zero. Treat him as a leading candidate, not a proven improver.

**Ishan Kishan:** 32 sixes in 330 balls over 15 innings, versus 49/763 in 2023–25. Early and late rates are similar at 10.1% and 9.3%; excluding his highest-six innings leaves 9.1%. This persistence is encouraging. His role changed: all 2026 balls here are at No. 3, while most older balls were as an opener. The same-role baseline is 14/205 = 6.8%, still below 9.7%, but with fewer historical observations. Adjusted evidence remains uncertain. This is a candidate for sustained higher six-hitting, not proof of a technique or underlying ability change.

**Rohit Sharma:** 21/179 = 11.7%, compared with 7.6% in 2023–25 and 7.9% in 2025. Both halves are higher and the rate remains 10.4% after excluding his highest-six innings. Almost all historical and current balls are as an opener, which helps comparability. However, the latest sample is only nine innings, so uncertainty is substantial.

**Sai Sudharsan:** 30/454 = 6.6%, after approximately 4.3% in each of 2023–2025. There is substantial latest-season exposure, but the rate fell from 8.7% early to 5.0% late. His historical opening-only rate was already 4.9%, compared with 6.6% opening in 2026. This is a plausible modest increase, less convincing as a strong sustained new level than the season average alone suggests.

## Rebounds are not necessarily newly acquired ability

- **Rajat Patidar:** 6.5% in 2025 to 16.2% in 2026, a large rebound. But he had already reached 14.8% in 2024. Both 2026 halves exceed the pooled historical rate; this is a strong return to high six-hitting, with uncertain evidence of an unprecedented underlying level.
- **Nitish Kumar Reddy:** 2.6% in 2025 to 11.9% in 2026, versus 9.9% in 2024. Again, a clear observed rebound from a poor year, but a weaker case for a newly acquired ability. The late-half sample is only 56 balls.
- **Abhishek Sharma:** 12.4% in 2025 to 15.6% in 2026, below his 17.4% in 2024. He remains a high six-rate player; “improved on 2025” is different from “established a new level”.
- **Ajinkya Rahane:** 7.6% in 2025 versus 7.3% in 2026. The earlier in-form screen reflected a forecast above a much longer-term reference; it was not evidence that he improved during the latest season.
- **Travis Head:** 6.6% to 9.6%, but 10.9% in 2024. The same rebound distinction applies.

Newer players with limited prior exposure are listed as insufficient-history cases, not dismissed as unimproved. For example, V Suryavanshi has a striking 72/326 latest-season record but only 121 earlier balls in this comparison. That is too little historical evidence for this audit's established-player change screen.

## What the stronger checks say

We use an exploratory linear probability regression to estimate player-specific 2026 changes while accounting for role–phase combinations, venue, opposition and innings. It includes a separate historical level for each player and a separate 2026 change for each player. Uncertainty keeps deliveries from the same match together.

| Player | Adjusted rise versus 2023–25 | 95% match-cluster interval |
| --- | --- | --- |
| KL Rahul | +2.83 percentage points | -0.29 to +5.95 points |
| Ishan Kishan | +3.26 percentage points | -0.56 to +7.07 points |
| Rohit Sharma | +3.77 percentage points | -0.61 to +8.15 points |
| Sai Sudharsan | +2.61 percentage points | -0.22 to +5.45 points |

These intervals all include zero. In fact, **no positive adjusted player change has an individual two-sided 95% interval entirely above zero even before multiple-search correction**. Benjamini–Hochberg adjustment across the 40 players strengthens the caution; no adjusted rise survives. A second comparison subtracts the 2026-ball-weighted adjusted change among the other 39 players; no player clears that screen either.

The analysis adjusts observed associations; it does not prove causes. Context effects are assumed common across players and periods. Bowler quality, health, tactical intent and every possible dependency are not modelled. The linear probability model is an explanatory diagnostic, not a replacement for the main probability forecaster; its adjusted differences are not guarantees of future rates.

This is not the same check as the main project's context test. The main test asked whether a recent-history feature improves future predictions on average. This audit asks which players' latest-season rates differ from their earlier rates after adjustment. Both remain limited by sample size and observational data.

## Reproducibility and interpretation

The [audit protocol](PLAYER_IMPROVEMENT_PROTOCOL.md) was written before fitting these comparisons, but follows prior exploration of the same data; it is not a preregistered confirmatory study. Run `python -m sixes.player_improvement` to rebuild the tables. The original forecasting experiments and their saved results remain unchanged.

- [All 40 player comparisons](../results/player_improvement/player_changes.csv)
- [Annual counts and rates](../results/player_improvement/annual_rates.csv)
- [All 110 players and eligibility](../results/player_improvement/coverage.csv)
- [Role and phase exposure](../results/player_improvement/role_phase_exposure.csv)
- [Context-adjusted differences](../results/player_improvement/context_adjusted_changes.csv)
- [Model assumptions and rank audit](../results/player_improvement/model_audit.json)
- [Count and coverage verification](../results/player_improvement/audit.json)

The complete test suite now passes 41 tests, including new checks of the multiple-comparison calculation, innings resampling and recovery of known player-change contrasts. This report's numbers describe six-hitting frequency, not all-round batting performance or proven permanent talent changes.
