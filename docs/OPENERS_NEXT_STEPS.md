# Plan for the rest of the project after the opener experiment

**13 October 2026. Planning only: no subsequent experiment, forecast or watchlist has been changed.**

## Decision supported by this experiment

Retain opener-specific empirical-Bayes fitting as the primary opener version of experiment 1, with opener pooling alongside it. It improves evaluation log loss by 0.44% relative to the original all-batter shrinkage model on identical opening deliveries. Its advantage over opener pooling is uncertain. The opener prior is usually narrower, and every tested deliberate widening worsens 2025–2026 aggregate log loss relative to the fitted opener width. The 2024-selected widening rule fails to transfer.

The next hypothesis should therefore be **“Does a more relevant prior centre explain apparent form and improve forecasts?”**, followed by **“Does player-specific recent information help after that adjustment?”** It should not be “shrink less until the model wins.” Nor should the entire project be restricted to openers: that would discard its broader batting question without evidence that the restriction improves every horizon.

Maintain two explicit tracks: the existing all-batter reference and an opening-role extension. Compare forecasts only within a common target set and information horizon. A better score on opening deliveries than a score on all batting deliveries says nothing by itself about model quality.

## 1. Freeze the evidence and define future evaluation

**Purpose:** prevent the follow-up from turning previously observed outcomes into apparently fresh validation.

1. Preserve the original protocol, results, model selections and 12 October 2026 watchlist as version 1. Preserve this opener follow-up as version 2 evidence for experiment 1 only. Record source snapshot and code hashes.
2. Declare that 2025–2026 have now been used to generate hypotheses. Their further results are exploratory, even when features are chronologically valid. Do not relabel another already-inspected period as an untouched test.
3. For developing future candidates, use rolling chronological folds available before the evaluation period: for example, model choices assessed across 2018–2023 after training on earlier years, followed by a frozen 2024 check. These are historical development exercises, not a way to erase the present knowledge of 2025–2026.
4. Limit the candidate family in advance. Record primary loss, cohort, prediction time, treatment of role changes, opportunity weighting, uncertainty method, and a practical improvement threshold before the next run. Continue to publish equal-player scores and Brier alongside delivery log loss.
5. Reserve a genuinely future period, such as available 2027 outcomes after predictions have been frozen, for prospective confirmation. Historical 2025–2026 comparisons can still inform modelling, with honest exploratory labels.

**Deliverable:** a dated extension protocol, experiment matrix and evaluation calendar. **Gate:** no model promotion based merely on the most favourable 2025–2026 variant.

## 2. Build a role-aware prior without assuming less shrinkage

**Purpose:** target the main mechanism suggested by the opener results while avoiding unstable tiny cohorts.

Use a small, interpretable model ladder:

| Candidate | Population centre | Individual history | Purpose |
|---|---|---|---|
| Existing baseline | All-batter prior | Existing history rule | Preserve reference |
| Role population | Opening-role mean only | None | Essential pooling benchmark |
| Role-centred prior | Opening-role mean; shared concentration | Same history as baseline | Isolate centre |
| Role-specific prior | Opening-role mean and learned concentration | Same history | Test width separately |
| Role history | Same role prior | Opening-only history | Test whether other-role evidence helps |
| Partially pooled roles | Role centres shrunk toward a league centre | Explicit role-aware history | Stabilise small groups if simple fits vary too much |

Keep the original annual lookback discipline for sequential work initially: year t's prior uses only t−3 through t−1. Do not import the first experiment's current-season early-half prior into pre-season or early-season predictions. If later testing a prior that adapts within a season, make it a separate, named update rule using strictly earlier dates and apply it equally to all history candidates.

Diagnose prior means, concentration, latent-player spread, players, effective counts and individual-data weights by fold. Record fits near the pooled limit, optimiser bounds and sensitivity to leaving out one player. A future hierarchical model should account for uncertainty in the prior, not just return a plug-in Beta interval. Do not fix large concentrations purely to make plots look less shrunk.

Avoid adding many position groups immediately. Decide any additional roles from information available at the forecast time; first test the binary opening/non-opening distinction with partial pooling. Establish a past-only fallback for very small or absent role samples.

**Deliverable:** a common prior-fitting interface and centre/strength/history ablation table. **Gate:** identifiable gains over role pooling across historical folds, or retain the simpler pooled/role-centred alternative.

## 3. Update the sequential adaptation experiment

**Purpose:** determine whether recency still helps once the comparison group is appropriate.

1. Reproduce the unchanged sequential baseline on exactly the same opening innings. This measures the starting point without retraining on a different target and calling that a gain.
2. Define timing carefully. If the intended forecast is before the match date, the actual opening pair may not be known. Either use only historical role information at that time or explicitly change the task to “conditional on this player opening.” An at-innings-start model can use the announced/observed opening pair, but must be compared at that same later information time. Do not quietly insert realised role into a pre-date forecast.
3. Compare all-history, current-season, rolling-ball and exponential histories under the **same** role prior, training eligibility and scored opportunities. Start with the original finite grids; do not simultaneously search a large role × width × window × context space.
4. Keep every same-date outcome out of pre-date history. Define opening-only ball windows as counts of eligible balls faced while opening. Decide explicitly whether calendar decay runs during a player's non-opening spells; a natural default is yes, since elapsed time still passes. Label it as a modelling assumption and assess it historically.
5. Compare opening-only history with all-role history under a fixed role prior. A player promoted to open should not automatically lose all useful batting evidence. A partially pooled personal role effect is a later candidate if the simple comparison warrants it.
6. Select recency rules across multiple chronological development folds, report performance by season and history size, and include the role-only population benchmark. Evaluate calibration and both weighting schemes. Use paired date blocks for a direct continuation; add player/season dependence and refitting sensitivity before making stronger uncertainty claims.

**Deliverable:** an updated sequential comparison with a clear decomposition: population centre, personal role history and recency. **Gate:** recency must add useful predictive value over a well-centred expanding forecast, not merely compensate for league or role drift. A null result should lead to a simpler model.

## 4. Separate context and player form

**Purpose:** avoid labelling role/phase/environment changes as individual ability changes.

The opener-only powerplay check favoured pooling in its point estimate. Openers' full-innings observations also depend on surviving into later phases. Therefore an opening role does not create identical conditions or independent outcomes.

1. Extend the existing past-trained context model with the same small role definition. Retain venue, opposition, innings, nominal phase and legitimate pre-delivery state; fit encoders, mappings and scaling on past data only.
2. Fit a historical league/season scoring environment component and distinguish changes common to the league from changes specific to a player. If estimating current-season league conditions, use only previous dates and describe the additional information set.
3. Test adding the chosen recent-minus-baseline signal **after** role and environment are represented. Compare contextual models to each other on identical balls; do not compare their absolute score to a pre-innings model as if the information sets matched.
4. Show phase-specific calibration and loss differences, including powerplay-only and full opening innings. Keep these prespecified diagnostics, not alternative headlines chosen after inspecting significance.
5. For player illustrations, standardise predictions to a documented common distribution of contexts using an appropriately fitted model. Do not mistake a different future context mix for a changed player effect. Fixed-context comparisons remain predictive summaries, not causal skill estimates.

**Deliverable:** context-adjusted incremental value of recency and a clear explanation of what remains unmeasured, including bowler strength if not modelled. **Gate:** claim “extra predictive form information” only if recency adds stable evidence beyond these controls; otherwise explain its simpler forecasting role.

## 5. Reassess the separate preseason experiment

**Purpose:** test persistence through an off-season; within-season success is insufficient.

1. Fit all priors and personal history using seasons strictly before the target season. Never carry an early-half target-season prior into a preseason prediction.
2. Compare all-batter, opener-centred and partially pooled alternatives on a fixed conditional-opening target. Explicitly disclose that actual later opening is used to define evaluation exposure, not to construct preseason features.
3. If an unconditional preseason forecast is desired, separately estimate the chance of batting and role assignment from contemporaneously available squad/role evidence. Do not assume opening status known from the eventual scorecard. Until that model exists, publish conditional predictions.
4. Select off-season decay independently of the sequential experiment, using multiple earlier validation folds. Compare the existing rolling-300 reference, expanding history and a small frozen set of decay alternatives. Do not carry the failed 0.1 concentration multiplier forward.
5. Assess established openers, new IPL players and role switchers separately, with sample sizes and uncertainty. Report proper scores conditional on observed opportunities and clarify that this does not assess selection prediction.

**Deliverable:** separate preseason selection and evaluation tables, including role-uncertainty limitations. **Gate:** update the prospective application only after this horizon supports the proposed model.

## 6. Update watchlists only after model decisions

**Purpose:** translate a validated forecast into understandable candidate lists without rewriting prior predictions.

Preserve the existing 12 October 2026 forecasts permanently. If a new model is later approved, publish a separately dated version alongside them and explain what changed.

For each player show: the forecasted outcome and assumed role; eligible history in that role and all roles; prior mean and strength; recent and long-term estimates; forecast probability and interval; and whether a changed role or league baseline accounts for the difference. Compare a player with their own baseline under the **same reference environment and role**, or separately report the environment/role contribution. The opener result shows why comparing forecasts with differently centred priors can exaggerate “improvement.”

Keep improver and rebound definitions separate. A rebound projection is not a probability of returning to baseline. To make probability statements, first define the event and exposure horizon, then use a model that represents realised-season outcome uncertainty. Estimate baseline and forecast jointly because their histories overlap. Include prior refitting, player/innings dependence and role uncertainty as appropriate; present model/regime uncertainty separately if it remains unquantified. With many ranked players, use multiplicity-aware reporting or make clear that rankings are exploratory and avoid significance labels.

Retain transparent history thresholds and show excluded/low-history players. Evaluate both conditional rates and eventual exposure where possible. At the end of the future season, score the frozen versions before making further changes; do not replace a disappointing forecast retrospectively.

**Deliverable:** versioned candidate lists and an evaluation plan. **Gate:** every headline claim must match the forecast horizon, conditioning information and uncertainty actually modelled.

## 7. Reporting, tests and interview narrative

- Add role-detection tests for illegal first balls, non-striker dismissals, no eligible balls and role changes. Add cutoff perturbation tests to demonstrate that same-day/future outcomes cannot affect earlier forecasts. Assert identical evaluation rows across competing models.
- Reconstruct selected configurations from development/validation output and verify that no evaluation score enters selection. Record checksums and reproducible commands. Keep experiment outputs in distinct versioned locations.
- Present the research as a tested hypothesis: “Restricting to opening batters improved forecasts against the original all-batter model, but not because the prior became wider. Appropriate population centring mattered, and deliberately weakening shrinkage failed to generalise.”
- Keep the distinction between shrinkage beating raw rates and beating an appropriate population baseline. Report the latter's uncertainty, the powerplay reversal and validation instability.
- Update the main report and any CV only after the corresponding work is actually completed. Do not turn this plan into a claim that role-aware sequential, contextual or 2027 forecasts already exist. Financial relevance remains signal estimation, hierarchical pooling, regime changes and honest validation—not a claim about trading returns.

## Suggested implementation order and stopping rules

1. **Protocol and role/prior interface**: freeze scope, timing and fallback rules; reproduce the baseline on the opening target.
2. **Sequential and centre/strength ablations**: test the simplest role correction before adding model complexity.
3. **Context analysis**: determine whether apparent recency is mostly role/environment information.
4. **Separate preseason validation**: choose a model for the actual next-season application.
5. **Versioned watchlists and write-up**: publish only after the preceding gates, then evaluate prospectively.

Stop adding complexity when improvements are unstable across folds, disappear against role pooling, worsen calibration or mainly benefit a few high-exposure players without matching the intended target. If opening-specific gains do not transfer to a later experiment, retain its original model and publish the negative result. The project is stronger when it explains where a plausible idea works and where it fails.
