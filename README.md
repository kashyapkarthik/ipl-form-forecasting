# IPL six-rate forecasting

### How much should a forecast trust recent form?

A Python research project adapting Brown's early-to-late-season baseball forecasting question to IPL **six probabilities**, then testing how much recent performance should change a forecast.

**Main finding:** a last-300-ball shrinkage model, selected on 2024, improves next-innings log loss by **0.24% over full-history shrinkage** in 2025–2026. Very short and raw recent histories perform worse. However, player-cluster uncertainty weakens the gain, recency adds no clear value after context adjustment, and the separately selected preseason short-history model does not beat full history. Useful recent information is not proof of permanent player improvement.

**[Read the digestible write-up](docs/SIXES_REPORT.md)** · [Executed notebook](03_six_rate_forecasting.ipynb) · [All 2027 forecasts](results/sixes/forecast_2027_all.csv) · [In-form candidates](results/sixes/watchlist_in_form.csv) · [Bounce-back candidates](results/sixes/watchlist_bounce_back.csv)

![Six-rate evidence](figures/sixes/evidence.png)

## What the experiments show

| Question | Result on 2025–2026 | Interpretation |
|---|---|---|
| Early-season shrinkage vs raw | 21.33% lower log loss | Shrinkage protects sparse estimates; the magnitude depends on raw clipping |
| Early-season shrinkage vs position population | 0.52% lower log loss; interval spans zero | Personal half-season evidence adds little reliably detectable value |
| Selected last-300 shrinkage vs full history, updated before each match | 0.24% lower log loss | A modest gain; date interval excludes zero, player interval does not |
| Selected 300-ball exponential decay vs full history | 0.23% lower log loss | Retaining older observations with lower weights also helps |
| Selected 120-ball preseason model vs full history | 0.11% higher log loss | In-season gains do not automatically survive an off-season |
| Extra recency after context adjustment | No clear gain | Avoid claiming demonstrated genuine ability improvement |

All percentages are relative changes in log loss, not accuracy or probability-point changes. Six rate captures one aspect of batting. The dataset and evaluation years were already inspected during earlier outcome exploration, so these are exploratory historical results.

A follow-up [individual-player improvement audit](docs/PLAYER_IMPROVEMENT_REPORT.md) examines 2026 changes, persistence, context and multiple-player screening. Rahul and Kishan are promising descriptive candidates; no player has a clearly positive adjusted 95% change interval.

The [shrinkage and memory qualification](docs/PLAYER_MEMORY_REPORT.md) holds the prior, role mix and memory rule constant across 2026. It tests nine histories for all 40 eligible players, with paired and simultaneous uncertainty; no change is confirmed across the full search.

## Data and modelling

- Checksum-pinned Cricsheet archive: **1,243 matches, 295,732 recorded events**, 2008–2026, through 31 May 2026.
- Six-rate cohort: **267,994 legal deliveries, 14,916 sixes**, observed batting positions 1–7. Initial striker/non-striker are combined as openers; positions 3–7 are separate.
- A success is six batter runs unless marked non-boundary. Exclude wides, no-balls and Super Overs; retain non-sixes and legal byes/leg-byes as zero outcomes.
- Registry IDs, explicit event order, strictly earlier-date histories and population priors fitted on past seasons prevent common data errors.
- Population, raw and empirical-Bayes forecasts; expanding, rolling and exponential histories; separate chronological selection for sequential and preseason predictions.
- Log loss, unclipped Brier, calibration, date/player uncertainty, venue/phase/opposition checks and retrospective temporary-form examples.
- **110 dated 2027 forecasts**, with expected-rate stability ranges, past-role mixtures and a full-history alternative. In-form and bounce-back labels are candidate screens, not confirmed improvement or availability predictions.

![Temporary form](figures/sixes/temporary_form.png)

## Reproduce

Python 3.13 is the tested interpreter. The raw archive is included with its attribution; no live download or API key is needed for analysis.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
```

On Windows use `.venv\Scripts\activate`. The default workflow creates processed data if needed, runs the complete six-rate project, verifies results and runs tests. It generates the report, figures and executed notebook. Use `python run_sixes.py` for the same six-rate workflow explicitly. Use `python run_all.py --legacy` to reproduce the original boundary-rate project while retaining the current README.

Open **[docs/sixes.html](docs/sixes.html)** locally for the portable viewer and searchable forecast table. GitHub displays the Markdown and notebook directly; its repository preview does not execute the HTML viewer.

## Project map

| Location | Purpose |
|---|---|
| `pipeline.py`, `data/raw/` | Shared pinned source and audited event ingestion |
| `sixes/core.py`, `sixes/history.py` | Six targets, priors, metrics and past-only histories |
| `sixes/study.py` | Within-season, sequential and preseason tests |
| `sixes/context.py`, `sixes/cases.py` | Context checks and temporary-form examples |
| `sixes/forecasts.py` | Conditional next-season forecasts and uncertainty |
| `sixes/verify.py` | Independent raw-JSON and forecast reconstruction |
| `sixes/report.py` | Figures, readable reports, viewer and executed notebook |
| `config/sixes_protocol.json`, `docs/SIXES_PROTOCOL.md` | Choices frozen and published before the full study |
| `results/sixes/`, `reports/sixes_verification.json` | Complete numerical evidence and verification |
| `docs/SIXES_CV_ALIGNMENT.md` | Evidence-backed CV wording and interview preparation |

The user's CV has not been edited. The report supports a cricket adaptation of Brown, not replication of every baseball estimator or proof that a hot streak was caused by skill rather than luck.

## Historical studies

These are preserved for the research record; the six-rate study above is now the main project.

- [Original boundary project](docs/RESEARCH_REPORT.md)
- [Opening-batter comparison](docs/OPENERS_REPORT.md)
- [Separate batting-position comparison](docs/POSITIONS_REPORT.md)
- [Exploratory outcome comparison that nominated six rate](docs/OUTCOMES_REPORT.md)

## References and attribution

[Brown (2008)](https://arxiv.org/abs/0803.3697) motivates the forecasting/shrinkage question. [Cricsheet](https://cricsheet.org/) provides the data under the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/); see [DATA_LICENSE.md](DATA_LICENSE.md). Cleaning, aggregation and analysis are this project's transformations; Cricsheet does not endorse the results. Code is MIT licensed. Built with Codex assistance, with reproducible code and checks provided.
