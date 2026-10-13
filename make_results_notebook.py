from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[
md('''# IPL form forecasting: completed research results

**Kashyap Karthik · 12 October 2026**

This notebook reads the saved outputs of `python run_all.py`. It distinguishes the within-season shrinkage experiment, sequential cross-season adaptation, and forecasts frozen before the next season. Read `docs/RESEARCH_REPORT.md` for the complete methods and limitations.

The primary finding is modest predictive improvement from recent weighting. The context-adjusted incremental effect is uncertain; this is not proof of permanent player improvement.'''),
code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
R = ROOT / 'results'
selection = json.loads((R / 'selection.json').read_text())
selection'''),
md('''## 1. Check what was selected, and when

2024 chooses 365-day exponential decay for forecasts updated before match dates and a 300-ball window for preseason forecasts. 2025–2026 evaluate those fixed choices. Selecting a different method because it wins on the test set would invalidate that separation.'''),
code('''pd.read_csv(R / 'sequential_validation_grid.csv')[['model','log_loss','brier']]'''),
md('''## 2. Early-season records predict later-season boundaries

The ≥11-opportunity subset avoids relying on very tiny samples for the transformed diagnostic. The more inclusive all-player results are also saved. Beta-binomial shrinkage outperforms raw rates, while its advantage over the pooled probability is uncertain.'''),
code('''brown = pd.read_csv(R / 'brown_metrics.csv')
brown.query("period == 'test' and subset == 'min11_both_halves'")[['model','N','log_loss','brier','noise_adjusted_arcsine_mse']]'''),
code('''pd.read_csv(R / 'brown_uncertainty.csv')'''),
md('''## 3. How much history should the forecast remember?

Every forecast below uses only outcomes on earlier dates. Annual priors use the preceding three seasons. Compare both opportunity-weighted and equal-player scores: the research answer depends on the intended objective.'''),
code('''pd.read_csv(R / 'sequential_metrics.csv').query("period == 'test'")'''),
code('''display(Image(filename=str(ROOT / 'figures/model_comparison.png')))'''),
code('''pd.read_csv(R / 'sequential_uncertainty.csv')'''),
md('''## 4. The off-season is a separate prediction problem

Preseason probabilities remain fixed throughout the target season. This is stricter than updating after every match. The 300-ball rolling model remains the selected forecast even though another candidate scores slightly better after the test outcomes are revealed.'''),
code('''pd.read_csv(R / 'preseason_metrics.csv').query("period == 'test'")'''),
md('''## 5. Context and uncertainty change the interpretation

The contextual comparison controls for venue group, opposition team, nominal phase, innings, observed batting-entry order, pre-delivery state and a season trend. Both contextual models use a richer information set than the main pre-innings forecasts. Their *difference* tests the extra recency signal. Its bootstrap interval includes zero.'''),
code('''pd.read_csv(R / 'context_uncertainty.csv')'''),
code('''display(Image(filename=str(ROOT / 'figures/uncertainty.png')))
display(Image(filename=str(ROOT / 'figures/calibration.png')))'''),
code('''pd.read_csv(R / 'subgroups_phase.csv')'''),
md('''## 6. Dated 2027 watchlists

These predict boundary rates, not overall cricketing quality or squad selection. Eligibility requires ≥120 opportunities in 2026 and ≥300 earlier. Baseline is the shrunk pre-2026 historical rate. Intervals describe conditional block-resampling stability, not the full range of possible realised 2027 performance. Candidate labels can overlap.

A player-specific change cannot be separated completely from league conditions or changing role. Lists are a transparent forecasting application, not declarations of proven improvement.'''),
code('''columns=['player','pre_2026_N','latest_N','baseline','latest_rate','last_120_rate','forecast','forecast_low','forecast_high','change_from_baseline']
display(pd.read_csv(R / 'watchlist_improvers.csv')[columns].head(10))
display(pd.read_csv(R / 'watchlist_rebounds.csv')[columns].head(10))'''),
code('''display(Image(filename=str(ROOT / 'figures/watchlists_2027.png')))
display(Image(filename=str(ROOT / 'figures/player_trajectories.png')))'''),
md('''## 7. What the research supports

- Shrinkage protects against overconfident estimates from small samples.
- Recent weighting gives modest gains in the chosen held-out, opportunity-weighted forecasting task.
- Benefits vary with horizon, scoring weights and match context.
- The current study does not identify permanent or causal skill changes.
- The 2027 forecasts are prospective and must remain dated when their eventual outcomes are evaluated.

Sources: [Cricsheet](https://cricsheet.org/), [Brown (2008)](https://www.stat.berkeley.edu/users/aldous/157/Papers/brown_batting.pdf). Data license and transformation attribution are in `DATA_LICENSE.md`.''')])
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
NotebookClient(nb,timeout=120,resources={'metadata':{'path':str(ROOT)}}).execute()
nbf.write(nb,ROOT/'02_research_results.ipynb')
print('Executed results notebook')
