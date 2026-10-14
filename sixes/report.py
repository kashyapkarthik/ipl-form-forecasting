"""Digestible report, scientific figures, portable viewer and executed results notebook."""
import base64
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from .core import ROOT,OUT,C

FIG=ROOT/'figures/sixes'
NAMES={'population':'Position population','raw':'Raw early rate','shrinkage':'Early-season shrinkage','raw_expanding':'Raw full history','expanding':'Full-history shrinkage',
       'rolling_300':'Last 300 balls + shrinkage','ew_balls_300':'300-ball half-life + shrinkage','raw_recent':'Raw last 120 balls','rolling_30':'Last 30 balls + shrinkage',
       'current':'Current-season shrinkage','rolling_120':'Last 120 balls + shrinkage','ew_balls_1200':'1,200-ball half-life + shrinkage',
       'context':'Context + long history','context_recent':'Context + long history + recency'}
FULL_NAMES={'RG Sharma':'Rohit Sharma','AM Rahane':'Ajinkya Rahane','SS Iyer':'Shreyas Iyer','SV Samson':'Sanju Samson','N Pooran':'Nicholas Pooran','H Klaasen':'Heinrich Klaasen','C Green':'Cameron Green','TM Head':'Travis Head','JC Buttler':'Jos Buttler'}

def load(name):return pd.read_csv(OUT/f'{name}.csv')
def name(x):return FULL_NAMES.get(x,x)
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def scoretable(frame):return mdtable(['Forecast','Log loss ↓','Brier ↓','Equal-player log loss ↓'],[[NAMES.get(r.model,r.model),f'{r.log_loss:.6f}',f'{r.brier:.6f}',f'{r.player_macro_log_loss:.6f}']for r in frame.itertuples()])
def fit_table(frame):return mdtable(['Player','2026 sixes / 100 balls','2027 forecast','90% stability range','Full-history alternative'],[[name(r.player),f'{100*r.latest_rate:.1f}',f'{100*r.forecast:.1f}',f'{100*r.forecast_low:.1f}–{100*r.forecast_high:.1f}',f'{100*r.full_history_forecast:.1f}']for r in frame.itertuples()])

def figure(fig,stem):
    for ext in ['png','svg']:
        fig.savefig(FIG/f'{stem}.{ext}',dpi=180,bbox_inches='tight',metadata={'Date':None}if ext=='svg'else{})
    plt.close(fig)
    path=FIG/f'{stem}.svg';path.write_text('\n'.join(x.rstrip()for x in path.read_text().splitlines())+'\n')


def make_figures(cases,forecast):
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.right':False,'axes.spines.top':False,'svg.hashsalt':'sixes-20261014'})
    selections=json.loads((OUT/'selection.json').read_text())
    specs=[('brown','shrinkage','population','Within season: shrinkage vs population'),
           ('sequential','rolling_300','expanding','Next innings: last 300 vs full history'),
           ('sequential','ew_balls_300','expanding','Next innings: exponential vs full history'),
           ('preseason','rolling_120','expanding','Next season: last 120 vs full history'),
           ('context','context_recent','context','Extra recency after context adjustment')]
    points=[]
    for f,a,b,label in specs:
        u=load(f+'_uncertainty');r=u.loc[u.model.eq(a)&u.reference.eq(b)&u.unit.eq('date')].iloc[0];points.append((label,r))
    fig,ax=plt.subplots(figsize=(11,5.4));ys=np.arange(len(points))
    for y,(label,r) in zip(ys,points):
        color='#087e8b'if r.skill_pct>0 else '#b66b40'
        ax.errorbar(r.skill_pct,y,xerr=[[r.skill_pct-r.skill_low],[r.skill_high-r.skill_pct]],fmt='o',color=color,lw=2,capsize=4)
        ax.text(1.25,y,f'{r.skill_pct:+.2f}%',va='center',ha='right',fontsize=10)
    ax.axvline(0,color='#94a3b8',lw=1);ax.set_yticks(ys,[x[0]for x in points]);ax.set_ylim(4.7,-.7);ax.set_xlim(-.55,1.35)
    ax.set_xlabel('Log-loss improvement (%) — positive means better');ax.grid(axis='x',alpha=.15)
    ax.set_title('Recent form helps modestly within a season',loc='left',fontweight='bold',fontsize=17,pad=24)
    fig.text(.01,-.03,'2025–26 evaluation · 95% date-bootstrap ranges · Player-cluster uncertainty weakens the main recency result.\nPreseason choices were selected separately. Context uses additional information available before each ball.',fontsize=10,color='#475569')
    figure(fig,'evidence')
    fig,axes=plt.subplots(1,len(cases),figsize=(12,4.7),sharey=True)
    for ax,r in zip(np.atleast_1d(axes),cases.itertuples()):
        vals=[r.baseline,r.early_rate,r.early_only_shrinkage,r.history_retaining_shrinkage,r.late_rate]
        ax.bar(np.arange(5),100*np.array(vals),color=['#94a3b8','#e5944a','#c4a05a','#087e8b','#16324f'],width=.72)
        ax.set_xticks(range(5),['Prior','Early','Early\nonly','History','Later'],fontsize=9)
        role_label='opener' if r.role==0 else f'No. {r.role}'
        ax.set_title(f'{name(r.player)} · {r.season}\n{role_label}',fontsize=11)
        ax.set_ylim(0,16);ax.grid(axis='y',alpha=.12)
        for i,v in enumerate(vals):ax.text(i,100*v+.4,f'{100*v:.1f}',ha='center',fontsize=9)
        ax.text(.5,-.2,f'Early: {r.early_B}/{r.early_N} sixes · later: {r.late_B}/{r.late_N}',transform=ax.transAxes,ha='center',fontsize=9)
    axes[0].set_ylabel('Sixes per 100 legal balls')
    fig.suptitle('An early spike need not persist',x=.07,ha='left',fontweight='bold',fontsize=18)
    fig.text(.07,-.06,'Early-only and History are shrunk forecasts; Early and Later are observed rates.\nRetrospective examples illustrate reversion; they do not prove luck caused it.',fontsize=10,color='#475569')
    fig.subplots_adjust(top=.82,bottom=.18,wspace=.12);figure(fig,'temporary_form')
    chosen=pd.concat([forecast.loc[forecast.in_form_candidate].sort_values('change_from_baseline',ascending=False).head(3),
                      forecast.loc[forecast.bounce_back_candidate].sort_values('forecast',ascending=False).head(3)])
    fig,ax=plt.subplots(figsize=(10,5.5))
    for i,r in enumerate(chosen.itertuples()):
        ax.errorbar(100*r.forecast,i,xerr=[[100*(r.forecast-r.forecast_low)],[100*(r.forecast_high-r.forecast)]],fmt='o',color='#087e8b',capsize=4,lw=2,label='Selected preseason forecast'if i==0 else None)
        ax.scatter(100*r.full_history_forecast,i,marker='D',color='#64748b',s=35,label='Full-history alternative'if i==0 else None)
    ax.set_yticks(range(len(chosen)),[name(r.player)+(' · in-form'if r.in_form_candidate else ' · bounce-back')for r in chosen.itertuples()]);ax.invert_yaxis()
    ax.set_xlabel('Expected sixes per 100 legal balls');ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',fontsize=9)
    ax.set_title('2027 forecasts are scenarios, not guarantees',loc='left',fontsize=17,fontweight='bold',pad=20)
    fig.text(.01,-.03,'90% stability ranges from 400 prior/innings resamples. Fixed 2026 role mix; conditional on playing.\nThe selected short-history preseason model did not beat full history in 2025–26. Model choice adds uncertainty.',fontsize=10,color='#475569')
    figure(fig,'forecasts')
    cal=load('sequential_calibration');fig,ax=plt.subplots(figsize=(6.5,5.5))
    ax.plot([0,.3],[0,.3],ls='--',color='#94a3b8',lw=1,label='Perfect agreement')
    for model,color in [('population','#94a3b8'),('expanding','#416c9b'),('rolling_300','#087e8b')]:
        g=cal.loc[cal.model.eq(model)&cal.N.ge(100)]
        ax.plot(g.predicted,g.observed,'o-',color=color,label=NAMES[model],markersize=5)
    ax.set(xlim=(0,.3),ylim=(0,.3),xlabel='Forecast six probability',ylabel='Observed six frequency',title='Calibration: do forecast rates match outcomes?')
    ax.legend(fontsize=9);ax.grid(alpha=.15);fig.text(.03,-.01,'2025–26; predefined bins with at least 100 balls. Descriptive, not a calibration guarantee.',fontsize=9)
    figure(fig,'calibration')


def main():
    audit=json.loads((OUT/'data_audit.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    scores={label:load(label+'_metrics').query('period=="evaluation"')for label in ['brown','sequential','preseason']}
    cases=load('temporary_form_cases');forecasts=load('forecast_2027_all');in_form=load('watchlist_in_form');bounce=load('watchlist_bounce_back')
    bootstrap_priors=load('forecast_prior_resamples');bootstrap_bound_count=int(bootstrap_priors.at_strength_bound.sum())
    make_figures(cases,forecasts)
    case_table=mdtable(['Player / season / role','Pre-season baseline','Early observed','Early-only shrinkage','History retained','Later observed'],
        [[f'{name(r.player)} / {r.season} / '+('opener'if r.role==0 else str(r.role)),f'{100*r.baseline:.1f}%',f'{r.early_B}/{r.early_N} = {100*r.early_rate:.1f}%',
          f'{100*r.early_only_shrinkage:.1f}%',f'{100*r.history_retaining_shrinkage:.1f}%',f'{r.late_B}/{r.late_N} = {100*r.late_rate:.1f}%']for r in cases.itertuples()])
    phase=load('subgroups_phase');phase_table=mdtable(['Phase','Balls','Observed six rate','Recency gain [95% range]'],[[r.group,f'{r.N:,}',f'{100*r.observed_rate:.2f}%',f'{r.skill_pct:+.2f}% [{r.skill_low:+.2f}, {r.skill_high:+.2f}]']for r in phase.itertuples()])
    report=f'''# How much should a six-rate forecast trust recent form?

**IPL forecasting project · Kashyap Karthik · 14 October 2026**

## The answer in one minute

**Recent performances contain some useful predictive information, but the gain is small and does not establish permanent improvement.** The model selected on 2024 uses the last **300 same-role balls plus empirical-Bayes shrinkage** for the next innings. On 2025–2026 it improves log loss by **0.24% over full-history shrinkage**, **1.03% over a position-population forecast**, and **11.21% over raw full-history rates**.

A very short 30-ball shrunk history and an unsmoothed recent rate perform worse. However, it is not true that *any* removal of old personal history must hurt: the selected rolling 300-ball model discards older personal observations and still improves. The fitted population prior continues to supply information from three earlier seasons. An exponential model that retains all earlier balls with decreasing weight performs similarly well.

**Two findings limit the improvement claim.** The main advantage is uncertain under player-cluster resampling, and recency adds no clear value after venue/phase/opposition/match-state adjustment. Also, the independently selected **120-ball preseason model slightly underperforms full history** when asked to forecast a whole next season. The 2027 watchlists are therefore conditional forecasts to investigate, not validated claims of future improvement.

![Evidence](../figures/sixes/evidence.png)

## 1. Correct target, relevant comparisons

The pinned archive contains **1,243 matches and 295,732 recorded events**, from 2008 through 31 May 2026. There are **284,465 legal non-Super-Over deliveries** across all positions. This batting-focused project uses positions 1–7: **267,994 legal balls, 14,916 sixes and 495 players**.

A six means exactly six runs credited to the batter, with no non-boundary flag. All other eligible legal balls remain zeroes. Wides, no-balls and Super Overs are excluded; legal byes/leg-byes remain opportunities. This is a defined eligible-ball rate, not official scorecard balls faced. The extraction follows [Cricsheet's documented fields](https://cricsheet.org/format/json/) and is independently checked against every raw match JSON.

Each player is compared with others in the same innings role. Opening slots are combined because initial striker/non-striker ordering is arbitrary; positions 3–7 remain separate. A player changing role gets a different history for that role. This avoids comparing an opener with a finisher, but fragments histories and can miss transferable player ability. Roles are observed batting-entry ranks, not permanent labels.

## 2. Experiment 1: early-season records predict late-season sixes

A shared calendar midpoint divides each season into early and late dates. Fit the population distribution using early data only; freeze the forecasts; score every later eligible ball. Population is the pooled early role rate; raw is the player's early same-role rate; shrinkage averages personal evidence with the fitted role distribution. No-history raw uses population, and shrinkage uses the fitted prior mean.

`forecast = (prior six pseudo-count + player sixes) / (prior strength + player balls)`

Prior strength is learned from the spread of player records, not set equal to all balls in the database. This is a cricket adaptation of [Brown's early-to-late forecasting question](https://arxiv.org/abs/0803.3697), not reproduction of all his baseball estimators.

**2025–2026 late halves: 12,881 balls and 1,153 sixes. Lower scores are better.**

{scoretable(scores['brown'])}

Shrinkage reduces log loss **21.33% versus raw** and **0.52% versus population**. The population advantage has a 95% date-bootstrap range of **−0.03% to +1.09%**, spanning no gain. It is a large improvement over unstable raw estimates, not firm evidence that the within-season personal signal beats the population.

Raw zero forecasts are especially costly when a later six occurs. Log scoring clips raw probabilities at 0.000001; results at 0.0001 and 0.001 are also saved. Raw still loses, and Brier scoring, which keeps exact zero/one probabilities, also favours shrinkage. The size of the log-loss advantage over raw depends on the clipping convention.

Both robustness checks agree with the cautious interpretation: requiring 11 balls in each half gives +0.69% over population with a range crossing zero; a date-preserving match-count split gives +0.40%, also uncertain. Every season's results are published, not just the headline years.

## 3. Experiment 2: does recent form predict the next innings?

Every forecast uses only dates strictly before the match date. The annual role prior uses the previous three complete seasons, with one aggregated record per player. The same prior is used across all memory methods. Candidate rolling windows and exponential half-lives are fixed in the [protocol](SIXES_PROTOCOL.md). 2024 chooses the best rule; 2025–2026 evaluate it unchanged. Development years run from 2012–2023.

The main selected rule is **last 300 same-role balls plus shrinkage**. Within the exponential family, 2024 selects a **300-ball half-life**: a ball 300 observations old has half the weight of the newest one. It does not completely discard older data. Current-season-only, raw last-120 and shrunk last-30 histories test more aggressive forgetting.

**All methods score the same 30,980 balls, 2,591 sixes and 141 players in 2025–2026.**

{scoretable(scores['sequential'])}

The selected 300-ball forecast improves over full history by **0.24%**, with a 95% date-bootstrap range **+0.05% to +0.44%**. Its 300-ball exponential counterpart improves **0.23%**, range **+0.12% to +0.35%**. Equal-player log loss also improves slightly. The selected rolling model's gains are approximately **0.19% in 2025** and **0.28% in 2026**.

But the primary rolling result is less decisive under other assessments: the player-cluster range is **−0.03% to +0.52%**, and its Brier-score difference interval also crosses zero. Excluding documented nonstandard innings gives +0.26% versus full history. In aggregated 2012–2023 development data the 300-ball rolling rule was slightly worse than full history, while the exponential rule was better. This is not a universally optimal memory length.

**The proposed “form versus luck” argument is only partly supported.** The last-30-ball shrunk forecast is worse than the selected 300-ball forecast, and raw recent rates are substantially worse. Those results support damping noisy short spells. Yet current-season shrinkage is only slightly worse than the selected model, with an uncertain difference; rolling 300 actually wins despite discarding older personal balls. The defensible claim is that *the amount and treatment of recent evidence matter*, not that every old observation must always be retained.

![Calibration](../figures/sixes/calibration.png)

Log loss rewards accurate probabilities and strongly penalises confidently wrong ones. Brier is average squared probability error. The calibration plot asks whether, for example, groups forecast near 10% actually hit sixes about 10% of the time. None of these metrics alone establishes a permanent change in talent.

Calibration also exposes a limitation: the observed evaluation six rate is **8.36%**, while the population, full-history and selected recent forecasts average **7.10%, 7.36% and 7.69%** respectively. All three underpredict the aggregate rate. Part of recency's gain may reflect a changing league environment rather than player-specific improvement; the context check below is therefore important. These probabilities should not be described as perfectly calibrated.

## 4. Is it player improvement, or changing conditions?

There are **124 evaluation match dates**. The main uncertainty calculation resamples whole dates within each season, scoring each model on the same resampled observations. Think of its interval as the range of advantages obtained under variations in which observed match days occur. It is conditional on the fitted models and chosen windows. Player-cluster resampling preserves a player's whole observed trajectory and provides a complementary dependence check.

The selected model's advantage over full history varies by nominal innings phase:

{phase_table}

Powerplay means overs 1–6, middle 7–15, death 16–20. These are nominal bands, not verified rain-adjusted fielding restrictions. The clearest descriptive gain is in death overs. Venue differences include +0.93% at Eden Gardens and −0.38% at M Chinnaswamy; against Delhi Capitals the gain is +0.91%. These are exploratory subgroups with multiple comparisons and potentially different player mixes, not causal venue/team effects or betting signals. All venue and opposition results, including failures and small samples, are published.

The stronger check trains two contextual models on the preceding five seasons. Both use long-history ability, venue, opposition, phase, innings, role, year trend and pre-delivery score/wickets/balls. One also receives the recent-minus-long-history signal. Historical training signals themselves use memory rules selected before their season; there is no retrospective use of the 2024 choice to choose old training features.

Context + long history has log loss **0.283560**. Adding recency gives **0.283608**: a **−0.017% improvement**, i.e. a tiny deterioration, with a 95% range **−0.063% to +0.029%**. Thus recent form does not clearly add value after these controls. These forecasts also use more information, available before each ball rather than just before an innings. They cannot be presented as an equal-information win over the simple forecasts.

The full project therefore supports **modest predictive value in recent six-hitting records, with substantial uncertainty about independent player improvement**. Changes in role, conditions and league scoring can explain part of the pattern. Better predictions are evidence of useful information; they do not identify luck versus ability with certainty.

## 5. Examples of a hot spell that faded

An early-data rule flags a spike only when a player has at least 300 previous same-role balls, 60 early balls, and an early six rate four percentage points above their pre-season shrunk baseline. It flags **35 spells** across 2018–2026. We then select three distinct players whose later record reverts near that baseline, ordered by the size of the original spike.

**This final selection uses later outcomes deliberately: these are retrospective illustrations, not an unbiased performance test.** All 35 flags are retained. The population prior for each illustration is fitted before the season; early-only and history-retaining shrinkage use that same prior.

{case_table}

For Iyer, retaining the earlier record forecast about **6.1 sixes per 100 balls**, close to the later **6.1**, despite an early observed **12.6**. Samson and Kishan also fell well below their early rates. The examples explain why we should not promote every observed spike to an improvement claim. They cannot prove that luck was its sole cause; form, health, conditions and opposition may also change.

![Temporary form](../figures/sixes/temporary_form.png)

## 6. Next-season forecasts need their own test

For preseason prediction, all inputs freeze before the first match. To avoid assuming future batting order, each player gets a fixed mixture over their previous-season role exposure; if absent, use their latest earlier season, and debutants use the historical population mix. This is also how the 2027 player forecasts are produced.

2024 independently selected **last 120 balls plus shrinkage**. It only narrowly beat full-history shrinkage in validation (0.272370 versus 0.272394). We preserve that choice rather than replacing it with the hindsight winner.

{scoretable(scores['preseason'])}

The selected preseason model is **0.11% worse than full history**, with an improvement range **−0.34% to +0.13%**. Its +0.34% gain over population is also uncertain. The separately selected exponential-family model, with a 1,200-ball half-life, improves by about 0.08% over full history in this evaluation, but it was not the overall validation winner. This is evidence against treating the in-season result as automatically transferable across the off-season.

A conditional-on-actual-role diagnostic looks slightly better (+0.16% versus full history, uncertain), but future role is not known in a preseason deployment. The past-role-mixture result is the appropriate headline for the published player forecasts.

## 7. Dated 2027 predictions: expected sixes per 100 legal balls

**As of 14 October 2026**, using data through 31 May 2026. The assumed 28 March 2027 reference date is not an announced fixture date. All forecasts are conditional on the player batting in the IPL with the same role mix as 2026; selection, retirement, injuries and future opponents/venues are not forecast.

There are **110 player forecasts**, including **15 in-form screens** and **19 bounce-back screens**. Screens require at least 120 latest-season balls and 300 earlier balls. The common-prior pre-2026 baseline uses the same 2027 population priors and same role weights as the forecast, so a changed population prior or role mix alone cannot create the reported increase. It is a retrospective reference, not a prediction issued before 2026.

**In-form examples**, ranked by increase from that reference:

{fit_table(in_form.head(5))}

**Bounce-back examples**, ranked by forecast rate:

{fit_table(bounce.head(5))}

“In-form” requires a latest-season rate above baseline and a forecast at least one percentage point above it. “Bounce-back” means latest rate below baseline and forecast above latest rate; it can still remain below the career baseline. These are statistical candidate definitions, not claims that players will return to a peak.

The ranges come from **400 shared population-prior refits and within-player/season/role innings resamples**, holding the selected model and role mixture fixed. They are marginal ranges without adjustment for screening many players. A forecast of 10.2 with a range 7.3–12.7 describes uncertainty/stability in an **expected rate**, not a guarantee of 7–13 sixes in the next 100 balls. The full table separately gives an illustrative 100-ball outcome range adding binomial event variation. Real matches add dependence and varying conditions.

A numerical limitation also matters: {bootstrap_bound_count} of {len(bootstrap_priors):,} resampled role-prior fits reach the prior-strength bound. Such fits approach complete pooling and their strength can be poorly identified. Two full rebuilds gave unchanged point forecasts and conclusions, but slightly different bootstrap strengths; the largest stability-limit change was below 0.001 sixes per 100 balls. See the [rebuild audit](../reports/sixes_rebuild.json). These ranges should be interpreted at their displayed precision.

The full-history alternative is especially important: Rohit Sharma is about **10.2** under the selected short-history forecast but **6.3** under full history. This model-choice uncertainty is not contained in the narrow resampling range. Seven in-form candidates have a positive paired baseline-change range, conditional on this model; that does not override the weak preseason backtest or establish true improvement.

![Forecasts](../figures/sixes/forecasts.png)

[All forecasts](../results/sixes/forecast_2027_all.csv) · [In-form candidates](../results/sixes/watchlist_in_form.csv) · [Bounce-back candidates](../results/sixes/watchlist_bounce_back.csv) · [Forecast assumptions](../results/sixes/forecast_metadata.json)

## 8. What the project demonstrates, and what comes next

The strongest result is that population shrinkage makes sparse player rates more useful. A modest recency adjustment helps the simple next-innings forecast, while very short/no-shrinkage estimates are unstable. The weaker findings are just as important: a position population is hard to beat within one half-season; context can absorb recency's benefit; and the selected preseason short window did not beat full history.

For a next iteration, freeze a new-season evaluation before receiving its outcomes. Compare a shared player effect plus role effects against fully separate player-role histories, so moving position does not discard useful evidence. Test long-memory exponential models alongside the existing rolling rule with nested chronological selection. Use context-standardised forecasts before labelling a player an improver, and test watchlist calibration prospectively. These are future recommendations; this completed project reports the fixed comparisons above without tuning them to obtain a desired conclusion.

## 9. Reproducibility and CV claims

`python run_all.py` now reproduces the six-rate project. It builds six-specific targets, fits the experiments, evaluates context, creates player cases and dated forecasts, runs independent verification, and produces this report, a standalone HTML viewer and an executed notebook. `python run_all.py --legacy` retains the original boundary workflow.

The [frozen protocol](SIXES_PROTOCOL.md) was published before fitting this full study. The six-rate target itself was chosen from earlier exploration of the same data, so 2025–2026 are historical evaluation years, **not a fresh confirmatory holdout**. We do not claim externally preregistered evidence or exact reconstruction of each historical data vintage. The raw archive is checksum-pinned; source definitions, exclusions, keys, histories, forecast formulas, target equality, all reported scores and 2027 mixtures were checked. **38 tests pass.**

[Verification](../reports/sixes_verification.json) · [Executed notebook](../03_six_rate_forecasting.ipynb) · [All evidence](../results/sixes) · [CV and interview wording](SIXES_CV_ALIGNMENT.md)

The project supports claims about Python/pandas data work, a Brown-inspired empirical-Bayes comparison, chronological memory selection, probability scoring/calibration, uncertainty, context checks and player case studies. It does **not** support a claim that permanent player improvement was proved or that recent weighting always wins. The user's CV file has not been edited.

Data attribution: [Cricsheet](https://cricsheet.org/), under the Open Data Commons Attribution License; see [DATA_LICENSE.md](../DATA_LICENSE.md). Source data were transformed through cleaning and statistical analysis. Code is MIT licensed. Built with Codex assistance; code, frozen choices and checks are supplied for review.
'''
    (ROOT/'docs/SIXES_REPORT.md').write_text(report)
    cv='''# CV alignment and interview notes

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
'''
    (ROOT/'docs/SIXES_CV_ALIGNMENT.md').write_text(cv)
    build_html(scores,cases,forecasts,in_form,bounce)
    notebook()
    print('Built report, figures, standalone viewer, CV notes and executed notebook',flush=True)


def build_html(scores,cases,forecasts,in_form,bounce):
    images=''.join('<figure><img alt="'+label+'" src="data:image/png;base64,'+base64.b64encode((FIG/(stem+'.png')).read_bytes()).decode()+'"></figure>'for stem,label in [('evidence','Forecast comparison and uncertainty'),('temporary_form','Temporary-form case studies'),('forecasts','Dated player forecasts'),('calibration','Calibration')])
    tables=[]
    for label,title in [('brown','1. Early season predicts late season'),('sequential','2. Recent performance predicts the next innings'),('preseason','3. A separate next-season test')]:
        frame=scores[label][['model','log_loss','brier']].copy();frame['model']=frame.model.map(lambda x:NAMES.get(x,x));frame.columns=['Forecast','Log loss','Brier']
        tables.append('<h2>'+title+'</h2><div class="scroll">'+frame.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'</div>')
    rows=[]
    for r in forecasts.itertuples():
        kind=('form 'if r.in_form_candidate else '')+('bounce'if r.bounce_back_candidate else '')
        rows.append('<tr data-kind="'+kind+'"><td>'+html.escape(name(r.player))+f'</td><td>{r.latest_N}</td><td>{100*r.latest_rate:.1f}</td><td>{100*r.forecast:.1f}</td><td>{100*r.forecast_low:.1f}–{100*r.forecast_high:.1f}</td><td>{100*r.full_history_forecast:.1f}</td><td>{100*r.change_from_baseline:+.1f}</td><td>'+('Sufficient history'if r.eligible else 'Limited history')+'</td></tr>')
    doc='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IPL six-rate forecasting</title><style>body{font:17px/1.6 system-ui,sans-serif;background:#f3f7f8;color:#183447;margin:0}main{max-width:1120px;margin:auto;padding:40px 24px}h1{font-size:clamp(32px,5vw,55px);line-height:1.12}h2{margin-top:42px}p{max-width:950px}.eyebrow{color:#087e8b;font-size:13px;letter-spacing:.12em;font-weight:700}.note{background:#fff4e5;border-left:4px solid #cf8c43;padding:18px}figure{background:white;padding:15px;border-radius:12px;margin:30px 0}img{width:100%;height:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{text-align:left;padding:11px;border-bottom:1px solid #d7e2e7}th{background:#164b52;color:white}.scroll{overflow:auto;max-height:600px}input,select{font:inherit;padding:10px;border:1px solid #9ab7c0;border-radius:6px;margin:8px}a{color:#087e8b}code{background:#e1ebed;padding:2px 5px}</style></head><body><main><div class="eyebrow">CRICKET × FORECASTING · 14 OCTOBER 2026</div><h1>How much should a forecast trust recent form?</h1><p><strong>Shrinkage protects against noisy player rates. Recent weighting modestly improves simple next-innings forecasts, but does not prove genuine player improvement.</strong></p><p>The selected 300-ball model improves log loss by 0.24% over full-history shrinkage in 2025–2026. A very short 30-ball model is worse. After context adjustment, the extra recency signal is uncertain; the separately selected preseason model does not beat full history.</p><p><a href="SIXES_REPORT.md">Read the full findings</a> · <a href="../03_six_rate_forecasting.ipynb">Executed notebook</a> · <a href="../results/sixes/forecast_2027_all.csv">Download all forecasts</a></p><p class="note">This is exploratory research on previously examined data. Forecast skill does not identify permanent ability changes. The next-season forecasts are conditional scenarios, with a full-history alternative shown alongside.</p>'''+''.join(tables)+images+'''<h2>Explore the 2027 forecasts</h2><p>All rates below are sixes per 100 eligible legal balls. The 90% ranges show expected-rate stability under prior and innings resampling, not the range of realised season totals. Forecasts assume the 2026 role mix and do not predict squad selection, injury or availability.</p><label>Player <input id="search" placeholder="Search a player"></label><label>List <select id="kind"><option value="">All 2026 players</option><option value="form">In-form candidates</option><option value="bounce">Bounce-back candidates</option></select></label><p id="count" aria-live="polite"></p><div class="scroll"><table><thead><tr><th>Player</th><th>2026 balls</th><th>2026 observed</th><th>2027 forecast</th><th>90% range</th><th>Full-history alternative</th><th>Change vs baseline</th><th>History</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><h2>What we learned</h2><p>Use a population prior to temper sparse records. Compare recent and long-term histories on the same future observations. Distinguish in-season prediction from persistence across the off-season. Treat hot spells and bounce-back screens as hypotheses whose uncertainty and context must remain visible.</p><p>Reproduce with <code>python run_all.py</code>. <a href="SIXES_PROTOCOL.md">Protocol</a> · <a href="../reports/sixes_verification.json">Verification</a> · <a href="SIXES_CV_ALIGNMENT.md">CV and interview notes</a>.</p><p>Data: <a href="https://cricsheet.org/">Cricsheet</a>, Open Data Commons Attribution License 1.0. Modified through cleaning and analysis; see the repository's data licence. No external scripts or trackers.</p></main><script>const search=document.querySelector('#search'),kind=document.querySelector('#kind');function filter(){let n=0;document.querySelectorAll('tbody tr[data-kind]').forEach(r=>{const show=r.cells[0].textContent.toLowerCase().includes(search.value.toLowerCase())&&(!kind.value||r.dataset.kind.includes(kind.value));r.hidden=!show;if(show)n++});document.querySelector('#count').textContent=n+(n===1?' player shown':' players shown')}search.addEventListener('input',filter);kind.addEventListener('change',filter);filter();</script></body></html>'''
    (ROOT/'docs/sixes.html').write_text(doc)


def notebook():
    nb=nbformat.v4.new_notebook();cells=[]
    cells.append(nbformat.v4.new_markdown_cell('# IPL six-rate forecasting\n\nExecuted results companion, 14 October 2026. The question is whether recent form helps forecast future sixes without overreacting to noise. Read `docs/SIXES_REPORT.md` for interpretation and limitations. These evaluation years were previously examined during outcome exploration.'))
    cells.append(nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = Path.cwd()\nout = root / 'results/sixes'\nprint(json.loads((out / 'data_audit.json').read_text()))"))
    cells.append(nbformat.v4.new_markdown_cell('## The three probability forecasts\n\nPopulation uses comparable batting positions. Raw uses the player’s observed rate. Empirical Bayes shrinks the personal rate towards its fitted population. Every success is a six off the bat on an eligible legal delivery; non-sixes remain zeroes.'))
    for label,title in [('brown','Early-to-late within season'),('sequential','Forecasting the next innings'),('preseason','Forecasting a whole next season')]:
        cells.append(nbformat.v4.new_markdown_cell('## '+title+'\n\nLower scores are better. Model selection used 2024 only. Raw probabilities are clipped only for log scoring.'))
        cells.append(nbformat.v4.new_code_cell(f"display(pd.read_csv(out / '{label}_metrics.csv').query('period == \"evaluation\"'))\ndisplay(pd.read_csv(out / '{label}_uncertainty.csv'))"))
    cells.append(nbformat.v4.new_markdown_cell('## What does this say about genuine improvement?\n\nA modest simple-forecast gain is useful evidence, but player-cluster uncertainty and the context check weaken a permanent-improvement claim. Very short histories lose; not every use of a shorter history loses. The selected preseason rule underperforms full history.'))
    cells.append(nbformat.v4.new_code_cell("display(Image(filename=str(root / 'figures/sixes/evidence.png')))\ndisplay(pd.read_csv(out / 'context_metrics.csv'))\ndisplay(pd.read_csv(out / 'context_uncertainty.csv'))"))
    cells.append(nbformat.v4.new_markdown_cell('## Temporary-form examples\n\nThese cases were selected retrospectively from all early-data spike flags. Their later reversion does not prove luck caused the spike.'))
    cells.append(nbformat.v4.new_code_cell("display(pd.read_csv(out / 'temporary_form_cases.csv'))\ndisplay(Image(filename=str(root / 'figures/sixes/temporary_form.png')))"))
    cells.append(nbformat.v4.new_markdown_cell('## Conditional 2027 forecasts\n\nExpected sixes per 100 legal balls, assuming the 2026 role mix. Stability ranges exclude model-selection uncertainty, availability and new regimes. Compare the full-history alternative; the selected short-history preseason rule did not beat it in later evaluation.'))
    cells.append(nbformat.v4.new_code_cell("cols = ['player','latest_N','latest_rate','forecast','forecast_low','forecast_high','full_history_forecast']\ndisplay(pd.read_csv(out / 'watchlist_in_form.csv')[cols].head(8))\ndisplay(pd.read_csv(out / 'watchlist_bounce_back.csv')[cols].head(8))\ndisplay(Image(filename=str(root / 'figures/sixes/forecasts.png')))"))
    cells.append(nbformat.v4.new_code_cell("verification = json.loads((root / 'reports/sixes_verification.json').read_text())\nassert verification['passed']\nfor check in verification['checks']:\n    print('PASS:', check)"))
    nb.cells=cells;nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    NotebookClient(nb,timeout=120,resources={'metadata':{'path':str(ROOT)}}).execute()
    for cell in nb.cells:
        cell.metadata.pop('execution',None)
    nbformat.write(nb,ROOT/'03_six_rate_forecasting.ipynb')


if __name__=='__main__':main()
