"""Publication figures and compact evidence tables from saved experiment outputs."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from research import ROOT, OUT, read, save

FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
COLORS={'expanding':'#7b8795','rolling_300':'#d5853a','ew_balls_300':'#4d79a7','ew_days_365':'#167a72','ew_days_730':'#167a72',
    'population':'#b6a5c5','raw':'#c75a51','beta_binomial':'#167a72','current':'#9393be'}
LABELS={'expanding':'All-history shrinkage','rolling_300':'Last 300 balls + shrinkage',
    'ew_balls_300':'Exponential: 300-ball half-life','ew_days_365':'Exponential: 365-day half-life',
    'ew_days_730':'Exponential: 730-day half-life','population':'Population rate','raw':'Raw player rate',
    'raw_expanding':'Raw all-history rate','beta_binomial':'Beta-binomial shrinkage','current':'Current-season shrinkage'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelcolor':'#334155',
    'text.color':'#172e3e','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfcfd',
    'axes.facecolor':'#fbfcfd','grid.color':'#dde5e9','axes.edgecolor':'#b5c3cb','savefig.facecolor':'#fbfcfd'})

def export(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def performance():
    fig,axes=plt.subplots(1,3,figsize=(15,5.1),gridspec_kw={'width_ratios':[.85,1.2,1.2]})
    brown=pd.read_csv(OUT/'brown_metrics.csv').query("period == 'test' and subset == 'min11_both_halves'")
    selections=json.loads((OUT/'selection.json').read_text())
    for ax,label,g,order in [(axes[0],'Early → late season',brown,['raw','population','beta_binomial']),
        (axes[1],'Update before each match date',pd.read_csv(OUT/'sequential_metrics.csv').query("period=='test'"),['expanding','rolling_300','ew_balls_300','ew_days_365']),
        (axes[2],'Freeze before the next season',pd.read_csv(OUT/'preseason_metrics.csv').query("period=='test'"),['expanding','rolling_300','ew_balls_300','ew_days_730'])]:
        g=g.set_index('model').loc[order]
        ax.scatter(g.log_loss,range(len(order)),c=[COLORS[m] for m in order],s=75,zorder=3)
        ax.set_yticks(range(len(order)),[LABELS[m] for m in order],fontsize=8.5)
        ax.invert_yaxis(); ax.set_title(label,pad=16); ax.set_xlabel('Log loss · lower is better')
        xmin=g.log_loss.min()-.004; xmax=g.log_loss.max()+.004
        ax.set_xlim(xmin,xmax); ax.grid(axis='x',alpha=.6); ax.set_axisbelow(True)
        for i,value in enumerate(g.log_loss): ax.text(value+.0002,i,f'{value:.4f}',va='center',fontsize=8)
    fig.suptitle('How much history should a cricket forecast remember?',fontsize=20,x=.04,ha='left',y=1.03)
    fig.text(.04,-.03,'Held-out 2025–2026 • selected on 2024: 365-day decay during season; last 300 balls before season\nLeft panel requires ≥11 opportunities in both halves. Compare methods within panels; horizons and samples differ.',fontsize=9,color='#596979')
    fig.tight_layout(w_pad=2.5); export(fig,'model_comparison')


def uncertainty():
    seq=pd.read_csv(OUT/'sequential_uncertainty.csv').set_index('model')
    pre=pd.read_csv(OUT/'preseason_uncertainty.csv').set_index('model')
    ctx=pd.read_csv(OUT/'context_uncertainty.csv').iloc[0]
    vals=[('During season: 365-day decay',seq.loc['ew_days_365']),('Before season: last 300 balls',pre.loc['rolling_300']),('Recent history added to context model',ctx)]
    fig,ax=plt.subplots(figsize=(10,4.3))
    for i,(label,r) in enumerate(vals):
        v=r.log_loss_difference*1000; lo=r.log_loss_low*1000; hi=r.log_loss_high*1000
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#167a72' if i<2 else '#d5853a',capsize=5,ms=8)
    ax.axvline(0,color='#4c5966',lw=1,ls='--'); ax.set_yticks(range(3),[v[0] for v in vals]); ax.invert_yaxis()
    ax.set_xlabel('Log-loss difference × 1,000  ← better with recent history')
    ax.set_title('Recency helps the simple models; its extra value after context is uncertain',pad=18)
    ax.grid(axis='x',alpha=.5)
    fig.text(.02,-.03,'95% paired date-bootstrap intervals, stratified by test season. Conditional on fitted models and observed 2025–2026 seasons.',fontsize=8.5)
    fig.tight_layout(); export(fig,'uncertainty')


def calibration():
    p=pd.read_parquet(OUT/'sequential_predictions.parquet'); p=p.loc[p.season.isin([2025,2026])]
    chosen=['population','expanding','ew_days_365']; rec=[]
    bins=np.array([0,.08,.12,.16,.20,.24,.28,.36,1.01])
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.plot([0,.4],[0,.4],ls='--',color='#a0aeb8',label='Perfect calibration')
    for model in chosen:
        g=p.loc[p.model==model].copy(); g['bin']=pd.cut(g.p,bins,include_lowest=True)
        for label,x in g.groupby('bin',observed=True):
            rec.append(dict(model=model,bin=str(label),N=int(x.N.sum()),predicted=float(np.average(x.p,weights=x.N)),observed=float(x.B.sum()/x.N.sum())))
        z=pd.DataFrame(rec).query('model==@model and N>=200')
        ax.plot(z.predicted,z.observed,'o-',label=LABELS[model],color=COLORS[model])
    ax.set(xlabel='Forecast boundary probability',ylabel='Observed boundary frequency',title='Are forecast probabilities calibrated?  |  2025–2026',xlim=(.05,.36),ylim=(.05,.36))
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False,fontsize=9)
    ax.grid(alpha=.5); fig.tight_layout(); export(fig,'calibration'); save(pd.DataFrame(rec),'calibration_bins')


def era():
    d=read('model_deliveries'); league=d.groupby('season',as_index=False).agg(N=('is_boundary','size'),B=('is_boundary','sum'))
    league['rate']=league.B/league.N; save(league,'league_by_season')
    s=pd.read_csv(OUT/'sequential_by_season.csv').pivot(index='season',columns='model',values='log_loss')
    delta=s['ew_days_365']-s['expanding']
    fig,axes=plt.subplots(2,1,figsize=(11,6.7),sharex=True)
    axes[0].plot(league.season,league.rate,'o-',color='#167a72'); axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(ylabel='Observed boundary rate',title='The scoring environment changes too')
    axes[1].bar(delta.index,delta*1000,color=['#167a72' if v<0 else '#c75a51' for v in delta])
    axes[1].axhline(0,color='#596979',lw=.8); axes[1].set(xlabel='IPL season',ylabel='Decay − expanding\nlog loss × 1,000',title='Recency gains are not uniform across historical seasons')
    for ax in axes:
        ax.axvspan(2024.5,2026.5,color='#c9dbe2',alpha=.35); ax.grid(axis='y',alpha=.4)
    axes[0].text(2025,.11,'Held-out',fontsize=9)
    axes[1].set_xticks(range(2008,2027,2)); fig.tight_layout(); export(fig,'changing_environment')


def cases_and_lists():
    a=pd.read_csv(OUT/'watchlist_improvers.csv'); b=pd.read_csv(OUT/'watchlist_rebounds.csv')
    allf=pd.read_csv(OUT/'forecast_2027_all.csv')
    ids=list(dict.fromkeys(list(a.head(2).batter_id)+list(b.head(2).batter_id)))
    p=pd.read_parquet(OUT/'sequential_predictions.parquet')
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
    for ax,pid in zip(axes.flat,ids):
        x=p.loc[(p.batter_id==pid)&(p.season>=2022)]
        player=allf.loc[allf.batter_id==pid].iloc[0]
        for model in ['expanding','ew_days_365']:
            g=x.loc[x.model==model].sort_values('date')
            ax.plot(pd.to_datetime(g.date),g.p,label=LABELS[model],color=COLORS[model],lw=1.8)
        g=x.loc[x.model=='expanding']; ax.scatter(pd.to_datetime(g.date),g.B/g.N,s=np.clip(g.N,4,45),alpha=.22,color='#b1946c',label='Observed innings')
        ax.set_title(player.player,loc='left',fontweight='bold'); ax.set_ylim(0,.65); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y',alpha=.35)
        ax.tick_params(axis='x',labelsize=8)
    axes[0,0].legend(frameon=False,fontsize=8,loc='upper left'); fig.suptitle('Player case studies: observed innings versus pre-match forecasts',fontsize=16,y=1.01)
    fig.text(.02,-.02,'Illustrative cases selected after analysis. Lines are forecasts formed before each date; dots are subsequent outcomes. Lines across off-seasons connect observed dates.',fontsize=8)
    fig.tight_layout(); export(fig,'player_trajectories')
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    for ax,g,title in [(axes[0],a.head(6),'Potential boundary-rate improvers'),(axes[1],b.head(6),'Projected rebound candidates')]:
        for i,r in enumerate(g.itertuples()):
            ax.plot([r.baseline,r.forecast],[i,i],color='#c7d3d9',lw=3)
            ax.scatter(r.baseline,i,marker='s',color='#7b8795',s=25,label='Pre-2026 baseline' if i==0 else None)
            ax.scatter(r.latest_rate,i,marker='x',color='#d5853a',s=45,label='Observed 2026 rate' if i==0 else None)
            ax.errorbar(r.forecast,i,xerr=[[max(0,r.forecast-r.forecast_low)],[max(0,r.forecast_high-r.forecast)]],fmt='o',color='#167a72',capsize=3,label='2027 forecast + 95% interval' if i==0 else None)
        ax.set_yticks(range(len(g)),g.player); ax.invert_yaxis(); ax.set_title(title,loc='left',pad=18); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_xlabel('Boundary probability'); ax.grid(axis='x',alpha=.4)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('IPL 2027 watchlists  |  Forecasts dated 12 October 2026',fontsize=18,y=1.01)
    fig.text(.02,-.10,'Model: last 300 eligible deliveries + shrinkage, chosen on 2024 preseason validation. Intervals describe resampling stability, not realised-season prediction ranges.\nCandidates require ≥120 opportunities in 2026 and ≥300 beforehand. These are marginal boundary forecasts, not proof of skill change or squad-selection predictions.',fontsize=8.5)
    fig.tight_layout(w_pad=3); export(fig,'watchlists_2027')

if __name__=='__main__':
    performance(); uncertainty(); calibration(); era(); cases_and_lists()
    print('Created six figure sets and underlying calibration/season tables')
