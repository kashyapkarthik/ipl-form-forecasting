"""Post-protocol diagnostic: raw-rate clipping sensitivity, not model selection."""
import pandas as pd
from research import OUT,save
from models import summarise
p=pd.read_csv(OUT/'brown_predictions.csv')
p=p.loc[(p.model=='raw')&p.season.isin([2025,2026])].copy()
p['p']=(p.early_B+.5)/(p.early_N+1)
# Keep the documented population fallback for batters with no early observations.
pooled=pd.read_csv(OUT/'brown_predictions.csv').query("model == 'population'")
p=p.merge(pooled[['season','batter_id','p']],on=['season','batter_id'],suffixes=('','_pooled'),validate='one_to_one')
p.loc[p.early_N==0,'p']=p.loc[p.early_N==0,'p_pooled']
p['model']='jeffreys_smoothed_individual'
rows=[]
for label,g in [('all_late_players',p),('min11_both_halves',p.loc[(p.early_N>=11)&(p.N>=11)])]:
 s=summarise(g);s['subset']=label; rows.append(s)
save(pd.concat(rows),'brown_smoothed_sensitivity')
