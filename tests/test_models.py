import numpy as np
import pandas as pd
from models import fit_prior, posterior, losses, history_counts, snapshot_counts, paired_bootstrap
CFG={'rolling_balls':[2], 'ew_balls_halflives':[2], 'ew_days_halflives':[10]}

def deliveries():
    return pd.DataFrame({'batter_id':['a']*5,'date':['2023-01-01']*2+['2023-01-02']*2+['2024-01-01'],
        'season':[2023]*4+[2024], 'match_id':['1','1','2','2','3'],'innings':[1]*5,
        'delivery_sequence':[1,2,1,2,1],'is_boundary':[True,False,True,True,False]})

def test_shrinkage_and_no_history():
    prior={'alpha':15,'beta':85,'strength':100}
    np.testing.assert_allclose(posterior([6,60,0],[20,200,0],prior),[.175,.25,.15])

def test_histories_equal_direct_past_only_computation():
    d=deliveries(); h=history_counts(d,CFG)
    for row in h.to_dict('records'):
        counts=snapshot_counts(d,row['date'],CFG)
        for model,(b,n) in counts.items():
            np.testing.assert_allclose([row[model+'_B'],row[model+'_N']],[b,n])
    assert h.iloc[-1].current_N==0
    assert h.iloc[0].expanding_N==0

def test_same_day_and_future_changes_cannot_change_forecast():
    d=deliveries(); a=history_counts(d,CFG)
    d.loc[d.date>='2023-01-02','is_boundary']=False
    b=history_counts(d,CFG)
    pd.testing.assert_frame_equal(a.loc[a.date<='2023-01-02'],b.loc[b.date<='2023-01-02'])

def test_counts_match_manual_binary_scoring():
    ll,bs=losses([1],[3],[.2])
    y=np.array([1,0,0]); p=.2
    np.testing.assert_allclose(ll,-(y*np.log(p)+(1-y)*np.log(1-p)).sum())
    np.testing.assert_allclose(bs,((y-p)**2).sum())

def test_prior_recovers_simulated_population():
    rng=np.random.default_rng(44); n=np.full(3000,100)
    b=rng.binomial(n,rng.beta(4,16,size=len(n)))
    fit=fit_prior(b,n)
    assert abs(fit['mu']-.2)<.02
    assert 14<fit['strength']<27


def test_bootstrap_identical_models_zero_difference():
    rows=[]
    for model in ['a','b']:
        for i in range(3):
            rows.append(dict(model=model,season=2025,date=f'2025-01-0{i+1}',match_id=str(i),
                innings=1,batter_id='x',B=i,N=10,p=.2))
    result=paired_bootstrap(pd.DataFrame(rows),'a','b',repetitions=20)
    assert result['log_loss_low']==result['log_loss_high']==0
