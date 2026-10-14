import numpy as np
import pandas as pd
from sixes.core import target,losses,C
from sixes.history import features,snapshot,selected_weights
from sixes.study import role_weights,memory_forecasts
from models import fit_prior


def data():
    rows=[]
    for year in [2023,2024]:
        for day in ['04-01','04-02']:
            for role in [1,2,5,8]:
                for j in range(4):
                    rows.append(dict(season=year,date=f'{year}-{day}',match_id=f'{year}-{day}',innings=1,
                                     batter_id='a',batting_entry_order=role,delivery_sequence=role*10+j,
                                     delivery_id=f'{year}-{day}-{role}-{j}',batter_runs=[0,4,6,6][j],non_boundary=j==3,
                                     primary_eligible=True,is_super_over=False,batter_name='A'))
    return pd.DataFrame(rows)


def test_six_target_is_not_boundary_and_keeps_zeroes():
    raw=data();raw.loc[0,'primary_eligible']=False;raw.loc[1,'is_super_over']=True
    d=target(raw)
    assert d.role.isin([0,5]).all() and len(d)==46
    assert d.is_six.sum()==12
    assert d.loc[d.batter_runs.eq(4),'is_six'].eq(0).all()
    assert d.loc[d.non_boundary,'is_six'].eq(0).all()


def test_incremental_histories_match_independent_snapshot_and_exclude_same_day():
    d=target(data());h=features(d)
    for row in h.itertuples():
        player=d.loc[d.batter_id.eq(row.batter_id)&d.role.eq(row.role)]
        expected=snapshot(player,row.date)
        for name,(b,n) in expected.items():
            np.testing.assert_allclose([getattr(row,name+'_B'),getattr(row,name+'_N')],[b,n],atol=1e-12)
    first=h.loc[h.date.eq('2023-04-01')]
    assert first.expanding_N.eq(0).all()
    new=h.loc[h.date.eq('2024-04-01')]
    assert new.current_N.eq(0).all() and new.expanding_N.gt(0).all()


def test_future_outcomes_and_other_roles_cannot_change_history():
    d=target(data());first=features(d)
    d.loc[d.date.ge('2024-04-01'),'is_six']=1
    d.loc[d.role.eq(5),'is_six']=1
    second=features(d)
    cols=[c for c in first if c.endswith('_B') or c.endswith('_N')]
    mask=first.role.eq(0)&first.date.le('2024-04-01')
    np.testing.assert_allclose(first.loc[mask,cols],second.loc[mask,cols])


def test_role_mix_uses_only_completed_past_seasons():
    d=target(data());w,_=role_weights(d,'a',2024)
    d.loc[d.season.eq(2024),'role']=7
    after,_=role_weights(d,'a',2024)
    pd.testing.assert_series_equal(w,after)
    assert w[0]==2/3 and w[5]==1/3


def test_weighted_counts_agree_with_snapshot():
    d=target(data()).query('role==0').sort_values(['date','match_id','innings','delivery_sequence'])
    cutoff='2025-03-28';s=snapshot(d,cutoff)
    for name,(b,n) in s.items():
        w=selected_weights(d,name,cutoff)
        np.testing.assert_allclose([w@d.is_six.to_numpy(),w.sum()],[b,n])


def test_cold_start_and_raw_zero_probabilities_are_distinct():
    d=target(data());s=snapshot(d.iloc[:0],'2024-01-01')
    p=fit_prior([1,7,12],[30,50,60]);f=memory_forecasts(s,p)
    assert f['p_raw_expanding']==p['pooled'] and f['p_expanding']==p['mu']
    s=snapshot(d.assign(is_six=0),'2025-01-01');f=memory_forecasts(s,p)
    assert f['p_raw_expanding']==0 and f['p_expanding']>0
    ll,bs=losses([1,0],[1,1],[0,1]);assert np.isfinite(ll).all();np.testing.assert_array_equal(bs,[1,1])
