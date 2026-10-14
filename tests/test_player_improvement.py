"""Checks for the exploratory player-change audit, separate from forecasts."""
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sixes import player_improvement as audit


def test_multiple_search_adjustment_matches_independent_implementation():
    p=np.array([.04,.001,.2,.015,.04,.8])
    np.testing.assert_allclose(audit.bh(p),false_discovery_control(p,method='bh'))


def test_innings_bootstrap_preserves_a_constant_rate():
    x=pd.DataFrame({'season':[2024,2024,2025,2025],'B':[1,2,3,4],'N':[10,20,30,40]})
    np.testing.assert_allclose(audit.draw_rate(x,np.random.default_rng(12)),.1)


def test_context_model_recovers_known_absolute_and_peer_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'OUT',tmp_path)
    rows=[]
    for season in [2025,2026]:
        for match in range(12):
            for pid in ['a','b']:
                sixes=10+(10 if pid=='a' and season==2026 else 0)+(1 if match%2 else -1)
                for ball in range(100):
                    rows.append(dict(batter_id=pid,season=season,role=0,phase='middle',venue='Example',bowling_team='Example',innings=1,
                                     match_id=f'{season}-{match}',is_six=int(ball<sixes)))
    result=audit.adjusted_fit(pd.DataFrame(rows),['a','b']).set_index(['batter_id','comparison'])
    np.testing.assert_allclose([result.loc[('a','absolute'),'change'],result.loc[('b','absolute'),'change']], [.1,0],atol=1e-12)
    np.testing.assert_allclose([result.loc[('a','relative'),'change'],result.loc[('b','relative'),'change']], [.1,-.1],atol=1e-12)
