from copy import deepcopy
import pandas as pd
from pipeline import flatten, prior_day_features

def fixture():
    def ball(runs, extras=None, **kwargs):
        ext = extras or {}
        return dict(batter='A', non_striker='B', bowler='C',
            runs=dict(batter=runs, extras=sum(ext.values()), total=runs+sum(ext.values())),
            extras=ext, **kwargs)
    balls = [ball(4), ball(0, {'byes': 4}), ball(0, {'wides': 1}), ball(6, {'noballs': 1}), ball(4),
        ball(0, wickets=[{'player_out': 'B', 'kind': 'run out'}]), ball(1)]
    balls[4]['runs']['non_boundary'] = True
    return {'meta': {'data_version': '1.2.0', 'revision': 1},
        'info': dict(gender='male', match_type='T20', event={'name': 'Indian Premier League'},
            registry={'people': {'A': 'a', 'B': 'b', 'C': 'c'}}, dates=['2020-09-20'], season='2020/21',
            venue='Ground', overs=20, balls_per_over=6, teams=['X', 'Y'],
            players={'X': ['A','B'], 'Y': ['C']}, toss={'winner': 'X', 'decision': 'bat'}, outcome={'winner': 'X'}),
        'innings': [{'team': 'X', 'overs': [{'over': 0, 'deliveries': balls}]}]}

def test_boundary_denominator_and_event_order():
    match, rows, _, wickets, _ = flatten('1', fixture())
    assert match['season'] == 2020 and match['season_source'] == '2020/21'
    assert [r['is_boundary'] for r in rows] == [True, False, False, True, False, False, False]
    assert [r['primary_eligible'] for r in rows] == [True, True, False, False, True, True, True]
    assert len({r['delivery_id'] for r in rows}) == 7
    assert rows[4]['legal_balls_before'] == 2
    assert rows[6]['wickets_before'] == 1 and wickets[0]['player_out_id'] == 'b'
    assert rows[1]['score_before'] == 4

def test_super_over_and_retired_hurt():
    obj=fixture()
    obj['innings'][0]['super_over']=True
    obj['innings'][0]['overs'][0]['deliveries'][5]['wickets'][0]['kind']='retired hurt'
    _, rows, _, _, _=flatten('2',obj)
    assert not any(r['primary_eligible'] for r in rows)
    assert rows[6]['wickets_before']==0

def test_no_future_or_same_day_history():
    x=pd.DataFrame({'batter_id':['a','a','a','a','b'],
        'date':['2020-01-01','2020-01-02','2020-01-02','2020-01-03','2020-01-02'],
        'N':[10,20,30,40,100], 'B':[1,2,3,4,10]})
    hist=prior_day_features(x).set_index(['batter_id','date'])
    assert hist.loc[('a','2020-01-02'),'prior_N']==10
    assert hist.loc[('a','2020-01-03'),'prior_N']==60
    assert hist.loc[('b','2020-01-02'),'prior_N']==0
    y=x.copy(); y.loc[y.date=='2020-01-03','B']=40
    pd.testing.assert_frame_equal(prior_day_features(x),prior_day_features(y))

def test_missing_id_fails():
    import pytest
    obj=fixture(); del obj['info']['registry']['people']['A']
    with pytest.raises(KeyError): flatten('3',obj)

def test_shortened_conditions_preserved():
    obj=fixture(); obj['innings'][0]['target']={'runs':80,'overs':10}
    _, rows, meta, _, _=flatten('4',obj)
    assert not rows[0]['standard_conditions']
    assert rows[0]['primary_eligible']
    assert meta[0]['target_overs']==10
