import numpy as np
import pandas as pd
from outcomes_experiment import appearances, samples, history_sample, forecast, losses, uncertainty


def test_full_innings_includes_noball_runs_and_zero_ball_partner():
    rows = []
    for seq, runs in enumerate([6, 6, 6, 6, 6, 6, 4, 4, 6]):
        rows.append(dict(match_id='m', innings=1, season=2024, date='2024-01-01',
                         delivery_sequence=seq, is_super_over=False, batter_id='a', non_striker_id='b',
                         batter_runs=runs, primary_eligible=seq<8, batting_entry_order=1,
                         non_boundary=False, is_boundary=True))
    rows.append(dict(rows[0], innings=2, is_super_over=True, batter_runs=6))
    d = pd.DataFrame(rows)
    a = appearances(d).set_index('batter_id')
    assert a.loc['a', 'total_runs'] == 50
    assert a.loc['a', 'legal_balls'] == 8
    assert a.loc['b', 'total_runs'] == 0 and a.loc['b', 'legal_balls'] == 0
    assert a.loc['b', 'position_group'] == 2
    splits = pd.DataFrame(dict(match_id=['m'], half=['early'], cutoff_date=['2024-01-02']))
    s = samples(d, a.reset_index(), splits)
    assert s['score50'].B.sum() == 1 and s['score50'].N.sum() == 2
    assert s['score30'].B.sum() == 1
    assert s['boundary'].N.sum() == 8 and s['six'].B.sum() == 6


def test_nonboundary_six_is_not_six_and_byes_are_batter_dots():
    d = pd.DataFrame(dict(season=[2024]*3, date=['2024-01-01']*3, match_id=['m']*3,
                         innings=[1]*3, batter_id=['a']*3, batting_entry_order=[1]*3,
                         batter_runs=[6, 0, 1], non_boundary=[True, False, False],
                         is_boundary=[False]*3, primary_eligible=[True]*3, is_super_over=[False]*3))
    a = pd.DataFrame(dict(season=[2024], date=['2024-01-01'], match_id=['m'], innings=[1],
                         batter_id=['a'], position_group=[1], total_runs=[7]))
    split = pd.DataFrame(dict(match_id=['m'], half=['early'], cutoff_date=['2024-01-02']))
    s = samples(d, a, split)
    assert s['six'].B.item() == 0 and s['dot'].B.item() == 1


def history_fixture():
    rows = []
    for year in range(2020, 2026):
        for half, date in [('early', f'{year}-04-01'), ('late', f'{year}-05-01')]:
            for pos in [1, 3]:
                for j, player in enumerate(['a', 'b', 'c']):
                    rows.append(dict(season=year, half=half, date=date, cutoff_date=f'{year}-04-15',
                                     match_id=date, innings=1, batter_id=player, position_group=pos,
                                     N=30, B=2+j*5 if pos == 1 else 1+j))
    return pd.DataFrame(rows)


def test_three_previous_seasons_excludes_old_and_future_data():
    x = history_fixture()
    train = history_sample(x, 2024, 'previous3_plus_current')
    assert set(train.season) == {2021, 2022, 2023, 2024}
    assert train.loc[train.season.eq(2024), 'half'].eq('early').all()
    assert len(history_sample(x, 2024, 'current')) == 6


def test_forecasts_exclude_future_outcomes_and_other_positions():
    x = history_fixture()
    f, _ = forecast(x, 'boundary', 'current', years=[2024])
    x.loc[x.half.eq('late'), 'B'] = 29
    x.loc[x.position_group.eq(3), 'B'] = 25
    second, _ = forecast(x, 'boundary', 'current', years=[2024])
    cols = [c for c in f if c.startswith('p_')]+['early_N', 'early_B']
    np.testing.assert_allclose(f.loc[f.position_group.eq(1), cols], second.loc[second.position_group.eq(1), cols])
    assert f.early_N.eq(30).all()
    long, _ = forecast(x, 'boundary', 'previous3_plus_current', years=[2024])
    assert long.early_N.eq(210).all()


def test_unclipped_brier_and_finite_log_loss_at_extremes():
    ll, bs = losses([1, 0], [1, 1], [0, 1])
    assert np.isfinite(ll).all()
    np.testing.assert_array_equal(bs, [1, 1])


def test_common_date_bootstrap_identical_forecasts_have_zero_effect():
    x = history_fixture().query('season==2025 and half=="late"').copy()
    x['outcome'], x['window'] = 'six', 'current'
    for model in ['beta', 'population', 'raw', 'jeffreys', 'prior_mean', 'current_population']:
        x[f'p_{model}'] = .2
    u = uncertainty(x)
    np.testing.assert_allclose(u[['log_loss_difference', 'log_loss_low', 'log_loss_high', 'skill_low', 'skill_high']], 0)
