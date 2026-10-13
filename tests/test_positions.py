import numpy as np
import pandas as pd
from models import fit_prior
from positions_experiment import (position_sample, counts, position_forecasts, filter_subset,
                                  block_comparisons, losses)


def data():
    return pd.DataFrame(dict(delivery_id=list('abcdef'), match_id=['e']*3+['l']*3,
                             season=[2024]*6, batter_id=['x', 'x', 'y', 'x', 'x', 'z'],
                             batting_entry_order=[1, 3, 7, 2, 4, 8],
                             primary_eligible=[True]*6, is_boundary=[True, False, True, False, True, True]))


def test_positions_are_innings_roles_not_permanent_player_labels():
    d = position_sample(data())
    assert set(d.delivery_id)==set('abcde')
    splits = pd.DataFrame(dict(match_id=['e', 'l'], half=['early', 'late']))
    h = counts(d, splits)
    early_x = h.loc[h.half.eq('early')&h.batter_id.eq('x')]
    assert set(early_x.position_group)=={1, 3}
    assert early_x.N.sum()==2
    assert not h.loc[h.half.eq('early')].position_group.isin([2, 4]).any()


def test_combining_openers_preserves_targets_and_separates_other_positions():
    d = position_sample(data(), combine_openers=True)
    assert d.loc[d.batting_entry_order.le(2), 'position_group'].eq(0).all()
    assert set(d.loc[d.position_group.ne(0), 'position_group'])=={3, 4, 7}
    assert len(d)==len(position_sample(data()))


def test_new_position_uses_prior_despite_history_elsewhere():
    early = pd.DataFrame(dict(batter_id=['a', 'b', 'c'], B=[3, 10, 6], N=[30, 40, 50]))
    all_early = pd.concat([early, pd.DataFrame(dict(batter_id=['switcher'], B=[50], N=[100]))])
    ap = fit_prior(all_early.B, all_early.N)
    f, prior = position_forecasts(early, ['a', 'switcher', 'debut'], all_early, ap)
    f = f.set_index('batter_id')
    assert f.loc['switcher', 'early_N']==0 and f.loc['switcher', 'all_early_N']==100
    np.testing.assert_allclose(f.loc['switcher', 'p_beta_binomial'], prior['mu'])
    np.testing.assert_allclose(f.loc['switcher', 'p_raw'], prior['pooled'])
    assert f.loc['switcher', 'p_original_beta'] != f.loc['debut', 'p_original_beta']
    assert prior['position_records']==3  # Early b/c remain in the prior without late appearances.


def test_late_outcomes_cannot_change_position_forecasts():
    early = pd.DataFrame(dict(batter_id=['a', 'b', 'c'], B=[3, 10, 6], N=[30, 40, 50]))
    ap = fit_prior(early.B, early.N)
    late = pd.DataFrame(dict(batter_id=['a', 'new'], B=[1, 0], N=[10, 4]))
    first, _ = position_forecasts(early, late.batter_id, early, ap)
    late['B'], late['N'] = [90, 80], [100, 100]
    second, _ = position_forecasts(early, late.batter_id, early, ap)
    pd.testing.assert_frame_equal(first, second)


def test_min11_restriction_is_half_exposure_not_innings_exposure():
    innings = pd.DataFrame(dict(early_N=[20, 20, 5], late_N=[20, 5, 20], N=[2, 5, 20]))
    assert filter_subset(innings, 'min11_both_halves', innings=True).index.tolist()==[0]


def test_joint_date_resampling_preserves_aggregate_cancellation():
    # Opposing position effects on every date exactly cancel in the aggregate.
    rows = []
    for day in ['2025-01-01', '2025-01-02']:
        for pos, pa, pb in [(1, .1, .3), (2, .3, .1)]:
            rows.append(dict(season=2025, date=day, match_id=day, innings=1, batter_id=str(pos),
                             position_group=pos, B=2, N=10, p_a=pa, p_b=pb))
    result = block_comparisons(pd.DataFrame(rows), [('a', 'b')]).set_index('position_group')
    np.testing.assert_allclose(result.loc['all', ['log_loss_difference', 'log_loss_low', 'log_loss_high']].to_numpy(float), 0, atol=1e-15)
    assert result.loc['1', 'log_loss_difference'] == -result.loc['2', 'log_loss_difference']


def test_position_scoring_preserves_raw_zero_one_brier():
    ll, bs = losses([1, 2], [3, 3], [0, 1])
    np.testing.assert_array_equal(bs, [1, 1])
    assert np.isfinite(ll).all()
