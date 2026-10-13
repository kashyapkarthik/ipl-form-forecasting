"""Role, leakage, comparison and loss checks specific to the opener extension."""
import numpy as np
import pandas as pd
import pytest
from openers_experiment import (opening_deliveries, match_count_splits, halves,
                                fit_season, binary_losses, paired_interval, scaled_prior)


def test_opening_pair_uses_illegal_first_event_and_non_striker_identity():
    # b opens without facing either of the first two legal balls. c is a replacement.
    raw = pd.DataFrame(dict(
        match_id=['m']*4, innings=[1]*4, delivery_sequence=[1, 2, 3, 4],
        delivery_id=['1', '2', '3', '4'], batter_id=['a', 'c', 'c', 'b'],
        non_striker_id=['b']*3+['c'], is_super_over=[False]*4,
        is_legal=[False, True, True, True], primary_eligible=[False, True, True, True],
        is_boundary=[False, False, True, True], batting_entry_order=[1, 3, 3, 2], over=[0]*4))
    opening, audit = opening_deliveries(raw, raw.loc[raw.primary_eligible])
    assert opening.delivery_id.tolist() == ['4']
    assert audit['first_event_illegal_innings'] == 1
    assert audit['opening_pair_slots'] == 2
    assert audit['eligible_opening_innings'] == 1


def test_role_is_innings_specific_not_career_identity():
    raw = pd.DataFrame(dict(
        match_id=['m']*3, innings=[1, 2, 2], delivery_sequence=[1, 1, 2],
        delivery_id=['1', '2', '3'], batter_id=['a', 'c', 'a'],
        non_striker_id=['b', 'd', 'd'], is_super_over=[False]*3,
        is_legal=[True]*3, primary_eligible=[True]*3, is_boundary=[True]*3,
        batting_entry_order=[1, 1, 3], over=[0]*3))
    result, _ = opening_deliveries(raw, raw)
    assert set(result.delivery_id) == {'1', '2'}


def test_match_count_cut_never_splits_double_header_date():
    matches = pd.DataFrame(dict(match_id=list('abcde'), season=[2024]*5,
                                date=['2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-04']))
    split = match_count_splits(matches)
    assert split.cutoff_date.unique().tolist() == ['2024-01-02']
    assert split.half.tolist() == ['early', 'early', 'early', 'late', 'late']


def test_forecasts_cannot_use_late_outcomes_or_filter_training_by_late_roster():
    # d has early history and then disappears: its early record still fits the prior.
    d = pd.DataFrame(dict(season=[2024]*10, half=['early']*4+['late']*6,
                         batter_id=list('abcdabcabe'), N=[25]*10, B=[3, 7, 2, 10, 9, 8, 7, 6, 5, 4]))
    early, late = d.loc[d.half=='early'], d.loc[d.half=='late']
    a, prior_a, _ = fit_season(early, late.batter_id.unique(), diagnostics=False)
    d.loc[d.half=='late', 'B'] = 0
    b, prior_b, _ = fit_season(d.loc[d.half=='early'], late.batter_id.unique(), diagnostics=False)
    pd.testing.assert_frame_equal(a, b)
    assert prior_a == prior_b and prior_a['records'] == 4
    entrant = a.loc[a.batter_id=='e'].set_index('model')
    assert entrant.loc['raw', 'p'] == prior_a['pooled']
    assert entrant.loc['beta_binomial', 'p'] == prior_a['mu']


def test_widening_preserves_mean_and_increases_data_weight():
    prior = dict(alpha=20, beta=80, strength=100, mu=.2)
    wide = scaled_prior(prior, .25)
    assert wide['alpha']/wide['strength'] == .2
    assert 30/(30+wide['strength']) > 30/(30+prior['strength'])


def test_brier_is_unclipped_and_aggregated_scores_match_deliveries():
    ll, bs = binary_losses([1, 2], [3, 3], [0, 1])
    np.testing.assert_array_equal(bs, [1, 1])
    assert np.isfinite(ll).all()
    ll, bs = binary_losses([1], [3], [.2])
    y = np.array([1, 0, 0])
    np.testing.assert_allclose(bs, ((y-.2)**2).sum())
    np.testing.assert_allclose(ll, -(y*np.log(.2)+(1-y)*np.log(.8)).sum())


def test_bootstrap_rejects_different_opportunities():
    base = pd.DataFrame(dict(season=[2025, 2025], date=['2025-01-01']*2,
                             match_id=['m']*2, innings=[1]*2, batter_id=['p']*2,
                             model=['a', 'b'], p=[.2]*2, N=[10, 11], B=[2, 2]))
    with pytest.raises(AssertionError):
        paired_interval(base, 'a', 'b', repetitions=10)
    base.loc[1, 'N'] = 10
    r = paired_interval(base, 'a', 'b', repetitions=10)
    assert r['log_loss_low'] == r['log_loss_high'] == r['brier_difference'] == 0
    base.loc[1, 'batter_id'] = 'different'
    with pytest.raises(AssertionError):
        paired_interval(base, 'a', 'b', repetitions=10)


def test_same_date_cutoff_applies_to_all_players():
    d = pd.DataFrame(dict(match_id=['a', 'a', 'b', 'b'], season=[2024]*4,
                         batter_id=['x', 'y', 'x', 'y'], is_boundary=[True, False, False, True]))
    splits = pd.DataFrame(dict(match_id=['a', 'b'], half=['early', 'late']))
    h = halves(d, splits)
    assert h.N.sum() == 4 and h.B.sum() == 2
    assert h.loc[(h.half=='early')&(h.batter_id=='x'), 'B'].item() == 1
