import numpy as np
import pandas as pd
from sixes.player_memory import aligned_weights,primary_simultaneous_bands


def sample():
    return pd.DataFrame({'date':['2025-05-01']*160+['2026-05-01']*160,'season':[2025]*160+[2026]*160})


def test_matching_memory_endpoints_excludes_future_and_reacts_to_new_history():
    x=sample();old=aligned_weights(x,'rolling_120','2026-01-01');new=aligned_weights(x,'rolling_120','2026-06-01')
    assert old.sum()==new.sum()==120
    assert old[160:].sum()==0 and new[:160].sum()==0
    changed=x.copy();changed.loc[160:,'date']='2027-01-01'
    np.testing.assert_array_equal(aligned_weights(changed,'rolling_120','2026-01-01'),old)


def test_expanding_changes_are_not_a_switch_of_memory_rule():
    x=sample();before=aligned_weights(x,'expanding','2026-01-01');after=aligned_weights(x,'expanding','2026-06-01')
    assert before.sum()==160 and after.sum()==320
    assert np.all(before<=after)


def test_simultaneous_band_covers_family_and_unchanging_contrast():
    rng=np.random.default_rng(1);draw=rng.normal(size=(2000,20));draw[:,0]=0
    point=np.zeros(20);low,high,c=primary_simultaneous_bands(point,draw)
    assert low[0]==high[0]==0 and c>2
    z=np.abs((draw[:,1:]-draw[:,1:].mean(axis=0))/draw[:,1:].std(axis=0,ddof=1))
    coverage=np.mean(np.max(z,axis=1)<=c)
    assert .94<=coverage<=.96
