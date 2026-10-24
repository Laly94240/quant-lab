import numpy as np
import pandas as pd
import pytest

from quantlab.strategies import (
    BuyAndHold,
    CrossSectionalMomentum,
    LookaheadError,
    Strategy,
    TrendFollowing,
    check_no_lookahead,
    hold_between_rebalances,
    rebalance_mask,
)

STRATS = [
    TrendFollowing(lookback=100),
    TrendFollowing(lookback=100, weighting="inv_vol"),
    CrossSectionalMomentum(lookback=126, top_k=2),
    CrossSectionalMomentum(lookback=126, top_k=2, rebalance="D"),
    BuyAndHold(),
]


@pytest.mark.parametrize("s", STRATS, ids=lambda s: f"{s.name}-{s.params}")
def test_strategy_has_no_lookahead_and_valid_weights(s, small_prices):
    check_no_lookahead(s, small_prices)
    w = s.target_weights(small_prices)
    assert w.shape == small_prices.shape and not w.isna().any().any()
    assert (w >= 0).all().all() and w.sum(axis=1).max() <= 1 + 1e-9


def test_lookahead_detector_catches_future_peeking(small_prices):
    class PeekNextDay(Strategy):
        name = "peek"
        def target_weights(self, prices):
            return (prices.shift(-1) > prices).astype(float) / prices.shape[1]

    class FullSampleNormalised(Strategy):
        name = "fullsample"
        def target_weights(self, prices):
            z = (prices - prices.mean()) / prices.std()       # uses the whole sample
            return (z > 0).astype(float) / prices.shape[1]

    class CenteredWindow(Strategy):
        name = "centered"
        def target_weights(self, prices):
            sma = prices.rolling(21, center=True, min_periods=1).mean()
            return (prices > sma).astype(float) / prices.shape[1]

    for bad in (PeekNextDay(), FullSampleNormalised(), CenteredWindow()):
        with pytest.raises(LookaheadError):
            check_no_lookahead(bad, small_prices)


def test_trend_is_flat_until_enough_history(small_prices):
    w = TrendFollowing(lookback=200, rebalance="D").target_weights(small_prices)
    assert (w.iloc[:199] == 0).all().all()


def test_trend_signal_logic_on_toy_series():
    idx = pd.bdate_range("2020-01-01", periods=40)
    up = pd.DataFrame({"U": np.linspace(100, 140, 40), "D": np.linspace(140, 100, 40)}, index=idx)
    w = TrendFollowing(lookback=10, rebalance="D").target_weights(up)
    assert w["U"].iloc[-1] == 0.5 and w["D"].iloc[-1] == 0.0     # 1/n slot when above SMA, else cash


def test_momentum_selects_top_k_and_respects_abs_filter():
    idx = pd.bdate_range("2020-01-01", periods=60)
    p = pd.DataFrame({
        "W": np.linspace(100, 200, 60), "M": np.linspace(100, 130, 60),
        "L": np.linspace(100, 90, 60), "L2": np.linspace(100, 80, 60)}, index=idx)
    s = CrossSectionalMomentum(lookback=40, skip=5, top_k=3, rebalance="D", abs_filter=True)
    last = s.target_weights(p).iloc[-1]
    assert last["W"] == last["M"] == pytest.approx(1 / 3) and last["L"] == last["L2"] == 0.0
    s2 = CrossSectionalMomentum(lookback=40, skip=5, top_k=3, rebalance="D", abs_filter=False)
    assert (s2.target_weights(p).iloc[-1] > 0).sum() == 3


def test_rebalance_mask_and_hold():
    idx = pd.bdate_range("2020-01-29", periods=8)               # spans a month end
    m = rebalance_mask(idx, "M")
    assert m[0] and m.sum() == 2 and idx[m][1].month == 2 and idx[m][1].day <= 3
    w = pd.DataFrame({"A": np.arange(8, dtype=float)}, index=idx)
    held = hold_between_rebalances(w, "M")
    first_feb = np.flatnonzero(m)[1]
    assert (held["A"].iloc[first_feb:] == float(first_feb)).all()
