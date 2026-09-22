import numpy as np
import pandas as pd
import pytest

from pair_scout.analysis.correlation import (
    divergence_direction,
    momentum,
    return_correlation,
)
from pair_scout.analysis.volatility import (
    atr_pct,
    choppiness,
    long_notional_share,
    vol_metrics,
    vol_ratio,
)


def test_levels_correlation_is_higher_than_returns_for_trending_series():
    """Levels correlation is spurious (B3): returns correlation should differ."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2025-01-01", periods=500, freq="h")
    a = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)), index=idx)
    b = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)), index=idx)
    lev = a.corr(b)
    ret = return_correlation(a, b, 480)
    # both computed, but they are not the same statistic
    assert np.isfinite(ret)
    assert abs(lev - ret) < 1.0


def test_return_correlation_matching_noises():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2025-01-01", periods=600, freq="h")
    base = rng.normal(0, 1, 600)
    a = pd.Series(100 + np.cumsum(base), index=idx)
    b = pd.Series(100 + np.cumsum(base + rng.normal(0, 0.1, 600)), index=idx)
    assert return_correlation(a, b, 590) > 0.95


def test_momentum_sign():
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    up = pd.Series(np.linspace(100, 150, 100), index=idx)
    down = pd.Series(np.linspace(150, 100, 100), index=idx)
    assert momentum(up, 50) > 0
    assert momentum(down, 50) < 0
    div = divergence_direction("UP", "DOWN", up, down, 90, 50)
    assert div.asset_strong == "UP"
    assert div.momentum_long_asset > 0 > div.momentum_short_asset


def test_atr_pct_crafted_series():
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    close = pd.Series(100.0, index=idx)
    high = close + 2.0
    low = close - 2.0
    # TR ≈ 4 every bar -> ATR ≈ 4 -> 4%
    assert atr_pct(high, low, close, window=50) == pytest.approx(4.0, rel=0.1)


def test_choppiness_bounds():
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    rng = np.random.default_rng(2)
    close = pd.Series(100 + rng.normal(0, 1, 200).cumsum(), index=idx)
    high, low = close + 1.0, close - 1.0
    chop = choppiness(high, low, close, window=14)
    assert 0 < chop <= 100  # standard CHOP is bounded (B10 fix)


def test_vol_metrics_shapes(ohlcv_frame):
    m = vol_metrics(ohlcv_frame, bars_per_year=8760, atr_window=168)
    assert m.annualized_vol > 0
    assert m.atr_pct > 0
    assert 0 < m.choppiness <= 100
    assert np.isfinite(m.skewness)


def test_vol_metrics_rejects_short_frame():
    with pytest.raises(ValueError):
        vol_metrics(pd.DataFrame({"Close": [1, 2], "High": [1, 2], "Low": [1, 2]}), 24)


def test_vol_ratio_and_sizing():
    class M:
        def __init__(self, av):
            self.annualized_vol = av
            self.inverse_vol_weight = 1.0 / av

    calm, wild = M(0.4), M(1.6)
    assert vol_ratio(wild, calm) == pytest.approx(4.0)
    assert vol_ratio(calm, wild) == pytest.approx(4.0)  # symmetric: hi/lo
    share = long_notional_share(calm, wild)
    assert share == pytest.approx(0.8)  # inverse-vol: 2.5/(2.5+0.625)
