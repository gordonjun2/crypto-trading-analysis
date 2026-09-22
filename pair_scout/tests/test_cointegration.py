import numpy as np
import pandas as pd
import pytest

from pair_scout.analysis.cointegration import (
    engle_granger,
    half_life_bars,
    hedge_ratio,
    hurst_exponent,
    rolling_zscore,
)


def test_cointegrated_pair_detected(cointegrated_pair):
    """Y = 2*X + noise: cointegrated in levels; in log space beta ~ 1."""
    y, x, _ = cointegrated_pair
    eg = engle_granger("Y", "X", y, x)
    assert eg.pvalue < 0.01
    assert eg.hedge_ratio == pytest.approx(1.0, abs=0.15)  # log(2x) ~ log x + ln 2
    assert eg.hurst < 0.6  # mean-reverting spread


def test_independent_walks_not_cointegrated(independent_walks):
    a, b = independent_walks
    eg = engle_granger("A", "B", a, b)
    assert eg.pvalue > 0.05


def test_hedge_ratio_recovery(cointegrated_pair):
    y, x, _ = cointegrated_pair
    ly, lx = np.log(y), np.log(x)
    assert hedge_ratio(ly, lx) == pytest.approx(1.0, abs=0.15)


def test_half_life_ar1(ar1_series):
    s, phi = ar1_series
    expected = -np.log(2) / np.log(phi)
    assert half_life_bars(s) == pytest.approx(expected, rel=0.15)


def test_half_life_explosive_trend_is_infinite():
    rng = np.random.default_rng(7)
    trend = pd.Series(np.arange(1000.0) + rng.normal(0, 0.5, 1000).cumsum())
    hl = half_life_bars(trend)
    assert np.isinf(hl)


def test_hurst_mean_reverting_below_trending(ar1_series):
    s, _ = ar1_series
    rng = np.random.default_rng(3)
    trend = pd.Series(np.cumsum(rng.normal(0.05, 1.0, 2000)))
    assert hurst_exponent(s) < hurst_exponent(trend)


def test_rolling_zscore_is_causal():
    rng = np.random.default_rng(11)
    s = pd.Series(rng.normal(size=100).cumsum())
    z = rolling_zscore(s, window=20)
    # z at t must not depend on future values
    s2 = s.copy()
    s2.iloc[-1] += 1e6
    z2 = rolling_zscore(s2, window=20)
    assert z.iloc[50] == pytest.approx(z2.iloc[50])
    assert np.isnan(z.iloc[18])  # window not yet full
    assert np.isfinite(z.iloc[19])


def test_orientation_symmetry(cointegrated_pair):
    """Both orientations should find the same relationship (B9 fix)."""
    y, x, _ = cointegrated_pair
    fwd = engle_granger("Y", "X", y, x)
    bwd = engle_granger("X", "Y", x, y)
    assert fwd.pvalue == pytest.approx(bwd.pvalue, abs=0.02)
