"""Shared fixtures: synthetic markets with known statistical properties."""

import numpy as np
import pandas as pd
import pytest

rng = np.random.default_rng(42)


def make_index(n: int, freq: str = "h") -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq=freq)


@pytest.fixture
def cointegrated_pair() -> tuple[pd.Series, pd.Series, float]:
    """Y = 2*X + stationary noise; X a random walk. beta_hat should be ~2."""
    n = 1500
    x = 100 + np.cumsum(rng.normal(0, 0.5, n))
    noise = rng.normal(0, 0.3, n)
    y = 2 * x + noise
    idx = make_index(n)
    return pd.Series(y, index=idx), pd.Series(x, index=idx), 2.0


@pytest.fixture
def independent_walks() -> tuple[pd.Series, pd.Series]:
    n = 1500
    a = 100 + np.cumsum(rng.normal(0, 1.0, n))
    b = 100 + np.cumsum(rng.normal(0, 1.0, n))
    idx = make_index(n)
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


@pytest.fixture
def ar1_series() -> tuple[pd.Series, float]:
    """AR(1) with phi=0.9 -> half-life = -ln2/ln(0.9) ≈ 6.58 bars."""
    n = 4000
    phi = 0.9
    eps = rng.normal(0, 1.0, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return pd.Series(x), phi


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    """Crafted OHLC frame for volatility tests: rising then falling closes."""
    n = 300
    close = pd.Series(
        np.concatenate([np.linspace(100, 120, n // 2), np.linspace(120, 95, n - n // 2)])
    )
    high = close * 1.02
    low = close * 0.98
    return pd.DataFrame({"Close": close, "High": high, "Low": low})
