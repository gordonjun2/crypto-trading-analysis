"""Return-based correlation and momentum divergence (correlation notebook, corrected).

The notebook correlated price *levels* (spurious, B3); here correlation is computed
on log returns, and the markdown-only "long the stronger asset, short the weaker"
direction rule (M5) is implemented explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Divergence:
    return_correlation: float
    momentum_long_asset: float
    momentum_short_asset: float
    asset_strong: str


def return_correlation(close_a: pd.Series, close_b: pd.Series, window_bars: int) -> float:
    """Pearson correlation of log returns over the trailing window."""
    n = min(window_bars, len(close_a))
    if n < 30:
        return float("nan")
    la = np.log(close_a).diff().dropna().iloc[-n:]
    lb = np.log(close_b).diff().dropna().iloc[-n:]
    joined = pd.concat([la, lb], axis=1, keys=["a", "b"]).dropna()
    if len(joined) < 30:
        return float("nan")
    corr = joined["a"].corr(joined["b"])
    return float(corr) if np.isfinite(corr) else float("nan")


def momentum(close: pd.Series, lookback_bars: int) -> float:
    """Simple momentum: last close / close N bars ago - 1."""
    n = min(lookback_bars, len(close) - 1)
    base = float(close.iloc[-(n + 1)])
    if base <= 0:
        return float("nan")
    return float(close.iloc[-1]) / base - 1.0


def divergence_direction(
    asset_a: str,
    asset_b: str,
    close_a: pd.Series,
    close_b: pd.Series,
    window_bars: int,
    momentum_lookback_bars: int,
) -> Divergence:
    corr = return_correlation(close_a, close_b, window_bars)
    mom_a = momentum(close_a, momentum_lookback_bars)
    mom_b = momentum(close_b, momentum_lookback_bars)
    strong, weak = (
        (asset_a, asset_b) if (mom_a if np.isfinite(mom_a) else 0) >= (mom_b if np.isfinite(mom_b) else 0)
        else (asset_b, asset_a)
    )
    mom_strong = mom_a if strong == asset_a else mom_b
    mom_weak = mom_b if strong == asset_a else mom_a
    return Divergence(
        return_correlation=corr,
        momentum_long_asset=float(mom_strong),
        momentum_short_asset=float(mom_weak),
        asset_strong=strong,
    )
