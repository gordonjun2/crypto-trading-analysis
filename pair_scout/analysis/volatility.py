"""Per-asset volatility/risk metrics (from volatility-trading.ipynb, corrected).

Raw metrics only — the notebook's MinMax "Total Volatility Score" is not ported
(it double-counts volatility and is outlier-sensitive; plan §2 M8). CHOP uses the
standard formula (B10 fix); cumulative vol replaced by annualized vol (B11 fix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolMetrics:
    ret_std: float  # std of simple returns over the window
    annualized_vol: float  # std of log returns * sqrt(bars_per_year)
    atr_pct: float  # Wilder ATR / last close * 100
    price_range_pct: float  # mean (High-Low)/Close * 100 over the window
    skewness: float  # skewness of simple returns
    choppiness: float  # standard CHOP in [0, 100]

    @property
    def inverse_vol_weight(self) -> float:
        return 1.0 / self.annualized_vol if self.annualized_vol > 0 else 0.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 168) -> float:
    """Wilder-smoothed ATR as a percentage of the last close."""
    n = min(window, max(len(close) - 1, 2))
    tr = true_range(high, low, close).iloc[-n:]
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean().iloc[-1]
    last = float(close.iloc[-1])
    if last <= 0 or not np.isfinite(atr):
        return float("nan")
    return float(atr / last * 100.0)


def choppiness(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    """Standard Choppiness Index: 100 * log10(sum(TR,n) / range_n) / log10(n)."""
    n = min(window, len(close) - 1)
    if n < 2:
        return float("nan")
    tr = true_range(high, low, close).iloc[-n:]
    hh = float(high.iloc[-n:].max())
    ll = float(low.iloc[-n:].min())
    rng = hh - ll
    tr_sum = float(tr.sum())
    if rng <= 0 or tr_sum <= 0:
        return float("nan")
    return float(100.0 * np.log10(tr_sum / rng) / np.log10(n))


def vol_metrics(df: pd.DataFrame, bars_per_year: int, atr_window: int = 168) -> VolMetrics:
    """Compute per-asset risk metrics from a frame with Close/High/Low columns."""
    if not {"Close", "High", "Low"}.issubset(df.columns):
        raise ValueError("frame must contain Close/High/Low columns")
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    if len(close) < 30 or close.isna().any():
        raise ValueError("need >= 30 clean bars for volatility metrics")
    rets = close.pct_change().dropna()
    log_rets = np.log(close).diff().dropna()
    return VolMetrics(
        ret_std=float(rets.std()),
        annualized_vol=float(log_rets.std() * np.sqrt(bars_per_year)),
        atr_pct=atr_pct(high, low, close, window=atr_window),
        price_range_pct=float(((high - low) / close).mean() * 100.0),
        skewness=float(rets.skew()),
        choppiness=choppiness(high, low, close),
    )


def vol_ratio(a: VolMetrics, b: VolMetrics) -> float:
    """Higher-vol leg / lower-vol leg (>= 1)."""
    lo, hi = sorted((a.annualized_vol, b.annualized_vol))
    if lo <= 0:
        return float("inf")
    return hi / lo


def long_notional_share(a: VolMetrics, b: VolMetrics) -> float:
    """Vol-balanced weight of leg ``a`` in [0, 1] (inverse-vol sizing)."""
    wa, wb = a.inverse_vol_weight, b.inverse_vol_weight
    total = wa + wb
    if total <= 0:
        return 0.5
    return wa / total
