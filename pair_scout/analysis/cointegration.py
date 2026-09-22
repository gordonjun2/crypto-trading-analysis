"""Engle-Granger cointegration analysis (cointegration notebook, corrected).

Fixes (plan §2):
- B9: EG is asymmetric — both orientations are tested, the better one is kept;
- B4/B12: z-scores are rolling-only, and the long/short direction follows one
  consistent rule derived from the spread orientation;
- B5: the backtest module executes signals on the next bar (see backtest/spread.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EGResult:
    """Result of the better Engle-Granger orientation.

    Spread convention: ``spread = log(Y) - hedge_ratio * log(X)``.
    z > 0  => Y is rich relative to X  =>  short Y, long X.
    z < 0  => Y is cheap relative to X =>  long Y, short X.
    """

    asset_y: str
    asset_x: str
    pvalue: float
    adf_stat: float
    hedge_ratio: float
    spread: pd.Series
    zscore: float
    half_life_bars: float
    hurst: float


def hedge_ratio(log_y: pd.Series, log_x: pd.Series) -> float:
    """OLS slope of log_y on log_x."""
    var = float(np.var(log_x.to_numpy()))
    if var <= 0:
        return float("nan")
    cov = float(np.cov(log_y.to_numpy(), log_x.to_numpy())[0, 1])
    return cov / var


def spread_series(log_y: pd.Series, log_x: pd.Series, beta: float) -> pd.Series:
    return log_y - beta * log_x


def half_life_bars(spread: pd.Series) -> float:
    """OU half-life from an AR(1) regression of the spread on its lag."""
    s = spread.dropna()
    s_lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    common = s_lag.index.intersection(ds.index)
    if len(common) < 30:
        return float("nan")
    x = s_lag.loc[common].to_numpy()
    y = ds.loc[common].to_numpy()
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 0:
        return float("nan")
    b = float(np.dot(x_centered, y - y.mean())) / denom
    one_plus_b = 1.0 + b
    if one_plus_b <= 0:
        return float("inf")  # oscillating/explosive: no mean reversion
    lam = np.log(one_plus_b)
    if lam >= 0:
        return float("inf")  # non-mean-reverting at this horizon
    return float(-np.log(2.0) / lam)


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Hurst exponent via the variance-of-differences (aggregate variance) method."""
    s = pd.Series(series, dtype=float).dropna().to_numpy()
    if len(s) < max_lag * 4:
        return float("nan")
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags])
    ok = np.isfinite(tau) & (tau > 0)
    if ok.sum() < 5:
        return float("nan")
    slope, _ = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)
    return float(slope)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling z-score (no full-sample statistics — B4 fix)."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def engle_granger(
    asset_y: str,
    asset_x: str,
    close_y: pd.Series,
    close_x: pd.Series,
    zscore_window: int = 15,
) -> EGResult:
    """Run EG in both orientations and keep the better one (lower p-value, B9 fix)."""
    log_y = np.log(close_y.astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
    log_x = np.log(close_x.astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
    common = log_y.index.intersection(log_x.index)
    if len(common) < 100:
        raise ValueError(f"insufficient overlap for {asset_y}/{asset_x}")
    ly, lx = log_y.loc[common], log_x.loc[common]

    p_fwd, stat_fwd = _eg_once(ly, lx)
    p_bwd, stat_bwd = _eg_once(lx, ly)
    if p_bwd < p_fwd:
        asset_y, asset_x, ly, lx = asset_x, asset_y, lx, ly
        pvalue, adf_stat = p_bwd, stat_bwd
    else:
        pvalue, adf_stat = p_fwd, stat_fwd

    beta = hedge_ratio(ly, lx)
    if not np.isfinite(beta):
        raise ValueError(f"degenerate hedge ratio for {asset_y}/{asset_x}")
    spread = spread_series(ly, lx, beta)
    z = rolling_zscore(spread, zscore_window)
    return EGResult(
        asset_y=asset_y,
        asset_x=asset_x,
        pvalue=float(pvalue),
        adf_stat=float(adf_stat),
        hedge_ratio=beta,
        spread=spread,
        zscore=float(z.iloc[-1]) if np.isfinite(z.iloc[-1]) else float("nan"),
        half_life_bars=half_life_bars(spread),
        hurst=hurst_exponent(spread),
    )


def _eg_once(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    try:
        stat, pvalue, _ = coint(y.to_numpy(), x.to_numpy())
    except Exception as exc:  # noqa: BLE001 — degenerate series can raise inside sm
        logger.debug("coint failed (%s vs %s): %s", y.name, x.name, exc)
        return 1.0, float("nan")
    return float(pvalue), float(stat)
