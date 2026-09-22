"""Spread-trading simulation on two legs (plan §5).

Consolidated engine (arms 2/3): rolling z-score signal, NEXT-bar execution (B5 fix),
per-leg fees + slippage (M2 fix), vol-balanced or equal leg sizing.
Notebook engine (arm 1): faithful-but-runnable reimplementation of the original
logic — full-sample z (look-ahead B4), same-bar execution, no fees, long-only.

Leg convention mirrors analysis/cointegration.py: spread = log(Y) - beta*log(X);
pos=+1 means long Y / short X (entered when z <= -entry), pos=-1 the mirror.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pair_scout.analysis.cointegration import rolling_zscore


@dataclass(frozen=True)
class Trade:
    entry_idx: int
    exit_idx: int
    direction: int  # +1 long-spread, -1 short-spread
    pnl: float


@dataclass(frozen=True)
class SimResult:
    bar_returns: pd.Series
    trades: list[Trade] = field(default_factory=list)
    total_return: float = 0.0
    sharpe: float = float("nan")
    max_drawdown: float = 0.0
    hit_rate: float = float("nan")
    n_trades: int = 0


def leg_weights(vol_y: float, vol_x: float, sizing: str) -> tuple[float, float]:
    """Notional weights for legs Y and X summing to 1."""
    if sizing == "equal":
        return 0.5, 0.5
    if vol_y <= 0 or vol_x <= 0 or not math.isfinite(vol_y) or not math.isfinite(vol_x):
        return 0.5, 0.5
    wy, wx = 1.0 / vol_y, 1.0 / vol_x
    total = wy + wx
    return wy / total, wx / total


def _positions(z: pd.Series, entry_z: float, exit_z: float, stop_z: float) -> pd.Series:
    pos = 0
    out = np.zeros(len(z), dtype=int)
    zvals = z.to_numpy()
    for i, zv in enumerate(zvals):
        if not math.isfinite(zv):
            out[i] = pos
            continue
        if pos == 0:
            if zv <= -entry_z:
                pos = 1
            elif zv >= entry_z:
                pos = -1
        elif pos == 1:
            if zv >= -exit_z or zv <= -stop_z:
                pos = 0
        else:  # pos == -1
            if zv <= exit_z or zv >= stop_z:
                pos = 0
        out[i] = pos
    return pd.Series(out, index=z.index)


def simulate_consolidated(
    log_y: pd.Series,
    log_x: pd.Series,
    beta: float,
    vol_y: float,
    vol_x: float,
    sizing: str = "vol_balanced",
    zscore_window: int = 15,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    bars_per_year: int = 8760,
) -> SimResult:
    """Rolling-z spread strategy, next-bar execution, two legs, fees on turnover."""
    spread = log_y - beta * log_x
    z = rolling_zscore(spread, zscore_window)
    pos = _positions(z, entry_z, exit_z, stop_z)
    pos_prev = pos.shift(1).fillna(0).astype(int)  # B5 fix: execute next bar

    wy, wx = leg_weights(vol_y, vol_x, sizing)
    dly = log_y.diff()
    dlx = log_x.diff()
    gross = pos_prev * (wy * dly - wx * dlx)

    turnover = pos_prev.diff().abs().fillna(pos_prev.abs())
    cost = turnover * (wy + wx) * (fee_bps + slippage_bps) / 1e4
    net = (gross - cost).dropna()

    trades = _segment_trades(net, pos_prev)
    return _finalize(net, trades, bars_per_year)


def simulate_notebook(
    log_y: pd.Series,
    log_x: pd.Series,
    beta: float,
    entry_z: float = 1.0,
    exit_z: float = 1.0,
    bars_per_year: int = 8760,
) -> SimResult:
    """Faithful arm: full-sample z (look-ahead), same-bar execution, no fees, long-only.

    Reproduces signals_zscore_evolution semantics (buy z<=-entry, sell z>=-exit at 1)
    with the label bug (B2) fixed so it actually runs.
    """
    spread = log_y - beta * log_x
    z = (spread - spread.mean()) / spread.std()  # full-sample stats = look-ahead (B4)
    pos = 0
    out = np.zeros(len(z), dtype=int)
    for i, zv in enumerate(z.to_numpy()):
        if not math.isfinite(zv):
            out[i] = pos
            continue
        if pos == 0 and zv <= -entry_z:
            pos = 1
        elif pos == 1 and zv >= exit_z:
            pos = 0
        out[i] = pos
    pos = pd.Series(out, index=z.index)  # same-bar execution (B5: intentionally kept)
    net = (pos * spread.diff()).dropna()
    trades = _segment_trades(net, pos)
    return _finalize(net, trades, bars_per_year)


def _segment_trades(net: pd.Series, pos: pd.Series) -> list[Trade]:
    trades: list[Trade] = []
    positions = pos.to_numpy()
    rets = net.to_numpy()
    in_trade = False
    start = 0
    direction = 0
    for i in range(len(positions)):
        if not in_trade and positions[i] != 0:
            in_trade, start, direction = True, i, positions[i]
        elif in_trade and positions[i] == 0:
            trades.append(
                Trade(start, i - 1, direction, float(np.nansum(rets[start:i])))
            )
            in_trade = False
    if in_trade:  # still open at the end
        trades.append(
            Trade(start, len(positions) - 1, direction, float(np.nansum(rets[start:])))
        )
    return trades


def _finalize(net: pd.Series, trades: list[Trade], bars_per_year: int) -> SimResult:
    total = float(net.sum())
    std = float(net.std())
    sharpe = float(net.mean() / std * math.sqrt(bars_per_year)) if std > 0 else float("nan")
    equity = net.cumsum()
    dd = float((equity - equity.cummax()).min()) if len(equity) else 0.0
    hit = (
        sum(1 for t in trades if t.pnl > 0) / len(trades)
        if trades
        else float("nan")
    )
    return SimResult(
        bar_returns=net,
        trades=trades,
        total_return=total,
        sharpe=sharpe,
        max_drawdown=dd,
        hit_rate=hit,
        n_trades=len(trades),
    )
