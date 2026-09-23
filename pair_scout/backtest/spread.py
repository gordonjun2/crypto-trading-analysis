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
from dataclasses import dataclass, field, replace

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
    gross_returns: pd.Series | None = None  # pre-fee returns (before size scaling)
    cost_returns: pd.Series | None = None  # fee/slippage drag (before size scaling)
    funding_returns: pd.Series | None = None  # perp funding paid/received
    mean_scale: float = 1.0  # mean notional multiplier applied (1.0 = unscaled)
    positions: pd.Series | None = None  # pos_prev series (for trade re-segmentation)
    signal: pd.Series | None = None  # entry momentum series (for conviction sizing)


def leg_weights(vol_y: float, vol_x: float, sizing: str,
                beta_y: float | None = None, beta_x: float | None = None) -> tuple[float, float]:
    """Notional weights for legs Y and X summing to 1.

    beta_balanced: w ∝ 1/beta -> w_y*beta_y ≈ w_x*beta_x, i.e. the long-short
    combo's BTC beta is ~0 (requires positive betas; falls back to equal).
    """
    if sizing == "equal":
        return 0.5, 0.5
    if sizing == "beta_balanced":
        if (
            beta_y is not None and beta_x is not None
            and math.isfinite(beta_y) and math.isfinite(beta_x)
            and beta_y > 0 and beta_x > 0
        ):
            wy, wx = 1.0 / beta_y, 1.0 / beta_x
            total = wy + wx
            return wy / total, wx / total
        return 0.5, 0.5
    # vol_balanced
    if vol_y <= 0 or vol_x <= 0 or not math.isfinite(vol_y) or not math.isfinite(vol_x):
        return 0.5, 0.5
    wy, wx = 1.0 / vol_y, 1.0 / vol_x
    total = wy + wx
    return wy / total, wx / total


def combo_beta(beta_y: float, beta_x: float, w_y: float, w_x: float) -> float:
    """Beta of the long-Y/short-X book: w_y*beta_y - w_x*beta_x."""
    if not all(math.isfinite(v) for v in (beta_y, beta_x)):
        return float("nan")
    return w_y * beta_y - w_x * beta_x


def _positions(z: pd.Series, entry_z: float, exit_z: float, stop_z: float,
               max_hold_bars: int | None = None) -> pd.Series:
    pos = 0
    held = 0
    out = np.zeros(len(z), dtype=int)
    zvals = z.to_numpy()
    for i, zv in enumerate(zvals):
        if pos != 0:
            held += 1
            if max_hold_bars is not None and held >= max_hold_bars:
                pos, held = 0, 0
        if not math.isfinite(zv):
            out[i] = pos
            continue
        if pos == 0:
            if zv <= -entry_z:
                pos, held = 1, 0
            elif zv >= entry_z:
                pos, held = -1, 0
        elif pos == 1:
            if zv >= -exit_z or zv <= -stop_z:
                pos, held = 0, 0
        else:  # pos == -1
            if zv <= exit_z or zv >= stop_z:
                pos, held = 0, 0
        out[i] = pos
    return pd.Series(out, index=z.index)


def _positions_trend(
    momentum: pd.Series,
    enter_level: float,
    exit_level: float,
    max_hold_bars: int | None = None,
    z_entry: float | None = None,
    z_lookback_bars: int = 720,
    spread: pd.Series | None = None,
    stop_pct: float = 0.0,
    entry_allowed: pd.Series | None = None,
    min_entry_i: int = 0,
    cooldown_bars: int = 0,
    trail_pct: float = 0.0,
) -> pd.Series:
    """Long the spread while its momentum is strong; multiple exits.

    Entry: absolute (momentum >= enter_level) or adaptive z-score
    (momentum z >= z_entry over z_lookback_bars) AND momentum >= enter_level.
    Exits: momentum fade (<= exit_level), hard time stop (max_hold_bars),
    adverse spread move (stop_pct on log spread), a trailing stop
    (trail_pct below the highest spread since entry; 0 = off), or a
    disallowed entry day (regime gate) never OPENS a position.
    """
    n = len(momentum)
    pos = 0
    held = 0
    cooldown = 0
    entry_level = float("nan")
    peak_level = float("nan")
    mom_vals = momentum.to_numpy()
    spread_vals = spread.to_numpy() if spread is not None else None
    if entry_allowed is not None:
        # align to (possibly warmup-extended) sim index; warmup region -> False
        allowed = (
            entry_allowed.reindex(momentum.index)
            .fillna(False).infer_objects(copy=False).astype(bool).to_numpy()
        )
    else:
        allowed = None
    if z_entry is not None:
        roll_mean = momentum.rolling(z_lookback_bars).mean().to_numpy()
        roll_std = momentum.rolling(z_lookback_bars).std().to_numpy()
    else:
        roll_mean = roll_std = np.full(n, np.nan)
    out = np.zeros(n, dtype=int)
    for i in range(n):
        if pos == 1:
            held += 1
            stopped = False
            if max_hold_bars is not None and held >= max_hold_bars:
                stopped = True
            elif (
                spread_vals is not None
                and stop_pct > 0
                and math.isfinite(spread_vals[i])
                and spread_vals[i] - entry_level <= -stop_pct
            ):
                stopped = True
            elif (
                spread_vals is not None
                and trail_pct > 0
                and math.isfinite(spread_vals[i])
            ):
                if not math.isfinite(peak_level):
                    peak_level = spread_vals[i]
                elif spread_vals[i] > peak_level:
                    peak_level = spread_vals[i]
                elif peak_level - spread_vals[i] >= trail_pct:
                    stopped = True
            if stopped:
                pos, held = 0, 0
                entry_level, peak_level = float("nan"), float("nan")
                cooldown = cooldown_bars
        mv = mom_vals[i]
        if math.isfinite(mv):
            if pos == 0:
                gate_ok = allowed is None or (i < len(allowed) and bool(allowed[i]))
                if not gate_ok or i < min_entry_i or cooldown > 0:
                    if cooldown > 0:
                        cooldown -= 1
                    out[i] = 0
                    continue
                if z_entry is not None:
                    z_ok = (
                        math.isfinite(roll_mean[i])
                        and roll_std[i] > 0
                        and (mv - roll_mean[i]) / roll_std[i] >= z_entry
                    )
                else:
                    z_ok = True
                if z_ok and mv >= enter_level:
                    pos, held = 1, 0
                    entry_level = (
                        spread_vals[i]
                        if spread_vals is not None and math.isfinite(spread_vals[i])
                        else float("nan")
                    )
                    peak_level = entry_level
            elif mv <= exit_level:
                pos, held = 0, 0
                entry_level, peak_level = float("nan"), float("nan")
                cooldown = cooldown_bars
        out[i] = pos
    return pd.Series(out, index=momentum.index)


def simulate_divergence(
    log_long: pd.Series,
    log_short: pd.Series,
    beta_long: float,
    beta_short: float,
    sizing: str = "beta_balanced",
    signal_window_bars: int = 168,
    entry_mom_per_bar: float = 0.02 / 168,
    exit_mom_per_bar: float = 0.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    bars_per_year: int = 8760,
    max_hold_bars: int | None = None,
    entry_mode: str = "zscore",
    z_entry: float = 1.0,
    z_lookback_bars: int = 720,
    momentum_shift_bars: int = 0,
    spread_stop_pct: float = 0.0,
    entry_allowed: pd.Series | None = None,
    trade_start: pd.Timestamp | None = None,
    cooldown_bars: int = 0,
    size_scale: pd.Series | None = None,
    funding_long: pd.Series | None = None,
    funding_short: pd.Series | None = None,
    funding_stress: float = 1.0,
    spread_trail_pct: float = 0.0,
    signal_override: pd.Series | None = None,
) -> SimResult:
    """Short-horizon divergence book: long the strong leg, short the weak leg.

    Signal: mean per-bar change of log(strong) - log(weak) over the signal window,
    optionally skipping the most recent ``momentum_shift_bars`` bars (short-term
    reversal hedge) — or an externally supplied ``signal_override`` series (e.g.
    a funding differential), which replaces the computed momentum entirely.
    Entry is adaptive (momentum z-score) or absolute. Exits: momentum fade, hard
    time stop, adverse spread stop, regime gate on entries. Weights balance the
    book's BTC beta to ~0 (beta_balanced) or split equally.
    """
    if signal_override is not None:
        momentum = signal_override.astype(float)
    else:
        diff = (log_long - log_short).diff()
        momentum = diff.rolling(signal_window_bars).mean()
        if momentum_shift_bars > 0:
            momentum = momentum.shift(momentum_shift_bars)
    spread = log_long - log_short
    pos = _positions_trend(
        momentum,
        entry_mom_per_bar,
        exit_mom_per_bar,
        max_hold_bars=max_hold_bars,
        z_entry=z_entry if entry_mode == "zscore" else None,
        z_lookback_bars=z_lookback_bars,
        spread=spread,
        stop_pct=spread_stop_pct,
        entry_allowed=entry_allowed,
        min_entry_i=(
            spread.index.searchsorted(trade_start) if trade_start is not None else 0
        ),
        cooldown_bars=cooldown_bars,
        trail_pct=spread_trail_pct,
    )
    if trade_start is not None:
        start_i = spread.index.searchsorted(trade_start)
        pos = pos.iloc[start_i:]
        log_long = log_long.iloc[start_i:]
        log_short = log_short.iloc[start_i:]
    pos_prev = pos.shift(1).fillna(0).astype(int)  # B5: execute next bar
    w_long, w_short = leg_weights(
        float("nan"), float("nan"), sizing, beta_y=beta_long, beta_x=beta_short
    )
    dll = log_long.diff()
    dls = log_short.diff()
    # size_scale: continuous notional multiplier (dispersion-scaled sizing);
    # both PnL and fees scale with notional, so net = (gross - cost) * scale.
    if size_scale is None:
        scale = pd.Series(1.0, index=pos_prev.index)
    else:
        scale = size_scale.reindex(pos_prev.index).fillna(1.0).astype(float)
    gross = scale * pos_prev * (w_long * dll - w_short * dls)
    turnover = pos_prev.diff().abs().fillna(pos_prev.abs())
    cost = (
        scale
        * turnover
        * (w_long + w_short)
        * (fee_bps + slippage_bps)
        / 1e4
    )
    # perpetual funding: paid/received by positions held at settlement times
    if funding_long is not None or funding_short is not None:
        f_long = (
            funding_long.reindex(pos_prev.index).fillna(0.0).astype(float)
            if funding_long is not None
            else pd.Series(0.0, index=pos_prev.index)
        )
        f_short = (
            funding_short.reindex(pos_prev.index).fillna(0.0).astype(float)
            if funding_short is not None
            else pd.Series(0.0, index=pos_prev.index)
        )
        # funding COST series (positive = paid by the book): longs pay f>0,
        # shorts receive it -> cost = w_l*f_long - w_s*f_short
        funding = (
            scale
            * pos_prev
            * (w_long * f_long - w_short * f_short)
            * funding_stress
        )
    else:
        funding = pd.Series(0.0, index=pos_prev.index)
    net = (gross - cost - funding).dropna()
    trades = _segment_trades(net, pos_prev)
    result = _finalize(net, trades, bars_per_year)
    return replace(
        result,
        gross_returns=gross,
        cost_returns=cost,
        funding_returns=funding,
        mean_scale=float(scale.mean()) if len(scale) else 1.0,
        positions=pos_prev,
        signal=momentum,
    )


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
    beta_y: float | None = None,
    beta_x: float | None = None,
    max_hold_bars: int | None = None,
) -> SimResult:
    """Rolling-z spread strategy, next-bar execution, two legs, fees on turnover."""
    spread = log_y - beta * log_x
    z = rolling_zscore(spread, zscore_window)
    pos = _positions(z, entry_z, exit_z, stop_z, max_hold_bars=max_hold_bars)
    pos_prev = pos.shift(1).fillna(0).astype(int)  # B5 fix: execute next bar

    wy, wx = leg_weights(vol_y, vol_x, sizing, beta_y=beta_y, beta_x=beta_x)
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
    if std > 0:
        sharpe = float(net.mean() / std * math.sqrt(bars_per_year))
    else:
        sharpe = 0.0  # flat book: no information, do not poison fold means
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
