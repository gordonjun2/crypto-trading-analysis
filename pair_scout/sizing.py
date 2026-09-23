"""Dispersion-scaled position sizing — continuous risk allocation, not binary gates.

The JEV regime GATE (hard on/off) failed OOS validation. This module implements the
alternative: keep trading always on, but scale each book's notional by how favorable
the market state is, computed from strictly-past data:

- wide cross-sectional momentum dispersion -> more raw material for long-strong /
  short-weak books -> size up;
- low average pairwise return correlation (no correlated chop / single macro move)
  -> size up.

Causality: the multiplier applied to day D uses only bars up to the close of D-1
(trailing metrics + trailing z-score normalization), so it is decision-time safe
when a position opened during day D is sized with it.

The multiplier is symmetric around 1.0 and clipped to [scale_min, scale_max], so
average exposure stays ~1x notional: risk is reallocated across time, not added.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pair_scout.analysis.volatility import true_range
from pair_scout.config import AppConfig
from pair_scout.data.loader import Panel

logger = logging.getLogger(__name__)

METRICS = ("dispersion", "avg_corr", "chop")


def _clip(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return float(min(max(value, lo), hi))


def market_daily_metrics(
    panel: Panel,
    mom_days: float = 7.0,
    corr_days: float = 3.0,
    chop_window_bars: int = 24,
) -> pd.DataFrame:
    """Per-UTC-day market-state metrics, each using only data before that day.

    dispersion: cross-sectional std of trailing ``mom_days`` token returns
        (wide = clear leaders and laggards).
    avg_corr:   mean pairwise correlation of ``corr_days``-window returns
        (high = the universe moves as one asset = correlated chop).
    chop:       mean Choppiness Index across the universe over
        ``chop_window_bars`` (100 = pure chop, lower = directional).
    """
    day = panel.bars_per_day
    closes = panel.closes
    stamps = pd.DatetimeIndex(sorted(set(closes.index.floor("D"))))
    mom_bars = max(int(mom_days * day), 2)
    corr_bars = max(int(corr_days * day), 10)
    tr_cache = {
        p: true_range(panel.frames[p]["High"], panel.frames[p]["Low"],
                      panel.frames[p]["Close"])
        for p in panel.pairs
    }
    rows = []
    for stamp in stamps:
        past = closes.loc[:stamp - pd.Timedelta(nanoseconds=1)]
        if len(past) < mom_bars + 1:
            rows.append((stamp, np.nan, np.nan, np.nan))
            continue
        base = past.iloc[-(mom_bars + 1)]
        moms = (past.iloc[-1] / base - 1.0).dropna()
        dispersion = float(moms.std()) if len(moms) >= 5 else np.nan
        sample = past.tail(corr_bars).pct_change().dropna(how="all")
        if sample.shape[0] >= 10 and sample.shape[1] >= 5:
            corr = sample.corr(min_periods=corr_bars // 2)
            n = len(corr)
            mat = corr.to_numpy()
            avg_corr = float((mat.sum() - n) / max(n * (n - 1), 1))
            if not math.isfinite(avg_corr):
                avg_corr = np.nan
        else:
            avg_corr = np.nan
        chops = []
        chop_slice = past.index[-chop_window_bars:]
        for p in panel.pairs:
            tr = tr_cache[p].reindex(chop_slice)
            hi = panel.frames[p]["High"].reindex(chop_slice)
            lo = panel.frames[p]["Low"].reindex(chop_slice)
            rng = float(hi.max() - lo.min()) if hi.notna().any() else 0.0
            tr_sum = float(tr.sum()) if tr.notna().any() else 0.0
            n = int(tr.notna().sum())
            if rng > 0 and tr_sum > 0 and n >= 2:
                chops.append(100.0 * np.log10(tr_sum / rng) / np.log10(n))
        chop = float(np.mean(chops)) if len(chops) >= 5 else np.nan
        rows.append((stamp, dispersion, avg_corr, chop))
    return pd.DataFrame(
        rows, columns=["date", "dispersion", "avg_corr", "chop"]
    ).set_index("date")


def _rolling_z(series: pd.Series, lookback_days: int, min_periods: int) -> pd.Series:
    mean = series.rolling(lookback_days, min_periods=min_periods).mean()
    std = series.rolling(lookback_days, min_periods=min_periods).std()
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.clip(-3.0, 3.0)


@dataclass(frozen=True)
class ScaleSpec:
    """Fully describes a sizing variant (research grid + BacktestConfig share this)."""

    mode: str = "off"  # off | market | vol_target
    metric: str = "disp_corr"  # disp_corr | disp | corr | chop  (market mode)
    scale_min: float = 0.5
    scale_max: float = 1.5
    slope: float = 0.3  # multiplier change per 1 sigma of the combined z
    z_lookback_days: int = 60
    disp_weight: float = 0.5  # market mode, disp_corr: rest goes to inverse corr
    vol_target: float = 0.10  # vol_target mode: annualized target for the book

    @property
    def label(self) -> str:
        if self.mode == "off":
            return "off"
        bounds = f"[{self.scale_min:g},{self.scale_max:g}]"
        if self.mode == "vol_target":
            return f"vol_target({self.vol_target:.0%}){bounds}"
        if self.mode == "efficiency":
            return f"efficiency(k{self.slope:g}){bounds}"
        if self.mode == "conviction":
            return f"conviction(k{self.slope:g}){bounds}"
        if self.mode == "jev_regime_soft":
            return f"jev_soft{bounds}"
        if self.mode == "combo":
            return f"combo(jevxvt{self.vol_target:.0%}){bounds}"
        return (
            f"{self.mode}:{self.metric}{bounds}k{self.slope:g}"
            f"z{self.z_lookback_days}w{self.disp_weight:g}"
        )


def market_scale_series(
    panel: Panel, spec: ScaleSpec, metrics: pd.DataFrame | None = None
) -> pd.Series:
    """Per-bar notional multiplier from market-state z-scores (causal)."""
    if metrics is None:
        metrics = market_daily_metrics(panel)
    if spec.metric == "disp":
        z = _rolling_z(
            metrics["dispersion"], spec.z_lookback_days, max(spec.z_lookback_days // 3, 10)
        )
    elif spec.metric == "corr":
        z = -_rolling_z(
            metrics["avg_corr"], spec.z_lookback_days, max(spec.z_lookback_days // 3, 10)
        )
    elif spec.metric == "chop":
        z = -_rolling_z(
            metrics["chop"], spec.z_lookback_days, max(spec.z_lookback_days // 3, 10)
        )
    else:  # disp_corr
        w = spec.disp_weight
        z = w * _rolling_z(
            metrics["dispersion"], spec.z_lookback_days, max(spec.z_lookback_days // 3, 10)
        ) + (1.0 - w) * -_rolling_z(
            metrics["avg_corr"], spec.z_lookback_days, max(spec.z_lookback_days // 3, 10)
        )
    daily = (1.0 + spec.slope * z).map(
        lambda v: _clip(v, spec.scale_min, spec.scale_max)
    )
    return daily


def vol_target_scale_series(
    candidate_returns: pd.Series, panel: Panel, spec: ScaleSpec
) -> pd.Series:
    """Per-bar multiplier: scale up when the book's realized vol is below target.

    Uses trailing realized vol of the candidate's own net returns (annualized),
    shifted so day D's multiplier only sees returns through D-1.
    """
    day = panel.bars_per_day
    ann = np.sqrt(day * 365)
    realized = candidate_returns.rolling(3 * day, min_periods=day).std() * ann
    # shift(1): day D's multiplier uses realized vol through the close of D-1
    daily = (spec.vol_target / realized).resample("1D").last().shift(1).dropna()
    daily = daily.map(lambda v: _clip(v, spec.scale_min, spec.scale_max))
    return daily


def expand_to_bars(daily: pd.Series, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Map a per-UTC-day multiplier onto bars: bars of day D get day D's value.

    Day D's value is computed from data through D-1 (see module docstring), so a
    position held during day D is sized with information already known when the
    day started.
    """
    keys = pd.Series(bar_index.floor("D"), index=bar_index)
    scale = keys.map(daily)
    return scale.fillna(1.0).astype(float)


def gates_scale_series(
    gates: dict[str, float], bar_index: pd.DatetimeIndex, spec: ScaleSpec
) -> pd.Series:
    """JEV regime probability as a SOFT size: scale = a + b * P(momentum regime).

    The binary gate version (trade only when P >= threshold) failed OOS
    validation; here P is used continuously: high conviction -> full size, low
    conviction -> reduced (never zero) size.
    """
    keys = pd.Series(bar_index.floor("D"), index=bar_index) - pd.Timedelta(days=1)
    probs = keys.map(lambda d: gates.get(str(d.date())))
    daily = probs.map(
        lambda p: _clip(
            spec.scale_min + (spec.scale_max - spec.scale_min) * p
            if p is not None
            else float("nan"),
            spec.scale_min,
            spec.scale_max,
        )
    )
    return daily.fillna(1.0).astype(float)


def efficiency_scale_series(
    candidate_returns: pd.Series, panel: Panel, spec: ScaleSpec
) -> pd.Series:
    """Per-bar multiplier from the book's Kaufman-style efficiency ratio.

    ER = |sum(returns, n)| / sum(|returns|, n) over a trailing window: near 1 =
    the book is trending cleanly, near 0 = chop. Scale up trending books, down
    choppy ones. Shifted so day D's multiplier only sees returns through D-1.
    """
    day = panel.bars_per_day
    window = max(2 * day, 12)
    gross = candidate_returns.rolling(window).sum()
    mag = candidate_returns.abs().rolling(window).sum()
    er = (gross.abs() / mag.replace(0.0, np.nan)).clip(0.0, 1.0)
    z = _rolling_z(er, spec.z_lookback_days, max(spec.z_lookback_days // 3, 10))
    daily = (1.0 + spec.slope * z).map(
        lambda v: _clip(v, spec.scale_min, spec.scale_max)
    )
    return daily.resample("1D").last().shift(1).dropna()


def conviction_scale_series(
    signal: pd.Series, z_entry: float, z_lookback_bars: int
) -> pd.Series:
    """Per-bar multiplier from entry-signal conviction (pair-level momentum z).

    The entry gate fires at momentum z >= z_entry; conviction sizing scales
    linearly beyond that: scale = 1 + slope * (z - z_entry), clipped. A z=1
    entry trades 1x, a z=2 entry trades (1 + slope)x — the 'momentum spread is
    wide' component at the pair level.
    """
    lookback = signal.rolling(z_lookback_bars).mean()
    std = signal.rolling(z_lookback_bars).std()
    z = (signal - lookback) / std.replace(0.0, np.nan)
    return 1.0 + z.sub(z_entry)  # slope applied by caller via clip bounds


def combo_scale_series(
    gate_bars: pd.Series, candidate_returns: pd.Series, panel: Panel, spec: ScaleSpec
) -> pd.Series:
    """Product of the JEV soft regime size and the vol-target size, clipped.

    Both components are continuous (never zero) — the book keeps trading, just
    with risk reallocated toward favorable states (research winner, 2026-09).
    """
    vt = expand_to_bars(
        vol_target_scale_series(candidate_returns, panel, spec),
        candidate_returns.index,
    )
    out = gate_bars.reindex(vt.index).fillna(1.0) * vt
    return out.clip(spec.scale_min, spec.scale_max).fillna(1.0).astype(float)


def weekend_scale_series(panel: Panel, weekend_size_scale: float) -> pd.Series:
    """Per-bar overlay: multiply size on Sat/Sun UTC (1.0 = off)."""
    if weekend_size_scale >= 1.0:
        return pd.Series(1.0, index=panel.index)
    dow = panel.index.dayofweek
    return pd.Series(np.where(dow >= 5, weekend_size_scale, 1.0), index=panel.index)


def scale_for_config(
    panel: Panel,
    cfg: AppConfig,
    bar_index: pd.DatetimeIndex,
    metrics: pd.DataFrame | None = None,
    candidate_returns: pd.Series | None = None,
    gates: dict[str, float] | None = None,
    candidate_signal: pd.Series | None = None,
) -> pd.Series:
    """Dispatch on BacktestConfig.size_scaling; returns a per-bar multiplier."""
    mode = cfg.backtest.size_scaling
    if mode == "off":
        return pd.Series(1.0, index=bar_index)
    spec = ScaleSpec(
        mode=mode,
        metric=cfg.backtest.scale_metric,
        scale_min=cfg.backtest.scale_min,
        scale_max=cfg.backtest.scale_max,
        slope=cfg.backtest.scale_slope,
        z_lookback_days=cfg.backtest.scale_z_lookback_days,
        disp_weight=cfg.backtest.scale_disp_weight,
        vol_target=cfg.backtest.scale_vol_target,
    )

    def finish(series: pd.Series) -> pd.Series:
        """Apply the weekend overlay then clip to the declared bounds."""
        if cfg.backtest.weekend_size_scale < 1.0:
            wk = weekend_scale_series(panel, cfg.backtest.weekend_size_scale)
            series = series * wk.reindex(series.index).fillna(1.0)
        return series.clip(spec.scale_min, spec.scale_max).fillna(1.0)
    if mode == "market":
        return finish(expand_to_bars(market_scale_series(panel, spec, metrics), bar_index))
    if mode == "vol_target":
        if candidate_returns is None:
            return pd.Series(1.0, index=bar_index)
        return finish(expand_to_bars(
            vol_target_scale_series(candidate_returns, panel, spec), bar_index
        ))
    if mode == "efficiency":
        if candidate_returns is None:
            return pd.Series(1.0, index=bar_index)
        return finish(expand_to_bars(
            efficiency_scale_series(candidate_returns, panel, spec), bar_index
        ))
    if mode == "conviction":
        if candidate_signal is None:
            return pd.Series(1.0, index=bar_index)
        raw = conviction_scale_series(
            candidate_signal, cfg.backtest.z_entry,
            int(cfg.backtest.z_lookback_days * panel.bars_per_day),
        )
        return finish(((raw - 1.0) * spec.slope + 1.0))
    if mode in ("jev_soft", "combo"):
        if not gates:
            base = pd.Series(1.0, index=bar_index)
        else:
            base = gates_scale_series(gates, bar_index, spec)
        if mode == "jev_soft":
            return finish(base)
        if candidate_returns is None:
            return pd.Series(1.0, index=bar_index)
        return finish(combo_scale_series(base, candidate_returns, panel, spec))
    raise ValueError(f"unknown size_scaling mode: {mode}")


def live_sizing_hint(cfg: AppConfig, panel: Panel, regime_prob: float | None) -> str | None:
    """One-line report hint: the sizing multiplier in effect right now.

    Uses only decision-safe data (the JEV regime read and trailing market
    metrics); per-book vol-target components are described, not simulated.
    """
    mode = cfg.backtest.size_scaling
    if mode == "off":
        return None
    spec = ScaleSpec(
        mode="jev_soft",
        scale_min=cfg.backtest.scale_min,
        scale_max=cfg.backtest.scale_max,
    )
    bounds = f"{cfg.backtest.scale_min:.2f}x–{cfg.backtest.scale_max:.2f}x"
    if mode in ("jev_soft", "combo"):
        prob = regime_prob
        if prob is None:  # JEV disabled/unavailable: use the latest cached read
            from pair_scout.jev.regime import DEFAULT_CACHE

            if DEFAULT_CACHE.exists():
                try:
                    import json

                    cache = json.loads(DEFAULT_CACHE.read_text())
                    if cache:
                        prob = cache[max(cache)]
                except (OSError, ValueError):
                    prob = None
        if prob is not None:
            gate = _clip(
                spec.scale_min + (spec.scale_max - spec.scale_min) * prob,
                spec.scale_min,
                spec.scale_max,
            )
            if mode == "jev_soft":
                return (
                    f"sizing: jev_soft {bounds} — current JEV regime size "
                    f"{gate:.2f}x (P={prob:.2f})"
                )
            wk = (
                f", weekend {cfg.backtest.weekend_size_scale:.1f}x"
                if cfg.backtest.weekend_size_scale < 1.0
                else ""
            )
            return (
                f"sizing: combo {bounds} — JEV regime component {gate:.2f}x "
                f"(P={prob:.2f}), vol-target {cfg.backtest.scale_vol_target:.0%} "
                f"per book{wk}"
            )
        note = "JEV read unavailable — vol-target component only" \
            if mode == "combo" else "JEV read unavailable — 1x"
        return f"sizing: {mode} {bounds} ({note})"
    if mode == "market":
        metrics = market_daily_metrics(panel)
        daily = market_scale_series(
            panel,
            ScaleSpec(
                mode="market",
                metric=cfg.backtest.scale_metric,
                scale_min=cfg.backtest.scale_min,
                scale_max=cfg.backtest.scale_max,
                slope=cfg.backtest.scale_slope,
                z_lookback_days=cfg.backtest.scale_z_lookback_days,
                disp_weight=cfg.backtest.scale_disp_weight,
            ),
            metrics,
        )
        return (
            f"sizing: market/{cfg.backtest.scale_metric} {bounds} — current "
            f"multiplier {float(daily.iloc[-1]):.2f}x"
        )
    if mode == "vol_target":
        return (
            f"sizing: vol_target({cfg.backtest.scale_vol_target:.0%}) {bounds} "
            "per book from trailing realized vol"
        )
    return f"sizing: {mode} {bounds} (continuous, no gates)"


def apply_scale_to_result(result, scale: pd.Series, bars_per_year: int):
    """Recompute a SimResult's net stats under a per-bar notional multiplier.

    Both PnL and fees scale linearly with notional, so net_scaled = (gross - cost)
    * scale exactly — no re-simulation needed. Positions/trade counts are unchanged
    by construction (sizing never gates entries).
    """
    from pair_scout.backtest.spread import SimResult, _segment_trades

    if result.gross_returns is None or result.cost_returns is None:
        return result
    aligned = scale.reindex(result.bar_returns.index).fillna(1.0).astype(float)
    net = (result.gross_returns - result.cost_returns).dropna()
    if result.funding_returns is not None:
        net = net - result.funding_returns
    net = net * aligned.reindex(net.index).fillna(1.0)
    positions = (
        result.positions.reindex(net.index).fillna(0).astype(int)
        if result.positions is not None
        else None
    )
    trades = (
        _segment_trades(net, positions)
        if positions is not None
        else list(result.trades)
    )
    total = float(net.sum())
    std = float(net.std())
    sharpe = (
        float(net.mean() / std * math.sqrt(bars_per_year)) if std > 0 else 0.0
    )
    equity = net.cumsum()
    dd = float((equity - equity.cummax()).min()) if len(equity) else 0.0
    hit = sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else float("nan")
    return SimResult(
        bar_returns=net,
        trades=trades,
        total_return=total,
        sharpe=sharpe,
        max_drawdown=dd,
        hit_rate=hit,
        n_trades=len(trades),
        gross_returns=result.gross_returns,
        cost_returns=result.cost_returns,
        mean_scale=float(aligned.mean()) if len(aligned) else 1.0,
    )
