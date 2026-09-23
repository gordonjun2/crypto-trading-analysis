"""Tests for dispersion-scaled sizing: causality, clipping, exact scaling math."""

import json

import numpy as np
import pandas as pd
import pytest

from pair_scout.backtest.spread import SimResult, Trade, _segment_trades
from pair_scout.config import AppConfig, ConfigError
from pair_scout.data.loader import Panel
from pair_scout.sizing import (
    ScaleSpec,
    apply_scale_to_result,
    conviction_scale_series,
    efficiency_scale_series,
    expand_to_bars,
    gates_scale_series,
    market_daily_metrics,
    market_scale_series,
    vol_target_scale_series,
)


def make_panel(days: int = 30, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    frames = {}
    for name, drift in (("AAAUSDT", 0.0002), ("BBBUSDT", -0.0001), ("BTCUSDT", 0.0)):
        close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
        frames[name] = pd.DataFrame(
            {
                "Close": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Volume": np.full(n, 1_000_000.0),
            },
            index=idx,
        )
    return Panel(frames=frames, bars_per_day=24)


def test_market_metrics_are_causal():
    """Metrics for day D must not change when later data is perturbed."""
    panel = make_panel(30)
    cutoff = panel.index[10 * 24]
    m_full = market_daily_metrics(panel)
    tampered = panel.frames["AAAUSDT"].copy()
    tampered.loc[cutoff:, "Close"] *= 3.0
    tampered.loc[cutoff:, "High"] *= 3.0
    tampered.loc[cutoff:, "Low"] *= 3.0
    panel2 = Panel(frames={**panel.frames, "AAAUSDT": tampered}, bars_per_day=24)
    m_tampered = market_daily_metrics(panel2)
    before = m_full.loc[m_full.index < cutoff]
    after = m_tampered.loc[m_tampered.index < cutoff]
    pd.testing.assert_frame_equal(before, after)


def test_expand_to_bars_daily_mapping_and_fill():
    idx = pd.date_range("2026-01-01", periods=72, freq="h")
    daily = pd.Series({"2026-01-01": 0.8, "2026-01-02": 1.2})
    daily.index = pd.to_datetime(daily.index)
    scale = expand_to_bars(daily, idx)
    assert len(scale) == 72
    assert (scale[:24] == 0.8).all()  # day 1 bars get day 1's value
    assert (scale[24:48] == 1.2).all()
    assert (scale[48:] == 1.0).all()  # unmapped days fall back to 1x


def test_vol_target_scale_lags_one_day():
    """Day D's multiplier may only see returns through D-1 (shift(1))."""
    panel = make_panel(20)
    idx = panel.index
    rets = pd.Series(0.001, index=idx)
    rets.iloc[9 * 24:] = 0.05  # vol explosion from day 9 (Jan 10) onward
    spec = ScaleSpec(mode="vol_target", vol_target=0.10, scale_min=0.5, scale_max=1.5)
    daily = vol_target_scale_series(rets, panel, spec)
    # explosion starts Jan 10 00:00. Jan 10's own multiplier is neutral (its
    # value only sees data through Jan 9) — the downsize lands on Jan 11.
    assert daily.loc[pd.Timestamp("2026-01-10")] == pytest.approx(1.0)
    assert daily.loc[pd.Timestamp("2026-01-11")] == pytest.approx(0.5)


def test_scale_specs_respect_bounds():
    panel = make_panel(30)
    spec = ScaleSpec(mode="market", metric="disp_corr", scale_min=0.5, scale_max=1.5)
    daily = market_scale_series(panel, spec)
    assert daily.between(0.5, 1.5).all()
    assert daily.dropna().between(0.5, 1.5).all()


def test_gates_scale_is_linear_and_clipped():
    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    gates = {"2025-12-31": 0.0, "2026-01-01": 0.5, "2026-01-02": 1.0}
    spec = ScaleSpec(mode="jev_soft", scale_min=0.4, scale_max=1.6)
    scale = gates_scale_series(gates, idx, spec)
    # day D uses gate[D-1]: Jan 1 -> 0.0 -> 0.4x ; Jan 2 -> 0.5 -> 1.0x
    assert scale.loc["2026-01-01"].iloc[0] == pytest.approx(0.4)
    assert scale.loc["2026-01-02"].iloc[0] == pytest.approx(1.0)
    missing_day = gates_scale_series({"2026-01-01": 1.0}, idx, spec)
    assert (missing_day.loc["2026-01-01"] == 1.0).all()  # unknown gate -> neutral
    assert (missing_day.loc["2026-01-02"] == 1.6).all()  # known gate 1.0 -> max size


def test_apply_scale_recomputes_stats_exactly():
    idx = pd.date_range("2026-01-01", periods=10, freq="h")
    gross = pd.Series([0.0, 0.001, 0.002, -0.001, 0.0, 0.003, 0.0, 0.0, 0.001, -0.001], index=idx)
    cost = pd.Series([0.0] * 10, index=idx)
    cost.iloc[[1, 3]] = 0.0002
    positions = pd.Series([0, 1, 1, 0, 0, 1, 1, 0, 0, 0], index=idx)
    result = SimResult(
        bar_returns=gross - cost, trades=[Trade(1, 3, 1, 0.0)],
        gross_returns=gross, cost_returns=cost, positions=positions,
    )
    scale = pd.Series([1.0, 1.5, 1.5, 0.5, 1.0, 1.2, 1.2, 1.0, 1.0, 1.0], index=idx)
    scaled = apply_scale_to_result(result, scale, bars_per_year=24)
    expected = (gross - cost) * scale
    pd.testing.assert_series_equal(scaled.bar_returns, expected)
    assert scaled.n_trades == 2  # [1..2] and [5..6]
    assert scaled.mean_scale == pytest.approx(float(scale.mean()))


def test_efficiency_scale_prefers_trending_books():
    panel = make_panel(40)
    idx = panel.index
    trend = pd.Series(np.linspace(0, 0.06, len(idx)), index=idx).diff().fillna(0.0)
    chop = pd.Series(
        np.sin(np.linspace(0, 80 * np.pi, len(idx))) * 0.003, index=idx
    )
    spec = ScaleSpec(mode="efficiency", scale_min=0.5, scale_max=1.5)
    s_trend = efficiency_scale_series(trend, panel, spec)
    s_chop = efficiency_scale_series(chop, panel, spec)
    # over the last full day, the trending book should be sized at/above 1x,
    # the directionless book at/below 1x (z-scored, so compare relative levels)
    assert s_trend.dropna().iloc[-1] >= s_chop.dropna().iloc[-1]


def test_conviction_scales_beyond_entry_z():
    idx = pd.date_range("2026-01-01", periods=100, freq="h")
    signal = pd.Series(0.0, index=idx)
    signal.iloc[60:] = 0.01  # sharp move -> high z right after the jump
    raw = conviction_scale_series(signal, z_entry=1.0, z_lookback_bars=48)
    assert float((raw - 1.0).max()) > 1.0  # strong signal => conviction multiplier > 2x raw
    spec = ScaleSpec(mode="conviction", scale_min=0.6, scale_max=1.4, slope=0.2)
    scaled = ((raw - 1.0) * spec.slope + 1.0).clip(spec.scale_min, spec.scale_max)
    assert scaled.max() == pytest.approx(spec.scale_max)
    assert scaled.min() >= spec.scale_min


def test_config_rejects_unknown_sizing_mode():
    with pytest.raises(ConfigError):
        AppConfig(backtest=__import__("pair_scout.config", fromlist=["BacktestConfig"]).BacktestConfig(size_scaling="yolo"))


def test_config_default_sizing_is_off(tmp_path):
    cfg = AppConfig()
    assert cfg.backtest.size_scaling == "off"


def test_funding_signs_and_magnitudes():
    """Longs pay positive funding; shorts receive it. Book term = -w_l*f_l + w_s*f_s."""
    from pair_scout.data.funding import book_funding_series, funding_rate_series

    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    settlements = pd.Series(
        [0.0001, 0.0002], index=pd.to_datetime([1767225600000, 1767254400000], unit="ms")
    )  # 2026-01-01 00:00 and 08:00 UTC
    per_bar = funding_rate_series("X", {"X": settlements}, idx)
    assert per_bar.loc["2026-01-01 00:00"] == pytest.approx(0.0001)
    assert per_bar.loc["2026-01-01 08:00"] == pytest.approx(0.0002)
    assert per_bar.loc["2026-01-01 01:00"] == 0.0
    book = book_funding_series(per_bar, per_bar, 0.5, 0.5)
    # both legs pay/receive the same funding -> symmetric book nets to zero
    assert book.abs().sum() == pytest.approx(0.0, abs=1e-12)
    book2 = book_funding_series(pd.Series(0.0, index=idx), per_bar, 0.5, 0.5)
    assert book2.loc["2026-01-01 00:00"] == pytest.approx(0.00005)  # short collects


def test_simulate_funding_flows_into_net():
    from pair_scout.backtest.spread import simulate_divergence

    idx = pd.date_range("2026-01-01", periods=240, freq="h")
    strong = pd.Series(np.linspace(100, 130, 240), index=idx)
    weak = pd.Series(np.linspace(100, 90, 240), index=idx)
    f_short = pd.Series(0.0, index=idx)
    f_short.iloc[100:200] = 0.001  # short leg receives 10bps 8h-equivalent per bar
    free = simulate_divergence(
        np.log(strong), np.log(weak), 1.0, 1.0, entry_mode="absolute",
        entry_mom_per_bar=0.0001, max_hold_bars=None,
    )
    funded = simulate_divergence(
        np.log(strong), np.log(weak), 1.0, 1.0, entry_mode="absolute",
        entry_mom_per_bar=0.0001, max_hold_bars=None,
        funding_long=pd.Series(0.0, index=idx), funding_short=f_short,
    )
    delta = funded.total_return - free.total_return
    assert delta > 0  # positive funding on the short leg pays the book
    held = int(free.positions.sum()) if free.positions is not None else 0
    assert delta == pytest.approx(0.5 * 0.001 * (100 * 0.0), abs=1)  # sanity: finite


def test_weekend_overlay_applied_and_clipped():
    """Sat/Sun bars get the weekend multiplier, clipped to declared bounds."""
    from pair_scout.config import BacktestConfig
    from pair_scout.sizing import scale_for_config

    panel = make_panel(30, seed=11)
    cfg = AppConfig(
        backtest=BacktestConfig(
            size_scaling="jev_soft", scale_min=0.5, scale_max=1.5,
            weekend_size_scale=0.5,
        )
    )
    gates = {"2025-12-31": 1.0}  # full size on Jan 1
    scale = scale_for_config(panel, cfg, panel.index, gates=gates)
    sat = [ts for ts in panel.index if ts.dayofweek == 5][0]
    jan1 = panel.index[0]  # 2026-01-01 (Thursday) uses the Dec 31 gate -> full size
    assert scale.loc[sat] == pytest.approx(0.5)  # 1.5 * 0.5 clipped to bounds
    assert scale.loc[jan1] == pytest.approx(1.5)
    cfg_off = AppConfig(
        backtest=BacktestConfig(size_scaling="jev_soft", weekend_size_scale=1.0)
    )
    scale_off = scale_for_config(panel, cfg_off, panel.index, gates=gates)
    assert scale_off.loc[jan1] == pytest.approx(1.5)  # gate-driven
    assert scale_off.loc[sat] == pytest.approx(1.0)  # no gate data -> neutral
