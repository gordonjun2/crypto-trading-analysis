import numpy as np
import pandas as pd
import pytest

from pair_scout.backtest.spread import (
    _positions_trend,
    combo_beta,
    leg_weights,
    simulate_divergence,
)
from pair_scout.config import AppConfig
from pair_scout.data.loader import Panel
from pair_scout.screen import screen_candidates


def test_beta_balanced_weights_zero_combo_beta():
    w_long, w_short = leg_weights(0.0, 0.0, "beta_balanced", beta_y=0.5, beta_x=2.0)
    # w_long ∝ 1/0.5 = 2, w_short ∝ 1/2 = 0.5 -> 0.8 / 0.2
    assert w_long == pytest.approx(0.8)
    assert w_short == pytest.approx(0.2)
    assert combo_beta(0.5, 2.0, w_long, w_short) == pytest.approx(0.0, abs=1e-12)


def test_beta_balanced_falls_back_when_betas_missing():
    assert leg_weights(0.0, 0.0, "beta_balanced", beta_y=None, beta_x=1.0) == (0.5, 0.5)
    assert leg_weights(0.0, 0.0, "beta_balanced", beta_y=-1.0, beta_x=1.0) == (0.5, 0.5)


def test_trend_state_machine_hysteresis():
    idx = pd.date_range("2025-01-01", periods=7, freq="h")
    mom = pd.Series([0.0, 0.0002, 0.0001, -0.0001, 0.0003, 0.0, -0.001], index=idx)
    pos = _positions_trend(mom, enter_level=0.0002, exit_level=0.0)
    assert list(pos) == [0, 1, 1, 0, 1, 0, 0]


def test_divergence_sim_earns_when_trend_persists():
    """Strong persistent uptrend in the spread -> positive net return."""
    n = 2000
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    t = np.arange(n)
    strong = 100 * np.exp(0.0005 * t)  # ~ +2.7%/day drift
    weak = 100 * np.exp(0.0000 * t)
    result = simulate_divergence(
        pd.Series(np.log(strong), index=idx),
        pd.Series(np.log(weak), index=idx),
        beta_long=1.0,
        beta_short=1.0,
        sizing="beta_balanced",
        signal_window_bars=168,
        entry_mom_per_bar=0.0002,
        exit_mom_per_bar=0.0,
        fee_bps=5.0,
        slippage_bps=2.0,
    )
    assert result.n_trades >= 1
    assert result.total_return > 0
    assert all(t.direction == 1 for t in result.trades)  # always long the strong leg


def test_divergence_sim_flat_in_chop():
    """Mean-reverting spread -> no trend entry (or round trips ~zero)."""
    n = 2000
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    rng = np.random.default_rng(4)
    a = np.cumsum(rng.normal(0, 0.001, n))
    b = np.cumsum(rng.normal(0, 0.001, n))
    strong = 100 * np.exp(a + np.sin(np.arange(n) / 24) * 0.02)
    weak = 100 * np.exp(b)
    result = simulate_divergence(
        pd.Series(np.log(strong), index=idx),
        pd.Series(np.log(weak), index=idx),
        1.0, 1.0,
        entry_mom_per_bar=0.0005,
        exit_mom_per_bar=0.0,
        signal_window_bars=168,
    )
    assert result.total_return < 0.05  # chop does not pay


class TestDivergenceScreening:
    def _panel(self):
        rng = np.random.default_rng(10)
        bars = 24 * 40
        idx = pd.date_range("2025-06-01", periods=bars, freq="h")
        btc = np.cumsum(rng.normal(0, 0.004, bars))  # market factor

        def frame(log_prices, volume=5_000_000.0):
            close = pd.Series(100 * np.exp(log_prices), index=idx)
            return pd.DataFrame(
                {
                    "Close": close,
                    "High": close * 1.004,
                    "Low": close * 0.996,
                    "Volume": pd.Series(volume, index=idx),
                }
            )

        frames = {
            # BTC loads on the market; MOON outruns it; DOG lags it
            "BTCUSDT": frame(btc),
            "MOONUSDT": frame(btc * 0.3 + np.cumsum(rng.normal(0.004, 0.003, bars))),
            "DOGUSDT": frame(btc * 0.3 + np.cumsum(rng.normal(-0.004, 0.003, bars))),
            "ETHUSDT": frame(btc * 1.1 + np.cumsum(rng.normal(0, 0.002, bars))),
        }
        return Panel(frames=frames, bars_per_day=24)

    def test_planted_divergence_is_screened(self):
        panel = self._panel()
        cfg = AppConfig()  # default mode = divergence
        cfg = cfg.__class__(
            data=cfg.data,
            screen=type(cfg.screen)(**{**cfg.screen.__dict__, "min_momentum_spread": 0.05}),
            jev=cfg.jev,
            backtest=cfg.backtest,
            eval=cfg.eval,
        )
        out = screen_candidates(panel, cfg)
        assert out.candidates, "expected divergence candidates to be generated"
        assert all(c.strategy == "divergence" for c in out.candidates)
        # long leg must be the stronger token in every candidate
        assert all(c.momentum_long >= c.momentum_short for c in out.candidates)
        # beta balancing must roughly zero the combo
        strong_pairs = [c for c in out.passed]
        for c in strong_pairs:
            assert abs(c.combo_beta) < 0.16

    def test_long_leg_is_stronger_after_ranking(self):
        panel = self._panel()
        cfg = AppConfig()
        out = screen_candidates(panel, cfg)
        moon_mom = panel.frames["MOONUSDT"]["Close"].iloc[-1] / panel.frames["MOONUSDT"]["Close"].iloc[-14 * 24] - 1
        dog_mom = panel.frames["DOGUSDT"]["Close"].iloc[-1] / panel.frames["DOGUSDT"]["Close"].iloc[-14 * 24] - 1
        assert moon_mom > dog_mom
        pair = [c for c in out.candidates if {c.asset_long, c.asset_short} == {"MOONUSDT", "DOGUSDT"}]
        assert pair and pair[0].asset_long == "MOONUSDT" and pair[0].asset_short == "DOGUSDT"
