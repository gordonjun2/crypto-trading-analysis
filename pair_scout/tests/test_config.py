import numpy as np
import pandas as pd
import pytest

from pair_scout.config import (
    AppConfig,
    BacktestConfig,
    ConfigError,
    DataConfig,
    JevConfig,
    load_config,
)
from pair_scout.data.loader import Panel, slice_days
from pair_scout.screen import screen_candidates


class TestConfig:
    def test_defaults_valid(self):
        cfg = AppConfig()
        assert cfg.jev.weight_reversion == 0.45

    def test_env_secrets_picked_up(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
        cfg = AppConfig()
        assert cfg.typesafe_api_key == "test-key"

    @pytest.mark.parametrize(
        "cls, bad",
        [
            (DataConfig, dict(top_n_volume=1)),
            (DataConfig, dict(max_nan_fraction=1.5)),
            (JevConfig, dict(weight_reversion=0.9)),  # weights no longer sum to 1
            (JevConfig, dict(min_composite=2.0)),
            (BacktestConfig, dict(entry_z=0.1)),  # entry must exceed exit
        ],
    )
    def test_invalid_values_rejected(self, cls, bad):
        with pytest.raises(ConfigError):
            cls(**bad)

    def test_load_config_missing_file(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(tmp_path / "nope.toml")

    def test_load_config_roundtrip(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text(
            "[data]\ntop_n_volume = 10\n[screen]\npvalue_max = 0.02\n"
        )
        cfg = load_config(toml)
        assert cfg.data.top_n_volume == 10
        assert cfg.screen.pvalue_max == 0.02

    def test_unknown_key_rejected(self, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text("[data]\nnope = 1\n")
        with pytest.raises(ConfigError):
            load_config(toml)


class TestPanel:
    def _panel(self, n_days=5, pairs=("AAAUSDT", "BBBUSDT")):
        bars = n_days * 24
        idx = pd.date_range("2025-01-01", periods=bars, freq="h")
        frames = {}
        for i, p in enumerate(pairs):
            close = pd.Series(100 + i + np.sin(np.arange(bars)) * 2, index=idx)
            frames[p] = pd.DataFrame(
                {
                    "Close": close,
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Volume": pd.Series(1000.0 + i, index=idx),
                }
            )
        return Panel(frames=frames, bars_per_day=24)

    def test_slice_days(self):
        panel = self._panel()
        sub = slice_days(panel, 0, 2)
        assert len(sub.index) == 48
        assert sub.pairs == panel.pairs

    def test_slice_drops_nan_pairs(self):
        panel = self._panel()
        panel.frames["BBBUSDT"].iloc[0:24, 0] = np.nan  # day 0 all-NaN closes
        sub = slice_days(panel, 0, 2)
        assert "BBBUSDT" not in sub.pairs
        assert "AAAUSDT" in sub.pairs

    def test_days_property(self):
        assert self._panel().days == 5


class TestScreenIntegration:
    def test_screen_finds_planted_pair(self):
        """In a panel with one cointegrated duo, screening should surface it."""
        rng = np.random.default_rng(123)
        bars = 24 * 45
        idx = pd.date_range("2025-06-01", periods=bars, freq="h")
        common = 10 + np.cumsum(rng.normal(0, 0.1, bars))
        noise = rng.normal(0, 0.05, bars)

        def frame(close):
            return pd.DataFrame(
                {
                    "Close": close,
                    "High": close * 1.005,
                    "Low": close * 0.995,
                    "Volume": pd.Series(5_000_000.0, index=idx),
                }
            )

        frames = {
            "AAAUSDT": frame(pd.Series(common + noise, index=idx)),
            "BBBUSDT": frame(pd.Series(2 * common + noise, index=idx)),
            "CCCUSDT": frame(pd.Series(50 + np.cumsum(rng.normal(0, 1.0, bars)), index=idx)),
        }
        panel = Panel(frames=frames, bars_per_day=24)
        cfg = AppConfig()
        out = screen_candidates(panel, cfg)
        assert len(out.candidates) == 3
        keys = {c.key for c in out.passed}
        # AAA/BBB is strongly cointegrated; it must not be rejected on cointegration
        assert any("AAAUSDT" in k and "BBBUSDT" in k for k in keys | {c.key for c in out.candidates if c.pvalue < 0.05})
