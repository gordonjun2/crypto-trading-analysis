"""Configuration: TOML file + environment variables (secrets only from env/.env).

Precedence: CLI flags > config.toml > defaults; secrets (API keys) come exclusively
from the environment (.env via python-dotenv).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op when .env is absent; real env vars keep precedence


class ConfigError(ValueError):
    """Raised when configuration values are invalid."""


@dataclass(frozen=True)
class DataConfig:
    cex: str = "binance"
    interval: str = "1h"
    data_dir: str = "saved_data"
    top_n_volume: int = 30
    min_history_bars: int = 500
    max_nan_fraction: float = 0.02
    trailing_volume_days: int = 7

    def __post_init__(self) -> None:
        if self.top_n_volume < 2:
            raise ConfigError("data.top_n_volume must be >= 2")
        if self.min_history_bars < 50:
            raise ConfigError("data.min_history_bars must be >= 50")
        if not 0 < self.max_nan_fraction < 1:
            raise ConfigError("data.max_nan_fraction must be in (0, 1)")
        if self.trailing_volume_days < 1:
            raise ConfigError("data.trailing_volume_days must be >= 1")


@dataclass(frozen=True)
class ScreenConfig:
    pvalue_max: float = 0.05
    half_life_min_bars: int = 4
    half_life_max_days: float = 30.0
    entry_z: float = 1.5
    watch_z: float = 1.0  # near-miss tier: everything else OK but |z| < entry_z
    zscore_window_bars: int = 15
    corr_lookback_days: float = 30.0
    momentum_lookback_days: float = 30.0
    min_dollar_volume: float = 1_000_000.0
    atr_pct_max: float = 15.0
    skew_abs_max: float = 4.0
    vol_ratio_max: float = 4.0

    def __post_init__(self) -> None:
        if not 0 < self.pvalue_max < 1:
            raise ConfigError("screen.pvalue_max must be in (0, 1)")
        if self.half_life_min_bars < 2:
            raise ConfigError("screen.half_life_min_bars must be >= 2")
        if self.half_life_max_days <= 0:
            raise ConfigError("screen.half_life_max_days must be > 0")
        if self.entry_z <= 0:
            raise ConfigError("screen.entry_z must be > 0")
        if not 0 < self.watch_z <= self.entry_z:
            raise ConfigError("screen.watch_z must be in (0, entry_z]")
        if self.zscore_window_bars < 5:
            raise ConfigError("screen.zscore_window_bars must be >= 5")
        if self.min_dollar_volume < 0 or self.atr_pct_max <= 0 or self.skew_abs_max <= 0:
            raise ConfigError("screen thresholds must be positive")
        if self.vol_ratio_max < 1:
            raise ConfigError("screen.vol_ratio_max must be >= 1")


@dataclass(frozen=True)
class JevConfig:
    enabled: bool = True
    model: str | None = None  # None -> SDK default (jev-latest)
    weight_reversion: float = 0.45
    weight_entry: float = 0.35
    weight_executability: float = 0.20
    min_composite: float = 0.60
    min_confidence: float = 0.40
    min_direction_noul: float = 0.60
    max_red_flag_noul: float = 0.50
    max_workers: int = 4
    top_k: int = 5
    max_candidates: int = 12
    timeout_seconds: float = 60.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        w = self.weight_reversion + self.weight_entry + self.weight_executability
        if abs(w - 1.0) > 1e-9:
            raise ConfigError("jev weights must sum to 1.0")
        if min(self.weight_reversion, self.weight_entry, self.weight_executability) < 0:
            raise ConfigError("jev weights must be non-negative")
        if not 0 <= self.min_composite <= 1:
            raise ConfigError("jev.min_composite must be in [0, 1]")
        if not 0 <= self.min_confidence <= 1:
            raise ConfigError("jev.min_confidence must be in [0, 1]")
        if not 0 <= self.min_direction_noul <= 1:
            raise ConfigError("jev.min_direction_noul must be in [0, 1]")
        if not 0 <= self.max_red_flag_noul <= 1:
            raise ConfigError("jev.max_red_flag_noul must be in [0, 1]")
        if self.max_workers < 1 or self.top_k < 1 or self.max_candidates < 1:
            raise ConfigError("jev.max_workers/top_k/max_candidates must be >= 1")
        if self.timeout_seconds <= 0 or self.max_retries < 0:
            raise ConfigError("jev.timeout_seconds must be > 0 and max_retries >= 0")


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    entry_z: float = 1.5
    exit_z: float = 0.5
    stop_z: float = 3.5
    zscore_window_bars: int = 15
    sizing: str = "vol_balanced"  # "vol_balanced" | "equal"

    def __post_init__(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ConfigError("backtest fees/slippage must be non-negative")
        if not 0 < self.exit_z < self.entry_z < self.stop_z:
            raise ConfigError("backtest must satisfy 0 < exit_z < entry_z < stop_z")
        if self.zscore_window_bars < 5:
            raise ConfigError("backtest.zscore_window_bars must be >= 5")
        if self.sizing not in ("vol_balanced", "equal"):
            raise ConfigError("backtest.sizing must be 'vol_balanced' or 'equal'")


@dataclass(frozen=True)
class EvalConfig:
    train_days: int = 40
    test_days: int = 22
    roll_days: int = 10  # second fold rolls the train end forward by this many days
    top_k: int = 3

    def __post_init__(self) -> None:
        if self.train_days < 10 or self.test_days < 2 or self.roll_days < 0:
            raise ConfigError("eval.day windows are invalid")


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    jev: JevConfig = field(default_factory=JevConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    typesafe_api_key: str | None = field(
        default_factory=lambda: os.environ.get("TYPESAFE_API_KEY") or None
    )
    telegram_bot_token: str | None = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN") or None
    )
    telegram_group_id: str | None = field(
        default_factory=lambda: os.environ.get("TELEGRAM_GROUP_ID") or None
    )

    def with_jev(self, enabled: bool) -> "AppConfig":
        return replace(self, jev=replace(self.jev, enabled=enabled))


def _build_section(cls, section: dict) -> object:
    valid = {f for f in cls.__dataclass_fields__ if not f.endswith(("_token", "_key"))}
    kwargs = {}
    for k, v in section.items():
        if k not in cls.__dataclass_fields__:
            raise ConfigError(f"unknown config key: {k}")
        kwargs[k] = v
    del valid
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load AppConfig from an optional TOML file. Env supplies secrets."""
    if path is None:
        return AppConfig()
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with open(p, "rb") as fh:
        raw = tomllib.load(fh)
    kwargs = {}
    for name, cls in (
        ("data", DataConfig),
        ("screen", ScreenConfig),
        ("jev", JevConfig),
        ("backtest", BacktestConfig),
        ("eval", EvalConfig),
    ):
        section = raw.get(name, {})
        if not isinstance(section, dict):
            raise ConfigError(f"[{name}] must be a TOML table")
        kwargs[name] = _build_section(cls, section)
    return AppConfig(**kwargs)
