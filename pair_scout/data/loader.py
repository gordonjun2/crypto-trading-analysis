"""Leakage-safe loading of the saved_data pkl panels.

Fixes relative to ``data_manager.process_data``/``sanitize_data`` (see plan §2, M3):
- forward-fill only (never backfill with future prices);
- trailing volume is measured over the trailing N days of *each pair's own history*
  (decision-time-safe for a live screener; walk-forward folds slice by time);
- pairs with too many alignment gaps are dropped, not backfilled.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMNS = ("Close", "High", "Low", "Volume")


class DataError(RuntimeError):
    """Raised when no usable data can be loaded."""


@dataclass(frozen=True)
class Panel:
    """Aligned per-pair OHLCV frames on a shared DatetimeIndex."""

    frames: dict[str, pd.DataFrame]
    bars_per_day: int

    @property
    def pairs(self) -> list[str]:
        return sorted(self.frames)

    @property
    def index(self) -> pd.DatetimeIndex:
        return next(iter(self.frames.values())).index

    @property
    def closes(self) -> pd.DataFrame:
        return pd.DataFrame({p: f["Close"] for p, f in self.frames.items()})

    @property
    def days(self) -> int:
        return len(self.index) // self.bars_per_day


def _interval_to_bars_per_day(interval: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    if not interval or interval[-1] not in units:
        raise DataError(f"unsupported interval: {interval!r}")
    seconds = float(interval[:-1]) * units[interval[-1]]
    bars = int(round(86400 / seconds))
    if bars < 1:
        raise DataError(f"interval must be at most daily: {interval!r}")
    return bars


def _load_pair(path: Path) -> tuple[str, pd.DataFrame] | None:
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001 — corrupted cache files must not kill a run
        logger.warning("skipping unreadable pkl %s: %s", path.name, exc)
        return None
    df = payload.get("dataframe") if isinstance(payload, dict) else None
    meta = payload.get("metadata") if isinstance(payload, dict) else None
    if df is None or meta is None or "pair" not in meta:
        logger.warning("skipping malformed pkl %s", path.name)
        return None
    pair = str(meta["pair"])
    df = df.copy()
    df["Open Time"] = pd.to_datetime(df["Open Time"])
    df = df.set_index("Open Time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if "Volume" not in df.columns and "Volume in USDT" in df.columns:
        df = df.rename(columns={"Volume in USDT": "Volume"})
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        logger.warning("skipping %s: missing columns %s", pair, missing)
        return None
    out = df[list(COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return pair, out


def load_panel(
    cex: str = "binance",
    interval: str = "1h",
    data_dir: str | Path = "saved_data",
    top_n_volume: int = 30,
    min_history_bars: int = 500,
    max_nan_fraction: float = 0.02,
    trailing_volume_days: int = 7,
) -> Panel:
    """Load and align the saved pkl cache, keeping the top-N pairs by trailing volume."""
    directory = Path(data_dir) / cex / interval
    files = sorted(directory.glob("*.pkl"))
    if not files:
        raise DataError(
            f"no pkl files in {directory} — run `python data_manager.py -c {cex} -i {interval}` first"
        )
    bars_per_day = _interval_to_bars_per_day(interval)

    raw: dict[str, pd.DataFrame] = {}
    for f in files:
        loaded = _load_pair(f)
        if loaded is not None:
            pair, df = loaded
            if len(df) >= min_history_bars:
                raw[pair] = df
            else:
                logger.info("dropping %s: only %d bars", pair, len(df))
    if not raw:
        raise DataError("no pair has enough history to screen")

    window = trailing_volume_days * bars_per_day
    volumes = {
        p: float(df["Volume"].tail(window).mean()) for p, df in raw.items()
    }
    ranked = sorted(raw, key=lambda p: volumes[p], reverse=True)[:top_n_volume]

    union_index = sorted(set().union(*(set(df.index) for df in (raw[p] for p in ranked))))
    union_index = pd.DatetimeIndex(union_index).sort_values()

    frames: dict[str, pd.DataFrame] = {}
    dropped: list[str] = []
    for pair in ranked:
        df = raw[pair].reindex(union_index)
        gap_frac = float(df["Close"].isna().mean())
        if gap_frac > max_nan_fraction:
            dropped.append(f"{pair} ({gap_frac:.0%} gaps)")
            continue
        df = df.ffill()
        if df["Close"].isna().any():  # leading NaNs: no backfill allowed
            dropped.append(f"{pair} (leading NaNs)")
            continue
        frames[pair] = df
    if dropped:
        logger.info("dropped pairs after alignment: %s", ", ".join(dropped))
    if len(frames) < 2:
        raise DataError("need at least 2 aligned pairs to screen")
    logger.info("panel: %d pairs, %d bars (%s → %s)", len(frames), len(union_index),
                union_index[0].date(), union_index[-1].date())
    return Panel(frames=frames, bars_per_day=bars_per_day)


def slice_days(panel: Panel, start_day: int, end_day: int) -> Panel:
    """Slice a panel by day offsets [start_day, end_day) for walk-forward folds."""
    idx = panel.index
    start = idx[start_day * panel.bars_per_day]
    stop = idx[min(end_day * panel.bars_per_day, len(idx)) - 1]
    frames = {p: df.loc[start:stop].copy() for p, df in panel.frames.items()}
    frames = {p: df for p, df in frames.items() if not df["Close"].isna().any()}
    return Panel(frames=frames, bars_per_day=panel.bars_per_day)


def assert_no_nan(panel: Panel) -> None:
    for pair, df in panel.frames.items():
        if np.any(df.isna().to_numpy()):
            raise DataError(f"NaN values present in panel for {pair}")
