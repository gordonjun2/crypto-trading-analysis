"""Funding-rate data for perpetual futures (closes the biggest modeling gap).

Binance USDT-perps exchange funding every 8h by default (some pairs every 4h in
high-vol regimes). A long pays `rate * notional` when rate > 0; a short receives
it. The book's funding PnL for bar i (position held from bar i-1 close to bar i
close, i.e. AT bar i's open when settlements occur):

    funding_pnl[i] = pos * (-w_long * f_long[i] + w_short * f_short[i])

Settlements falling within a bar are summed and assigned to that bar.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_funding(path: str | Path) -> dict[str, pd.Series]:
    """Load the funding cache: {symbol: [[fundingTime_ms, rate], ...]}."""
    p = Path(path)
    if not p.exists():
        logger.warning("funding cache %s not found — funding costs NOT modeled", p)
        return {}
    raw = json.loads(p.read_text())
    out: dict[str, pd.Series] = {}
    for symbol, rows in raw.items():
        if not rows:
            continue
        idx = pd.to_datetime([r[0] for r in rows], unit="ms")
        out[symbol] = pd.Series([r[1] for r in rows], index=idx).sort_index()
    return out


def funding_rate_series(
    symbol: str, funding: dict[str, pd.Series], bar_index: pd.DatetimeIndex
) -> pd.Series:
    """Per-bar funding rate for `symbol`: settlements inside a bar are summed.

    A settlement at time T is paid to the position held at T, which is the
    position during the bar whose open is floor(T, hour).
    """
    series = funding.get(symbol)
    if series is None or series.empty:
        return pd.Series(0.0, index=bar_index)
    floored = series.groupby(series.index.floor("h")).sum()
    return floored.reindex(bar_index).fillna(0.0)


def book_funding_series(
    funding_long: pd.Series, funding_short: pd.Series, w_long: float, w_short: float
) -> pd.Series:
    """Book-level funding rate per bar: longs pay, shorts receive (rate > 0)."""
    return -w_long * funding_long + w_short * funding_short
