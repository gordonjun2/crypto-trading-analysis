"""Paginated kline refresh for Binance perps (extends cex_api's single-request fetcher).

Fetches up to N candles per pair by walking backward in <=1500-bar windows and
saves each pair in the exact `data_manager.save_ts_df` pkl layout, replacing the
pair's previous files so the PairScout loader picks up one clean series.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
MAX_PER_REQUEST = 1500
REQUEST_WEIGHT_SLEEP = 0.05  # weight 10/request; fapi allows 2400/min — very safe


def _fetch_window(symbol: str, interval: str, end_ms: int, limit: int,
                  retries: int = 4) -> list[list]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_ms:
        params["endTime"] = end_ms
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(KLINES_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            logger.warning("kline request failed (%s), retry %d", exc, attempt)
            time.sleep(2.0 * attempt)
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            wait = float(response.headers.get("Retry-After", 10))
            logger.warning("rate limited; sleeping %.0fs", wait)
            time.sleep(wait)
            continue
        raise RuntimeError(
            f"kline fetch for {symbol} returned HTTP {response.status_code}"
        )
    raise RuntimeError(f"kline fetch for {symbol} exhausted retries")


def fetch_klines_paginated(symbol: str, interval: str, bars: int,
                           end_ms: int = 0) -> list[list]:
    """Walk backward until `bars` candles are collected (or the API runs dry)."""
    collected: list[list] = []
    end = end_ms
    while len(collected) < bars:
        batch = _fetch_window(symbol, interval, end, min(MAX_PER_REQUEST, bars - len(collected)))
        if not batch:
            break
        collected = batch + collected  # batches arrive ascending
        end = int(batch[0][0]) - 1
        time.sleep(REQUEST_WEIGHT_SLEEP)
    # dedupe on open time, keep ascending order
    seen: dict[int, list] = {}
    for row in collected:
        seen[int(row[0])] = row
    return [seen[k] for k in sorted(seen)]


def _rows_for_save(raw_rows: list[list]) -> list[list]:
    """Keep the 6 columns data_manager.save_ts_df expects (quote volume for 'Volume in USDT')."""
    return [
        [row[0], row[1], row[2], row[3], row[4], row[7]] for row in raw_rows
    ]


def refresh_pair(symbol: str, interval: str, bars: int, dir_path: str | Path,
                 end_ms: int = 0) -> int:
    """Refresh one pair's pkl cache; returns the number of candles saved."""
    import pickle

    from cex_api.query_binance_data import get_binance_perpetual_futures_candlestick_data
    from data_manager import save_ts_df  # reuse the exact on-disk layout

    raw = fetch_klines_paginated(symbol, interval, bars, end_ms) if bars > MAX_PER_REQUEST else None
    if raw is None:
        # small request: use the original single-shot fetcher
        raw = get_binance_perpetual_futures_candlestick_data(
            symbol, interval, str(end_ms) if end_ms else "", bars
        )
        raw = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in raw]
    if not raw:
        logger.warning("%s: no candlestick data returned", symbol)
        return 0
    rows = _rows_for_save(raw)
    # remove stale files for this pair (their filenames embed old date ranges)
    d = Path(dir_path)
    if d.exists():
        for old in d.glob(f"{symbol}_*.pkl"):
            old.unlink()
    save_ts_df(rows, str(d), symbol)
    # sanity: file readable and expected length
    files = sorted(d.glob(f"{symbol}_*.pkl"))
    with open(files[-1], "rb") as fh:
        payload = pickle.load(fh)
    return len(payload["dataframe"])


def refresh_all(cex: str, interval: str, bars: int, data_dir: str | Path = "saved_data",
                end_ms: int = 0) -> dict[str, int]:
    """Refresh every Binance USDT perp pair; returns {pair: candles_saved}."""
    if cex != "binance":
        raise ValueError(
            "pair_scout refresh supports binance; use data_manager.py for okx/bybit"
        )
    from cex_api.query_binance_data import get_binance_perpetual_futures_pairs

    pairs = get_binance_perpetual_futures_pairs()
    dir_path = Path(data_dir) / cex / interval
    dir_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for i, pair in enumerate(sorted(pairs), 1):
        try:
            n = refresh_pair(pair, interval, bars, dir_path, end_ms)
            results[pair] = n
        except Exception as exc:  # noqa: BLE001 — one bad pair must not kill the refresh
            logger.warning("%s: refresh failed (%s)", pair, exc)
            results[pair] = 0
        if i % 10 == 0:
            logger.info("refreshed %d/%d pairs", i, len(pairs))
    ok = sum(1 for n in results.values() if n >= bars * 0.9)
    logger.info("refresh complete: %d/%d pairs with >=90%% of requested bars", ok, len(pairs))
    return results
