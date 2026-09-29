"""Backfill 24m of hourly klines WITH taker-buy fields (orderflow).

Saves 7-column pkls (adds 'Taker Buy USDT' = aggressive buy quote volume)
to saved_data_24m_flow/binance/1h/, aligned to the exact saved_data_24m
grid (last bar = 2026-09-27 13:00 UTC open, 17,520 bars) so flow columns
merge 1:1 onto the existing research panel.

Run:  ./venv/bin/python backtests/fetch_flow24.py
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/root/crypto-trading-analysis")

import pickle

import pandas as pd

from pair_scout.data.refresh import fetch_klines_paginated, _interval_ms

OUT_DIR = Path("saved_data_24m_flow/binance/1h")
BARS = 17_520
END_MS = int(datetime(2026, 9, 27, 13, 0, tzinfo=timezone.utc).timestamp() * 1000)
COLUMNS = ["Open Time", "Open", "High", "Low", "Close", "Volume in USDT",
           "Taker Buy USDT"]
SLEEP = 0.30


def save_flow(rows, pair: str) -> int:
    df = pd.DataFrame(rows, columns=range(12))
    out = pd.DataFrame({
        "Open Time": pd.to_datetime(df[0].astype("int64"), unit="ms"),
        "Open": pd.to_numeric(df[1]), "High": pd.to_numeric(df[2]),
        "Low": pd.to_numeric(df[3]), "Close": pd.to_numeric(df[4]),
        "Volume in USDT": pd.to_numeric(df[7]),
        "Taker Buy USDT": pd.to_numeric(df[9]),
    }).sort_values("Open Time").reset_index(drop=True)
    meta = {"pair": pair, "start_datetime": out["Open Time"].iloc[0],
            "end_datetime": out["Open Time"].iloc[-1]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob(f"{pair}_*.pkl"):
        old.unlink()
    path = OUT_DIR / f"{pair}_{meta['start_datetime']}_{meta['end_datetime']}.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"dataframe": out, "metadata": meta}, fh)
    return len(out)


def main() -> None:
    from cex_api.query_binance_data import get_binance_perpetual_futures_pairs
    pairs = sorted(get_binance_perpetual_futures_pairs())
    print(f"{len(pairs)} pairs; end={datetime.fromtimestamp(END_MS / 1000, timezone.utc)}",
          flush=True)
    done, failed = 0, []
    for i, pair in enumerate(pairs, 1):
        if list(OUT_DIR.glob(f"{pair}_*.pkl")):
            done += 1
            continue
        try:
            raw = fetch_klines_paginated(pair, "1h", BARS, END_MS)
            if len(raw) < 1000:
                failed.append(pair)
                print(f"{i}/{len(pairs)} {pair}: too short "
                      f"({len(raw)} bars) — skipped", flush=True)
                continue
            save_flow(raw, pair)
            done += 1
            if done % 25 == 0:
                print(f"{i}/{len(pairs)} pairs fetched ({done} ok, "
                      f"{len(failed)} skipped)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(pair)
            print(f"{i}/{len(pairs)} {pair}: FAILED {exc}", flush=True)
        time.sleep(SLEEP)
    print(f"DONE: {done} ok, {len(failed)} skipped/failed: {failed[:20]}",
          flush=True)


if __name__ == "__main__":
    main()
