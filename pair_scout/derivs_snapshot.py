"""Derivatives forward-collector: Binance futures OI + positioning ratios.

Free Binance futures-data endpoints only serve ~30 days of history, so we
SNAPSHOT hourly and build our own archive for future research:
  - openInterestHist (period=1h, top-150 pairs by kline $volume) -> oi_1h.jsonl
  - at 10:00 SGT also: globalLongShortAccountRatio, topLongShortPositionRatio,
    takerlongshortRatio (period=1d) -> ratios_1d.jsonl

State: {sym: last_ts} index in saved_data_live/derivs/index.json dedupes runs.

Run:  ./venv/bin/python -m pair_scout.derivs_snapshot [--data-dir saved_data_live]
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE = "https://fapi.binance.com"
OI_URL = f"{BASE}/futures/data/openInterestHist"
RATIO_URLS = {
    "global_ls": f"{BASE}/futures/data/globalLongShortAccountRatio",
    "top_ls": f"{BASE}/futures/data/topLongShortPositionRatio",
    "taker_ls": f"{BASE}/futures/data/takerlongshortRatio",
}
TOP_N = 150
SLEEP = 0.35


def _get(url: str, params: dict, retries: int = 3) -> list:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            logger.warning("%s HTTP %s (attempt %d)", url, r.status_code, attempt)
        except requests.RequestException as exc:
            logger.warning("%s failed (%s, attempt %d)", url, exc, attempt)
        time.sleep(2.0 * attempt)
    return []


def _top_pairs(limit: int = TOP_N) -> list[str]:
    from pair_scout.data.refresh import last_closed_bar_ms
    from cex_api.query_binance_data import get_binance_perpetual_futures_pairs

    end = last_closed_bar_ms("1h")
    rows = []
    for sym in get_binance_perpetual_futures_pairs():
        try:
            k = _get(f"{BASE}/fapi/v1/klines",
                     {"symbol": sym, "interval": "1d", "limit": 2})
            if k:
                rows.append((sym, float(k[-1][7])))
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.1)
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:limit]]


def snapshot(data_dir: str = "saved_data_live") -> None:
    d = Path(data_dir) / "derivs"
    d.mkdir(parents=True, exist_ok=True)
    idx_path = d / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    pairs = _top_pairs()
    new_rows = 0
    with open(d / "oi_1h.jsonl", "a") as fh:
        for sym in pairs:
            for row in _get(OI_URL, {"symbol": sym, "period": "1h", "limit": 3}):
                ts = int(row["timestamp"])
                if ts <= index.get(sym, 0):
                    continue
                fh.write(json.dumps({
                    "sym": sym, "ts": ts,
                    "oi": float(row["sumOpenInterest"]),
                    "oi_usdt": float(row["sumOpenInterestValue"]),
                }) + "\n")
                index[sym] = ts
                new_rows += 1
            time.sleep(SLEEP)
    idx_path.write_text(json.dumps(index))
    logger.info("oi_1h: +%d rows across %d pairs", new_rows, len(pairs))

    sgt_hour = int(datetime.now(timezone.utc).timestamp() // 3600 % 24)  # UTC hour
    if sgt_hour == 2:  # 10:00 SGT — daily positioning ratios
        with open(d / "ratios_1d.jsonl", "a") as fh:
            for name, url in RATIO_URLS.items():
                for sym in pairs[:50]:
                    for row in _get(url, {"symbol": sym, "period": "1d",
                                          "limit": 1}):
                        fh.write(json.dumps({
                            "kind": name, "sym": sym,
                            "ts": int(row["timestamp"]),
                            "long_short_ratio": float(row["longShortRatio"]),
                            "long_acc": row.get("longAccount"),
                            "short_acc": row.get("shortAccount"),
                        }) + "\n")
                    time.sleep(SLEEP)
        logger.info("ratios_1d appended")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="saved_data_live")
    args = ap.parse_args()
    snapshot(args.data_dir)
