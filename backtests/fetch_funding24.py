"""Fetch Binance USDT-perp funding history into funding_rates.json (24-month
backfill for saved_data_24m). Format matches load_funding: {sym: [[ts, rate]]}.

Run from repo root:  ./venv/bin/python backtests/fetch_funding24.py
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, "/root/crypto-trading-analysis")

URL = "https://fapi.binance.com/fapi/v1/fundingRate"
OUT = Path("saved_data_24m/funding_rates.json")
KL = Path("saved_data_24m/binance/1h")


def fetch_all(symbol: str) -> list[list]:
    rows, start = [], 0
    while True:
        params = {"symbol": symbol, "limit": 1000}
        if start:
            params["startTime"] = start
        attempt = 0
        while True:
            attempt += 1
            try:
                r = requests.get(URL, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2.0 * attempt)
                if attempt > 8:
                    raise
                continue
            if r.status_code == 200:
                break
            if r.status_code in (429, 418, 403):
                wait = float(r.headers.get("Retry-After", 120))
                print(f"  {symbol}: HTTP {r.status_code}, sleeping {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{symbol}: HTTP {r.status_code}")
        batch = r.json()
        if not batch:
            return rows
        rows.extend([e for e in batch])
        if len(batch) < 1000:
            return rows
        start = batch[-1]["fundingTime"] + 1
        time.sleep(0.5)


def main():
    syms = sorted({p.name.split("_")[0] for p in KL.glob("*.pkl")})
    have = {}
    if OUT.exists():
        have = json.loads(OUT.read_text())
    todo = [s for s in syms if s not in have or not have[s]]
    print(f"{len(syms)} symbols, {len(have)} cached, fetching {len(todo)}",
          flush=True)
    errs = {}
    for i, s in enumerate(todo, 1):
        try:
            rows = fetch_all(s)
            have[s] = [[e["fundingTime"], float(e["fundingRate"])] for e in rows]
        except Exception as exc:  # noqa: BLE001
            key = str(exc).split(":")[-1].strip()[:40]
            errs[key] = errs.get(key, 0) + 1
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} (errs: {errs})", flush=True)
        time.sleep(0.15)
    OUT.write_text(json.dumps(have))
    print(f"wrote {OUT} ({len(have)} symbols); errors by type: {errs}",
          flush=True)


if __name__ == "__main__":
    main()
