"""Prediction-market backfill + forward snapshot collector (round 32).

Sources (public, no auth):
  Polymarket gamma-api  : market discovery (closed + open)
  Polymarket clob       : prices-history (probability paths, 10-min ticks)
  Kalshi elections API  : KXBTCD/KXBTC strike ladders + Fed series

Modes:
  backfill  -- discover closed BTC/Fed markets overlapping the research
               window (2024-09 .. today), pull full price histories into
               saved_data_live/prediction/polymarket_backfill/<id>.jsonl
  snapshot  -- hourly collector: open BTC ladders + Fed meetings on both
               venues -> saved_data_live/prediction/snapshots.jsonl

Run:  ./venv/bin/python pair_scout/fetch_prediction.py backfill
      ./venv/bin/python pair_scout/fetch_prediction.py snapshot
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("saved_data_live/prediction")
BF_DIR = BASE / "polymarket_backfill"
SNAP = BASE / "snapshots.jsonl"
UA = {"User-Agent": "research/0.1"}


def get(url, retries=3):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception as e:
            if k == retries - 1:
                print(f"  FAIL {url[:90]}: {e}")
                return None
            time.sleep(2 * (k + 1))


def poly_search(q, status="all", n=20):
    r = get("https://gamma-api.polymarket.com/public-search"
            f"?q={urllib.parse.quote(q)}&limit_per_type={n}"
            f"&events_status={status}")
    return (r or {}).get("events", []) or []


def poly_event_markets(ev_id):
    return get(f"https://gamma-api.polymarket.com/markets?event_id={ev_id}"
               "&limit=50") or []


def poly_history(token_id):
    r = get(f"https://clob.polymarket.com/prices-history?market={token_id}"
            "&interval=all&fidelity=60")
    return (r or {}).get("history", []) or []


# ----------------------------------------------------------------- backfill
def backfill():
    BF_DIR.mkdir(parents=True, exist_ok=True)
    win0 = "2024-08-01"
    queries = ["bitcoin price", "bitcoin hit", "bitcoin above",
               "bitcoin 150", "bitcoin 200", "fed interest rate",
               "fed cut", "bitcoin december", "bitcoin january",
               "bitcoin march", "bitcoin june"]
    seen = set()
    n_saved = 0
    for q in queries:
        for ev in poly_search(q):
            title = ev.get("title", "")
            ev_id = ev.get("id")
            if not ev_id or ev_id in seen:
                continue
            seen.add(ev_id)
            end = (ev.get("endDate") or "")[:10]
            start = (ev.get("startDate") or "")[:10]
            # keep events whose life overlaps the research window
            if end < win0:
                continue
            for m in ev.get("markets", []) or []:
                toks = m.get("clobTokenIds")
                if not toks:
                    continue
                try:
                    tok = json.loads(toks)[0]  # YES token
                except Exception:
                    continue
                h = poly_history(tok)
                if len(h) < 200:
                    continue
                out = BF_DIR / f"{m['id']}.jsonl"
                with open(out, "w") as fh:
                    meta = {"question": m.get("question"),
                            "event": title, "start": start, "end": end,
                            "token": tok, "closed": m.get("closed"),
                            "outcome": m.get("outcomePrices")}
                    fh.write(json.dumps(meta) + "\n")
                    for pt in h:
                        fh.write(json.dumps(pt) + "\n")
                n_saved += 1
        print(f"[{q}] total saved so far: {n_saved}", flush=True)
        time.sleep(1)
    print(f"backfill complete: {n_saved} market paths -> {BF_DIR}")


# ----------------------------------------------------------------- snapshot
def kalshi_open(series):
    r = get("https://api.elections.kalshi.com/trade-api/v2/markets"
            f"?series_ticker={series}&status=open&limit=1000") or {}
    out = []
    for m in r.get("markets", []):
        out.append({"ticker": m.get("ticker"),
                    "yes_bid": m.get("yes_bid"),
                    "yes_ask": m.get("yes_ask"),
                    "last": m.get("last_price")})
    return out


def poly_open(query, n=10):
    out = []
    for ev in poly_search(query, status="active", n=n)[:n]:
        for m in poly_event_markets(ev.get("id")):
            toks = m.get("clobTokenIds")
            if not toks:
                continue
            try:
                tok = json.loads(toks)[0]
            except Exception:
                continue
            out.append({"q": m.get("question"),
                        "token": tok,
                        "bid": m.get("bestBid"),
                        "ask": m.get("bestAsk")})
        time.sleep(0.5)
    return out


def snapshot():
    BASE.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(),
           "poly_btc": poly_open("bitcoin price", 8),
           "poly_fed": poly_open("fed interest rate", 8),
           "kalshi_btcd": kalshi_open("KXBTCD"),
           "kalshi_btc": kalshi_open("KXBTC")}
    with open(SNAP, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"snapshot: poly_btc {len(rec['poly_btc'])} poly_fed "
          f"{len(rec['poly_fed'])} kx_btcd {len(rec['kalshi_btcd'])} "
          f"kx_btc {len(rec['kalshi_btc'])}")


if __name__ == "__main__":
    import urllib.parse  # noqa: F401 (used via poly_search)
    mode = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    if mode == "backfill":
        backfill()
    else:
        snapshot()
