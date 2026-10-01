"""Crypto strike-ladder backfill (Track C data).

Markets: Polymarket 'Bitcoin above ___ on DATE' / 'Ethereum above ___'
binary-strike ladders. Settlement: Binance 1m candle close at 12:00 ET.

  backfill_strikes()  active events (T..T+5d) + recent CLOSED events
                      (price-history ~7d window) -> strike_backfill.json
                      legs: question/kind/strike/token/hist(60min)
  backfill_settle()   Binance 1m kline close at 12:00 ET per event date
                      -> strike_settle.json  (settlement labels only)

Causality: model inputs = Binance hourly closes strictly before entry +
token hist <= entry; settle file used only as labels.
Run: ./venv/bin/python pair_scout/fetch_strikes.py backfill_strikes|backfill_settle
"""
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("saved_data_live/prediction")


def get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url,
                                         headers={"User-Agent": "research"})
            return json.load(urllib.request.urlopen(req, timeout=40))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def parse_strike(q):
    import re
    m = re.search(r"above\s+\$?([\d,]+(?:\.\d+)?)", q or "")
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def backfill_strikes():
    BASE.mkdir(parents=True, exist_ok=True)
    out = {"fetched": datetime.now(timezone.utc).isoformat(),
           "events": []}
    for q in ("bitcoin above on", "ethereum above on"):
        for status in ("active", "all"):
            evs = get("https://gamma-api.polymarket.com/public-search"
                      f"?q={urllib.parse.quote(q)}"
                      f"&limit_per_type=20&events_status={status}"
                      ).get("events", [])
            seen = {e["id"] for e in out["events"]}
            for ev in evs:
                if ev.get("id") in seen:
                    continue
                title = ev.get("title", "")
                if "above" not in title.lower():
                    continue
                sym = "BTC" if "bitcoin" in title.lower() else \
                    "ETH" if "ethereum" in title.lower() else None
                if sym is None:
                    continue
                legs = []
                for m in ev.get("markets", []):
                    toks = m.get("clobTokenIds")
                    if not toks:
                        continue
                    k = parse_strike(m.get("question") or "")
                    if k is None:
                        continue
                    tok = json.loads(toks)[0]
                    try:
                        h = get("https://clob.polymarket.com/"
                                "prices-history"
                                f"?market={tok}&interval=all&fidelity=1"
                                ).get("history", [])
                    except Exception:
                        h = []
                    time.sleep(0.2)
                    if h:
                        legs.append({"q": m.get("question"), "k": k,
                                     "token": tok,
                                     "hist": [[p["t"], p["p"]]
                                              for p in h]})
                if legs:
                    out["events"].append({
                        "id": ev.get("id"), "sym": sym, "title": title,
                        "end": str(ev.get("endDate"))[:10], "legs": legs})
                    print(f"  {title[:48]} | end {out['events'][-1]['end']}"
                          f" | {len(legs)} legs w/ hist", flush=True)
            time.sleep(0.4)
    path = BASE / "strike_backfill.json"
    json.dump(out, open(path, "w"))
    print(f"backfill_strikes: {len(out['events'])} events, "
          f"{sum(len(e['legs']) for e in out['events'])} legs -> {path}")


def settle_dt(title, end):
    """Settlement datetime (UTC) from title: '..., 10AM ET' or daily
    noon ET default."""
    from zoneinfo import ZoneInfo
    m = re.search(r"(\w+)\s+(\d{1,2})(?:,)?\s*(\d{1,2})(AM|PM)?\s*ET",
                  title or "")
    if m:
        mon, day, hour, ap = (m.group(1), int(m.group(2)),
                              int(m.group(3)), m.group(4))
        months = {mn: i for i, mn in enumerate(
            ("January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November",
             "December"), 1)}
        mon_n = months.get(mon)
        if mon_n:
            year = int(end[:4]) if mon_n <= int(end[5:7]) else \
                int(end[:4])
            if ap:
                h24 = hour % 12 + (12 if ap == "PM" else 0)
            else:
                h24 = 12  # noon ET default for daily ladders
            return (datetime(year, mon_n, day, h24, 1,
                             tzinfo=ZoneInfo("America/New_York"))
                    .astimezone(timezone.utc))
    # fallback: end date at noon ET
    try:
        from zoneinfo import ZoneInfo
        y, mo, dy = map(int, end.split("-"))
        return (datetime(y, mo, dy, 12, 1,
                         tzinfo=ZoneInfo("America/New_York"))
                .astimezone(timezone.utc))
    except Exception:
        return None


def backfill_settle():
    """Binance 1m close at each event's settlement minute."""
    d = json.load(open(BASE / "strike_backfill.json"))
    path = BASE / "strike_settle.json"
    try:
        cur = json.load(open(path))
    except Exception:
        cur = {"settle": {}}
    settle = cur.get("settle", {})
    pairs = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
    for ev in d["events"]:
        st = settle_dt(ev["title"], ev["end"])
        if st is None:
            continue
        key = f"{ev['sym']}:{st.isoformat()}"
        if key in settle:
            continue
        ms = int(st.timestamp() * 1000)
        u = ("https://api.binance.com/api/v3/klines?symbol="
             + pairs[ev["sym"]] + "&interval=1m&startTime="
             + str(ms) + "&limit=1")
        try:
            r = get(u)
            if r and abs(r[0][0] - ms) < 60_000:
                settle[key] = float(r[0][4])
        except Exception as e:
            print(f"  {key}: ERR {str(e)[:60]}")
        time.sleep(0.25)
    json.dump({"fetched": datetime.now(timezone.utc).isoformat(),
               "settle": settle}, open(path, "w"))
    print(f"backfill_settle: {len(settle)} labels -> {path}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("backfill_strikes", "all"):
        backfill_strikes()
    if cmd in ("backfill_settle", "all"):
        backfill_settle()
