"""Weather temp-ladder backfill (Track B data): market prices + NWP
forecast vintages + observed daily max. Three independent collectors:

  backfill_markets()  Polymarket temp ladders (8 cities), legs + 60-min
                      price history (server keeps ~30d) -> weather_backfill.json
  backfill_nwp()      open-meteo HISTORICAL-FORECAST api: hourly t2m as
                      issued + previous-day run vintages (d1/d2/d3) from
                      2026-03-01 -> nwp_hourly.json.gz   (forecast-as-issued,
                      NOT reanalysis — this is what a trader knew)
  backfill_obs()      open-meteo archive api (ERA5) daily max + recent
                      days from forecast api past_days -> obs_daily.json

Causality note: entry decisions use nwp vintages (forecast known at time)
and market prices before entry; obs are settlement labels only.
Run: ./venv/bin/python pair_scout/fetch_weather.py backfill_markets|backfill_nwp|backfill_obs
"""
import gzip
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("saved_data_live/prediction")
CITIES = {  # poly ladder city -> (open-meteo lat, lon, local tz)
    "Paris": (48.85, 2.35, "Europe/Paris"),
    "New York": (40.71, -74.01, "America/New_York"),
    "Los Angeles": (34.05, -118.24, "America/Los_Angeles"),
    "Seoul": (37.57, 126.98, "Asia/Seoul"),
    "Hong Kong": (22.30, 114.17, "Asia/Hong_Kong"),
    "Shanghai": (31.23, 121.47, "Asia/Shanghai"),
    "Singapore": (1.35, 103.82, "Asia/Singapore"),
    "London": (51.51, -0.13, "Europe/London"),
    "Beijing": (39.90, 116.41, "Asia/Shanghai"),
    "Qingdao": (36.07, 120.38, "Asia/Shanghai"),
    "Wellington": (-41.29, 174.78, "Pacific/Auckland"),
    "Seattle": (47.61, -122.33, "America/Los_Angeles"),
    "San Francisco": (37.77, -122.42, "America/Los_Angeles"),
    "Taipei": (25.03, 121.57, "Asia/Taipei"),
    "Tokyo": (35.68, 139.69, "Asia/Tokyo"),
    "Chicago": (41.88, -87.63, "America/Chicago"),
    "Miami": (25.76, -80.19, "America/New_York"),
    "Toronto": (43.65, -79.38, "America/Toronto"),
    "Moscow": (55.76, 37.62, "Europe/Moscow"),
    "Mexico City": (19.43, -99.13, "America/Mexico_City"),
    "Sao Paulo": (-23.55, -46.63, "America/Sao_Paulo"),
    "Buenos Aires": (-34.60, -58.38, "America/Argentina/Buenos_Aires"),
    "Manila": (14.60, 120.98, "Asia/Manila"),
    "Kuala Lumpur": (3.14, 101.69, "Asia/Kuala_Lumpur"),
    "Karachi": (24.86, 67.01, "Asia/Karachi"),
    "Istanbul": (41.01, 28.98, "Europe/Istanbul"),
    "Madrid": (40.42, -3.70, "Europe/Madrid"),
    "Warsaw": (52.23, 21.01, "Europe/Warsaw"),
    "Amsterdam": (52.37, 4.90, "Europe/Amsterdam"),
    "Helsinki": (60.17, 24.94, "Europe/Helsinki"),
    "Houston": (29.76, -95.37, "America/Chicago"),
    "Denver": (39.74, -104.99, "America/Denver"),
    "Atlanta": (33.75, -84.39, "America/New_York"),
    "Dallas": (32.78, -96.80, "America/Chicago"),
    "Austin": (30.27, -97.74, "America/Chicago"),
}


def get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            return json.load(urllib.request.urlopen(req, timeout=40))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def parse_leg(q):
    """'Will the highest temperature in Paris be 19°C on Sep 30?' ->
    ('eq', 19); '... 18°C or below' -> ('lo', 18); '... or higher' -> ('hi', k)"""
    import re
    ql = q.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*°?c\s+or\s+(below|less|higher|above|more)", ql)
    if m:
        k = float(m.group(1))
        return ("lo" if m.group(2) in ("below", "less") else "hi", k)
    m = re.search(r"be\s+(\d+(?:\.\d+)?)\s*°?c", ql)
    if m:
        return ("eq", float(m.group(1)))
    return (None, None)


def backfill_markets(min_end="2026-08-25"):
    BASE.mkdir(parents=True, exist_ok=True)
    out = {"fetched": datetime.now(timezone.utc).isoformat(), "events": []}
    for city in CITIES:
        s = get("https://gamma-api.polymarket.com/public-search"
                f"?q={urllib.parse.quote('highest temperature in ' + city)}"
                "&limit_per_type=20&events_status=all")
        for ev in s.get("events", []):
            title = ev.get("title", "")
            if "temperature" not in title.lower() or city.lower() not in title.lower():
                continue
            end = str(ev.get("endDate"))[:10]
            if end < min_end:
                continue
            legs = []
            for m in ev.get("markets", []):
                toks = m.get("clobTokenIds")
                if not toks:
                    continue
                tok = json.loads(toks)[0]
                kind, k = parse_leg(m.get("question") or "")
                if kind is None:
                    continue
                try:
                    h = get(f"https://clob.polymarket.com/prices-history"
                            f"?market={tok}&interval=all&fidelity=60"
                            ).get("history", [])
                except Exception:
                    h = []
                time.sleep(0.25)
                if h:
                    legs.append({"q": m.get("question"), "kind": kind, "k": k,
                                 "token": tok,
                                 "hist": [[p["t"], p["p"]] for p in h]})
            if legs:
                out["events"].append({"city": city, "title": title,
                                      "end": end, "legs": legs})
                print(f"  {title[:52]} | end {end} | {len(legs)} legs "
                      f"with history", flush=True)
        time.sleep(0.4)
    path = BASE / "weather_backfill.json"
    json.dump(out, open(path, "w"))
    ne = len(out["events"])
    nl = sum(len(e["legs"]) for e in out["events"])
    print(f"backfill_markets: {ne} events, {nl} legs -> {path}")


def backfill_nwp(start="2026-03-01"):
    BASE.mkdir(parents=True, exist_ok=True)
    from datetime import date
    today = date.today().isoformat()
    out = {"fetched": datetime.now(timezone.utc).isoformat(), "cities": {}}
    for city, (lat, lon, tz) in CITIES.items():
        u = ("https://historical-forecast-api.open-meteo.com/v1/forecast"
             f"?latitude={lat}&longitude={lon}"
             "&hourly=temperature_2m,temperature_2m_previous_day1,"
             "temperature_2m_previous_day2,temperature_2m_previous_day3"
             f"&start_date={start}&end_date={today}&timezone="
             f"{urllib.parse.quote(tz)}")
        r = get(u)
        h = r.get("hourly", {})
        out["cities"][city] = {
            "tz": tz, "time": h.get("time"),
            "t2m": h.get("temperature_2m"),
            "d1": h.get("temperature_2m_previous_day1"),
            "d2": h.get("temperature_2m_previous_day2"),
            "d3": h.get("temperature_2m_previous_day3")}
        print(f"  {city}: {len(h.get('time') or [])}h "
              f"{(h.get('time') or ['-'])[0]}..{(h.get('time') or ['-'])[-1]}",
              flush=True)
        time.sleep(1.5)
    path = BASE / "nwp_hourly.json.gz"
    with gzip.open(path, "wt") as fh:
        json.dump(out, fh)
    print(f"backfill_nwp: {len(out['cities'])} cities -> {path}")


def backfill_obs():
    BASE.mkdir(parents=True, exist_ok=True)
    from datetime import date, timedelta
    today = date.today()
    out = {"fetched": datetime.now(timezone.utc).isoformat(), "cities": {}}
    for city, (lat, lon, tz) in CITIES.items():
        u = ("https://archive-api.open-meteo.com/v1/archive"
             f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max"
             f"&start_date=2025-01-01&end_date={today - timedelta(days=6)}"
             f"&timezone={urllib.parse.quote(tz)}")
        r = get(u).get("daily", {})
        # recent days: forecast api past_days covers the archive lag
        u2 = ("https://api.open-meteo.com/v1/forecast"
              f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max"
              f"&past_days=8&forecast_days=1&timezone={urllib.parse.quote(tz)}")
        r2 = get(u2).get("daily", {})
        time.sleep(1.2)
        merged = dict(zip(r.get("time", []), r.get("temperature_2m_max", [])))
        for d, v in zip(r2.get("time", []), r2.get("temperature_2m_max", [])):
            merged.setdefault(d, v)
        out["cities"][city] = {"tz": tz, "tmax": merged}
        print(f"  {city}: {len(merged)} days obs", flush=True)
    path = BASE / "obs_daily.json"
    json.dump(out, open(path, "w"))
    print(f"backfill_obs -> {path}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("backfill_markets", "all"):
        backfill_markets()
    if cmd in ("backfill_nwp", "all"):
        backfill_nwp()
    if cmd in ("backfill_obs", "all"):
        backfill_obs()
