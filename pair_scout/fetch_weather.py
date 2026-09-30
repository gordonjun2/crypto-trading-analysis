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


def _station_icao(city):
    """Extract the settlement station ICAO from a market description."""
    import re
    s = get("https://gamma-api.polymarket.com/public-search"
            f"?q={urllib.parse.quote('highest temperature in ' + city)}"
            "&limit_per_type=3&events_status=active")
    for ev in s.get("events", []):
        for m in ev.get("markets", []):
            d = (m.get("description") or "").lower()
            hit = re.search(r"site=([a-z]{3,4})", d)
            if hit:
                return hit.group(1).upper()
            hit = re.search(r"\b([kceldrnzwyubgmprsftv][a-z]{3})\b\s*"
                            r"(airport|station|intl|international)", d)
            if hit:
                return hit.group(1).upper()
    return FALLBACK_ICAO.get(city)


FALLBACK_ICAO = {
    "New York": "KLGA", "Los Angeles": "KLAX", "Chicago": "KORD",
    "Miami": "KMIA", "Houston": "KHOU", "Dallas": "KDAL",
    "Atlanta": "KATL", "Denver": "KBKF", "Austin": "KAUS",
    "Seattle": "KSEA", "San Francisco": "KSFO", "Toronto": "CYYZ",
    "Paris": "LFPB", "London": "EGLC", "Tokyo": "RJTT",
    "Seoul": "RKSI", "Hong Kong": "VHHH", "Singapore": "WSSS",
    "Shanghai": "ZSPD", "Beijing": "ZBAA", "Qingdao": "ZSQD",
    "Taipei": "RCSS", "Wellington": "NZWN", "Moscow": "UUEE",
    "Mexico City": "MMMX", "Sao Paulo": "SBGR",
    "Buenos Aires": "SAEZ", "Manila": "RPLL", "Kuala Lumpur": "WMKK",
    "Karachi": "OPKC", "Istanbul": "LTFM", "Madrid": "LEMD",
    "Warsaw": "EPWA", "Amsterdam": "EHAM", "Helsinki": "EFHK",
}


def backfill_stations(start="2026-08-10"):
    """Settlement-true daily max per city from ASOS/METAR archives
    (Iowa State mesonet, public). Daily max mimics NOAA hourly Temp:
    only readings at minutes 40-59 of each local hour (the hourly
    report), falling back to any 5-min reading."""
    from datetime import date
    BASE.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out = {"fetched": datetime.now(timezone.utc).isoformat(), "cities": {}}
    for city in CITIES:
        icao = _station_icao(city)
        if not icao:
            print(f"  {city}: no station")
            continue
        u = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
             f"station={icao}&data=tmpc&year1={start[:4]}&month1={start[5:7]}"
             f"&day1={start[8:10]}&year2={today[:4]}&month2={today[5:7]}"
             f"&day2={today[8:10]}&tz=etc%2FUTC&format=onlycomma&latlon=no"
             "&missing=M&trace=T")
        txt = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    u, headers={"User-Agent": "research"})
                txt = urllib.request.urlopen(req, timeout=120).read().decode()
                if "station,valid" in txt:
                    break
            except Exception:
                pass
            time.sleep(65 * (attempt + 1))
        if not txt:
            print(f"  {city}/{icao}: failed after retries")
            continue
        daily = {}
        by_day_hour = {}
        for line in txt.strip().split("\n")[1:]:
            parts = line.split(",")
            if len(parts) < 3 or parts[2] in ("M", ""):
                continue
            ts = parts[1]  # UTC
            day = ts[:10]
            minute = int(ts[14:16])
            try:
                v = float(parts[2])
            except ValueError:
                continue
            hour = ts[11:13]
            key = (day, hour)
            if 40 <= minute <= 59:
                by_day_hour[key] = max(by_day_hour.get(key, -99), v)
            daily[day] = max(daily.get(day, -99), v)
        # prefer hourly-report max; fall back to 5-min max per day
        hours_by_day = {}
        for (day, hour), v in by_day_hour.items():
            hours_by_day[day] = max(hours_by_day.get(day, -99), v)
        merged = {d: hours_by_day.get(d, v) for d, v in daily.items()}
        out["cities"][city] = {"icao": icao, "tmax_c": merged}
        print(f"  {city}/{icao}: {len(merged)} days "
              f"({min(merged) if merged else '-'}.."
              f"{max(merged) if merged else '-'})", flush=True)
        time.sleep(35)
    path = BASE / "stations_daily.json"
    json.dump(out, open(path, "w"))
    print(f"backfill_stations -> {path}")


MODELS = ("ecmwf_ifs025", "gfs_seamless", "icon_seamless")


def backfill_models(start="2026-03-01"):
    """Per-model daily max (lead-0) + lead-1 vintage daily max for the
    EMOS-style multi-model blend."""
    from datetime import date
    BASE.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out = {"fetched": datetime.now(timezone.utc).isoformat(), "cities": {}}
    for city, (lat, lon, tz) in CITIES.items():
        u = ("https://historical-forecast-api.open-meteo.com/v1/forecast"
             f"?latitude={lat}&longitude={lon}"
             "&daily=temperature_2m_max&hourly=temperature_2m_previous_day1"
             f"&models={','.join(MODELS)}&start_date={start}"
             f"&end_date={today}&timezone={urllib.parse.quote(tz)}")
        try:
            r = get(u)
        except Exception as e:
            print(f"  {city}: ERR {str(e)[:60]}")
            continue
        cd = {"tz": tz, "lead0": {}, "lead1": {}}
        d = r.get("daily", {})
        for m in MODELS:
            vals = d.get(f"temperature_2m_max_{m}", [])
            for dt, v in zip(d.get("time", []), vals):
                if v is not None:
                    cd["lead0"].setdefault(dt, {})[m] = v
        h = r.get("hourly", {})
        t = h.get("time", [])
        for m in MODELS:
            vals = h.get(f"temperature_2m_previous_day1_{m}", [])
            if not vals:
                continue
            acc = {}
            for ts, v in zip(t, vals):
                if v is None:
                    continue
                acc.setdefault(ts[:10], -99)
                acc[ts[:10]] = max(acc[ts[:10]], v)
            for dt, v in acc.items():
                if v > -90:
                    cd["lead1"].setdefault(dt, {})[m] = v
        out["cities"][city] = cd
        n0 = sum(1 for v in cd["lead0"].values() if len(v) >= 2)
        print(f"  {city}: lead0 {n0}d, lead1 {len(cd['lead1'])}d",
              flush=True)
        time.sleep(1.5)
    path = BASE / "models_daily.json"
    json.dump(out, open(path, "w"))
    print(f"backfill_models -> {path}")


def backfill_ensemble(start="2026-07-01"):
    """ECMWF 50-member ensemble daily-max spread per city (lead-0).
    Chunked pulls (api caps response size)."""
    from datetime import date, timedelta
    import pandas as pd
    BASE.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out = {"fetched": datetime.now(timezone.utc).isoformat(), "cities": {}}
    # 10-day chunks
    chunks = []
    d0 = pd.Timestamp(start).date()
    while d0.isoformat() < today:
        d1 = min(d0 + timedelta(days=9), pd.Timestamp(today).date())
        chunks.append((d0.isoformat(), d1.isoformat()))
        d0 = d1 + timedelta(days=1)
    for city, (lat, lon, tz) in CITIES.items():
        spread = {}
        nfail = 0
        for cs, ce in chunks:
            u = (f"https://ensemble-api.open-meteo.com/v1/ensemble"
                 f"?latitude={lat}&longitude={lon}&hourly=temperature_2m"
                 f"&models=ecmwf_ifs025&start_date={cs}&end_date={ce}"
                 f"&timezone={urllib.parse.quote(tz)}")
            r = None
            for att in range(4):
                try:
                    r = get(u, tries=1)
                    if r.get("hourly", {}).get("time"):
                        break
                except Exception:
                    pass
                time.sleep(5 * (att + 1))
            if r is None or not r.get("hourly", {}).get("time"):
                nfail += 1
                print(f"    {city} chunk {cs}..{ce} FAILED", flush=True)
                continue
            h = r.get("hourly", {})
            members = [k for k in h
                       if k.startswith("temperature_2m_member")]
            acc = {}
            t = h.get("time", [])
            for mkey in members:
                for ts, v in zip(t, h[mkey]):
                    if v is None:
                        continue
                    day = ts[:10]
                    dacc = acc.setdefault(day, {})
                    dacc[mkey] = max(dacc.get(mkey, -99), v)
            for day, mm in acc.items():
                vals = list(mm.values())
                if len(vals) >= 20:
                    spread[day] = float(np_std(vals))
            time.sleep(1.2)
        out["cities"][city] = {"tz": tz, "spread": spread}
        print(f"  {city}: {len(spread)}d spread "
              f"({min(spread) if spread else '-'}.."
              f"{max(spread) if spread else '-'})", flush=True)
    path = BASE / "ens_spread.json"
    json.dump(out, open(path, "w"))
    print(f"backfill_ensemble -> {path}")


def np_std(vals):
    import math
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


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
    if cmd in ("backfill_stations", "all"):
        backfill_stations()
    if cmd in ("backfill_models", "all"):
        backfill_models()
    if cmd in ("backfill_ensemble", "all"):
        backfill_ensemble()
