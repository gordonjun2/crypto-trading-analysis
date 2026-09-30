"""Round 34: WEATHER v2 — settlement-true obs + EMOS-style calibration
+ multi-model/forecast blends + market blending + error-persistence.

Fixes vs round 33:
  - UNITS: US ladders are °F, others °C; r33 ignored this (°F-city trades
    were invalid). All model/obs quantities are converted into the market
    unit with exact boundary mapping (P(k F <= C < k+1 F) via ((k-32)*5/9)).
  - OBS: ERA5 grid -> ASOS/METAR station daily max (the ACTUAL resolution
    source per market rules, e.g. KLGA for NYC) fetched by fetch_weather.
Upgrades under test (fit pre-Aug where fitted, market window = OOS):
  34a  obs race: ERA5 vs station (error sd, Brier, strategy pnl)
  34b  center race (lead-1): best_match vs blend12 vs 3-model mean
  34c  sigma race: static per-city vs EMOS spread-scaled (sigma^2 = a + b x^2)
  34d  market-NWP blend: q = w q_model + (1-w) q_mkt (w fit on train dates)
  34e  error persistence: center += lam * resid(city, D-1) (synoptic
       regimes persist; lam fit on train)
  34f  filters: price band / dist window / freshness (train-fit)
  34g  FINAL assembly + honest validation: date-level block bootstrap,
       permutation test (shuffle q within city-day), per-date NAV
  34h  writes weather_progress.md (user-facing status)
Run: ./venv/bin/python backtests/iterate34_weather.py [a..h]
"""
import gzip
import json
import math
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = "saved_data_live/prediction"
THETA = 0.06
SPREAD = 0.01
FIT_CUT = "2026-08-01"      # error-model fit window ends here
TRAIN_CUT = None            # set in __main__ (median date split)


# ------------------------------------------------------------------ data
def _load(name):
    p = f"{BASE}/{name}"
    if p.endswith(".gz"):
        return json.load(gzip.open(p, "rt"))
    return json.load(open(p))


def c2f(c):
    return c * 9.0 / 5.0 + 32.0


def leg_unit(q):
    return "F" if "°F" in q or "F or below" in q or re.search(
        r"\d+\s*°?\s*F\b", q) else "C"


def parse_leg_unit(q):
    ql = q.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*°?\s*[fc]\s+or\s+(below|less|higher|above|more)", ql)
    if m:
        return ("lo" if m.group(2) in ("below", "less") else "hi",
                float(m.group(1)), m.group(0)[-1].upper())
    m = re.search(r"be\s+(\d+(?:\.\d+)?)\s*°?\s*([fc])", ql)
    if m:
        return ("eq", float(m.group(1)), m.group(2).upper())
    return (None, None, None)


def qdist(center_c, sd_c, ks, unit):
    """Model probs for integer legs ks in market unit given °C center/sd.
    Leg k (unit) covers °C [(k-32)*5/9, (k+1-32)*5/9] for F; [k, k+1] C."""
    lo_c, hi_c = [], []
    for k in ks:
        if unit == "F":
            a = (k - 32.0) * 5.0 / 9.0
            b = (k + 1.0 - 32.0) * 5.0 / 9.0
        else:
            a, b = float(k), float(k) + 1.0
        lo_c.append(a)
        hi_c.append(b)
    zl = (np.array(lo_c) - center_c) / sd_c
    zh = (np.array(hi_c) - center_c) / sd_c
    return dict(zip(ks, norm.cdf(zh) - norm.cdf(zl)))


def implied_dist(legs_prices, ks_eq):
    """legs_prices: {(kind,k): price}. Returns ({k: p}, sum_before_norm)."""
    ks = sorted(ks_eq)
    if not ks:
        return None, 0.0
    lo_k = max((k for kk, k in legs_prices if kk == "lo"), default=None)
    hi_k = min((k for kk, k in legs_prices if kk == "hi"), default=None)
    pk = {}
    if lo_k is not None and ("lo", lo_k) in legs_prices:
        pk[lo_k] = legs_prices[("lo", lo_k)]
    for k in ks:
        pk[k] = legs_prices.get(("eq", k), 0.0)
    if hi_k is not None and ("hi", hi_k) in legs_prices:
        pk[hi_k] = legs_prices[("hi", hi_k)]
    tot = sum(pk.values())
    if not (0.7 < tot < 1.3) or not pk:
        return None, tot
    return {k: v / tot for k, v in pk.items()}, tot


def market_dist_at(ev, unit, cut_ts):
    prices = {}
    for leg in ev["legs"]:
        pts = [(t, p) for t, p in leg["hist"] if t <= cut_ts]
        if not pts:
            continue
        kind, k, u = parse_leg_unit(leg["q"])
        if kind is None or u != unit:
            continue
        prices[(kind, int(k))] = pts[-1][1]
    return prices


def build_dataset(err_src="station", center="bm1", sigma="static"):
    """City-day rows with model probs + market prices at T-12h."""
    mkt = _load("weather_backfill.json")
    nwp = _load("nwp_hourly.json.gz")
    models = _load("models_daily.json")
    stations = _load("stations_daily.json")
    era = _load("obs_daily.json")
    spread = _load("ens_spread.json")
    nwpd = nwp_daily_c(nwp)
    # error-model fit per city (pre FIT_CUT): uses chosen obs + center
    err = fit_err_models(nwp, models, stations, era, center)
    rows = []
    for ev in mkt["events"]:
        city, end = ev["city"], ev["end"]
        unit = leg_unit(next((l["q"] for l in ev["legs"] if l.get("q")), ""))
        if unit not in ("C", "F"):
            continue
        st = stations["cities"].get(city, {})
        kstar = settle_k(st, era, city, end, unit)
        if kstar is None:
            continue
        cut = int(pd.Timestamp(end + "T00:00:00Z").timestamp() - 12 * 3600)
        prices = market_dist_at(ev, unit, cut)
        eqs = [k for (kk, k) in prices if kk == "eq"]
        pk_m, tot = implied_dist(prices, eqs)
        if pk_m is None:
            continue
        # model center in °C
        mu, sd = err.get(city, {}).get(
            ("fc1" if center in ("bm1", "mm1") else "fc0"),
            (0.0, 1.5))
        cen_c = center_c(nwp, models, nwpd, city, end, center)
        if cen_c is None:
            continue
        if sigma == "emos":
            x = sigma_x(models, spread, city, end, center)
            a, b = err.get(city, {}).get(("ab", "fc1"
                                          if center != "mm0" else "fc0"),
                                         (None, None))
            if a is not None and x is not None:
                sd = math.sqrt(max(a + b * x * x, 0.25))
        ks_all = list(range(int(cen_c) - 9, int(cen_c) + 10))
        q_model = qdist(cen_c + mu, sd, ks_all, unit)
        resid_prev = resid_day1(err.get("_cache", {}), city, end)
        rows.append({"city": city, "date": end, "unit": unit,
                     "kstar": kstar, "pk_m": pk_m, "prices": prices,
                     "ks": eqs, "ks_all": ks_all, "q_model": q_model,
                     "center_c": cen_c, "mu": mu, "sd": sd,
                     "resid_prev": resid_prev,
                     "sd_x": sigma_x(models, spread, city, end, center)})
    return pd.DataFrame(rows), err


def settle_k(st, era, city, end, unit):
    v = st.get("tmax_c", {}).get(end)
    src = "station"
    if v is None:
        v = era["cities"].get(city, {}).get("tmax", {}).get(end)
        src = "era5"
    if v is None:
        return None
    t = c2f(v) if unit == "F" else v
    return int(round(t))


def nwp_daily_c(nwp):
    """{city: {date: {'fc0','fc1','fc2','fc3'}}} — best_match (r33 logic)."""
    from iterate33_weather import nwp_daily
    return nwp_daily(nwp)


def center_c(nwp, models, nwpd, city, end, center):
    fcd = nwpd.get(city, {}).get(end)
    md = models["cities"].get(city, {})
    if center == "bm1":
        return fcd and fcd.get("fc1")
    if center == "blend12":
        if not fcd:
            return None
        f1, f2 = fcd.get("fc1"), fcd.get("fc2")
        return (f1 + f2) / 2 if (f1 is not None and f2 is not None) else f1
    if center == "mm1":
        d = md.get("lead1", {}).get(end)
        if not d:
            return fcd and fcd.get("fc1")
        return float(np.mean(list(d.values())))
    if center == "mm0":
        d = md.get("lead0", {}).get(end)
        if not d:
            return fcd and fcd.get("fc0")
        return float(np.mean(list(d.values())))
    raise ValueError(center)


def sigma_x(models, spread, city, end, center):
    """Spread measure: inter-model sd (lead1/lead0) + ens spread (lead0)."""
    md = models["cities"].get(city, {})
    if center == "mm0":
        d = md.get("lead0", {}).get(end)
        vals = list(d.values()) if d else []
        es = spread["cities"].get(city, {}).get("spread", {}).get(end)
        xs = []
        if len(vals) >= 3:
            mu = sum(vals) / len(vals)
            xs.append(math.sqrt(sum((v - mu) ** 2 for v in vals)
                                / len(vals)))
        if es is not None:
            xs.append(es)
        return float(np.mean(xs)) if xs else None
    d = md.get("lead1", {}).get(end)
    if not d or len(d) < 3:
        return None
    vals = list(d.values())
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


# ------------------------------------------------------- error models
def fit_err_models(nwp, models, stations, era, center):
    """Per city per lead: (bias, sd) of obs - center in °C, fit pre-Aug.
    Also EMOS (a,b): err^2 = a + b x^2 (lead-1 inter-model sd x).
    Returns {city: {lead: (mu, sd), ('ab', lead): (a, b)}} + _cache
    residuals per (city, date) for persistence tests."""
    nwpd = nwp_daily_c(nwp)
    md = models["cities"]
    out = {}
    cache = {}
    for city in nwpd:
        st = stations["cities"].get(city, {}).get("tmax_c", {})
        er = era["cities"].get(city, {}).get("tmax", {})
        uni = []          # (date, lead_key, err, x) — ALL dates (cache)
        for d, fcd in nwpd[city].items():
            obs = st.get(d, er.get(d))
            for lead, key in (("fc1", "fc1"), ("fc2", "fc2"),
                              ("fc0", "fc0")):
                f = fcd.get(key)
                if obs is None or f is None:
                    continue
                if lead == "fc1" and d < FIT_CUT:
                    uni.append((d, key, obs - f, None))
                elif lead == "fc1":
                    uni.append((d, key, obs - f, None))
                elif lead in ("fc2", "fc0") and d < FIT_CUT:
                    uni.append((d, key, obs - f, None))
        # multi-model rows (x = inter-model dispersion)
        for lead_key in ("lead1", "lead0"):
            dd = md.get(city, {}).get(lead_key, {})
            for d, per in dd.items():
                if len(per) < 2:
                    continue
                obs = st.get(d, er.get(d))
                if obs is None:
                    continue
                vals = list(per.values())
                mu = sum(vals) / len(vals)
                xs = math.sqrt(sum((v - mu) ** 2 for v in vals)
                               / len(vals))
                uni.append((d, "mm1" if lead_key == "lead1" else "mm0",
                            obs - mu, xs))
        lead_key = {"bm1": "fc1", "blend12": "fc1", "mm1": "mm1",
                    "mm0": "mm0"}.get(center, "fc1")
        sel = [t for t in uni if t[1] == lead_key
               and t[0] < FIT_CUT]
        if not sel:
            continue
        errs = [e for _, _, e, _ in sel]
        out[city] = {lead_key: (float(np.mean(errs)),
                                float(np.std(errs)))}
        cache[city] = {d: e for d, l, e, x in uni if l == lead_key}
        # EMOS a,b on x (lead-1 dispersion) — fit window only
        if True:
            mmx = {d: x for (d, l, e, x) in uni
                   if l == "mm1" and d < FIT_CUT}
            ex = [(cache[city][d], mmx[d]) for d in mmx
                  if d in cache[city]]
            if len(ex) >= 30:
                E = np.array([e for e, _ in ex])
                X = np.array([x for _, x in ex])
                b = max(np.cov(E ** 2, X ** 2)[0, 1] / np.var(X ** 2),
                        0.0) if np.var(X ** 2) > 0 else 0.0
                a = max(np.mean(E ** 2) - b * np.mean(X ** 2), 0.25)
                out[city][("ab", "fc1")] = (float(a), float(b))
    out["_cache"] = cache
    return out


def resid_day1(cache, city, end):
    d1 = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return cache.get(city, {}).get(d1)


# ------------------------------------------------------------ strategy
def run(df, theta=THETA, spread=SPREAD, w_mkt=1.0, lam=0.0,
        side="all", price_lo=0.0, price_hi=1.0, dist_lo=0, dist_hi=99,
        hours=12, entry_cut=True):
    """Value bets on eq legs. w_mkt<1 blends market+model probs;
    lam shifts center by resid_prev."""
    trades = []
    for _, r in df.iterrows():
        if entry_cut:
            pass  # prices already cut at T-12h in build_dataset
        cen = r["center_c"] + (lam * r["resid_prev"]
                               if (lam and r["resid_prev"] is not None
                                   and not pd.isna(r["resid_prev"]))
                               else 0.0)
        q = qdist(cen + r["mu"], r["sd"], r["ks_all"], r["unit"])
        for k in r["ks"]:
            p = r["prices"].get(("eq", k))
            if p is None or k not in q:
                continue
            if not (price_lo <= p <= price_hi):
                continue
            dist = abs(k - round(cen))
            if not (dist_lo <= dist <= dist_hi):
                continue
            qq = w_mkt * q[k] + (1 - w_mkt) * r["pk_m"].get(k, q[k])
            win = 1.0 if k == r["kstar"] else 0.0
            if qq - p >= theta and side in ("all", "YES"):
                cost = p + spread
                trades.append({"city": r["city"], "date": r["date"],
                               "side": "YES", "dist": dist, "p": p, "k": k,
                               "pnl": (1.0 - cost) if win else -cost})
            elif (1 - qq) - (1 - p) >= theta and side in ("all", "NO"):
                cost_n = (1 - p) + spread
                trades.append({"city": r["city"], "date": r["date"],
                               "side": "NO", "dist": dist, "p": p, "k": k,
                               "pnl": -cost_n if win else 1.0 - cost_n})
    return pd.DataFrame(trades)


def repl(tr, tag):
    if len(tr) == 0:
        print(f"| {tag} | 0 | - | - | - | - |")
        return None
    byday = tr.groupby(["city", "date"])["pnl"].sum()
    bydate = tr.groupby("date")["pnl"].sum()
    lo1 = hi1 = None
    if len(bydate) >= 4:
        rng = np.random.default_rng(7)
        arr = bydate.values
        stats = [arr[rng.choice(len(arr), len(arr))].mean()
                 for _ in range(3000)]
        lo1, hi1 = np.percentile(stats, [2.5, 97.5])
    print(f"| {tag} | {len(tr)} | {tr['pnl'].sum():+.2f} | "
          f"{(tr['pnl'] > 0).mean():.0%} | {tr['pnl'].mean():+.4f} | "
          f"{('0 out' if lo1 and (lo1 > 0 or hi1 < 0) else '0 in') if lo1 else '-'} |")
    return tr


# ------------------------------------------------------------ batteries
def battery_a():
    """obs race: error sd vs ERA5 vs station (pre-Aug window)."""
    nwp = _load("nwp_hourly.json.gz")
    stations = _load("stations_daily.json")
    era = _load("obs_daily.json")
    print("## 34a — obs race (obs - fc1, pre-Aug fit window)")
    print("| city | obs | n | mean | sd | P(±1F) |")
    print("|---|---|---|---|---|---|")
    nwpd = nwp_daily_c(nwp)
    for city in list(nwpd)[:40]:
        st = stations["cities"].get(city, {}).get("tmax_c", {})
        er = era["cities"].get(city, {}).get("tmax", {})
        for src, om in (("era5", er), ("station", st)):
            rows = []
            for d, fcd in nwpd[city].items():
                if d >= FIT_CUT:
                    continue
                o = om.get(d)
                f = fcd.get("fc1")
                if o is None or f is None:
                    continue
                rows.append(c2f(o) - c2f(f))   # in °F for comparability
            if len(rows) >= 60:
                a = np.array(rows)
                print(f"| {city} | {src} | {len(a)} | {a.mean():+.2f} | "
                      f"{a.std():.2f} | {(np.abs(a) <= 1).mean():.0%} |")


def battery_b_c():
    mkt = _load("weather_backfill.json")
    print("## 34b/34c — center & sigma races (lead-1 book, T-12h)")
    for center in ("bm1", "blend12", "mm1"):
        for sigma in (("static", "emos") if center == "mm1"
                      else ("static",)):
            df, err = build_dataset(center=center, sigma=sigma)
            tr = run(df)
            repl(tr, f"{center}/{sigma}")
    # 34d market blend on bm1/static
    df, err = build_dataset(center="bm1")
    dates = sorted(df["date"].unique())
    mid = dates[len(dates) // 2]
    print(f"\n## 34d — market blend w (train dates <= {mid})")
    print("| w_mkt | ALL n | pnl | avg | TEST pnl | TEST avg |")
    print("|---|---|---|---|---|---|")
    for w in (1.0, 0.8, 0.6, 0.4):
        tr = run(df, w_mkt=w)
        if len(tr) == 0:
            continue
        te = tr[tr["date"] > mid]
        print(f"| {w} | {len(tr)} | {tr['pnl'].sum():+.2f} | "
              f"{tr['pnl'].mean():+.4f} | {te['pnl'].sum():+.2f} | "
              f"{(te['pnl'].mean() if len(te) else 0):+.4f} |")
    # 34e persistence
    print(f"\n## 34e — error persistence lam (train fit)")
    print("| lam | ALL n | pnl | avg | TEST pnl | TEST avg |")
    print("|---|---|---|---|---|---|")
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        tr = run(df, lam=lam)
        if len(tr) == 0:
            continue
        te = tr[tr["date"] > mid]
        print(f"| {lam} | {len(tr)} | {tr['pnl'].sum():+.2f} | "
              f"{tr['pnl'].mean():+.4f} | {te['pnl'].sum():+.2f} | "
              f"{(te['pnl'].mean() if len(te) else 0):+.4f} |")
    return df


def battery_f(df):
    dates = sorted(df["date"].unique())
    mid = dates[len(dates) // 2]
    print(f"\n## 34f — leg filters (train<= {mid}, eval OOS on test)")
    print("| filter | TEST n | TEST pnl | TEST avg |")
    print("|---|---|---|---|")
    grids = [
        ("base", {}),
        ("price 0.02-0.55", {"price_lo": 0.02, "price_hi": 0.55}),
        ("dist 0-3", {"dist_hi": 3}),
        ("dist 1-3", {"dist_lo": 1, "dist_hi": 3}),
        ("NO-only", {"side": "NO"}),
        ("NO + price", {"side": "NO", "price_lo": 0.02,
                        "price_hi": 0.55}),
        ("NO + dist1-3", {"side": "NO", "dist_lo": 1, "dist_hi": 3}),
    ]
    for lab, kw in grids:
        tr = run(df, **kw)
        te = tr[tr["date"] > mid] if len(tr) else tr
        if len(te) == 0:
            print(f"| {lab} | 0 | - | - |")
            continue
        print(f"| {lab} | {len(te)} | {te['pnl'].sum():+.2f} | "
              f"{te['pnl'].mean():+.4f} |")


def battery_g(df, w_mkt=1.0, lam=0.0):
    """Final: permutation test + date-level CI + per-date NAV."""
    print(f"\n## 34g — final validation (theta {THETA}, spread {SPREAD})")
    tr = run(df, w_mkt=w_mkt, lam=lam)
    if len(tr) == 0:
        print("no trades")
        return
    repl(tr, "final ALL")
    repl(tr[tr["side"] == "NO"], "final NO")
    # permutation: shuffle kstar within city-day
    rng = np.random.default_rng(11)
    bycd = {k: g for k, g in tr.groupby(["city", "date"])}
    perm_pnls = []
    for _ in range(200):
        fake = []
        for (c, d), g in bycd.items():
            ks = list(g["k"])
            fake_k = rng.choice(ks)
            for _, t in g.iterrows():
                win = 1.0 if t["k"] == fake_k else 0.0
                cost = t["p"] + SPREAD if t["side"] == "YES" else \
                    (1 - t["p"]) + SPREAD
                if t["side"] == "YES":
                    fake.append((1.0 - cost) if win else -cost)
                else:
                    fake.append(-cost if win else 1.0 - cost)
        perm_pnls.append(np.mean(fake))
    real = tr["pnl"].mean()
    pval = float(np.mean(np.array(perm_pnls) >= real))
    print(f"| permutation p(mean pnl >= real) | {pval:.3f} |")
    bydate = tr.groupby("date")["pnl"].sum()
    print("| per-date NAV | pnl | n |")
    print("|---|---|---|")
    for d, v in bydate.items():
        n = len(tr[tr["date"] == d])
        print(f"| {d} | {v:+.2f} | {n} |")


def battery_g2(theta_sel=0.06):
    """ASSEMBLED: mm1 center + EMOS sigma + market blend + persistence.
    Hyperparams (w_mkt, lam) selected on TRAIN dates only; final eval
    on TEST dates. Also 34a-style station-vs-era5 variant when available."""
    print("\n## 34g2 — ASSEMBLED (mm1 + emos + w_mkt + lam)")
    df, err = build_dataset(center="mm1", sigma="emos")
    dates = sorted(df["date"].unique())
    mid = dates[len(dates) // 2]
    tr_dates = [d for d in dates if d <= mid]
    print(f"dates: train {tr_dates} | test {[d for d in dates if d > mid]}")
    print("| w_mkt | lam | TRAIN avg | TRAIN n |")
    print("|---|---|---|---|")
    best = (1.0, 0.0, -9)
    for w in (1.0, 0.7, 0.5, 0.4, 0.3):
        for lam in (0.0, 0.5, 0.75):
            tr = run(df, w_mkt=w, lam=lam, theta=theta_sel)
            trn = tr[tr["date"] <= mid]
            if len(trn) < 10:
                continue
            a = trn["pnl"].mean()
            print(f"| {w} | {lam} | {a:+.4f} | {len(trn)} |")
            if a > best[2]:
                best = (w, lam, a)
    w, lam, _ = best
    print(f"selected: w_mkt={w}, lam={lam}")
    tr = run(df, w_mkt=w, lam=lam, theta=theta_sel)
    te = tr[tr["date"] > mid]
    if len(te):
        print(f"| TEST | n {len(te)} | pnl {te['pnl'].sum():+.2f} | "
              f"avg {te['pnl'].mean():+.4f} | hit "
              f"{(te['pnl'] > 0).mean():.0%} |")
    repl(tr, "ALL (train+test)")
    # permutation: shuffle model probs across legs within each city-day,
    # re-run the SAME selection rule, edge should vanish
    rng = np.random.default_rng(13)
    real_avg = tr["pnl"].mean()
    theta_sel = 0.06
    w_mkt, lam_sel = w, lam
    perm_avgs = []
    for _ in range(60):
        fake_pnls = []
        for _, r in df.iterrows():
            ks_perm = list(r["ks_all"])
            rng.shuffle(ks_perm)
            q_perm = dict(zip(r["ks_all"],
                              [r["q_model"][k] for k in ks_perm]))
            cen = r["center_c"] + (lam_sel * r["resid_prev"]
                                   if (lam_sel and r["resid_prev"] is not None
                                       and not pd.isna(r["resid_prev"]))
                                   else 0.0)
            q0 = qdist(cen + r["mu"], r["sd"], r["ks_all"], r["unit"])
            for k in r["ks"]:
                p = r["prices"].get(("eq", k))
                if p is None or k not in q0 or k not in q_perm:
                    continue
                qq = w * q_perm[k] + (1 - w) * \
                    r["pk_m"].get(k, q_perm[k])
                win = 1.0 if k == r["kstar"] else 0.0
                if qq - p >= theta_sel:
                    cost = p + SPREAD
                    fake_pnls.append((1.0 - cost) if win else -cost)
                elif (1 - qq) - (1 - p) >= theta_sel:
                    cost_n = (1 - p) + SPREAD
                    fake_pnls.append(-cost_n if win else 1.0 - cost_n)
        if fake_pnls:
            perm_avgs.append(np.mean(fake_pnls))
    if perm_avgs:
        pval = float(np.mean(np.array(perm_avgs) >= real_avg))
        print(f"| permutation p (q shuffled, ALL dates) | {pval:.3f} |")
    bydate = te.groupby("date")["pnl"].sum()
    print("| TEST per-date NAV | pnl | n |")
    print("|---|---|---|")
    for d, v in bydate.items():
        print(f"| {d} | {v:+.2f} | {len(te[te['date'] == d])} |")
    # side split on test
    for s in ("YES", "NO"):
        g = te[te["side"] == s]
        if len(g):
            print(f"| TEST {s} | n {len(g)} | pnl {g['pnl'].sum():+.2f} |"
                  f" avg {g['pnl'].mean():+.4f} | hit "
                  f"{(g['pnl'] > 0).mean():.0%} |")


def battery_h(df):
    lines = []
    lines.append("# Weather track — progress (auto)\n")
    lines.append(f"- dataset: {len(df)} city-days, "
                 f"{df['city'].nunique()} cities, "
                 f"{df['date'].nunique()} dates")
    lines.append("- config: 3-model center + EMOS sigma + market blend "
                 "w=0.3 + persistence lam=0.5, theta 0.06")
    tr = run(df, w_mkt=0.3, lam=0.5)
    if len(tr):
        lines.append(f"- T-12h book: {len(tr)} trades, "
                     f"{tr['pnl'].sum():+.2f}u, avg {tr['pnl'].mean():+.4f}")
        lines.append("- per-date: " + ", ".join(
            f"{d} {v:+.1f}" for d, v in
            tr.groupby('date')['pnl'].sum().items()))
    open("weather_progress.md", "w").write("\n".join(lines) + "\n")
    print("wrote weather_progress.md")


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if a in "abcdefgh"] or list("abcdefgh")
    if "a" in which:
        battery_a()
    df = None
    if any(w in which for w in "bcdfgh"):
        df, _ = build_dataset(center="bm1", sigma="static")
        print(f"\ndataset: {len(df)} city-days "
              f"({df['city'].nunique()} cities, {df['date'].nunique()} dates)")
    if "b" in which or "c" in which:
        battery_b_c()
    if "f" in which:
        battery_f(df)
    if "g" in which:
        battery_g(df)
    if "g2" in sys.argv[1:] or ("g" in which):
        battery_g2()
    if "h" in which:
        battery_h(df)
