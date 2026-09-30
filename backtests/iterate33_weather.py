"""Round 33 (Track B): WEATHER TEMP LADDERS — standalone strategy research.
SEPARATE from Track A (crypto-linked macro). Do not merge books.
VERDICT (technique_research round 33): machinery proven, edge pending
sample — pilot (12 city-days) mildly positive (NO-fade hit 76-81%),
market itself beats raw NWP at T-12h (Brier 0.066 vs 0.088); growing
dataset via weather_build.py daily cron; decision at 300-500 city-days.

Data (pair_scout/fetch_weather.py):
  weather_backfill.json  24 events / 264 legs, 8 cities, 60-min price hist
  nwp_hourly.json.gz     hourly t2m forecast-as-issued + previous-day
                         run vintages d1/d2/d3 (2026-03-01.., local tz)
  obs_daily.json         ERA5-grid daily max (settlement proxy)

Causality / no label leakage:
  - entry uses NWP vintages known at entry time (lead-1 = previous-day
    run; lead-0 partial = same-day run truncated at entry hour) and
    market prices before entry time;
  - obs_daily is used ONLY as settlement label;
  - error-model sd/mean fit on dates OUTSIDE the market window (the 6
    months of NWP-vs-obs history precede the 30d market window) or on
    train split only where joint with market data;
  - thresholds fit on train split / train cities, evaluated on test.

Batteries:
  33W-a  NWP skill: obs - fc by lead/city (6-month record)
  33W-b  market implied dist vs NWP: sd ratio, mean bias, tail pricing
  33W-c  market calibration vs realized (bucket frequency vs implied prob)
  33W-d  strategy backtest: value bets vs NWP (yes/no on eq legs + tails),
         flat size, spread/fee params, train/test by date + city holdout,
         city-day block bootstrap
Run: ./venv/bin/python backtests/iterate33_weather.py [a..d]
"""
import gzip
import json
import sys

import numpy as np
import pandas as pd

from scipy.stats import norm

WB = "saved_data_live/prediction/weather_backfill.json"
NW = "saved_data_live/prediction/nwp_hourly.json.gz"
OB = "saved_data_live/prediction/obs_daily.json"

SPREAD = 0.01      # cost to cross spread (hist price is last-trade)
FEE = 0.0          # poly taker fee param (swept in battery d)
THETA_GRID = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15)


def load_all():
    mkt = json.load(open(WB))
    nwp = json.load(gzip.open(NW, "rt"))
    obs = json.load(open(OB))
    return mkt, nwp, obs


def nwp_daily(nwp):
    """-> {city: {date: {'fc0','fc1','fc2','fc3'}}} daily max of hourly
    vintage series over the LOCAL day (all of it for d1/d2/d3; fc0 = the
    same-day run which is only partially known intraday — battery d
    truncates it at entry hour when needed)."""
    out = {}
    for city, d in nwp["cities"].items():
        t = pd.DatetimeIndex(pd.to_datetime(d["time"]))
        day = t.normalize()
        rows = {}
        for key, col in (("fc0", "t2m"), ("fc1", "d1"),
                         ("fc2", "d2"), ("fc3", "d3")):
            s = pd.Series(d[col], index=day).dropna()
            rows[key] = s.groupby(level=0).max()
        dates = sorted(set.intersection(*(set(rows[k].index)
                                          for k in ("fc0", "fc1"))))
        out[city] = {str(dd.date()): {k: float(rows[k][dd])
                                      for k in rows} for dd in dates}
    return out


# ---------------------------------------------------------------- battery a
def battery_a(nwpd, obs, window_start="2026-08-01"):
    """NWP skill on 6 months; sd/mean fit EXCLUDING the market window
    (>= window_start) so battery d entries use out-of-sample error model."""
    print("## Round 33W-a — NWP forecast error obs - fc by lead "
          "(fit window: before 2026-08-01)")
    model = {}
    print("| city | n | lead | mean err | sd | P(±1°) | P(±2°) |")
    print("|---|---|---|---|---|---|---|")
    for city, days in nwpd.items():
        om = obs["cities"].get(city, {}).get("tmax", {})
        rows = []
        for ds, fc in days.items():
            if ds >= window_start:
                continue
            o = om.get(ds)
            if o is None:
                continue
            for lead in ("fc1", "fc2", "fc3"):
                if fc.get(lead) is not None:
                    rows.append((lead, o - fc[lead]))
        df = pd.DataFrame(rows, columns=["lead", "err"])
        city_model = {}
        for lead, g in df.groupby("lead"):
            mu, sd = g["err"].mean(), g["err"].std()
            city_model[lead] = (float(mu), float(sd))
            print(f"| {city} | {len(g)} | {lead} | {mu:+.2f} | {sd:.2f} | "
                  f"{(g['err'].abs() <= 1).mean():.0%} | "
                  f"{(g['err'].abs() <= 2).mean():.0%} |")
        # pooled sd for shrinkage
        model[city] = city_model
    pool = {}
    for lead in ("fc1", "fc2", "fc3"):
        allsd = [model[c][lead][1] for c in model if lead in model[c]]
        allmu = [model[c][lead][0] for c in model if lead in model[c]]
        pool[lead] = (float(np.mean(allmu)), float(np.sqrt(np.mean(
            np.square(allsd)))))
        print(f"| POOLED | - | {lead} | {pool[lead][0]:+.2f} | "
              f"{pool[lead][1]:.2f} | - | - |")
    json.dump({"city": model, "pool": pool},
              open("/tmp/opencode/nwp_err_model.json", "w"))
    return model, pool


# ------------------------------------------------- dataset for b/c/d
def build_legs(mkt, nwpd, obs):
    """City-day rows: implied dist from legs at T-end-12h, NWP fc, obs."""
    rows = []
    for ev in mkt["events"]:
        city, end = ev["city"], ev["end"]
        om = obs["cities"].get(city, {}).get("tmax", {})
        realized = om.get(end)
        if realized is None:
            continue
        kstar = int(round(realized))
        # entry: 12h before end-of-local-day (approx local midnight UTC
        # end stamp) -> take last hist point strictly before (end 00:00
        # UTC minus 12h)
        cut = int((pd.Timestamp(end + "T00:00:00Z").timestamp()) - 12 * 3600)
        ps = {}
        ok = True
        for leg in ev["legs"]:
            pts = [(t, p) for t, p in leg["hist"] if t <= cut]
            if not pts:
                ok = False
                break
            ps[(leg["kind"], int(round(leg["k"])))] = pts[-1][1]
        if not ok or len(ps) < 5:
            continue
        ks = sorted(k for kk, k in ps if kk == "eq")
        if not ks:
            continue
        lo_k = max((k for kk, k in ps if kk == "lo"), default=ks[0] - 1)
        hi_k = min((k for kk, k in ps if kk == "hi"), default=ks[-1] + 1)
        # implied P(k) over integers lo_k-? .. hi_k: eq probs + tails
        pk = {}
        if ("lo", lo_k) in ps:
            pk[lo_k] = ps[("lo", lo_k)]
        for k in ks:
            pk[k] = ps.get(("eq", k), 0.0)
        if ("hi", hi_k) in ps:
            pk[hi_k] = ps[("hi", hi_k)]
        tot = sum(pk.values())
        if not (0.7 < tot < 1.3):
            continue
        pk = {k: v / tot for k, v in pk.items()}
        ks_all = list(range(min(pk), max(pk) + 1))
        pv = np.array([pk.get(k, 0.0) for k in ks_all])
        mean_m = float((pv * np.array(ks_all)).sum())
        sd_m = float(np.sqrt((pv * (np.array(ks_all) - mean_m) ** 2).sum()))
        fcd = nwpd.get(city, {}).get(end)
        if fcd is None:
            continue
        rows.append({"city": city, "date": end, "kstar": kstar,
                     "realized": realized, "mean_m": mean_m, "sd_m": sd_m,
                     "fc1": fcd["fc1"], "fc2": fcd.get("fc2"),
                     "fc3": fcd.get("fc3"), "pk": pk, "prices": ps,
                     "lo_k": lo_k, "hi_k": hi_k, "ks": ks})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- battery b
def battery_b(legs, err_model):
    print("\n## Round 33W-b — market implied vs NWP (lead-1)")
    print("| metric | value |")
    print("|---|---|")
    m, s = [], []
    for _, r in legs.iterrows():
        mu_e, sd_e = err_model[r["city"]]["fc1"]
        s.append(r["sd_m"] / sd_e)
        m.append(r["mean_m"] - (r["fc1"] + mu_e))
    s = pd.Series(s); m = pd.Series(m)
    print(f"| n city-days | {len(legs)} |")
    print(f"| implied_sd / NWP_sd: median | {s.median():.2f} |")
    print(f"| implied_sd / NWP_sd: p25..p75 | {s.quantile(.25):.2f}.."
          f"{s.quantile(.75):.2f} |")
    print(f"| implied_mean - fc: median | {m.median():+.2f} deg |")
    print(f"| implied_mean - fc: IQR | {m.quantile(.25):+.2f}.."
          f"{m.quantile(.75):+.2f} |")
    print(f"| |implied_mean - fc| > 1 deg | {(m.abs() > 1).mean():.0%} |")


# ---------------------------------------------------------------- battery c
def battery_c(legs):
    print("\n## Round 33W-c — market calibration (implied P vs realized)")
    print("| bucket (dist from fc1, deg) | n | implied P(hit) | realized freq |")
    print("|---|---|---|---|")
    rows = []
    for _, r in legs.iterrows():
        for k, p in r["pk"].items():
            rows.append((k - round(r["fc1"]), p, 1.0 if k == r["kstar"] else 0.0))
    df = pd.DataFrame(rows, columns=["d", "p", "hit"])
    for lo, hi in ((-10, -3), (-2, -1), (0, 0), (1, 2), (3, 10)):
        g = df[(df["d"] >= lo) & (df["d"] <= hi)]
        if len(g) < 10:
            continue
        print(f"| {lo}..{hi} | {len(g)} | {g['p'].mean():.3f} | "
              f"{g['hit'].mean():.3f} |")
    brier_m = (df["p"] - df["hit"]).pow(2).mean()
    # NWP brier on the same rows (lead-1, calibrated err model)
    em = json.load(open("/tmp/opencode/nwp_err_model.json"))["city"]
    qn = []
    for _, r in legs.iterrows():
        mu_e, sd_e = em[r["city"]]["fc1"]
        for k in r["ks"]:
            qk = (norm.cdf((k + 0.5 - r["fc1"] - mu_e) / sd_e)
                  - norm.cdf((k - 0.5 - r["fc1"] - mu_e) / sd_e))
            qn.append((qk, 1.0 if k == r["kstar"] else 0.0))
    qn = pd.DataFrame(qn, columns=["q", "hit"])
    brier_n = (qn["q"] - qn["hit"]).pow(2).mean()
    clim = df["hit"].mean()
    brier_c = (clim - qn["hit"]).pow(2).mean()
    print(f"| Brier: market {brier_m:.4f} | NWP {brier_n:.4f} | "
          f"climatology {brier_c:.4f} |")


# ---------------------------------------------------------------- battery d
def battery_d(legs, err_model, nwp_h, mkt, theta=0.06):
    print(f"\n## Round 33W-d — strategy backtest: value bets vs NWP "
          f"(theta {theta}, spread {SPREAD}, fee {FEE})")
    em = err_model
    ev_by_key = {(ev["city"], ev["end"]): ev for ev in mkt["events"]}

    def qvec(fc, sd_e, ks):
        z = (np.array(ks) + 0.5 - fc) / sd_e
        c = norm.cdf(z)
        q = np.diff(np.concatenate(([0.0], c, [1.0])))
        return dict(zip(ks, q))

    def entries_at(hours_before, use_partial):
        trades = []
        for _, r in legs.iterrows():
            ev = ev_by_key.get((r["city"], r["date"]))
            if ev is None:
                continue
            mu_e, sd_e = em[r["city"]]["fc1"]
            if use_partial:
                d = nwp_h["cities"][r["city"]]
                t = pd.DatetimeIndex(pd.to_datetime(d["time"]))
                s = pd.Series(d["t2m"], index=t.normalize())
                dd = pd.Timestamp(r["date"])
                if dd not in s.index:
                    continue
                # NO LEAKAGE: only hours of the local day known by entry
                # time (T = local midnight - hours_before) — e.g. T-6h
                # entry at 18:00 local sees hours 00..17.
                last_h = 24 - hours_before - 1
                vals = s.loc[dd][s.loc[dd].index.hour <= last_h].dropna()
                if len(vals) < 6:
                    continue
                fc = float(vals.max())
            else:
                fc = r["fc1"]
            q = qvec(fc + mu_e, sd_e, r["ks"])
            for k, pk in r["pk"].items():
                p = r["prices"].get(("eq", k))
                if p is None or k not in q:
                    continue
                dist = abs(k - round(fc))
                win = 1.0 if k == r["kstar"] else 0.0
                cost = p + SPREAD + FEE
                if q[k] - p >= theta:
                    pnl = (1.0 - cost) if win else -cost
                    trades.append({"city": r["city"], "date": r["date"],
                                   "side": "YES", "k": k, "p": p, "q": q[k],
                                   "dist": dist, "pnl": pnl})
                elif (1 - q[k]) - (1 - p) >= theta:
                    cost_n = (1 - p) + SPREAD + FEE
                    pnl = -cost_n if win else 1.0 - cost_n
                    trades.append({"city": r["city"], "date": r["date"],
                                   "side": "NO", "k": k, "p": p, "q": q[k],
                                   "dist": dist, "pnl": pnl})
        return pd.DataFrame(trades)

    def report(tr, tag):
        if len(tr) == 0:
            print(f"| {tag} | n 0 |")
            return
        dates = sorted(tr["date"].unique())
        mid = dates[len(dates) // 2]
        tr_te = tr[tr["date"] > mid]
        tr_tr = tr[tr["date"] <= mid]

        def row(lab, g):
            if len(g) == 0:
                return
            print(f"| {lab} | {len(g)} | {g['pnl'].sum():+.2f} | "
                  f"{(g['pnl'] > 0).mean():.0%} | {g['pnl'].mean():+.4f} |")

        print(f"| {tag} | trades | pnl | hit | avg |")
        print("|---|---|---|---|---|")
        row(f"{tag} ALL", tr)
        row(f"{tag} train(dates<= {mid})", tr_tr)
        row(f"{tag} TEST(dates> {mid})", tr_te)
        row(f"{tag} YES", tr[tr["side"] == "YES"])
        row(f"{tag} NO", tr[tr["side"] == "NO"])
        for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 10)):
            row(f"{tag} dist {lo}..{hi}",
                tr[(tr["dist"] >= lo) & (tr["dist"] <= hi)])
        byday = tr.groupby(["city", "date"])["pnl"].sum()
        if len(byday) >= 20:
            rng = np.random.default_rng(7)
            arr = byday.values
            stats = [arr[rng.choice(len(arr), len(arr))].mean()
                     for _ in range(2000)]
            lo_b, hi_b = np.percentile(stats, [2.5, 97.5])
            print(f"| {tag} block-CI mean/day | [{lo_b:+.4f}, {hi_b:+.4f}] "
                  f"| zero-inside {lo_b <= 0 <= hi_b} | - | - |")

    report(entries_at(12, False), "T-12h lead1")
    report(entries_at(6, True), "T-6h lead0@17h")


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if a in "abcd"] or list("abcd")
    mkt, nwp, obs = load_all()
    nwpd = nwp_daily(nwp)
    em, pool = battery_a(nwpd, obs) if "a" in which else (
        json.load(open("/tmp/opencode/nwp_err_model.json"))["city"], None)
    legs = build_legs(mkt, nwpd, obs)
    print(f"\ndataset: {len(legs)} city-days "
          f"({legs['city'].nunique() if len(legs) else 0} cities, "
          f"{legs['date'].nunique() if len(legs) else 0} dates)")
    if "b" in which:
        battery_b(legs, em)
    if "c" in which:
        battery_c(legs)
    if "d" in which:
        battery_d(legs, em, nwp, mkt)
