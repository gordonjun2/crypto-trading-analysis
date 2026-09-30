"""Daily weather-market dataset builder (Track B forward validation).

Consumes saved_data_live/prediction/snapshots.jsonl (hourly poly_weather
quotes: bid/ask per leg) + nwp_hourly.json.gz (forecast vintages) +
obs_daily.json (settlement) -> appends settled city-day leg records to
saved_data_live/prediction/weather_dataset.jsonl.

Pricing = executable quotes from snapshots (taker YES at ask, NO at
1-bid); NO last-trade guessing. Causality: a leg's entry quote for entry
time T is the LAST snapshot quote strictly before T; NWP uses vintages
known at T; obs only for settlement.

  daily_build()   consolidate new settled city-days (cron, ~01:20 UTC)
  evaluate()      battery b/c/d re-run on the accumulated dataset
Run: ./venv/bin/python pair_scout/weather_build.py build|evaluate
"""
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

from iterate33_weather import nwp_daily  # noqa: E402

BASE = Path("saved_data_live/prediction")
DS = BASE / "weather_dataset.jsonl"
SNAP = BASE / "snapshots.jsonl"
SPREAD = 0.0          # quotes are executable: taker pays the spread already
FEE = 0.0
THETA = 0.06


def _norm_city(c):
    return {"New York City": "New York"}.get(c, c)


def daily_build():
    # refresh forecast vintages + obs first (overwrites whole files)
    import pair_scout.fetch_weather as FW
    FW.backfill_nwp()
    FW.backfill_obs()
    nwp = json.load(gzip.open(BASE / "nwp_hourly.json.gz", "rt"))
    nwpd = nwp_daily(nwp)
    obs = json.load(open(BASE / "obs_daily.json"))
    # settle-guard: a city-day is settled only 30h past its local end
    # (all tz offsets + obs availability) — avoids partial-day obs
    now = pd.Timestamp.utcnow().tz_localize(None).timestamp()
    done = set()
    if DS.exists():
        for line in open(DS):
            r = json.loads(line)
            done.add((r["city"], r["date"]))
    # 1) collect quotes: (city, ends, q) -> list (ts, bid, ask)
    quotes = {}
    for line in open(SNAP):
        rec = json.loads(line)
        ts = rec["ts"]
        for leg in rec.get("poly_weather", []):
            if leg.get("bid") is None or leg.get("ask") is None:
                continue
            key = (leg.get("city"), leg.get("ends"), leg.get("q"))
            quotes.setdefault(key, []).append(
                (ts, leg["bid"], leg["ask"]))
    # 2) parse legs, group per city-day
    days = {}
    for (city, ends, q), pts in quotes.items():
        city = _norm_city(city)
        ql = (q or "").lower()
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*°?c\s+or\s+(below|less|higher|above|more)", ql)
        if m:
            kind, k = ("lo" if m.group(2) in ("below", "less") else "hi",
                       int(float(m.group(1))))
        else:
            m = re.search(r"be\s+(\d+(?:\.\d+)?)\s*°?c", ql)
            if not m:
                continue
            kind, k = "eq", int(float(m.group(1)))
        days.setdefault((city, ends), {})[(kind, k)] = pts
    # 3) settled days -> records
    n_new = 0
    with open(DS, "a") as fh:
        for (city, ends), legs in sorted(days.items()):
            if (city, ends) in done:
                continue
            end_ts = pd.Timestamp(ends + "T00:00:00Z").timestamp()
            if now < end_ts + 30 * 3600:
                continue  # not settled yet (partial-day obs hazard)
            om = obs["cities"].get(city, {}).get("tmax", {})
            realized = om.get(ends)
            if realized is None:
                continue  # not settled yet
            fcd = nwpd.get(city, {}).get(ends)
            if fcd is None:
                continue
            kstar = int(round(realized))
            rec = {"city": city, "date": ends, "kstar": kstar,
                   "realized": realized, "fc1": fcd["fc1"],
                   "fc2": fcd.get("fc2"), "fc3": fcd.get("fc3"),
                   "built": datetime.now(timezone.utc).isoformat(),
                   "legs": {}}
            for (kind, k), pts in legs.items():
                for hours, tag in ((12, "T12"), (6, "T6")):
                    cut = end_ts - hours * 3600
                    prior = [(ts, b, a) for ts, b, a in pts
                             if pd.Timestamp(ts).timestamp() <= cut]
                    if not prior:
                        continue
                    ts, bid, ask = prior[-1]
                    rec["legs"][f"{kind}:{k}:{tag}"] = {
                        "bid": bid, "ask": ask, "ts": ts}
            # NWP vintage t2m truncated at 17h local for T6 model prob
            d = nwp["cities"][city]
            t = pd.DatetimeIndex(pd.to_datetime(d["time"]))
            s = pd.Series(d["t2m"], index=t.normalize())
            dd = pd.Timestamp(ends)
            if dd in s.index:
                vals = s.loc[dd][s.loc[dd].index.hour <= 17].dropna()
                rec["fc0_17"] = float(vals.max()) if len(vals) >= 6 else None
            fh.write(json.dumps(rec) + "\n")
            n_new += 1
    print(f"daily_build: +{n_new} city-days -> {DS} "
          f"(total {len(done) + n_new})")


def evaluate():
    if not DS.exists():
        print("no dataset yet")
        return
    rows = [json.loads(line) for line in open(DS)]
    print(f"## Weather forward dataset: {len(rows)} city-days")
    em = json.load(open("/tmp/opencode/nwp_err_model.json"))["city"]
    df = pd.DataFrame(rows)

    # calibration vs realized (market quotes at T-12h)
    print("| bucket (dist from fc1) | n | implied | realized |")
    print("|---|---|---|---|")
    allp, allh = [], []
    for _, r in df.iterrows():
        lo = max((k for kk, k in [(x.split(":")[0], int(x.split(":")[1]))
                                  for x in r["legs"] if x.endswith(":T12")
                                  and x.split(":")[0] == "lo"]),
                 default=None)
        hi = min((k for kk, k in [(x.split(":")[0], int(x.split(":")[1]))
                                  for x in r["legs"] if x.endswith(":T12")
                                  and x.split(":")[0] == "hi"]),
                 default=None)
        eqs = {int(x.split(":")[1]): r["legs"][x]["ask"]
               for x in r["legs"] if x.endswith(":T12")
               and x.split(":")[0] == "eq"}
        ks = sorted(eqs)
        if not ks:
            continue
        pk = {}
        if lo is not None and f"lo:{lo}:T12" in r["legs"]:
            pk[lo] = r["legs"][f"lo:{lo}:T12"]["bid"]
        for k in ks:
            pk[k] = eqs[k]
        if hi is not None and f"hi:{hi}:T12" in r["legs"]:
            pk[hi] = r["legs"][f"hi:{hi}:T12"]["bid"]
        tot = sum(pk.values())
        if not (0.7 < tot < 1.3):
            continue
        for k, p in pk.items():
            allp.append(p / tot)
            allh.append(1.0 if k == r["kstar"] else 0.0)
    if allp:
        g = pd.DataFrame({"d": [0] * len(allp), "p": allp, "hit": allh})
        print(f"| all legs (T-12h) | {len(g)} | {g['p'].mean():.3f} | "
              f"{g['hit'].mean():.3f} |")
        print(f"| Brier market {(g['p'] - g['hit']).pow(2).mean():.4f} | "
              f"vs clim {(g['hit'].mean() - g['hit']).pow(2).mean():.4f} |")

    # strategy: value bets with executable quotes
    for tag, hours, use_fc0 in (("T-12h blend12", 12, False),
                                ("T-6h lead0@17h", 6, True)):
        trades = []
        for _, r in df.iterrows():
            mu_e, sd_e = em.get(r["city"], em["Paris"])["fc1"]
            if use_fc0:
                fc = r["fc0_17"] if r.get("fc0_17") else r["fc1"]
                sd = sd_e
            else:
                # blend lead1+lead2 (round 33W-f: +28% pnl/trade vs fc1)
                mu2, sd2 = em.get(r["city"], em["Paris"]).get(
                    "fc2", (0.0, sd_e))
                fc = (r["fc1"] + (r["fc2"] if r.get("fc2")
                                  is not None else r["fc1"])) / 2
                sd = (sd_e + sd2) / 2
            center = fc + mu_e
            ks_all = range(int(center) - 8, int(center) + 9)
            z = (np.array(list(ks_all)) + 0.5 - center) / sd_e
            c = norm.cdf(z)
            q = np.diff(np.concatenate(([0.0], c, [1.0])))
            qd = dict(zip(ks_all, q))
            for key, leg in r["legs"].items():
                kind, kk, hh = key.split(":")
                k = int(kk)
                if hh != ("T6" if use_fc0 else "T12") or kind != "eq":
                    continue
                ask, bid = leg["ask"], leg["bid"]
                if k not in qd or qd[k] is None:
                    continue
                win = 1.0 if k == r["kstar"] else 0.0
                if qd[k] - ask >= THETA:
                    cost = ask + FEE
                    trades.append({"date": r["date"], "side": "YES",
                                   "dist": abs(k - round(fc)),
                                   "pnl": (1.0 - cost) if win else -cost})
                elif (1 - qd[k]) - (1 - bid) >= THETA:
                    cost = (1 - bid) + FEE
                    trades.append({"date": r["date"], "side": "NO",
                                   "dist": abs(k - round(fc)),
                                   "pnl": -cost if win else 1.0 - cost})
        t = pd.DataFrame(trades)
        if len(t) == 0:
            print(f"| {tag} | n 0 |")
            continue

        def row(lab, g):
            if len(g) == 0:
                return
            print(f"| {lab} | {len(g)} | {g['pnl'].sum():+.2f} | "
                  f"{(g['pnl'] > 0).mean():.0%} | {g['pnl'].mean():+.4f} |")

        print(f"| {tag} | trades | pnl | hit | avg |")
        print("|---|---|---|---|---|")
        row("ALL", t)
        row("YES", t[t["side"] == "YES"])
        row("NO", t[t["side"] == "NO"])
        for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 10)):
            row(f"dist {lo}..{hi}",
                t[(t["dist"] >= lo) & (t["dist"] <= hi)])
        byday = t.groupby("date")["pnl"].sum()
        if len(byday) >= 20:
            rng = np.random.default_rng(7)
            arr = byday.values
            stats = [arr[rng.choice(len(arr), len(arr))].mean()
                     for _ in range(2000)]
            lo_b, hi_b = np.percentile(stats, [2.5, 97.5])
            print(f"| block-CI mean/day | [{lo_b:+.4f}, {hi_b:+.4f}] | "
                  f"zero-inside {lo_b <= 0 <= hi_b} | - | - |")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        daily_build()
    elif cmd == "evaluate":
        evaluate()
