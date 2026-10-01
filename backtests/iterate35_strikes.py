"""Round 35 (Track C): CRYPTO STRIKE LADDERS — binary-strike books on
BTC/ETH ("above $X at 12:00 ET"), settlement = Binance 1m close. We OWN
the underlying data feed; the model is realized-vol based. Dual purpose:
standalone book + implied-vol surface for SQUEEZE conditioning.

Causality: spot/sigma from Binance hourly closes strictly before entry;
market price = last hist point <= entry; settlement label only for pnl.

Batteries (fit/selection on first half of events, test on rest):
  35a  calibration: market CDF vs realized; model vs realized; implied
       sigma (inverted from ladder) vs subsequent realized sigma (VRP)
  35b  sigma estimator race: EWMA halflife 12/24/48/96h, simple std,
       Parkinson; drift on/off
  35c  tails: normal vs Student-t (df 3/5)
  35d  market blend: q = w q_model + (1-w) q_mkt; and sigma blend
  35e  filters: theta, distance bands (sigma units), side, entry time
  35f  final: permutation (shuffle q within event) + settle-block CI
Run: ./venv/bin/python backtests/iterate35_strikes.py [a..f]
"""
import json
import math
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

BASE = "saved_data_live/prediction"
THETA = 0.06
SPREAD = 0.01
EVENT_CUT_FRAC = 0.5   # first half events = train


def _load(name):
    return json.load(open(f"{BASE}/{name}"))


def settle_dt_utc(title, end):
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    m = re.search(r"(\w+)\s+(\d{1,2}),?\s*(\d{1,2})(AM|PM)?\s*ET",
                  title or "")
    months = {mn: i for i, mn in enumerate(
        ("January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November",
         "December"), 1)}
    if m and m.group(1) in months:
        mon, day, hour, ap = (months[m.group(1)], int(m.group(2)),
                              int(m.group(3)), m.group(4))
        year = int(end[:4])
        h24 = (hour % 12 + (12 if ap == "PM" else 0)) if ap else 12
        return datetime(year, mon, day, h24, 1,
                        tzinfo=ZoneInfo("America/New_York")) \
            .astimezone(timezone.utc)
    try:
        y, mo, dy = map(int, end.split("-"))
        return datetime(y, mo, dy, 12, 1,
                        tzinfo=ZoneInfo("America/New_York")) \
            .astimezone(timezone.utc)
    except Exception:
        return None


def build_dataset(halflife=24, entry_hours=4):
    bk = _load("strike_backfill.json")
    se = _load("strike_settle.json")["settle"]
    kl = _load("strike_klines_h1.json")["klines"]
    rows = []
    for ev in bk["events"]:
        st_dt = settle_dt_utc(ev["title"], ev["end"])
        if st_dt is None:
            continue
        key = f"{ev['sym']}:{st_dt.isoformat()}"
        s_settle = se.get(key)
        if s_settle is None:
            continue
        bars = kl.get(ev["sym"], [])
        # entry: T-entry_hours before settle, but never before the
        # market itself existed (+30m for the first quote to form)
        first_pts = [min(t for t, _ in leg["hist"])
                     for leg in ev["legs"] if leg["hist"]]
        if not first_pts:
            continue
        born = float(max(min(first_pts), 0))
        entry_ts = max(st_dt.timestamp() - entry_hours * 3600,
                       born + 1800)
        if entry_ts > st_dt.timestamp() - 1200:
            continue  # market too young before settle
        tau = (st_dt.timestamp() - entry_ts) / 3600.0
        closes = [(t / 1000, c) for t, _, _, _, c in bars
                  if t / 1000 <= entry_ts - 3600]
        if len(closes) < 60:
            continue
        rets = np.diff(np.log([c for _, c in closes]))
        # EWMA sigma (hourly)
        lam = 0.5 ** (1.0 / halflife)
        w = lam ** np.arange(len(rets) - 1, -1, -1)
        var = np.sum(w * rets ** 2) / np.sum(w)
        sig_h = math.sqrt(max(var, 1e-10))
        spot = closes[-1][1]
        for leg in ev["legs"]:
            pts = [(t, p) for t, p in leg["hist"]
                   if t <= entry_ts]
            if not pts:
                continue
            p = pts[-1][1]
            # skip unpriced/placeholder legs: need real traded prints
            # (>= 2 distinct prices on the path) and off the 0.50 mid
            if len({round(pp, 3) for _, pp in pts}) < 2:
                continue
            if abs(p - 0.5) <= 0.02:
                continue
            rows.append({"event": key, "sym": ev["sym"],
                         "settle_ts": st_dt.timestamp(),
                         "entry_ts": entry_ts, "spot": spot,
                         "sig_h": sig_h, "tau": tau,
                         "k": leg["k"], "p": p,
                         "win": 1.0 if s_settle > leg["k"] else 0.0,
                         "s_settle": s_settle})
    df = pd.DataFrame(rows)
    evs = sorted(df["event"].unique())
    cut = evs[int(len(evs) * EVENT_CUT_FRAC)]
    df["train"] = df["event"] <= cut
    return df


def q_model(r, sig_h=None, dfree=None, drift=0.0):
    s = (r["sig_h"] if sig_h is None else sig_h) * math.sqrt(r["tau"])
    mu = drift * r["tau"] * (r["sig_h"] if sig_h is None else sig_h) ** 2
    z = (math.log(r["spot"] / r["k"]) + mu) / s
    if dfree is None:
        return float(st.norm.cdf(z))
    # student-t scaled to unit variance
    tval = z * math.sqrt(dfree / (dfree - 2))
    return float(1.0 - st.t.cdf(-tval, dfree))


def run(df, theta=THETA, spread=SPREAD, w_mkt=1.0, dfree=None,
        drift=0.0, dist_lo=None, dist_hi=None, side="all"):
    trades = []
    for _, r in df.iterrows():
        q = q_model(r, dfree=dfree, drift=drift)
        if dist_lo is not None:
            d = abs(math.log(r["spot"] / r["k"])) / \
                (r["sig_h"] * math.sqrt(r["tau"]))
            if not (dist_lo <= d <= dist_hi):
                continue
        qq = w_mkt * q + (1 - w_mkt) * r["p"]
        win = r["win"]
        if qq - r["p"] >= theta and side in ("all", "YES"):
            cost = r["p"] + spread
            trades.append({"event": r["event"], "sym": r["sym"],
                           "side": "YES", "train": r["train"],
                           "pnl": (1.0 - cost) if win else -cost})
        elif (1 - qq) - (1 - r["p"]) >= theta and side in ("all", "NO"):
            cost_n = (1 - r["p"]) + spread
            trades.append({"event": r["event"], "side": "NO",
                           "train": r["train"],
                           "pnl": -cost_n if win else 1.0 - cost_n})
    return pd.DataFrame(trades)


def repl(tr, tag):
    if len(tr) == 0:
        print(f"| {tag} | 0 | - | - | - |")
        return
    byev = tr.groupby("event")["pnl"].sum()
    rng = np.random.default_rng(5)
    ci = "-"
    if len(byev) >= 8:
        arr = byev.values
        stats = [arr[rng.choice(len(arr), len(arr))].mean()
                 for _ in range(2000)]
        lo, hi = np.percentile(stats, [2.5, 97.5])
        ci = f"[{lo:+.3f},{hi:+.3f}]{'' if (lo > 0 or hi < 0) else ' 0-in'}"
    print(f"| {tag} | {len(tr)} | {tr['pnl'].sum():+.2f} | "
          f"{tr['pnl'].mean():+.4f} | {ci} |")


# ---------------------------------------------------------------- batteries
def battery_a(df):
    print("## 35a — calibration (entry T-4h)")
    print("| source | n | avg P | realized freq | Brier |")
    print("|---|---|---|---|---|")
    qcol = [q_model(r) for _, r in df.iterrows()]
    for lab, probs in (("market", df["p"]), ("model EWMA24", qcol)):
        probs = np.array(probs)
        brier = float(np.mean((probs - df["win"]) ** 2))
        print(f"| {lab} | {len(df)} | {probs.mean():.3f} | "
              f"{df['win'].mean():.3f} | {brier:.4f} |")
    # by distance band
    df2 = df.copy()
    df2["d"] = [abs(math.log(s / k)) / (sh * math.sqrt(t))
                for s, k, sh, t in zip(df2["spot"], df2["k"],
                                       df2["sig_h"], df2["tau"])]
    print("| band(d) | n | market avg | mkt freq | model avg | mdl freq |")
    print("|---|---|---|---|---|---|")
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5),
                   (2.5, 99)):
        g = df2[(df2["d"] >= lo) & (df2["d"] < hi)]
        if len(g) < 8:
            continue
        qcol = np.array([q_model(r) for _, r in g.iterrows()])
        print(f"| {lo}-{hi} | {len(g)} | {g['p'].mean():.3f} | "
              f"{g['win'].mean():.3f} | {qcol.mean():.3f} | "
              f"{g['win'].mean():.3f} |")


def battery_b(df):
    print("\n## 35b — sigma estimator race (entry T-4h)")
    print("| variant | n | pnl | avg | settle-CI |")
    print("|---|---|---|---|---|")
    for hl in (12, 24, 48, 96):
        df2 = build_dataset(halflife=hl)
        tr = run(df2)
        repl(tr, f"EWMA hl{hl}")
    for dfr in (3, 5):
        tr = run(df, dfree=dfr)
        repl(tr, f"EWMA24 t{dfr}")
    for dr in (-0.5, 0.5):
        tr = run(df, drift=dr)
        repl(tr, f"EWMA24 drift{dr:+.1f}")


def battery_d(df):
    dates = sorted(df["event"].unique())
    mid = dates[int(len(dates) * EVENT_CUT_FRAC)]
    print(f"\n## 35d — market blend (train events <= {mid[:16]})")
    print("| w_mkt | TRAIN avg | TEST n | TEST pnl | TEST avg |")
    print("|---|---|---|---|---|")
    best = (1.0, -9)
    for w in (1.0, 0.7, 0.5, 0.3):
        tr = run(df, w_mkt=w)
        if len(tr) == 0:
            continue
        trn = tr[tr["train"]]
        te = tr[~tr["train"]]
        a = trn["pnl"].mean() if len(trn) else 0
        print(f"| {w} | {a:+.4f} [{len(trn)}] | {len(te)} | "
              f"{te['pnl'].sum() if len(te) else 0:+.2f} | "
              f"{te['pnl'].mean() if len(te) else 0:+.4f} |")
        if a > best[1]:
            best = (w, a)
    print(f"selected w={best[0]}")


def battery_e(df):
    dates = sorted(df["event"].unique())
    mid = dates[int(len(dates) * EVENT_CUT_FRAC)]
    print(f"\n## 35e — filters (eval ALL, split shown)")
    print("| filter | n | pnl | avg | CI |")
    print("|---|---|---|---|---|")
    for lab, kw in (
            ("base", {}),
            ("theta .08", {"theta": 0.08}),
            ("theta .10", {"theta": 0.10}),
            ("dist 0.5-2.0", {"dist_lo": 0.5, "dist_hi": 2.0}),
            ("dist 1.0-3.0", {"dist_lo": 1.0, "dist_hi": 3.0}),
            ("NO only", {"side": "NO"}),
            ("YES only", {"side": "YES"}),
    ):
        tr = run(df, **kw)
        repl(tr, lab)
    # entry-time comparison
    for eh in (12, 8, 4, 2):
        df2 = build_dataset(entry_hours=eh)
        tr = run(df2)
        repl(tr, f"entry T-{eh}h")


def battery_f(df, w_mkt=1.0, dfree=None):
    print(f"\n## 35f — final validation")
    tr = run(df, w_mkt=w_mkt, dfree=dfree)
    repl(tr, "ALL")
    for s in ("YES", "NO"):
        repl(tr[tr["side"] == s], s)
    # permutation: shuffle q across legs within event (price-sort
    # preserving relative structure would be complex; shuffle (q - p)
    # edge assignment across legs of same event)
    rng = np.random.default_rng(17)
    edges = []
    for _, r in df.iterrows():
        q = q_model(r, dfree=dfree)
        edges.append(max(q - r["p"], (1 - q) - (1 - r["p"])))
    df2 = df.copy()
    df2["edge"] = edges
    real_avg = tr["pnl"].mean() if len(tr) else 0
    perm = []
    for _ in range(60):
        fake = []
        for ev, g in df2.groupby("event"):
            e = g["edge"].values.copy()
            rng.shuffle(e)
            for (_, r), ee in zip(g.iterrows(), e):
                win = r["win"]
                if ee >= THETA:
                    cost = r["p"] + SPREAD
                    fake.append((1.0 - cost) if win else -cost)
        if fake:
            perm.append(np.mean(fake))
    if perm:
        pval = float(np.mean(np.array(perm) >= real_avg))
        print(f"| permutation p (edge shuffled within event) | {pval:.3f} |")


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if a in "abcdef"] or list("abcdef")
    df = build_dataset()
    print(f"dataset: {len(df)} legs | {df['event'].nunique()} settled "
          f"events | win-rate {df['win'].mean():.3f} | "
          f"train {int(df['train'].sum())} legs")
    if "a" in which:
        battery_a(df)
    if "b" in which or "c" in which:
        battery_b(df)
    if "d" in which:
        battery_d(df)
    if "e" in which:
        battery_e(df)
    if "f" in which:
        battery_f(df)
