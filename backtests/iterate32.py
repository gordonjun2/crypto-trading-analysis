"""Round 32: SENTIMENT — Fear & Greed + prediction markets (Kalshi/Polymarket).

Crypto is sentiment-driven; user directive: explore sentimental signals.
VERDICTS (see technique_research round 32): F&G level = no forward signal;
trade-level stress gradients (fade better in fear, both F&G and funding)
are vol compensation — gates REJECTED at book level; CLIM fear-reversal
hypothesis recorded (30 trades, noise); SQUEEZE regime masks rejected.
Polymarket/Kalshi histories pruned to ~30d -> forward-only archive
(fetch_prediction.py collector, cron :35).

32a  F&G event study: level regimes + shock days -> forward drift
32b  F&G trade-level conditioning (FADE/CLIM/SQUEEZE buckets)
32c  F&G gate tests (rejected; controls confirm bucket trap)
32d  Polymarket implied probs (insufficient history — forward only)
32e  Aggregate/BTC funding sentiment (stress gradient, gates rejected)
Data: saved_data_live/sentiment/feargreed.json (alternative.me, full history)
Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate32.py [a..e]
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import technique_research as TR
import family_scan_honest as FS
from iterate10 import (idx, closes, bpd, pc_np, fund_np, BTC, report)
from iterate14 import squeeze2, SQ_OVER
from iterate25 import load_flow
from iterate28 import cont_sim
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

FG_PATH = "saved_data_live/sentiment/feargreed.json"


def load_fg_strict():
    """Daily F&G Series (index = day the value describes). Causality: the
    point stamped day D is published at D 00:00 UTC; hourly bars on day D
    use the point stamped D-1 (known a full day). Returns hourly series."""
    d = json.load(open(FG_PATH))["data"]
    s = pd.Series({pd.Timestamp(datetime.fromtimestamp(
        int(x["timestamp"]), timezone.utc)).tz_localize(None).normalize():
        float(x["value"]) for x in d})
    s = s.sort_index()
    return s.shift(1).reindex(pd.Series(idx).dt.normalize().values).values


BUCKETS = ((0, 25, "extfear"), (25, 45, "fear"), (45, 56, "neutral"),
           (56, 76, "greed"), (76, 101, "extgreed"))


def bucket_of(v):
    for lo, hi, lab in BUCKETS:
        if lo <= v < hi:
            return lab
    return None


# ---------------------------------------------------------------- battery a
def battery_a():
    fg = load_fg_strict()
    days = pd.Series(fg, index=idx).resample("1D").last().dropna()
    btc_c = closes[BTC]
    # alt basket: median pct-change across liquid panel symbols per bar
    basket = pd.concat([pd.Series(pc_np[s], index=idx)
                        for s in list(TR.panel.pairs)[:400]
                        if s in pc_np], axis=1).median(axis=1)
    basket_d = basket.resample("1D").sum()

    print("## Round 32a — F&G level regimes: forward drift (24m, strict lag)")
    print("| regime | days | BTC +1d | BTC +3d | BTC +7d | basket +3d | basket +7d |")
    print("|---|---|---|---|---|---|---|")
    btc_cd = btc_c
    for lo, hi, lab in BUCKETS:
        m = (days >= lo) & (days < hi)
        if m.sum() < 5:
            continue
        ds = days.index[m]
        f1, f3, f7, b3, b7 = [], [], [], [], []
        for d0 in ds:
            i0 = idx.searchsorted(d0 + pd.Timedelta(hours=24))
            if i0 + 7 * 24 >= len(idx):
                continue
            p0 = float(btc_cd.iloc[i0 - 1])
            f1.append(float(btc_cd.iloc[i0 + 23]) / p0 - 1)
            f3.append(float(btc_cd.iloc[i0 + 71]) / p0 - 1)
            f7.append(float(btc_cd.iloc[i0 + 167]) / p0 - 1)
            j0 = basket_d.index.searchsorted(d0)
            if j0 + 7 < len(basket_d):
                b3.append(float(basket_d.iloc[j0 + 1:j0 + 4].sum()))
                b7.append(float(basket_d.iloc[j0 + 1:j0 + 8].sum()))
        f = lambda a: f"{np.mean(a):+.2%}" if len(a) else "-"
        print(f"| {lab} | {int(m.sum())} | {f(f1)} | {f(f3)} | {f(f7)} | "
              f"{f(b3)} | {f(b7)} |", flush=True)

    print()
    print("## Round 32a — F&G shocks |dfg| >= 15: forward drift")
    print("| shock | days | BTC +1d | BTC +3d | BTC +7d | basket +7d |")
    print("|---|---|---|---|---|---|")
    dchg = days.diff()
    for lab, m in (("down >=15", dchg <= -15), ("up >=15", dchg >= 15)):
        ds = days.index[m]
        f1, f3, f7, b7 = [], [], [], []
        for d0 in ds:
            i0 = idx.searchsorted(d0 + pd.Timedelta(hours=24))
            if i0 + 7 * 24 >= len(idx):
                continue
            p0 = float(btc_cd.iloc[i0 - 1])
            f1.append(float(btc_cd.iloc[i0 + 23]) / p0 - 1)
            f3.append(float(btc_cd.iloc[i0 + 71]) / p0 - 1)
            f7.append(float(btc_cd.iloc[i0 + 167]) / p0 - 1)
            j0 = basket_d.index.searchsorted(d0)
            if j0 + 7 < len(basket_d):
                b7.append(float(basket_d.iloc[j0 + 1:j0 + 8].sum()))
        f = lambda a: f"{np.mean(a):+.2%} [{len(a)}]" if len(a) else "-"
        print(f"| {lab} | {len(ds)} | {f(f1)} | {f(f3)} | {f(f7)} | "
              f"{f(b7)} |", flush=True)


# ---------------------------------------------------------------- battery b
def battery_b():
    fg = load_fg_strict()
    print("## Round 32b — production trades bucketed by F&G regime at entry")
    flow = load_flow()
    print(f"flow pairs: {len(flow)}")

    def day_lab(i):
        v = fg[i]
        return bucket_of(v) if np.isfinite(v) else None

    def bucket_table(name, trades, ret_of):
        rows = {lab: [] for _, _, lab in BUCKETS}
        for t in trades:
            lab = day_lab(t["i0"])
            if lab:
                rows[lab].append(ret_of(t))
        print(f"\n### {name}")
        print("| regime | trades | avg ret | win rate |")
        print("|---|---|---|---|")
        for _, _, lab in BUCKETS:
            rr = np.array(rows[lab])
            if len(rr):
                print(f"| {lab} | {len(rr)} | {rr.mean():+.2%} | "
                      f"{(rr > 0).mean():.0%} |", flush=True)

    # FADE v7 (1x sizing = clean trade attribution)
    _, ftr, frc = __import__("iterate25").fade_sim7(tf=flow, tf_max=0.60)
    bucket_table("FADE v7 shorts (ret = short pnl)", frc,
                 lambda r: r["ret"])
    # CLIM
    _, ctr = cont_sim(flow, tf_min=0.60, exit_mode="atr_trail", trail_atr=1.0)
    bucket_table("CLIM v1 longs", ctr,
                 lambda t: t["exit_px"] / t["px0"] - 1.0)
    # SQUEEZE
    sq = FS.sim(squeeze2(), False, None, SQ_OVER)
    bucket_table("SQUEEZE v2 (ret = signed move)", sq["trades"],
                 lambda t: (t["exit_px"] / t["px0"] - 1.0)
                 if t["dir"] == "LONG" else
                 (t["px0"] / max(t["exit_px"], 1e-12) - 1.0))


# ---------------------------------------------------------------- battery c
def battery_c():
    """F&G-gated production books at 1x (clean attribution)."""
    from iterate31 import fade_sim9, cont_sim8
    fg = load_fg_strict()
    flow = load_flow()
    print("## Round 32c — F&G-gated books @1x (24m)")
    print("| variant | SR(bar) | daily-block CI | ret | maxDD | trades |")
    print("|---|---|---|---|---|---|")

    def rep(label, net, n):
        lo, hi, pn = bootstrap_sharpe_ci(net)
        days = len(net) / 24
        print(f"| {label} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
              f"{max_dd_of(net):.1%} | {n} |", flush=True)

    base, btr, brc = fade_sim9(tf=flow, cluster_gate=3, lev=1.0)
    rep("FADE base (stop15+cluster3)", base, len(brc))
    net, tr, rc = fade_sim9(tf=flow, cluster_gate=3, lev=1.0,
                            fg=fg, fg_hi=55)
    rep("FADE fg<=55 (skip greed)", net, len(rc))
    net, tr, rc = fade_sim9(tf=flow, cluster_gate=3, lev=1.0,
                            fg=fg, fg_hi=45)
    rep("FADE fg<=45 (fear only)", net, len(rc))
    net, tr, rc = fade_sim9(tf=flow, cluster_gate=3, lev=1.0,
                            fg=fg, fg_lo=56)
    rep("FADE fg>=56 (greed only, CONTROL)", net, len(rc))
    net, tr, rc = fade_sim9(tf=flow, cluster_gate=3, lev=1.0,
                            fg=fg, fg_lo=46, fg_hi=55)
    rep("FADE 46-55 (neutral only, CONTROL)", net, len(rc))

    cbase, ctr, _ = cont_sim8(tf=flow, stop_pct=0.06, lev=1.0)
    rep("CLIM base (trail+floor6)", cbase, len(ctr))
    net, tr, _ = cont_sim8(tf=flow, stop_pct=0.06, lev=1.0, fg=fg, fg_hi=45)
    rep("CLIM fg<=45 (fear only)", net, len(tr))
    net, tr, _ = cont_sim8(tf=flow, stop_pct=0.06, lev=1.0, fg=fg, fg_hi=55)
    rep("CLIM fg<=55", net, len(tr))

    sig = squeeze2()
    mask = np.isfinite(fg) & (fg >= 25) & (fg <= 75)
    sig_g = {s: (la & mask, le, sa & mask, se)
             for s, (la, le, sa, se) in sig.items()}
    r = FS.sim(sig, False, None, SQ_OVER)
    rep("SQUEEZE base", r["net"], r["n"])
    r = FS.sim(sig_g, False, None, SQ_OVER)
    rep("SQUEEZE fg 25-75 (no extremes)", r["net"], r["n"])
    mask = np.isfinite(fg) & (fg >= 25) & (fg <= 55)
    sig_g = {s: (la & mask, le, sa & mask, se)
             for s, (la, le, sa, se) in sig.items()}
    r = FS.sim(sig_g, False, None, SQ_OVER)
    rep("SQUEEZE fg 25-55 (fear side)", r["net"], r["n"])


# ---------------------------------------------------------------- battery d
BF_DIR = "saved_data_live/prediction/polymarket_backfill"


def _path_of(q_exact):
    from pathlib import Path
    for f in Path(BF_DIR).glob("*.jsonl"):
        meta = json.loads(f.open().readline())
        if (meta.get("question") or "").strip() == q_exact:
            pts = [json.loads(x) for x in f.open().readlines()[1:]]
            s = pd.Series({pd.Timestamp(p["t"], unit="s"): p["p"]
                           for p in pts}).sort_index()
            return s
    return None


def load_pm_daily():
    """Daily series (strict lag: value known at next 00:00)."""
    from datetime import datetime, timezone as tz
    out = {}
    # market-implied bull gauge: P(BTC >= 150k by end-2026)
    p = _path_of("Will Bitcoin hit $150k by December 31, 2026?")
    if p is not None:
        out["p150"] = p.resample("1D").last()
    # E[cuts in 2026] from the count ladder
    ec = None
    for n in range(0, 13):
        p = _path_of(f"Will {n} Fed rate cut happen in 2026?" if n == 1
                     else f"Will {n} Fed rate cuts happen in 2026?")
        if p is None:
            continue
        ec = n * p if ec is None else ec.add(n * p, fill_value=0)
    if ec is not None:
        out["ecuts"] = ec.resample("1D").last()
    daily = {}
    for k, s in out.items():
        s.index = s.index.tz_localize(None).normalize()
        daily[k] = s.shift(1).reindex(
            pd.Series(idx).dt.normalize().unique()).values
    return daily


def battery_d():
    pm = load_pm_daily()
    for k, v in pm.items():
        fin = v[np.isfinite(v)]
        print(f"pm[{k}]: {len(fin)} days | mean {fin.mean():.3f} "
              f"min {fin.min():.3f} max {fin.max():.3f}")
    if "ecuts" in pm:
        ec = pm["ecuts"]
        d_ec = pd.Series(ec, index=idx).resample("1D").last().dropna()
        dchg = d_ec.diff()
        btc_c = closes[BTC]
        print("## Round 32d — Fed expectation shocks |dE[cuts]| >= 0.25")
        print("| shock | n | BTC +1d | BTC +3d | BTC +7d |")
        print("|---|---|---|---|---|")
        for lab, m in (("cuts repriced DOWN", dchg <= -0.25),
                       ("cuts repriced UP", dchg >= 0.25)):
            f1, f3, f7 = [], [], []
            for d0 in d_ec.index[m]:
                i0 = idx.searchsorted(d0 + pd.Timedelta(hours=24))
                if i0 + 7 * 24 >= len(idx):
                    continue
                p0 = float(btc_c.iloc[i0 - 1])
                f1.append(float(btc_c.iloc[i0 + 23]) / p0 - 1)
                f3.append(float(btc_c.iloc[i0 + 71]) / p0 - 1)
                f7.append(float(btc_c.iloc[i0 + 167]) / p0 - 1)
            f = lambda a: f"{np.mean(a):+.2%} [{len(a)}]" if len(a) else "-"
            print(f"| {lab} | {int(m.sum())} | {f(f1)} | {f(f3)} | "
                  f"{f(f7)} |", flush=True)
    if "p150" in pm:
        p150 = pm["p150"]
        qs = np.nanquantile(p150, (0.25, 0.5, 0.75))
        labs = ("p150 q1 (low)", "p150 q2", "p150 q3", "p150 q4 (euphoric)")
        for name, trades, rof in (
                ("FADE", __import__("iterate25").fade_sim7(
                    tf=load_flow(), tf_max=0.60)[1], None),):
            pass
        flow = load_flow()
        _, ftr, frc = __import__("iterate25").fade_sim7(tf=flow, tf_max=0.60)
        _, ctr = cont_sim(flow, tf_min=0.60, exit_mode="atr_trail",
                          trail_atr=1.0)
        sq = FS.sim(squeeze2(), False, None, SQ_OVER)
        print("## Round 32d — trades bucketed by market-implied euphoria"
              " (P(BTC 150k by 2026) quartile at entry)")
        for name, trades, rof in (
                ("FADE v7 shorts", frc, lambda r: r["ret"]),
                ("CLIM v1 longs", ctr,
                 lambda t: t["exit_px"] / t["px0"] - 1.0),
                ("SQUEEZE v2", sq["trades"],
                 lambda t: (t["exit_px"] / t["px0"] - 1.0)
                 if t["dir"] == "LONG" else
                 (t["px0"] / max(t["exit_px"], 1e-12) - 1.0))):
            rows = {i: [] for i in range(4)}
            for t in trades:
                v = p150[t["i0"]]
                if not np.isfinite(v):
                    continue
                q = 0 if v < qs[0] else 1 if v < qs[1] else \
                    2 if v < qs[2] else 3
                rows[q].append(rof(t))
            print(f"\n### {name}")
            print("| euphoria | trades | avg ret | win rate |")
            print("|---|---|---|---|")
            for i in range(4):
                rr = np.array(rows[i])
                if len(rr):
                    print(f"| {labs[i]} | {len(rr)} | {rr.mean():+.2%} | "
                          f"{(rr > 0).mean():.0%} |", flush=True)


# ---------------------------------------------------------------- battery e
def battery_e():
    """Aggregate funding = market-wide leverage sentiment (backtestable now).
    Cross-sectional median 8h funding across the panel, daily summed."""
    from iterate10 import fund_np
    fund_mat = pd.DataFrame({s: fund_np[s] for s in list(TR.panel.pairs)[:500]
                             if s in fund_np})
    agg8 = fund_mat.mean(axis=1)
    agg8.index = idx                        # per-bar mean funding (8h rate)
    agg_d = agg8.resample("1D").sum()       # daily funding paid (approx 3x)
    btc_fund_d = (pd.Series(fund_np[BTC], index=idx)
                  .resample("1D").sum())
    use = btc_fund_d if btc_fund_d.abs().quantile(0.9) > \
        agg_d.abs().quantile(0.9) else agg_d
    if use is btc_fund_d:
        print("(using BTC-only funding — better spread)")
    btc_c = closes[BTC]
    basket = pd.concat([pd.Series(pc_np[s], index=idx)
                        for s in list(TR.panel.pairs)[:400]
                        if s in pc_np], axis=1).median(axis=1)
    basket_d = basket.resample("1D").sum()
    print("## Round 32e — aggregate funding sentiment (24m)")
    print(f"median daily funding: {use.median():+.4%} | "
          f"p90 {use.quantile(0.9):+.4%} | p10 {use.quantile(0.1):+.4%}")
    print("| regime (daily funding) | days | BTC +3d | BTC +7d | basket +7d |")
    print("|---|---|---|---|---|")
    qs = use.quantile((0.2, 0.4, 0.6, 0.8))
    labs = ("very neg (capitulation)", "neg", "mid", "pos",
            "very pos (euphoric longs)")
    edges = [-1e9, qs[0.2], qs[0.4], qs[0.6], qs[0.8], 1e9]
    for i, lab in enumerate(labs):
        m = (use >= edges[i]) & (use < edges[i + 1])
        f3, f7, b7 = [], [], []
        for d0 in use.index[m]:
            i0 = idx.searchsorted(d0 + pd.Timedelta(hours=24))
            if i0 + 7 * 24 >= len(idx):
                continue
            p0 = float(btc_c.iloc[i0 - 1])
            f3.append(float(btc_c.iloc[i0 + 71]) / p0 - 1)
            f7.append(float(btc_c.iloc[i0 + 167]) / p0 - 1)
            j0 = basket_d.index.searchsorted(d0)
            if j0 + 7 < len(basket_d):
                b7.append(float(basket_d.iloc[j0 + 1:j0 + 8].sum()))
        f = lambda a: f"{np.mean(a):+.2%}" if len(a) else "-"
        print(f"| {lab} | {int(m.sum())} | {f(f3)} | {f(f7)} | {f(b7)} |",
              flush=True)
    # conditioning on production trades
    agg_h = use.shift(1).reindex(pd.Series(idx).dt.normalize().values).values
    flow = load_flow()
    _, ftr, frc = __import__("iterate25").fade_sim7(tf=flow, tf_max=0.60)
    _, ctr = cont_sim(flow, tf_min=0.60, exit_mode="atr_trail", trail_atr=1.0)
    sq = FS.sim(squeeze2(), False, None, SQ_OVER)
    print("## Round 32e — trades bucketed by funding quintile at entry")
    for name, trades, rof in (
            ("FADE v7 shorts", frc, lambda r: r["ret"]),
            ("CLIM v1 longs", ctr,
             lambda t: t["exit_px"] / t["px0"] - 1.0),
            ("SQUEEZE v2", sq["trades"],
             lambda t: (t["exit_px"] / t["px0"] - 1.0)
             if t["dir"] == "LONG" else
             (t["px0"] / max(t["exit_px"], 1e-12) - 1.0))):
        rows = {i: [] for i in range(5)}
        for t in trades:
            v = agg_h[t["i0"]]
            if not np.isfinite(v):
                continue
            q = 0 if v < qs[0.2] else 1 if v < qs[0.4] else \
                2 if v < qs[0.6] else 3 if v < qs[0.8] else 4
            rows[q].append(rof(t))
        print(f"\n### {name}")
        print("| funding quintile | trades | avg ret | win rate |")
        print("|---|---|---|---|")
        for i in range(5):
            rr = np.array(rows[i])
            if len(rr):
                print(f"| Q{i + 1} ({labs[i]}) | {len(rr)} | "
                      f"{rr.mean():+.2%} | {(rr > 0).mean():.0%} |",
                      flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "a"
    if mode == "a":
        battery_a()
    elif mode == "b":
        battery_b()
    elif mode == "c":
        battery_c()
    elif mode == "d":
        battery_d()
    elif mode == "e":
        battery_e()
