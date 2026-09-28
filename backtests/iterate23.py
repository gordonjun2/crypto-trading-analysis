"""Round 23: liquidity-tier split of the fade universe (24m).

Trailing 30d median DAILY dollar volume -> per-day rank. Compare fade v6
on: all (production), top-100, rank 100-200, top-150, 150-250.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate23.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import (idx, closes, bpd, in_u_all, pc_np, fund_np, SLOTS,
                       SLOT_FRAC, BTC, fee_rt, atr1_np, vol_raw, vol_ma, report)
from iterate14 import low_np, high_np
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

CLOSE_NP = {s: TR.panel.frames[s]["Close"].values for s in TR.panel.pairs}
VOL_NP = {s: TR.panel.frames[s]["Volume"].values for s in TR.panel.pairs}


def liq_tier(lo=None, hi=None):
    """Per-day rank by trailing-30d median daily $volume; hourly mask."""
    dv = pd.DataFrame({s: pd.Series(CLOSE_NP[s] * VOL_NP[s], index=idx)
                       for s in TR.panel.pairs})
    med = dv.resample("1D").median().rolling(30, min_periods=15).median()
    rk = med.rank(axis=1, ascending=False)
    uniq_days, inv = np.unique(idx.normalize().values, return_inverse=True)
    rk_mat = rk.reindex(uniq_days).values
    cols = {s: j for j, s in enumerate(rk.columns)}
    tier = {}
    for s in TR.panel.pairs:
        if s not in cols:
            continue
        r = rk_mat[inv, cols[s]]
        ok = np.isfinite(r)
        if lo is not None:
            ok &= r >= lo
        if hi is not None:
            ok &= r <= hi
        tier[s] = pd.Series(ok, index=idx)
    return tier


def fade_tier(tier, k=5.0, cap_h=42):
    slots = SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades, recs = [], [], []
    last_close_i = {}

    def accrue_to(i):
        nonlocal last_acc
        if i > last_acc:
            for t in open_tr:
                s = t["sym"]
                sgn = 1.0 if t["dir"] == "LONG" else -1.0
                lo = max(last_acc + 1, t["i0"] + 1)
                if lo <= i:
                    net_arr[lo:i + 1] += (t["w"] * sgn *
                                          (pc_np[s][lo:i + 1]
                                           - fund_np[s][lo:i + 1]))
            last_acc = i

    last_acc = 30 * bpd - 1

    def weight(symv, i):
        v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            return float(np.clip((mv / v) * SLOT_FRAC,
                                 0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
        return SLOT_FRAC

    def realize(t, i, frac, px):
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        net_arr[i] += frac * t["w0"] * sgn * (px / t["px0"] - 1.0)
        net_arr[i] -= frac * t["w0"] * fee_rt
        t["w"] -= frac * t["w0"]

    def close_trade(t, i, reason, px=None):
        c_i = float(closes[t["sym"]].iloc[i]) if px is None else px
        t.update(j0=i, exit_px=c_i, reason=reason)
        trades.append(t)
        last_close_i[t["sym"]] = i
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        recs.append({"sym": t["sym"], "dir": t["dir"], "i0": t["i0"], "j0": i,
                     "ret": sgn * (c_i / t["px0"] - 1.0)})

    V5_TR = [(0.5, 0.5), (0.75, 0.25)]
    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            expired = (i - t["i0"]) >= cap_h
            o, px0 = t["origin"], t["px0"]
            span = px0 - o if t["dir"] == "SHORT" else o - px0
            lo_i, hi_i = low_np[s][i], high_np[s][i]
            if t["w"] > 1e-12:
                for ti, (rf, sf) in enumerate(V5_TR):
                    if ti in t["done"]:
                        continue
                    lvl = px0 - rf * span if t["dir"] == "SHORT" \
                        else px0 + rf * span
                    if (lo_i <= lvl) if t["dir"] == "SHORT" \
                            else (hi_i >= lvl):
                        realize(t, i, sf, lvl)
                        t["done"].add(ti)
            o_hit = (lo_i <= o) if t["dir"] == "SHORT" else (hi_i >= o)
            if expired:
                close_trade(t, i, "time")
            elif o_hit:
                close_trade(t, i, "origin", px=o)
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        cands = []
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv not in pc_np:
                continue
            if symv in last_close_i and i - last_close_i[symv] < 12:
                continue
            if symv in tier and not bool(tier[symv].iloc[i]):
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a or r < 0:
                continue
            v0, vma = vol_raw[symv][i], vol_ma[symv][i]
            if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                    and v0 >= 3.0 * vma):
                continue
            cands.append((symv, a))
        if cands:
            room = slots - len(open_tr)
            for symv, a in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                w = weight(symv, i)
                net_arr[i] -= w * 2 * fee_rt
                open_tr.append({"sym": symv, "dir": "SHORT", "i0": i,
                                "px0": float(closes[symv].iloc[i]),
                                "origin": origin, "atr0": a, "w": w,
                                "w0": w, "done": set()})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, recs


def battery():
    print("## Round 23 — liquidity tiers (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, _, rc = fade_tier({})
    report("all (production, no tier filter)", net, len(rc))
    for lo, hi, lab in ((1, 100, "top-100"), (101, 200, "rank 101-200"),
                        (1, 150, "top-150"), (151, 250, "rank 151-250"),
                        (1, 50, "top-50")):
        tier = liq_tier(lo, hi)
        net, _, rc = fade_tier(tier)
        report(lab, net, len(rc))


if __name__ == "__main__":
    battery()
