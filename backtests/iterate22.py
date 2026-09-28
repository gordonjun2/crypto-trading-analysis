"""Round 22: v6-base re-validation + exit geometry round 3 (24m).

22a fine k x vol grid on the v6 base (old plateau was measured on v3/v4).
22b trailing last tranche: after the retrace plan is exhausted, exit the
    remainder when close falls back through the last touched level
    (shorts) instead of riding to origin/cap.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate22.py
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

V5_TR = [(0.5, 0.5), (0.75, 0.25)]


def fade_sim5(k=5.0, vol_min=3.0, cap_h=42, trail_last=False):
    """FADE v6 (touch fills, cd 12h) + optional trailing-last-tranche:
    once both tranches are realized, exit remainder on close crossing the
    75% level adversely (shorts) — vs production ride-to-origin."""
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
            if trail_last and len(t["done"]) == len(V5_TR) \
                    and t["w"] > 1e-12 and not o_hit:
                lvl75 = px0 - 0.75 * span if t["dir"] == "SHORT" \
                    else px0 + 0.75 * span
                if (c_i < lvl75) if t["dir"] == "SHORT" else (c_i > lvl75):
                    close_trade(t, i, "trail")
                    continue
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
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a:
                continue
            if r < 0:
                continue
            v0, vma = vol_raw[symv][i], vol_ma[symv][i]
            if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                    and v0 >= vol_min * vma):
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
    print("## Round 22 — v6-base fine grid + trailing last tranche (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, _, rc = fade_sim5()
    report("PRODUCTION v6 base (k5, vol3)", net, len(rc))
    for kk in (4.5, 5.5):
        net, _, rc = fade_sim5(k=kk)
        report(f"fine grid k={kk}xATR", net, len(rc))
    for vv in (2.75, 3.25):
        net, _, rc = fade_sim5(vol_min=vv)
        report(f"fine grid vol>={vv}x", net, len(rc))
    net, _, rc = fade_sim5(trail_last=True)
    report("22b trail last tranche at 75% level", net, len(rc))


if __name__ == "__main__":
    battery()
