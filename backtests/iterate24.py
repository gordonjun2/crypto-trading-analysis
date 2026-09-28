"""Round 24: fade event-conditioning (24m).

24a second-bar exhaustion: require the PREVIOUS hourly bar to also be a
    >= k*ATR move in the same direction (pump already 2 bars old).
24b weekend split: entries on Sat/Sun only vs Mon-Fri only.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate24.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import (idx, closes, bpd, in_u_all, pc_np, fund_np, SLOTS,
                       SLOT_FRAC, BTC, fee_rt, atr1_np, vol_raw, vol_ma, report)
from iterate22 import fade_sim5

PC2 = {s: TR.panel.frames[s]["Close"].pct_change() for s in TR.panel.pairs}


def fade_sim6(k=5.0, cap_h=42, two_bar=False, weekend=None):
    """fade_sim5 + two-bar exhaustion / weekday mask."""
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
                    pc = PC2[s].values
                    net_arr[lo:i + 1] += (t["w"] * sgn *
                                          (pc[lo:i + 1] - fund_np[s][lo:i + 1]))
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
        pc = PC2[t["sym"]].values
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        recs.append({"sym": t["sym"], "dir": t["dir"], "i0": t["i0"], "j0": i,
                     "ret": sgn * (c_i / t["px0"] - 1.0)})

    V5_TR = [(0.5, 0.5), (0.75, 0.25)]
    low_np = {s: TR.panel.frames[s]["Low"].values for s in TR.panel.pairs}
    high_np = {s: TR.panel.frames[s]["High"].values for s in TR.panel.pairs}
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
        wd = idx[i].dayofweek
        if weekend is True and wd < 5:
            continue
        if weekend is False and wd >= 5:
            continue
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
            if abs(r) <= k * a or r < 0:
                continue
            if two_bar:
                r1 = PC2[symv].iloc[i - 1]
                if not (np.isfinite(r1) and r1 > k * atr1_np[symv][i - 1]):
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
    print("## Round 24 — fade event conditioning (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, _, rc = fade_sim6()
    report("PRODUCTION v6 reference", net, len(rc))
    net, _, rc = fade_sim6(two_bar=True)
    report("24a second-bar exhaustion only", net, len(rc))
    net, _, rc = fade_sim6(weekend=True)
    report("24b weekend entries only", net, len(rc))
    net, _, rc = fade_sim6(weekend=False)
    report("24b weekday entries only", net, len(rc))


if __name__ == "__main__":
    battery()
