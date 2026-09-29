"""Round 28: continuation longs — join the buy climax (the mirror of v7's
toxic bucket). 28a is an event study: forward drift of spike bars by tf.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate28.py [a|b]
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import (idx, closes, bpd, in_u_all, pc_np, fund_np, SLOTS,
                       SLOT_FRAC, BTC, fee_rt, atr1_np, vol_raw, vol_ma)
from iterate25 import load_flow
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

CLOSE_NP = {s: TR.panel.frames[s]["Close"].values for s in TR.panel.pairs}


def battery_a():
    """Event study: spike bars (>= 5xATR, vol >= 3x) grouped by tf quintile;
    forward returns 1/3/6/12/24h (from event close, funding-adj)."""
    tf = load_flow()
    horizons = (1, 3, 6, 12, 24)
    buckets = 5
    rows = {h: [[] for _ in range(buckets)] for h in horizons}
    for sym in TR.panel.pairs:
        if sym == BTC or sym not in tf:
            continue
        if not bool(in_u_all[sym].iloc[-1]) and sym not in in_u_all.columns:
            continue
        tfa = tf[sym]
        pc = pc_np[sym]
        fund = fund_np[sym]
        a = atr1_np[sym]
        v0 = vol_raw[sym]
        vma = vol_ma[sym]
        c = CLOSE_NP[sym]
        in_u = in_u_all[sym].values
        for i in range(30 * bpd, len(idx) - 24):
            if not in_u[i]:
                continue
            r = pc[i]
            if not (np.isfinite(r) and np.isfinite(a[i]) and a[i] > 0):
                continue
            if abs(r) <= 5.0 * a[i] or r < 0:
                continue
            if not (np.isfinite(v0[i]) and np.isfinite(vma[i]) and vma[i] > 0
                    and v0[i] >= 3.0 * vma[i]):
                continue
            tfe = tfa[i]
            if not np.isfinite(tfe):
                continue
            q = min(int((tfe - 0.30) / 0.14), 4)  # 0.30..1.00 -> 5 buckets
            q = max(q, 0)
            px = c[i]
            for h in horizons:
                j = i + h
                if j >= len(idx):
                    continue
                fr = (c[j] / px - 1.0) - fund[i + 1:j + 1].sum()
                rows[h][q].append(fr)
    print("## Round 28a — spike-bar forward drift by taker fraction (24m)")
    print("| tf bucket | n | +1h | +3h | +6h | +12h | +24h |")
    print("|---|---|---|---|---|---|---|")
    edges = (0.30, 0.44, 0.58, 0.72, 0.86, 1.01)
    for q in range(5):
        line = f"| [{edges[q]:.2f},{edges[q + 1]:.2f}) | {len(rows[1][q])} "
        for h in horizons:
            arr = np.array(rows[h][q])
            line += f"| {arr.mean():+.2%} " if len(arr) else "| - "
        line += "|"
        print(line)


def cont_sim(tf=None, tf_min=0.60, exit_mode="trail_low", trail_atr=None,
             cap_h=24, k=5.0):
    """Continuation long: join spikes with tf >= tf_min; momentum exits."""
    slots = SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades = [], []

    def accrue_to(i):
        nonlocal last_acc
        if i > last_acc:
            for t in open_tr:
                s = t["sym"]
                lo = max(last_acc + 1, t["i0"] + 1)
                if lo <= i:
                    net_arr[lo:i + 1] += (t["w"] * (pc_np[s][lo:i + 1]
                                                    - fund_np[s][lo:i + 1]))
            last_acc = i

    last_acc = 30 * bpd - 1

    def weight(symv, i):
        v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            return float(np.clip((mv / v) * SLOT_FRAC,
                                 0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
        return SLOT_FRAC

    def close_trade(t, i, reason):
        c_i = float(closes[t["sym"]].iloc[i])
        t.update(j0=i, exit_px=c_i, reason=reason)
        trades.append(t)

    low_np = {s: TR.panel.frames[s]["Low"].values for s in TR.panel.pairs}
    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            t["hi"] = max(t["hi"], float(TR.panel.frames[s]["High"].iloc[i]))
            exit_now = False
            if exit_mode == "trail_low":
                exit_now = c_i < low_np[s][i - 1]
            elif exit_mode == "atr_trail":
                exit_now = c_i < t["hi"] - trail_atr * t["atr0"]
            if exit_now or (i - t["i0"]) >= cap_h:
                close_trade(t, i, "trail" if exit_now else "time")
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv not in pc_np:
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
            if tf is not None:
                tfe = tf[symv][i]
                if not (np.isfinite(tfe) and tfe >= tf_min):
                    continue
            if len(open_tr) >= slots:
                break
            w = weight(symv, i)
            net_arr[i] -= w * 2 * fee_rt
            open_tr.append({"sym": symv, "i0": i,
                            "px0": float(closes[symv].iloc[i]),
                            "atr0": a, "w": w, "w0": w,
                            "hi": float(TR.panel.frames[symv]["High"].iloc[i])})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades


def rep(label, net, n):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    print(f"| {label} [{n}] | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


def battery_b(tf):
    print("## Round 28b — continuation longs (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, tr = cont_sim()
    rep("raw spikes, trail-low, 24h", net, len(tr))
    for cap in (6, 12, 24):
        net, tr = cont_sim(tf=tf, tf_min=0.60, cap_h=cap)
        rep(f"join tf>=0.60, trail-low, {cap}h", net, len(tr))
    for ta in (1.0, 1.5):
        net, tr = cont_sim(tf=tf, tf_min=0.60, exit_mode="atr_trail",
                           trail_atr=ta, cap_h=24)
        rep(f"join tf>=0.60, ATR trail {ta}x, 24h", net, len(tr))
    net, tr = cont_sim(tf=tf, tf_min=0.55, cap_h=12)
    rep("join tf>=0.55, trail-low, 12h", net, len(tr))
    net, tr = cont_sim(tf=tf, tf_min=0.65, cap_h=12)
    rep("join tf>=0.65, trail-low, 12h", net, len(tr))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "a"
    if mode == "a":
        battery_a()
    else:
        battery_b(load_flow())
