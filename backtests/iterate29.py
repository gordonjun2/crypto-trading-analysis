"""Round 29: CLIM improvements + new long entries (24m).

29a CLIM exit variants: partial profit tranche (touch fills at +x ATR),
    tighter/looser trails, longer cap.
29b ignition definitions: tf >= 0.60 AND taker-buy $ >= Nx own 24h mean;
    k=6 spikes.
29c 4h buy-climax long, isolated (never tested — 25b only had sell-climax
    longs and buy-climax shorts).
29d post-climax pullback entry: buy the first red bar after the climax
    while it holds above the climax bar's low (better fills than the
    climax close).

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate29.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import (idx, closes, bpd, in_u_all, pc_np, fund_np, SLOTS,
                       SLOT_FRAC, BTC, fee_rt, atr1_np, vol_raw, vol_ma, report)
from iterate25 import load_flow, load_flow_df
from iterate14 import high_np, low_np
import family_scan_honest as FS
from pair_scout.research_ta_families import _states_to_signals

TB1, V1 = load_flow_df()
TB_NP = {s: TB1[s].values if s in TB1.columns else None for s in TR.panel.pairs}


def cont_sim2(tf=None, tf_min=0.60, trail_atr=1.0, cap_h=24, k=5.0,
              tp_atr=None, tp_frac=0.5, buyvol_min=None, pullback=False):
    """CLIM with tranche TP + entry variants. TP = resting limit filled on
    wick touch at px0 + tp_atr*atr0 (honest fills)."""
    slots = SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades = [], []
    pend = []  # pullback entries: (expire_i, sym, meta)

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

    def realize(t, i, frac, px):
        net_arr[i] += frac * t["w0"] * (px / t["px0"] - 1.0)
        net_arr[i] -= frac * t["w0"] * fee_rt
        t["w"] -= frac * t["w0"]

    def close_trade(t, i, reason, px=None):
        c_i = float(closes[t["sym"]].iloc[i]) if px is None else px
        t.update(j0=i, exit_px=c_i, reason=reason)
        trades.append(t)

    def clim_ok(symv, i, a):
        if not bool(in_u_all[symv].iloc[i]):
            return None
        r = pc_np[symv][i]
        if not (np.isfinite(r) and np.isfinite(a) and a > 0):
            return None
        if abs(r) <= k * a or r < 0:
            return None
        v0, vma = vol_raw[symv][i], vol_ma[symv][i]
        if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                and v0 >= 3.0 * vma):
            return None
        if tf is not None:
            tfe = tf[symv][i]
            if not (np.isfinite(tfe) and tfe >= tf_min):
                return None
        if buyvol_min is not None:
            tb = TB_NP[symv]
            if tb is None:
                return None
            lo = max(i - 23, 0)
            m = tb[lo:i + 1].mean()
            if not (np.isfinite(m) and m > 0 and tb[i] >= buyvol_min * m):
                return None
        return r

    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            hi_i = high_np[s][i]
            t["hi"] = max(t["hi"], hi_i)
            if t.get("tp") and t["w"] > 1e-12 and 0 not in t["done"]:
                lvl = t["px0"] + t["tp"]
                if hi_i >= lvl:
                    realize(t, i, tp_frac, lvl)
                    t["done"].add(0)
            ex = c_i < t["hi"] - trail_atr * t["atr0"]
            if ex or (i - t["i0"]) >= cap_h:
                close_trade(t, i, "trail" if ex else "time")
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        # pullback pending fills
        fill, keep = [], []
        for exp_i, symv, a in pend:
            if i > exp_i or symv in held:
                continue
            r = pc_np[symv][i]
            c_i = float(closes[symv].iloc[i])
            if r < 0 and c_i > low_np[symv][i - 1]:
                fill.append((symv, a))
            else:
                keep.append((exp_i, symv, a))
        pend = keep
        for symv, a in fill:
            w = weight(symv, i)
            net_arr[i] -= w * 2 * fee_rt
            open_tr.append({"sym": symv, "i0": i,
                            "px0": float(closes[symv].iloc[i]),
                            "atr0": a, "w": w, "w0": w, "done": set(),
                            "hi": float(high_np[symv][i]),
                            "tp": tp_atr * a if tp_atr else None})
            held.add(symv)
        for symv in TR.panel.pairs:
            if symv == BTC or symv in held or symv not in pc_np:
                continue
            a = atr1_np[symv][i]
            r = clim_ok(symv, i, a) if a is not None and np.isfinite(a) \
                else None
            if r is None:
                continue
            if pullback:
                pend.append((i + 2, symv, a))
                continue
            if len(open_tr) >= slots:
                break
            w = weight(symv, i)
            net_arr[i] -= w * 2 * fee_rt
            open_tr.append({"sym": symv, "i0": i,
                            "px0": float(closes[symv].iloc[i]),
                            "atr0": a, "w": w, "w0": w, "done": set(),
                            "hi": float(high_np[symv][i]),
                            "tp": tp_atr * a if tp_atr else None})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades


def battery():
    print("## Round 29 — CLIM improvements + long entries (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    tf = load_flow()
    net, tr = cont_sim2(tf=tf)
    report("CLIM v1 reference (trail 1x, 24h)", net, len(tr))
    for ta in (0.75, 1.25):
        net, tr = cont_sim2(tf=tf, trail_atr=ta)
        report(f"29a trail {ta}x", net, len(tr))
    net, tr = cont_sim2(tf=tf, cap_h=36)
    report("29a cap 36h", net, len(tr))
    for tp, lab in ((1.0, "TP 50% @ +1xATR"), (1.5, "TP 50% @ +1.5xATR")):
        net, tr = cont_sim2(tf=tf, tp_atr=tp)
        report(f"29a {lab}, trail rest", net, len(tr))
    net, tr = cont_sim2(tf=tf, buyvol_min=2.0)
    report("29b tf>=0.60 AND buy$ >= 2x mean24", net, len(tr))
    net, tr = cont_sim2(tf=tf, k=6.0)
    report("29b k=6xATR spikes", net, len(tr))
    net, tr = cont_sim2(tf=tf, pullback=True)
    report("29d post-climax pullback entry", net, len(tr))
    # 29c isolated 4h buy-climax long
    tb4 = TB1.resample("4h").sum()
    v4 = V1.resample("4h").sum()
    tf4 = tb4 / v4.where(v4 > 0)
    h4, l4, c4, _v4 = FS._resample_ohlcv(FS.panel)
    r4 = c4.pct_change()
    sa = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    la = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    trig = (tf4 >= 0.60) & (r4 > 0)
    hold = (tf4 >= 0.50) & (c4 >= c4.rolling(120).mean())
    for col in c4.columns:
        seg = trig[col].cumsum()
        ever_off = (~hold[col].fillna(False)).groupby(seg).cummax()
        la[col] = trig[col] | ((seg > 0) & ~ever_off)
    sig = _states_to_signals(la, ~la, sa, ~sa, idx, c4=c4)
    r = FS.sim(sig, False, None)
    report("29c 4h buy-climax long isolated", r["net"], r["n"])


if __name__ == "__main__":
    battery()
