"""Round 26: deeper orderflow exploitation (24m).

26a smooth flow sizing: scale fade entry weight by (0.65 - tf) instead of
    a hard veto at 0.60.
26b squeeze trigger aggression: do breakout entries need matching taker
    aggression (longs tf >= 0.55, shorts tf <= 0.45)?
26c veto + pre-pump interaction: 2-bar event tf (event + prior bar mean).

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate26.py
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
from iterate14 import low_np, high_np
import family_scan_honest as FS
from pair_scout.research_ta_families import _states_to_signals

V5_TR = [(0.5, 0.5), (0.75, 0.25)]


def fade_sim8(tf, cap_h=42, tf_sizing=False, tf2_max=None):
    """FADE v7 with (a) smooth flow sizing or (c) 2-bar tf veto."""
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

    def weight(symv, i, tfe=None):
        v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
        w = SLOT_FRAC
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            w = float(np.clip((mv / v) * SLOT_FRAC,
                              0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
        if tf_sizing and tfe is not None and np.isfinite(tfe):
            w *= float(np.clip((0.65 - tfe) / 0.25, 0.25, 1.0))
        return w

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
            if symv not in tf:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= 5.0 * a or r < 0:
                continue
            v0, vma = vol_raw[symv][i], vol_ma[symv][i]
            if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                    and v0 >= 3.0 * vma):
                continue
            tfe = tf[symv][i]
            if not np.isfinite(tfe):
                continue
            if tf2_max is not None:
                t0, t1 = tf[symv][i], tf[symv][max(i - 1, 0)]
                if np.isfinite(t1):
                    if min(t0, t1) >= tf2_max:
                        continue
                elif t0 >= tf2_max:
                    continue
            elif tfe >= 0.60:
                continue
            cands.append((symv, a, tfe))
        if cands:
            room = slots - len(open_tr)
            for symv, a, tfe in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                w = weight(symv, i, tfe)
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
    print("## Round 26 — orderflow deep cuts (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    tf = load_flow()
    net, _, rc = fade_sim8(tf)
    report("PRODUCTION v7 (hard veto 0.60)", net, len(rc))
    net, _, rc = fade_sim8(tf, tf_sizing=True)
    report("26a smooth flow sizing (no veto)", net, len(rc))
    net, _, rc = fade_sim8(tf, tf_sizing=True, tf2_max=0.6)
    report("26a smooth sizing + veto", net, len(rc))
    net, _, rc = fade_sim8(tf, tf2_max=0.6)
    report("26c 2-bar tf veto (min of last 2 < 0.6)", net, len(rc))
    # 26b squeeze trigger aggression
    tb1, v1 = load_flow_df()
    tb4 = tb1.resample("4h").sum()
    v4 = v1.resample("4h").sum()
    tf4 = tb4 / v4.where(v4 > 0)
    h4, l4, c4, _v4 = FS._resample_ohlcv(FS.panel)
    mid = FS._ema(c4, 120)
    atr = FS._atr(h4, l4, c4, 84)
    up, dn = mid + 2 * atr, mid - 2 * atr
    sd = c4.rolling(120).std(ddof=0)
    bup, bdn = mid + 2 * sd, mid - 2 * sd
    sq1 = ((bup < up) & (bdn > dn)).shift(1).fillna(False)
    tf4a = tf4.shift(1).fillna(0.5)
    la_trig = (c4 > bup) & sq1 & (tf4a >= 0.55)
    sa_trig = (c4 < bdn) & sq1 & (tf4a <= 0.45)
    hold_l, hold_s = c4 >= mid, c4 <= mid

    def persist(trig, hold):
        seg = trig.cumsum()
        ever_off = (~hold.fillna(False)).groupby(seg).cummax()
        return trig | ((seg > 0) & ~ever_off)

    la = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    sa = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    for col in c4.columns:
        la[col] = persist(la_trig[col], hold_l[col])
        sa[col] = persist(sa_trig[col], hold_s[col])
    r = FS.sim(_states_to_signals(la, ~la, sa, ~sa, idx, c4=c4),
               False, None, frozenset(("ivol", "voltarget", "riskoff")))
    report("26b squeeze w/ aggression-matched triggers", r["net"], r["n"])


if __name__ == "__main__":
    battery()
