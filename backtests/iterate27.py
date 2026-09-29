"""Round 27: LONG-book hunt — pair candidates for the short-biased pair.

27a mirrored dump-fade (hourly): LONG after dumps > 5xATR + vol 3x, same
    3-tranche resting-limit exit / 12h cooldown / 42h cap. Flow variants:
    raw, sell-climax veto (skip tf <= 0.40 dumps), join-flow (tf >= 0.45).
27b CVD momentum long-only (7d aggressive-buy-flow extreme -> LONG).
27c squeeze long-only with the flow veto removed (was -0.04 bare).

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate27.py
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
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

V5_TR = [(0.5, 0.5), (0.75, 0.25)]


def fade_long_sim(tf=None, tf_min=None, tf_max=None, k=5.0, cap_h=42):
    """Mirror of FADE v7: LONG after 1h dumps > k*ATR with vol >= 3x mean;
    tranches as resting limits on the upside (fills on wick touch)."""
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
        last_close_i[t["sym"]] = i
        recs.append({"sym": t["sym"], "i0": t["i0"], "j0": i,
                     "ret": c_i / t["px0"] - 1.0})

    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            expired = (i - t["i0"]) >= cap_h
            o, px0 = t["origin"], t["px0"]
            span = o - px0  # LONG: origin above entry
            lo_i, hi_i = low_np[s][i], high_np[s][i]
            if t["w"] > 1e-12:
                for ti, (rf, sf) in enumerate(V5_TR):
                    if ti in t["done"]:
                        continue
                    lvl = px0 + rf * span
                    if hi_i >= lvl:
                        realize(t, i, sf, lvl)
                        t["done"].add(ti)
            o_hit = hi_i >= o
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
            if abs(r) <= k * a or r > 0:
                continue
            v0, vma = vol_raw[symv][i], vol_ma[symv][i]
            if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                    and v0 >= 3.0 * vma):
                continue
            if tf is not None:
                tfe = tf[symv][i]
                if not np.isfinite(tfe):
                    continue
                if tf_min is not None and tfe < tf_min:
                    continue
                if tf_max is not None and tfe > tf_max:
                    continue
            cands.append((symv, a))
        if cands:
            room = slots - len(open_tr)
            for symv, a in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                w = weight(symv, i)
                net_arr[i] -= w * 2 * fee_rt
                open_tr.append({"sym": symv, "i0": i,
                                "px0": float(closes[symv].iloc[i]),
                                "origin": origin, "atr0": a, "w": w,
                                "w0": w, "done": set()})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, recs


def battery():
    print("## Round 27 — long-book hunt (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    tf = load_flow()
    net, _, rc = fade_long_sim()
    report("27a dump-fade LONG (raw mirror)", net, len(rc))
    net, _, rc = fade_long_sim(tf=tf, tf_min=0.40)
    report("27a + sell-climax veto (tf >= 0.40)", net, len(rc))
    net, _, rc = fade_long_sim(tf=tf, tf_min=0.45)
    report("27a + stricter veto (tf >= 0.45)", net, len(rc))
    net, _, rc = fade_long_sim(tf=tf, tf_max=0.40)
    report("27a join-flow (tf <= 0.40, capitulation long)", net, len(rc))

    # 27b CVD momentum long-only
    tb1, v1 = load_flow_df()
    tb4 = tb1.resample("4h").sum()
    v4 = v1.resample("4h").sum()
    h4, l4, c4, _ = FS._resample_ohlcv(FS.panel)
    delta = 2 * tb4 - v4
    d7 = delta.rolling(42).sum()
    la_trig = d7 > d7.rolling(252).quantile(0.95)
    sa = pd.DataFrame(False, index=c4.index, columns=c4.columns)
    r = FS.sim(_states_to_signals(la_trig, ~la_trig, sa, ~sa, idx, c4=c4),
               False, None)
    report("27b CVD mom LONG-only (7d flow p95)", r["net"], r["n"])
    r = FS.sim(_states_to_signals(la_trig, ~la_trig, sa, ~sa, idx, c4=c4),
               False, 80)
    report("27b CVD mom LONG-only gated", r["net"], r["n"])


if __name__ == "__main__":
    battery()
