"""Round 19: FADE v6 microstructure round 2 (24m).

19a fatigue: skip entry if symbol closed >= 2 trades in trailing 48h.
19b time-tranche: realize a fraction at fixed age (18h) regardless of
    retrace (converts decaying trades into earlier partial exits).
19c shape: pump-bar close position (close-low)/(high-low) — momentum-intact
    (>= 0.7) vs already-rejected (<= 0.3) at entry.
19d staggered entry: half size at flash close, half at next close.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate19.py
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


def fade_sim4(k=5.0, cap_h=42, slots=None, cooldown_h=None, fatigue_n=None,
              fatigue_h=48, shape_min=None, shape_max=None, stagger=False,
              tplan=None):
    """FADE v6 + round-19 knobs. tplan: [(age_h, frac)] realized at close
    once age >= age_h (retrace plan unchanged)."""
    slots = slots or SLOTS
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades, recs = [], [], []
    last_close_i = {}
    sym_closes = {}

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
        sym_closes.setdefault(t["sym"], []).append(i)
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        recs.append({"sym": t["sym"], "dir": t["dir"], "i0": t["i0"], "j0": i,
                     "ret": sgn * (c_i / t["px0"] - 1.0)})

    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            age = i - t["i0"]
            expired = age >= cap_h
            o, px0 = t["origin"], t["px0"]
            span = px0 - o if t["dir"] == "SHORT" else o - px0
            lo_i, hi_i = low_np[s][i], high_np[s][i]
            if t.get("tplan") and t["w"] > 1e-12:
                for ti, (ah, sf) in enumerate(t["tplan"]):
                    if ti in t["tdone"] or age < ah:
                        continue
                    realize(t, i, sf, c_i)
                    t["tdone"].add(ti)
            if t.get("plan") and t["w"] > 1e-12:
                for ti, (rf, sf) in enumerate(t["plan"]):
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
            if cooldown_h is not None and symv in last_close_i \
                    and i - last_close_i[symv] < cooldown_h:
                continue
            if fatigue_n is not None:
                hist = [j for j in sym_closes.get(symv, ()) if i - j <= fatigue_h]
                if len(hist) >= fatigue_n:
                    continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            r = pc_np[symv][i]
            a = atr1_np[symv][i]
            if not (np.isfinite(r) and np.isfinite(a) and a > 0):
                continue
            if abs(r) <= k * a:
                continue
            v0, vma = vol_raw[symv][i], vol_ma[symv][i]
            if not (np.isfinite(v0) and np.isfinite(vma) and vma > 0
                    and v0 >= 3.0 * vma):
                continue
            direction = "SHORT" if r > 0 else "LONG"
            if direction != "SHORT":
                continue
            if shape_min is not None or shape_max is not None:
                hi_v, lo_v, c_v = high_np[symv][i], low_np[symv][i], \
                    float(closes[symv].iloc[i])
                if not (np.isfinite(hi_v) and np.isfinite(lo_v)
                        and hi_v > lo_v):
                    continue
                shape = (c_v - lo_v) / (hi_v - lo_v)
                if direction == "SHORT":
                    if shape_min is not None and shape < shape_min:
                        continue
                    if shape_max is not None and shape > shape_max:
                        continue
            cands.append((symv, direction, a))
        if cands:
            room = slots - len(open_tr)
            for symv, direction, a in cands[:max(room, 0)]:
                origin = float(closes[symv].iloc[i - 1])
                w = weight(symv, i)
                if stagger:
                    w *= 0.5
                    net_arr[i] -= w * 2 * fee_rt
                    open_tr.append({"sym": symv, "dir": direction, "i0": i,
                                    "px0": float(closes[symv].iloc[i]),
                                    "origin": origin, "atr0": a, "w": w,
                                    "w0": w, "done": set(), "tdone": set(),
                                    "plan": list(V5_TR),
                                    "tplan": list(tplan) if tplan else None,
                                    "half2": i + 1})
                else:
                    net_arr[i] -= w * 2 * fee_rt
                    open_tr.append({"sym": symv, "dir": direction, "i0": i,
                                    "px0": float(closes[symv].iloc[i]),
                                    "origin": origin, "atr0": a, "w": w,
                                    "w0": w, "done": set(), "tdone": set(),
                                    "plan": list(V5_TR),
                                    "tplan": list(tplan) if tplan else None})
        if stagger:
            for t in open_tr:
                if t.get("half2") == i and t["w0"] < 0.5 * SLOT_FRAC * 1.99:
                    w2 = weight(t["sym"], i) * 0.5
                    net_arr[i] -= w2 * 2 * fee_rt
                    t["w"] += w2
                    t["w0"] += w2
                    t["px0"] = (t["px0"] + float(closes[t["sym"]].iloc[i])) / 2
                    t.pop("half2")
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades, recs


def battery():
    print("## Round 19 — FADE v6 microstructure round 2 (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, _, rc = fade_sim4(cooldown_h=12)
    report("PRODUCTION v6 (touch + cd12)", net, len(rc))
    for fn in (2, 3):
        net, _, rc = fade_sim4(cooldown_h=12, fatigue_n=fn)
        report(f"19a fatigue: skip if {fn}+ closes in 48h", net, len(rc))
    for tp, lab in ((((18, 0.25),), "19b time-tranche 25% @ 18h"),
                    (((12, 0.25),), "19b time-tranche 25% @ 12h")):
        net, _, rc = fade_sim4(cooldown_h=12, tplan=list(tp))
        report(lab, net, len(rc))
    for sm, sx, lab in ((0.7, None, "19c shape >= 0.7 (momentum intact)"),
                        (None, 0.3, "19c shape <= 0.3 (already rejected)")):
        net, _, rc = fade_sim4(cooldown_h=12, shape_min=sm, shape_max=sx)
        report(lab, net, len(rc))
    net, _, rc = fade_sim4(cooldown_h=12, stagger=True)
    report("19d staggered entry (half now, half next close)", net, len(rc))


if __name__ == "__main__":
    battery()
