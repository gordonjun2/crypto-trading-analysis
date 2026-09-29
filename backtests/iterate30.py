"""Round 30: long books from existing machinery (24m).

30a squeeze-LONG rescue: the long side of squeeze v2 is dead with the
    mid-band exit (-0.04) — try CLIM-style momentum exits (daily-trigger
    entries, hourly ATR trail).
30b single-asset trend: long-only Donchian 20d breakout on BTC (and ETH)
    with ATR trail — uncorrelated to alt-pump books; one-asset trend
    following is a different animal than the failed cross-sectional family.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate30.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import (idx, closes, bpd, pc_np, fund_np, SLOTS, SLOT_FRAC,
                       BTC, fee_rt, atr1_np, report)
from iterate14 import squeeze2, high_np, low_np
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def squeeze_long_trail(trail_atr=1.0, cap_h=24 * 14, sym_filter=None):
    """Enter on squeeze LONG activations (hourly grid), trail on closes."""
    sig = squeeze2()
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades = [], []
    last_acc = 30 * bpd - 1

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

    def weight(symv, i):
        v, mv = TR.vol_np[symv][i], TR.med_vol_bar[i]
        if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
            return float(np.clip((mv / v) * SLOT_FRAC,
                                 0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
        return SLOT_FRAC

    def close_trade(t, i, reason):
        t.update(j0=i, exit_px=float(closes[t["sym"]].iloc[i]), reason=reason)
        trades.append(t)

    for i in range(30 * bpd, n):
        accrue_to(i)
        still = []
        for t in open_tr:
            s = t["sym"]
            c_i = float(closes[s].iloc[i])
            t["hi"] = max(t["hi"], float(high_np[s][i]))
            if c_i < t["hi"] - trail_atr * t["atr0"] \
                    or (i - t["i0"]) >= cap_h:
                close_trade(t, i, "trail" if (i - t["i0"]) < cap_h else "time")
            else:
                still.append(t)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        for symv, (la, _le, _sa, _se) in sig.items():
            if symv == BTC or symv in held or symv not in pc_np:
                continue
            if sym_filter and symv not in sym_filter:
                continue
            if not bool(la.iloc[i]) or bool(la.iloc[i - 1]):
                continue  # fresh activation only
            if len(open_tr) >= SLOTS:
                break
            w = weight(symv, i)
            net_arr[i] -= w * 2 * fee_rt
            open_tr.append({"sym": symv, "i0": i,
                            "px0": float(closes[symv].iloc[i]),
                            "atr0": atr1_np[symv][i], "w": w, "w0": w,
                            "hi": float(high_np[symv][i])})
    accrue_to(len(idx) - 1)
    for t in open_tr:
        close_trade(t, len(idx) - 1, "time")
    return pd.Series(net_arr, index=idx), trades


def btc_trend(syms=("BTCUSDT",), don_d=20, trail_atr=2.0, cap_h=24 * 90):
    """Long-only Donchian breakout on the given symbols, ATR trail."""
    n = len(idx)
    net_arr = np.zeros(n)
    open_tr, trades = [], []
    last_acc = 30 * bpd - 1

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

    def close_trade(t, i, reason):
        t.update(j0=i, exit_px=float(closes[t["sym"]].iloc[i]), reason=reason)
        trades.append(t)

    don = 20 * bpd
    for s in syms:
        c = closes[s].values
        hi_n = high_np[s]
        a = atr1_np[s]
        w = 1.0 / len(syms)
        t = None
        hi_since = 0.0
        for i in range(don + 1, n):
            if t is not None:
                hi_since = max(hi_since, hi_n[i])
                c_i = c[i]
                age = i - t
                if (c_i < hi_since - trail_atr * a[i]) or age >= cap_h:
                    ret = w * (c_i / entry - 1.0)
                    net_arr[i] += ret
                    net_arr[i] -= w * 2 * fee_rt
                    trades.append({"i0": t, "j0": i,
                                   "ret": c_i / entry - 1.0})
                    t = None
            else:
                don_hi_prev = np.nanmax(c[max(i - don - 1, 0):i - 1])
                if c[i] > np.nanmax(c[i - don:i]) and c[i - 1] <= don_hi_prev \
                        and np.isfinite(a[i]) and a[i] > 0:
                    t = i
                    entry = c[i]
                    hi_since = hi_n[i]
    return pd.Series(net_arr, index=idx), trades


def battery():
    print("## Round 30 — long books round 3 (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    for ta in (1.0, 1.5):
        net, tr = squeeze_long_trail(trail_atr=ta)
        report(f"30a squeeze LONG + ATR trail {ta}x", net, len(tr))
    for syms, lab in ((("BTCUSDT",), "BTC"), (("BTCUSDT", "ETHUSDT"), "BTC+ETH")):
        net, tr = btc_trend(syms)
        report(f"30b {lab} Donchian 20d, trail 2xATR", net, len(tr))
    net, tr = btc_trend(("BTCUSDT",), trail_atr=3.0)
    report("30b BTC trail 3xATR", net, len(tr))


if __name__ == "__main__":
    battery()
