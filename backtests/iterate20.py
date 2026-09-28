"""Round 20: structural squeeze experiments + pair weights (24m).

20a squeeze on a 12h grid (states on 12h bars, decisions still daily).
20b trend alignment: long squeezes only above SMA(50d), shorts below.
20c trigger-bar volume >= 1.5x trailing mean (between dry-up and explosion).
20d pair weight sweep 30/70 .. 70/30 (fade/squeeze) on v6 books.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate20.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import idx
from iterate17 import fade_sim3
from iterate14 import squeeze2, SQ_OVER
import family_scan_honest as FS
from pair_scout.research_ta_families import _states_to_signals
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def rep(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


def squeeze_grid(bpd_grid=2, volx_min=None, trend=False):
    """squeeze2 generalized to a 12h (bpd_grid=2) state grid; optional
    trigger-volume floor and 50d-SMA trend alignment at trigger."""
    h4, l4, c4, v4 = FS._resample_ohlcv(FS.panel)
    if bpd_grid != 6:
        c4 = c4.resample(f"{24 // bpd_grid}h").last()
        h4 = h4.resample(f"{24 // bpd_grid}h").max()
        l4 = l4.resample(f"{24 // bpd_grid}h").min()
        v4 = v4.resample(f"{24 // bpd_grid}h").sum()
    mid = FS._ema(c4, 20 * bpd_grid)
    atr = FS._atr(h4, l4, c4, 14 * bpd_grid)
    up, dn = mid + 2 * atr, mid - 2 * atr
    sd = c4.rolling(20 * bpd_grid).std(ddof=0)
    bup, bdn = mid + 2 * sd, mid - 2 * sd
    sq = (bup < up) & (bdn > dn)
    sq1 = sq.shift(1).fillna(False)
    la_trig = (c4 > bup) & sq1
    sa_trig = (c4 < bdn) & sq1
    if volx_min is not None:
        vr = v4 / v4.rolling(6 * bpd_grid).mean()
        la_trig, sa_trig = la_trig & (vr >= volx_min), sa_trig & (vr >= volx_min)
    if trend:
        sma = c4.rolling(50 * bpd_grid).mean()
        la_trig, sa_trig = la_trig & (c4 > sma), sa_trig & (c4 < sma)
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
    return _states_to_signals(la, ~la, sa, ~sa, idx, c4=c4)


def battery():
    print("## Round 20 — squeeze structure + pair weights (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    r0 = FS.sim(squeeze2(), False, None, SQ_OVER)
    rep("PRODUCTION squeeze v2 (4h grid)", r0["net"], r0["n"])
    r = FS.sim(squeeze_grid(bpd_grid=2), False, None, SQ_OVER)
    rep("20a 12h state grid", r["net"], r["n"])
    r = FS.sim(squeeze_grid(bpd_grid=6), False, None, SQ_OVER)
    rep("20a 4h grid re-derivation (sanity)", r["net"], r["n"])
    r = FS.sim(squeeze_grid(bpd_grid=6, trend=True), False, None, SQ_OVER)
    rep("20b 50d-SMA trend alignment", r["net"], r["n"])
    r = FS.sim(squeeze_grid(bpd_grid=6, volx_min=1.5), False, None, SQ_OVER)
    rep("20c trigger vol >= 1.5x", r["net"], r["n"])
    f6, _, _ = fade_sim3(touch=True, cooldown_h=12)
    s_net = r0["net"]
    print("| pair weights (fade/squeeze) | SR | CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    for wf in (0.3, 0.4, 0.5, 0.6, 0.7):
        p = wf * f6 + (1 - wf) * s_net
        rep(f"20d {wf:.0%}/{1 - wf:.0%}", p)


if __name__ == "__main__":
    battery()
