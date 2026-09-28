"""Round 21: squeeze grid refinement + retest entries (24m).

21a squeeze states on a 2h grid (from hourly source; same causal mapping
    via _states_to_signals shift-by-grid-bar).
21b retest entry: enter not on the breakout close but on the first close
    back within 0.5xATR of the broken band (within 3 bars), same mid exit.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate21.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import idx
from iterate14 import squeeze2, SQ_OVER
import family_scan_honest as FS
from pair_scout.research_ta_families import _states_to_signals
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

BPD = 6  # hourly panel -> bars per 4h baseline


def ohlcv_grid(bpd_grid):
    """OHLCV resampled from the hourly panel to a bpd_grid bars/day grid."""
    src = TR.panel.frames
    rule = f"{24 // bpd_grid}h" if bpd_grid <= 24 else None
    c = pd.DataFrame({s: src[s]["Close"] for s in TR.panel.pairs})
    h = pd.DataFrame({s: src[s]["High"] for s in TR.panel.pairs})
    l = pd.DataFrame({s: src[s]["Low"] for s in TR.panel.pairs})
    v = pd.DataFrame({s: src[s]["Volume"] for s in TR.panel.pairs})
    if rule:
        c = c.resample(rule).last()
        h = h.resample(rule).max()
        l = l.resample(rule).min()
        v = v.resample(rule).sum()
    return h, l, c, v


def squeeze_grid(bpd_grid=6, retest=False):
    """Production squeeze on an arbitrary state grid; optional retest
    entry (close back within 0.5xATR of the broken band within 3 bars)."""
    h4, l4, c4, v4 = ohlcv_grid(bpd_grid)
    mid = FS._ema(c4, 20 * bpd_grid)
    atr = FS._atr(h4, l4, c4, 14 * bpd_grid)
    up, dn = mid + 2 * atr, mid - 2 * atr
    sd = c4.rolling(20 * bpd_grid).std(ddof=0)
    bup, bdn = mid + 2 * sd, mid - 2 * sd
    sq1 = ((bup < up) & (bdn > dn)).shift(1).fillna(False)
    la_trig = (c4 > bup) & sq1
    sa_trig = (c4 < bdn) & sq1
    if retest:
        brk_up = la_trig
        brk_dn = sa_trig
        near_up = (c4 > bup) & (c4 <= bup + 0.5 * atr)
        near_dn = (c4 < bdn) & (c4 >= bdn - 0.5 * atr)
        rec_up = brk_up.rolling(3, min_periods=1).max().fillna(0) > 0
        rec_dn = brk_dn.rolling(3, min_periods=1).max().fillna(0) > 0
        la_trig = near_up & rec_up
        sa_trig = near_dn & rec_dn
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


def rep(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


def battery():
    print("## Round 21 — squeeze grid refinement + retest (24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    r0 = FS.sim(squeeze2(), False, None, SQ_OVER)
    rep("PRODUCTION squeeze v2 (4h)", r0["net"], r0["n"])
    r = FS.sim(squeeze_grid(bpd_grid=6), False, None, SQ_OVER)
    rep("21a 4h from hourly source (sanity)", r["net"], r["n"])
    r = FS.sim(squeeze_grid(bpd_grid=12), False, None, SQ_OVER)
    rep("21a 2h state grid", r["net"], r["n"])
    r = FS.sim(squeeze_grid(bpd_grid=24), False, None, SQ_OVER)
    rep("21a 1h state grid", r["net"], r["n"])
    r = FS.sim(squeeze_grid(bpd_grid=6, retest=True), False, None, SQ_OVER)
    rep("21b retest entry (0.5xATR of band, 3 bars)", r["net"], r["n"])


if __name__ == "__main__":
    battery()
