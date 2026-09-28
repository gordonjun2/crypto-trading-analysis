"""Round 12: hunt the next breakthrough across all variants.

1. dump-side recheck WITH volume filter (the spike/dump split was measured
   pre-vol-filter; high-volume dump-fades may revert too)
2. k-plateau WITH vol filter (vol confirmation may rescue weaker thresholds)
3. hold-cap sweep for the origin exit (12/24/36/48h)
4. slot sweep for the fade book (8/12/16)
5. fade-side riskoff overlay (DD control) and its effect on the combo
6. daily PSAR direction split re-check under causal accounting

Run:  ./venv/bin/python backtests/iterate12.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import fade_sim, CHAMP

from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def report(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


def riskoff_overlay(net, thr=0.20, scale=0.5, back=0.10):
    eq = net.cumsum()
    emax = eq.cummax()
    dd = 1.0 - eq / emax.replace(0.0, np.nan)
    out = np.zeros(len(net))
    cur = 1.0
    vals, dds = net.values, dd.values
    for i in range(len(net)):
        out[i] = cur * vals[i]
        d = dds[i] if np.isfinite(dds[i]) else 0.0
        if d > thr:
            cur = scale
        elif d < back:
            cur = 1.0
    return pd.Series(out, index=net.index)


if __name__ == "__main__":
    base, _, rc_base = fade_sim(direction_filter="spike", exit_mode="origin",
                                vol_min=3.0)
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    report("BASE fade v3 (spike+vol, origin, ivol)", base, len(rc_base))

    # 1) dump side with volume filter
    net, tr, rc = fade_sim(direction_filter="dump", exit_mode="origin",
                           vol_min=3.0)
    report("dump-fades WITH vol>=3x", net, len(rc))
    both_net, both_tr, both_rc = fade_sim(exit_mode="origin", vol_min=3.0)
    report("both sides WITH vol>=3x", both_net, len(both_rc))

    # 2) k-plateau with vol filter
    for k in (3.0, 4.0, 8.0):
        net, tr, rc = fade_sim(k=k, direction_filter="spike",
                               exit_mode="origin", vol_min=3.0)
        report(f"spike+vol k={k}xATR", net, len(rc))

    # 3) hold-cap sweep
    for cap in (12, 36, 48):
        net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin",
                               vol_min=3.0, cap_h=cap)
        report(f"origin cap {cap}h", net, len(rc))

    # 4) slot sweep
    for sl in (12, 16):
        net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin",
                               vol_min=3.0, slots=sl)
        report(f"slots={sl}", net, len(rc))

    # 5) fade riskoff overlay + combos
    print()
    print("| combo variants | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    champ = TR.sim(False, 80, CHAMP)["net"]
    fade_ro = riskoff_overlay(base)
    report("fade v3 + riskoff20 overlay", fade_ro, len(rc_base))
    report("combo plain (50/50)", 0.5 * champ + 0.5 * base)
    report("combo 50/50 (fade riskoff'd)", 0.5 * champ + 0.5 * fade_ro)
    report("combo 50/50 (both books riskoff'd)",
           0.5 * champ + 0.5 * base + 0.5 * riskoff_overlay(champ) - 0.5 * champ)

    # 6) daily direction split re-check
    print()
    print("| daily direction split | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    r = TR.sim(False, 80, CHAMP | {"longonly"})
    report("daily gated LONG-only", r["net"], r["n"])
    r = TR.sim(False, 80, CHAMP | {"shortonly"})
    report("daily gated SHORT-only", r["net"], r["n"])
