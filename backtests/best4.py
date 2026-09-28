"""MAIN STRATEGY — canonical 5-variant table (production v6, 2026-09-28).

Round-18 production definition (24m-validated, cross-checked on 12m):
  v1 hourly gated   -> FADE v6 gated (tiny sample)
  v2 hourly nogate  -> FADE v6: spike >= 5xATR + vol >= 3x, ivol, 3-tranche
                        scale-out as RESTING LIMITS (50% @ 50% retrace,
                        25% @ 75%, 25% @ origin, wick-touch fills),
                        12h same-symbol re-entry cooldown, cap 42h
                        [PRODUCTION]
  v3 daily          -> SQUEEZE-breakout, true mid-band exit, + risk overlays
                        (PSAR RETIRED: fails 24m) [PRODUCTION]
  v4 daily nogate   -> SQUEEZE bare
  v5 COMBO          -> 60/40 capital: fade-hourly + squeeze-daily [PRODUCTION
                        PAIR: SR 5.73, P0 0%, maxDD -9.3%, worst month -0.5%]

Run:  ./venv/bin/python backtests/best4.py            # 5-variant table
      ./venv/bin/python backtests/best4.py --matrices # + reference matrix
Data: PS_DATA_DIR (default saved_data_12m; production = saved_data_24m)
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate17 import fade_sim3
from iterate14 import squeeze2, SQ_OVER
import family_scan_honest as FS

idx = TR.idx

from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def report(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


if __name__ == "__main__":
    matrices = "--matrices" in sys.argv

    # compute the books once (production v6 configs)
    f6_gated, _, rc_g = fade_sim3(touch=True, cooldown_h=12, gate=80)
    fade_net, _, rc_f = fade_sim3(touch=True, cooldown_h=12)
    sq = FS.sim(squeeze2(), False, None, SQ_OVER)
    sq_bare = FS.sim(squeeze2(), False, None)
    combo = 0.6 * fade_net + 0.4 * sq["net"]

    print("## MAIN STRATEGY — 5 variants (each at its best)")
    print("| variant | strategy | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")

    def row(label, strat, net, n=None):
        lo, hi, pn = bootstrap_sharpe_ci(net)
        days = len(net) / 24
        nn = f" [{n}]" if n is not None else ""
        print(f"| {label}{nn} | {strat} | {sharpe_of(net):.2f} | "
              f"[{lo:.2f},{hi:.2f}] P0 {pn:.0%} | "
              f"{net.sum() * 365 / days:+.0%}/yr | {max_dd_of(net):.1%} |",
              flush=True)

    row("v1 hourly gated", "FADE v6 gated (small sample)", f6_gated, len(rc_g))
    row("v2 hourly nogate",
        "FADE v6 vol>=3x + 3-tranche exit (PRODUCTION)", fade_net, len(rc_f))
    row("v3 daily", "SQUEEZE mid-exit + overlays (PRODUCTION)",
        sq["net"], sq["n"])
    row("v4 daily nogate", "SQUEEZE mid-exit bare", sq_bare["net"], sq_bare["n"])
    row("v5 COMBO 60/40", "production pair (v2 + v3)", combo)
    half = len(combo) // 2
    for name, seg in (("H1", combo.iloc[:half]), ("H2", combo.iloc[half:])):
        lo, hi, pn = bootstrap_sharpe_ci(seg)
        days = len(seg) / 24
        print(f"| v5 {name} | | {sharpe_of(seg):.2f} | [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} | {seg.sum() * 365 / days:+.0%}/yr | "
              f"{max_dd_of(seg):.1%} |", flush=True)
    m = combo.resample("MS").sum()
    print(f"  v5 months: {int((m > 0).sum())}/{len(m)} positive | worst "
          f"{m.min():+.1%} | median {m.median():+.1%}", flush=True)
    # fade-only validation (the durable core)
    half_f = len(fade_net) // 2
    for name, seg in (("H1", fade_net.iloc[:half_f]),
                      ("H2", fade_net.iloc[half_f:])):
        lo, hi, pn = bootstrap_sharpe_ci(seg)
        days = len(seg) / 24
        print(f"| fade-only {name} | | {sharpe_of(seg):.2f} | "
              f"[{lo:.2f},{hi:.2f}] P0 {pn:.0%} | "
              f"{seg.sum() * 365 / days:+.0%}/yr | {max_dd_of(seg):.1%} |",
              flush=True)
    m = fade_net.resample("MS").sum()
    print(f"  fade-only months: {int((m > 0).sum())}/{len(m)} positive | "
          f"worst {m.min():+.1%} | median {m.median():+.1%}", flush=True)

    if matrices:
        print()
        print("## Reference — production strategies across their cells")
        print("| variant | SR(bar) | daily-block CI | ret | maxDD |")
        print("|---|---|---|---|---|")
        report("FADE v6 — hourly nogate", fade_net, len(rc_f))
        report("FADE v6 — hourly gated", f6_gated, len(rc_g))
        report("SQUEEZE — daily + overlays", sq["net"], sq["n"])
        report("SQUEEZE — daily bare", sq_bare["net"], sq_bare["n"])
