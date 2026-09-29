"""MAIN STRATEGY — canonical 5-variant table (production v8, 2026-09-29).

Round-28 production definition (24m-validated, cross-checked on 12m):
  v1 hourly gated   -> FADE v7 gated (tiny sample)
  v2 hourly nogate  -> FADE v7: spike >= 5xATR + vol >= 3x, top-200 alt ->
                        FADE SHORT; ORDERFLOW VETO: skip buy climaxes
                        (event-bar taker-buy share >= 60% — the fade edge
                        is negative there); ivol; 3-tranche resting-limit
                        exit (50% @ 50% retrace, 25% @ 75%, 25% @ origin,
                        wick-touch fills), 12h re-entry cooldown, cap 42h
                        [PRODUCTION]
  v3 daily          -> SQUEEZE-breakout, true mid-band exit, + risk overlays
                        (PSAR RETIRED: fails 24m) [PRODUCTION]
  v4 daily nogate   -> SQUEEZE bare
  CLIM (hourly)     -> CLIM v1 long book: JOIN buy climaxes (same spike
                        >= 5xATR + vol >= 3x, tf >= 60%), ATR trail 1x,
                        cap 24h — first validated LONG book (SR 1.44 P0 2%;
                        +7%/yr standalone) [PRODUCTION]
  v5 COMBO          -> 60/25/15 capital: FADE + SQUEEZE + CLIM [PRODUCTION
                        TRIO: SR 6.42, P0 0%, +120%/yr, DD -8.9%; beats the
                        60/40 pair on every split; 12m xcheck 6.95]

Run:  ./venv/bin/python backtests/best4.py            # 5-variant table
      ./venv/bin/python backtests/best4.py --matrices # + reference matrix
Data: PS_DATA_DIR (default saved_data_12m; production = saved_data_24m)
Flow: saved_data_24m_flow (taker-buy volume, fetch_flow24.py)
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate17 import fade_sim3  # noqa: F401 (v6 reference)
from iterate25 import fade_sim7, load_flow
from iterate28 import cont_sim
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

    # compute the books once (production v8 configs)
    flow = load_flow()
    f7_gated, _, rc_g = fade_sim7(tf=flow, tf_max=0.60, gate=80)
    fade_net, _, rc_f = fade_sim7(tf=flow, tf_max=0.60)
    sq = FS.sim(squeeze2(), False, None, SQ_OVER)
    sq_bare = FS.sim(squeeze2(), False, None)
    clim_net, clim_tr = cont_sim(flow, tf_min=0.60, exit_mode="atr_trail",
                                 trail_atr=1.0)
    combo = 0.60 * fade_net + 0.25 * sq["net"] + 0.15 * clim_net

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

    row("v1 hourly gated", "FADE v7 gated (small sample)", f7_gated, len(rc_g))
    row("v2 hourly nogate",
        "FADE v7 vol>=3x + orderflow veto + 3-tranche exit (PRODUCTION)", fade_net, len(rc_f))
    row("v3 daily", "SQUEEZE mid-exit + overlays (PRODUCTION)",
        sq["net"], sq["n"])
    row("v4 daily nogate", "SQUEEZE mid-exit bare", sq_bare["net"], sq_bare["n"])
    row("CLIM long (hourly)",
        "CLIM v1: join buy climaxes tf>=0.60, ATR trail 1x, 24h (PRODUCTION)",
        clim_net, len(clim_tr))
    row("v5 COMBO 60/25/15", "production trio (FADE + SQUEEZE + CLIM)", combo)
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
        report("FADE v7 — hourly nogate", fade_net, len(rc_f))
        report("FADE v7 — hourly gated", f7_gated, len(rc_g))
        report("SQUEEZE — daily + overlays", sq["net"], sq["n"])
        report("SQUEEZE — daily bare", sq_bare["net"], sq_bare["n"])
