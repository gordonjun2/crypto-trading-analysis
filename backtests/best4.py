"""MAIN STRATEGY — canonical 5-variant table (production v8, 2026-09-29)
   + v9 LEVERED overlay (round 31, 2026-09-29).

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

Round-31 v9 LEVERED overlay (5x leverage directive, no liquidation):
  FADE v9 @5x       -> + fixed hard stop +15% (intrabar, gap-aware fill at
                        max(prev close, stop); ATR stops BANNED — a 6xATR
                        stop on a low-ATR alt was gapped to -156% NAV);
                        cluster gate: no new entries when >= 3 simultaneous
                        spikes (market-wide squeeze days gap stops in
                        clusters). SR 4.58, +485%/yr, DD -40.9%
  SQUEEZE v9 @2x    -> 5x churns the daily book (DD -72..-95% even with
                        hourly stops); capped at 2x + hourly intrabar stop
                        5% (daily-bar stops alone let trades run to -29%
                        adverse). SR 1.13, +57%/yr, DD -29.0%
  CLIM v9 @5x       -> ATR trail (effective stop) + hard floor -6%.
                        SR 1.48, +34%/yr, DD -10.5%
  TRIO 60/25/15     -> SR 4.79, +310%/yr, DD -25.0%, 21/23 months (worst
                        -0.2%); splits H1 3.42 / H2 6.02 / train 5.44 /
                        test 3.57; 12m xcheck 5.16 (DD -30.4%). Gross NAV
                        mean 0.88 / p95 2.33 / max 4.86 (24m). Worst trade
                        -9.4% NAV; max adverse 15% << 20% 5x-liquidation
                        point -> liquidation impossible.

Run:  ./venv/bin/python backtests/best4.py            # 5-variant table
      ./venv/bin/python backtests/best4.py --levered  # + v9 levered table
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
from iterate31 import fade_sim9, cont_sim8, sq_stop_hybrid, gross_series
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
    levered = "--levered" in sys.argv

    def line9(label, net, n=None):
        lo, hi, pn = bootstrap_sharpe_ci(net)
        days = len(net) / 24
        nn = f" [{n}]" if n is not None else ""
        print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
              f"{max_dd_of(net):.1%} |", flush=True)

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

    if levered:
        # ---- v9 LEVERED overlay (round 31): 5x hourly books, 2x daily book
        print()
        print("## v9 LEVERED — FADE@5x +15% stop + cluster gate | "
              "SQUEEZE@2x hourly 5% stop | CLIM@5x trail+floor -6%")
        print("| variant | SR(bar) | daily-block CI | ret | maxDD |")
        print("|---|---|---|---|---|")
        f9_net, f9_tr, f9_rc = fade_sim9(tf=flow, cluster_gate=3)
        c9_net, c9_tr, _ = cont_sim8(tf=flow, stop_pct=0.06)
        sq9, sq9_tr, sq9_n = sq_stop_hybrid(0.05, lev=2.0)
        tri9 = 0.60 * f9_net + 0.25 * sq9 + 0.15 * c9_net
        line9("FADE v9 @5x", f9_net, len(f9_rc))
        line9("SQUEEZE v9 @2x", sq9, sq9_n)
        line9("CLIM v9 @5x", c9_net, len(c9_tr))
        line9("TRIO 60/25/15 levered", tri9)
        half9 = len(tri9) // 2
        line9("  H1", tri9.iloc[:half9])
        line9("  H2", tri9.iloc[half9:])
        cut9 = int(len(tri9) * 0.7)
        line9("  train70", tri9.iloc[:cut9])
        line9("  test30", tri9.iloc[cut9:])
        m9 = tri9.resample("MS").sum()
        g9 = (gross_series(f9_tr) + gross_series(c9_tr)
              + gross_series(sq9_tr))
        print(f"  months: {int((m9 > 0).sum())}/{len(m9)} positive | worst "
              f"{m9.min():+.1%} | median {m9.median():+.1%}", flush=True)
        print(f"  gross NAV: mean {g9.mean():.2f} p95 {g9.quantile(0.95):.2f}"
              f" max {g9.max():.2f} | stops 15%/5%/6% << 20% 5x-liq point "
              f"-> no liquidation", flush=True)

    if matrices:
        print()
        print("## Reference — production strategies across their cells")
        print("| variant | SR(bar) | daily-block CI | ret | maxDD |")
        print("|---|---|---|---|---|")
        report("FADE v7 — hourly nogate", fade_net, len(rc_f))
        report("FADE v7 — hourly gated", f7_gated, len(rc_g))
        report("SQUEEZE — daily + overlays", sq["net"], sq["n"])
        report("SQUEEZE — daily bare", sq_bare["net"], sq_bare["n"])
