"""Round 10b: combine the round-10 wins and validate.

Fade upgrades to combine: spike-only (SHORT after up-flashes) +
revert-to-origin exit (+ optional limit entry). Then k-plateau re-check,
halves, months, and the new 50/50 combo with the daily champion.

Run:  ./venv/bin/python backtests/iterate10b.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import fade_sim, CHAMP

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
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin")
    report("spike-only + origin-exit", net, len(rc))
    net_l, tr_l, rc_l = fade_sim(direction_filter="spike", exit_mode="origin",
                                 entry_mode="limit")
    report("spike-only + origin + limit entry", net_l, len(rc_l))
    for k in (3.0, 8.0):
        net, tr, rc = fade_sim(k=k, direction_filter="spike", exit_mode="origin")
        report(f"spike+origin k={k}xATR", net, len(rc))

    # validation of the winner
    best_net, best_tr, best_rc = fade_sim(direction_filter="spike",
                                          exit_mode="origin")
    half = len(best_net) // 2
    for name, seg in (("H1", best_net.iloc[:half]), ("H2", best_net.iloc[half:])):
        lo, hi, pn = bootstrap_sharpe_ci(seg)
        days = len(seg) / 24
        print(f"| fade-upg {name} | | {sharpe_of(seg):.2f} | [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} | {seg.sum() * 365 / days:+.0%}/yr | "
              f"{max_dd_of(seg):.1%} |", flush=True)
    m = best_net.resample("MS").sum()
    print(f"  fade-upg months: {int((m > 0).sum())}/{len(m)} positive | worst "
          f"{m.min():+.1%} | median {m.median():+.1%}", flush=True)

    # new combo with the daily champion
    champ = TR.sim(False, 80, CHAMP)["net"]
    combo = 0.5 * best_net + 0.5 * champ
    print()
    report("COMBO 50/50 (champ + upgraded fade)", combo)
    for name, seg in (("H1", combo.iloc[:half]), ("H2", combo.iloc[half:])):
        lo, hi, pn = bootstrap_sharpe_ci(seg)
        days = len(seg) / 24
        print(f"| combo {name} | | {sharpe_of(seg):.2f} | [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} | {seg.sum() * 365 / days:+.0%}/yr | "
              f"{max_dd_of(seg):.1%} |", flush=True)
    m = combo.resample("MS").sum()
    print(f"  combo months: {int((m > 0).sum())}/{len(m)} positive | worst "
          f"{m.min():+.1%} | median {m.median():+.1%}", flush=True)
