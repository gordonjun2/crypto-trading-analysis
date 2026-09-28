"""Round 18: cooldown plateau + honest-execution validation (24m).

18a: cooldown sweep (touch fills) around the 12h winner from 17a.
18b: robustness of the winner — halves, user-split pseudo-OOS
     (train 2024-09..2025-12 / test 2026-01..09), 12m cross-check,
     monthly consistency. Adopt only if it beats production v5 there.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate18.py [a|b]
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import pandas as pd

import technique_research as TR
from iterate17 import fade_sim3, V5_TR
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def rep(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


def battery_a():
    print("## Round 18a — cooldown plateau (touch fills, 24m)")
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    net, _, rc = fade_sim3(touch=True)
    rep("PRODUCTION v5 honest-exec (touch)", net, len(rc))
    for ch in (8, 12, 18, 24):
        net, _, rc = fade_sim3(touch=True, cooldown_h=ch)
        rep(f"cooldown {ch}h", net, len(rc))


def battery_b():
    print("## Round 18b — cooldown-12h robustness (24m)")
    print("| variant | window | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    for lab, kw in (("PROD v5", {}), ("cooldown 12h", {"cooldown_h": 12})):
        net, _, rc = fade_sim3(touch=True, **kw)
        half = len(net) // 2
        rep(f"{lab} | full", net, len(rc))
        rep(f"{lab} | H1", net.iloc[:half], None)
        rep(f"{lab} | H2", net.iloc[half:], None)
        cut = pd.Timestamp("2026-01-01", tz="UTC")
        if net.index.tz is None:
            cut = cut.tz_localize(None)
        tr, te = net[net.index < cut], net[net.index >= cut]
        rep(f"{lab} | train<2026-01", tr, None)
        rep(f"{lab} | test>=2026-01", te, None)
        m = net.resample("MS").sum()
        print(f"  {lab}: months {int((m > 0).sum())}/{len(m)} pos | worst "
              f"{m.min():+.1%} | median {m.median():+.1%}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "a"
    if mode == "a":
        battery_a()
    elif mode == "b":
        battery_b()
