"""Round 18c — production v6 book table (24m).

FADE v6 = touch-fills + cooldown 12h. Squeeze v2 unchanged.
Pair = 50/50; also test vol-targeted pair (C1) on the new books.

Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate18.py c
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import technique_research as TR
from iterate14 import squeeze2, SQ_OVER
import family_scan_honest as FS
from iterate17 import fade_sim3
from iterate14 import squeeze2, SQ_OVER
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def rep(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


f6, _, rc_f = fade_sim3(touch=True, cooldown_h=12)
sq = FS.sim(squeeze2(), False, None, SQ_OVER)
s_net = sq["net"]
pair = 0.5 * s_net + 0.5 * f6

print("## Round 18c — production v6 books (24m)")
print("| book | trades | SR(bar) | daily-block CI | ret | maxDD |")
print("|---|---|---|---|---|---|")
rep("FADE v6 (touch + cd12)", f6, len(rc_f))
rep("SQUEEZE v2 + overlays", s_net, sq["n"])
rep("PAIR 50/50", pair)
half = len(pair) // 2
rep("PAIR H1", pair.iloc[:half])
rep("PAIR H2", pair.iloc[half:])
m = pair.resample("MS").sum()
print(f"  pair months: {int((m > 0).sum())}/{len(m)} pos | worst "
      f"{m.min():+.1%} | median {m.median():+.1%}", flush=True)
rv = pair.rolling(480).std()
scale = (rv.median() / rv).shift(24).clip(0.25, 1.5).fillna(1.0)
rep("PAIR vol-targeted", pair * scale)
