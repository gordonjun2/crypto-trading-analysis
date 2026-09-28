"""Round 11: squeeze further improvement out of the production pair.

1. Diagnostics on fade-v2 trades: mean trade PnL by event-hour bucket,
   volume signature, funding state, extremity tier.
2. Variants justified by diagnostics (coarse, a priori thresholds):
   hour_max=16, vol_min=3, fund_min=0.0005, limit 2xATR, mild extremity.
3. Combo-level DD throttle (causal exposure scaling on the 50/50 pair).
4. Final 5-variant table with upgrades.

Run:  ./venv/bin/python backtests/iterate11.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import fade_sim, CHAMP

idx = TR.idx
closes = TR.closes
pc_np = TR.pc_np
last_fund = TR.last_fund

from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def report(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


def riskoff_overlay(net, thr=0.20, scale=0.5, back=0.10):
    """Causal exposure throttle on a book's net series: scale exposure to
    `scale` while drawdown > thr, restore at full once dd < back. Exact for
    linear-weights books (scaling exposure scales returns)."""
    eq = net.cumsum()
    emax = eq.cummax()
    dd = 1.0 - eq / emax.replace(0.0, np.nan)
    out = np.zeros(len(net))
    cur = 1.0
    vals = net.values
    dds = dd.values
    for i in range(len(net)):
        out[i] = cur * vals[i]
        d = dds[i] if np.isfinite(dds[i]) else 0.0
        if d > thr:
            cur = scale
        elif d < back:
            cur = 1.0
    return pd.Series(out, index=net.index)


if __name__ == "__main__":
    # ---------- diagnostics ----------
    net0, tr0, recs = fade_sim(direction_filter="spike", exit_mode="origin")
    ret_of = {id(t): r["ret"] for t, r in zip(tr0, recs)}
    rows = []
    for t, r in zip(tr0, recs):
        s, i = t["sym"], t["i0"]
        r_flash = pc_np[s][i]
        a = float(0)  # extremity recovered from r/r below
        rows.append({
            "ret": r["ret"], "hour": idx[i].hour,
            "ext": None,
        })
    # rebuild extremity from stored data is messy; use volume + funding + hour
    vr, fund_v, ext_v = [], [], []
    vol_raw = {s: TR.panel.frames[s]["Volume"].values for s in TR.panel.pairs}
    vol_ma = {s: pd.Series(v).rolling(24).mean().values for s, v in vol_raw.items()}
    enriched = []
    for t, r in zip(tr0, recs):
        s, i = t["sym"], t["i0"]
        v0, vma = vol_raw[s][i], vol_ma[s][i]
        vratio = v0 / vma if (np.isfinite(v0) and np.isfinite(vma) and vma > 0) else np.nan
        lf = last_fund[s].iloc[i] if s in last_fund.columns else np.nan
        enriched.append({"ret": r["ret"], "hour": idx[i].hour,
                         "vr": vratio, "fund": lf})
    def bstat(rows_, key, lo, hi):
        sel = [r["ret"] for r in rows_
               if np.isfinite(r.get(key, np.nan)) and lo <= r[key] < hi]
        return (f"{len(sel)}: {np.mean(sel):+.2%}" if sel else "0: n/a")
    print("fade-v2 diagnostics (mean trade PnL):", flush=True)
    print(f"  hour 00-08 : {bstat(enriched, 'hour', 0, 8)}")
    print(f"  hour 08-16 : {bstat(enriched, 'hour', 8, 16)}")
    print(f"  hour 16-24 : {bstat(enriched, 'hour', 16, 24)}")
    print(f"  vol <3x    : {bstat(enriched, 'vr', 0, 3)}")
    print(f"  vol 3-8x   : {bstat(enriched, 'vr', 3, 8)}")
    print(f"  vol >8x    : {bstat(enriched, 'vr', 8, 1e9)}")
    print(f"  fund <0    : {bstat(enriched, 'fund', -1, 0)}")
    print(f"  fund 0-5bp : {bstat(enriched, 'fund', 0, 0.0005)}")
    print(f"  fund >5bp  : {bstat(enriched, 'fund', 0.0005, 1)}", flush=True)

    # ---------- variants ----------
    print()
    print("| variant | trades | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|---|")
    report("base fade-v2 (production)", net0, len(recs))
    net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin",
                           hour_max=16)
    report("hour_max=16 UTC", net, len(rc))
    net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin",
                           vol_min=3.0)
    report("vol_min=3x", net, len(rc))
    net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin",
                           fund_min=0.0005)
    report("fund_min=+5bp/8h", net, len(rc))
    net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin",
                           entry_mode="limit")
    report("limit entry 1xATR", net, len(rc))
    net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin",
                           entry_mode="limit")
    # deeper limit: approximate via 2xATR by overriding entry k? use limit + ext? keep 1xATR variant only
    net, tr, rc = fade_sim(direction_filter="spike", exit_mode="origin",
                           ext_size=True, ext_cap=1.5)
    report("extremity sizing cap1.5", net, len(rc))

    # ---------- combo DD throttle ----------
    print()
    print("| combo overlay | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    champ = TR.sim(False, 80, CHAMP)["net"]
    combo = 0.5 * champ + 0.5 * net0
    report("combo plain", combo)
    report("combo + riskoff20 overlay", riskoff_overlay(combo))
    best_f = fade_sim(direction_filter="spike", exit_mode="origin",
                      vol_min=3.0)[0]
    combo_b = 0.5 * champ + 0.5 * best_f
    report("combo (vol_min filter) plain", combo_b)
    report("combo (vol_min) + riskoff20", riskoff_overlay(combo_b))
