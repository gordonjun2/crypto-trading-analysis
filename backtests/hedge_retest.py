"""Hedge retest under CAUSAL accounting (2026-09-26).

The old iteration-10 "hedges cost more than they save" verdict was measured
with the lookahead-inflated sim. This retests static overlays on the CURRENT
champion book (post-causal-fix, blowoff+ivol+voltarget+riskoff):

- btc_h<f>  : short BTC perp at fixed fraction f of book gross (beta hedge)
- eth_h<f>  : short ETH perp at fraction f
- mom_h<f>  : short the equal-weight top-5 7d-momentum basket (pass-10 style,
              rebalanced daily, causal weights shifted one bar)

Overlay is exact for a constant-fraction overlay under the book's linear
(non-compounded) accounting:  net_h = net - f * (hedge_price_ret - hedge_fund)
+ amortized round-trip fee on the hedge leg.

Run:  ./venv/bin/python backtests/hedge_retest.py
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import numpy as np
import pandas as pd

import technique_research as TR

idx, closes, bpd = TR.idx, TR.closes, TR.bpd
pc_np, fund_np = TR.pc_np, TR.fund_np
BTC, ETH = TR.BTC, "ETHUSDT"
FEE, SLIP_BPS = TR.FEE, TR.SLIP_BPS
CHAMP = frozenset(("blowoff", "ivol", "voltarget", "riskoff"))

from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of


def mom_basket_weights(top_n=5):
    """Daily-rebalanced equal-weight top-N 7d-momentum basket, causal (weights
    formed at bar b are held over bar b+1), 1h bars."""
    mom7 = closes / closes.shift(7 * bpd) - 1.0
    ranks = mom7.rank(axis=1, ascending=False)
    w = (ranks <= top_n).astype(float)
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
    return w.shift(1).fillna(0.0)


def hedge_series(kind):
    if kind == "btc":
        return -pc_np[BTC] + fund_np[BTC]
    if kind == "eth":
        return -pc_np[ETH] + fund_np[ETH]
    wb = mom_basket_weights()
    rets = pd.DataFrame(pc_np, index=idx)
    fund = pd.DataFrame(fund_np, index=idx)
    basket = (wb * rets).sum(axis=1) - (wb * fund).sum(axis=1)
    return -basket.values


def report(label, net):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    print(f"| {label} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


if __name__ == "__main__":
    print("| book/hedge | SR(bar) | daily-block CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    for gate, lab in ((80, "dy/gated"), (None, "dy/nogate")):
        base = TR.sim(False, gate, CHAMP)
        report(f"{lab} unhedged", base["net"])
        for kind in ("btc", "eth", "mom"):
            hs = hedge_series(kind)
            for frac in (0.3, 0.5):
                net_h = base["net"].values - frac * hs
                net_h[-1] -= frac * 2 * (FEE + SLIP_BPS * 1e-4)  # one RT fee
                report(f"{lab} {kind}_h{frac}", pd.Series(net_h, index=idx))
