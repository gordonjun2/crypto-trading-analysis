"""Backtest health check for the live probe strategy (PSAR champion, pass-11).

Replicates the documented champion config and re-runs it on current data:
- 12m continuous dataset (saved_data_12m): headline + halves + slippage stress
  + af-neighborhood + direction split + monthly + recent-90d
- freshest ~91d dataset (saved_data_live): recency check (no funding cache there)

Run from repo root:  ./venv/bin/python backtests/champion_check.py
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, "/root/crypto-trading-analysis")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

from pair_scout.config import load_config
from pair_scout.data.funding import funding_rate_series, load_funding
from pair_scout.data.loader import load_panel
from pair_scout.research_ta_families import (
    build_signals, psar_signals_panel, run_family_sim, universe_mask,
)

SLOTS = 8
SLOT_FRAC = 1.0 / SLOTS  # 8 slots @ 1x gross
UNIV = 200
EXTRA_BPS = 2.0  # documented cost model: 7 bps/side taker + 2 bps slippage


def prep(data_dir: str):
    cfg = load_config(None)
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval, data_dir=data_dir,
        top_n_volume=658, min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    funding = load_funding(Path(data_dir) / "funding_rates.json")
    fund_cost = pd.DataFrame(
        {s: funding_rate_series(s, funding, panel.index) for s in panel.pairs}
    ).fillna(0.0)
    in_univ = universe_mask(panel, UNIV)
    return panel, fund_cost, in_univ


def sim(panel, fund_cost, in_univ, signals, extra_bps=EXTRA_BPS,
        direction="both", max_slots=SLOTS, slot_frac=SLOT_FRAC):
    return run_family_sim(panel, fund_cost, in_univ, signals,
                          direction=direction, extra_bps=extra_bps,
                          max_slots=max_slots, slot_frac=slot_frac)


def fmt(r, label, days=None):
    lo, hi, pn = r["ci"]
    d = days or len(r["net"]) / 24
    ann = r["ret"] * 365.0 / d
    return (f"| {label} | {r['sr']:.2f} [{lo:.2f},{hi:.2f}] | {pn:.0%} | "
            f"{r['ret']:+.1%} (~{ann:+.0%}/yr) | {r['dd']:.1%} | {r['trades']} | "
            f"{r['expectancy']:+.2%} | {r['win_rate']:.0%} |")


def slice_stats(net: pd.Series, lo_i: int, hi_i: int):
    from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of
    seg = net.iloc[lo_i:hi_i]
    lo, hi, pn = bootstrap_sharpe_ci(seg)
    return (f"SR {sharpe_of(seg):.2f} [{lo:.2f},{hi:.2f}] P0 {pn:.0%} "
            f"ret {seg.sum():+.1%} dd {max_dd_of(seg):.1%}")


def main():
    print("# Backtest health check — PSAR champion (live-probe strategy)\n")

    # ---------- 12m continuous ----------
    print("Loading saved_data_12m ...", flush=True)
    panel, fund_cost, in_univ = prep("saved_data_12m")
    print(f"panel: {len(panel.pairs)} pairs, {len(panel.index)} bars, "
          f"{panel.index[0].date()} → {panel.index[-1].date()}", flush=True)
    signals = build_signals("psar", panel)
    print(f"psar signals on {len(signals)} pairs; running champion sim ...", flush=True)

    r = sim(panel, fund_cost, in_univ, signals)
    net = r["net"]
    days = len(net) / 24
    print("\n## Headline (champion config, 12m, 8 slots @1x, +2bps slip)\n")
    print("| Variant | SR [90% CI] | P0 | Ret | DD | Trades | Expct | Win |")
    print("|---|---|---|---|---|---|---|---|")
    print(fmt(r, "psar champion (rerun)"), flush=True)
    print("documented        | SR [2.90,3.53] | 0% | ~+835%/yr | −7.9% | | | |")

    half = len(net) // 2
    print("\n## Halves\n")
    print(f"- H1 ({net.index[0].date()} → {net.index[half-1].date()}): {slice_stats(net, 0, half)}")
    print(f"- H2 ({net.index[half].date()} → {net.index[-1].date()}): {slice_stats(net, half, len(net))}")
    last90 = int(90 * 24)
    print(f"- Last 90d ({net.index[-last90].date()} → {net.index[-1].date()}): "
          f"{slice_stats(net, len(net) - last90, len(net))}")

    print("\n## Slippage stress (+45 bps/side extra)\n")
    r45 = sim(panel, fund_cost, in_univ, signals, extra_bps=45.0)
    print(fmt(r45, "champion @45bps extra"), "(documented: daily CI [1.83,2.47], P0 0%)", flush=True)

    print("\n## Direction split\n")
    rl = sim(panel, fund_cost, in_univ, signals, direction="long")
    rs = sim(panel, fund_cost, in_univ, signals, direction="short")
    print(fmt(rl, "long only"))
    print(fmt(rs, "short only"), flush=True)

    print("\n## PSAR neighborhood (af0 × afmax, corners + center)\n")
    for af0, afmax in ((0.01, 0.1), (0.01, 0.4), (0.04, 0.1), (0.04, 0.4), (0.02, 0.2)):
        sig = psar_signals_panel(panel, af0=af0, afmax=afmax)
        rn = sim(panel, fund_cost, in_univ, sig)
        lo, hi, pn = rn["ci"]
        print(f"- af0 {af0} afmax {afmax}: SR {rn['sr']:.2f} [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} ret {rn['ret']:+.1%} dd {rn['dd']:.1%}", flush=True)

    print("\n## Monthly returns (1x gross, net of costs)\n")
    monthly = net.groupby(net.index.to_period("M")).sum()
    for m, v in monthly.items():
        print(f"- {m}: {v:+.1%}")
    pos_months = int((monthly > 0).sum())
    print(f"- positive months: {pos_months}/{len(monthly)}")

    # exit-reason mix (with 8-slot champion run r)
    reasons = {}
    for t in r["trades_list"]:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    print("\n## Exit mix\n")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"- {k}: {v}")

    # ---------- recency: freshest data ----------
    print("\nLoading saved_data_live (freshest ~91d) ...", flush=True)
    panel2, fund2, univ2 = prep("saved_data_live")
    print(f"panel: {len(panel2.pairs)} pairs, {panel2.index[0].date()} → "
          f"{panel2.index[-1].date()} (funding cache missing there → funding=0)", flush=True)
    sig2 = build_signals("psar", panel2)
    r2 = sim(panel2, fund2, univ2, sig2)
    print("\n## Recency check (last ~91 days, champion config)\n")
    print(fmt(r2, "psar champion (fresh data)"), flush=True)


if __name__ == "__main__":
    main()
