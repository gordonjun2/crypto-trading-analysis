"""Exit-rule comparison for the PSAR champion: hard 7d cap vs flexible TA exits.

Variants (all keep SAR flip + 5% adverse stop):
  cap7    current champion: hard 7-day time stop
  notime  no time exit at all (pure TA exit: flip/stop only)
  loss7   at 7d, cut only if underwater; winners ride until flip/stop
  trail7  at 7d, arm a chandelier trail (3x ATR14(4h)) instead of cutting

Run from repo root:  ./venv/bin/python backtests/exit_variants.py
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
from pair_scout.research_ta_families import build_signals, universe_mask
from pair_scout.research_sizing import (
    bootstrap_sharpe_ci as bs,
    bootstrap_sharpe_ci, max_dd_of, sharpe_of,
)

FEE = 7e-4
BTC = "BTCUSDT"
SLOTS = 8
SLOT_FRAC = 1.0 / SLOTS
STOP = 0.05
CAP_DAYS = 7
TRAIL_MULT = 3.0
EXTRA_BPS = 2.0


def atr4_pct(panel):
    h = pd.DataFrame({s: panel.frames[s]["High"] for s in panel.pairs}).resample("4h").max()
    l = pd.DataFrame({s: panel.frames[s]["Low"] for s in panel.pairs}).resample("4h").min()
    c = pd.DataFrame({s: panel.frames[s]["Close"] for s in panel.pairs}).resample("4h").last()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14 * 6).mean()
    return (atr / c).reindex(panel.index).ffill()


def sim(panel, fund_cost, in_univ, signals, atrp, exit_mode, extra_bps=EXTRA_BPS):
    idx = panel.index
    closes = panel.closes
    rets = closes.pct_change()
    bpd = panel.bars_per_day
    start_i = 30 * bpd

    class T:
        __slots__ = ("sym", "d", "i0", "px0", "hh", "exit_i", "exit_px",
                     "reason", "hold_d")

    open_trades, trades = [], []
    net = pd.Series(0.0, index=idx[start_i:])
    pos_of = {t: k for k, t in enumerate(net.index)}
    exits_mix = {}

    for i in range(start_i, len(idx)):
        t = idx[i]
        still = []
        for tr in open_trades:
            la, _, sa, _ = signals[tr.sym]
            px = float(closes[tr.sym].iloc[i])
            tr.hh = max(tr.hh, px)
            reason = ""
            if tr.d > 0:
                if not bool(la.iloc[i]):
                    reason = "flip"
                elif px / tr.px0 - 1.0 <= -STOP:
                    reason = "stop"
            else:
                if not bool(sa.iloc[i]):
                    reason = "flip"
                elif px / tr.px0 - 1.0 >= STOP:
                    reason = "stop"
            hold_d = (i - tr.i0) / bpd
            if not reason:
                if exit_mode == "cap7" and hold_d >= CAP_DAYS:
                    reason = "time"
                elif exit_mode == "loss7" and hold_d >= CAP_DAYS and px / tr.px0 - 1.0 < 0:
                    reason = "time"
                elif exit_mode == "trail7" and hold_d >= CAP_DAYS:
                    a = float(atrp[tr.sym].iloc[i])
                    if np.isfinite(a) and px < tr.hh * (1 - TRAIL_MULT * a):
                        reason = "trail"
            if reason:
                exits_mix[reason] = exits_mix.get(reason, 0) + 1
                tr.exit_i, tr.exit_px, tr.reason, tr.hold_d = i, px, reason, hold_d
                trades.append(tr)
            else:
                still.append(tr)
        open_trades = still

        if len(open_trades) < SLOTS:
            held = {x.sym for x in open_trades}
            cands = []
            for sym in signals:
                if sym in held or sym == BTC:
                    continue
                if sym not in in_univ.columns or not bool(in_univ.at[t, sym]):
                    continue
                la, _, sa, _ = signals[sym]
                d = 0
                if la.iloc[i] and not la.iloc[i - 1]:
                    d = +1
                elif sa.iloc[i] and not sa.iloc[i - 1]:
                    d = -1
                if d:
                    cands.append((sym, d))
            for sym, d in cands:
                if len(open_trades) >= SLOTS:
                    break
                tr = T()
                tr.sym, tr.d, tr.i0 = sym, d, i
                tr.px0 = float(closes[sym].iloc[i])
                tr.hh = tr.px0
                open_trades.append(tr)
                held.add(sym)

        bar = 0.0
        for tr in open_trades:
            w = SLOT_FRAC * tr.d
            bar += w * rets[tr.sym].iloc[i]
            bar -= fund_cost[tr.sym].iloc[i] * w
        net.iloc[pos_of[t]] = bar

    for tr in open_trades:
        tr.exit_i, tr.exit_px, tr.reason = len(idx) - 1, float(closes[tr.sym].iloc[-1]), "open"
        tr.hold_d = (len(idx) - 1 - tr.i0) / bpd
        trades.append(tr)

    fee_rt = SLOT_FRAC * 2 * (FEE + extra_bps * 1e-4)
    net_net = net - fee_rt * len(trades) / max(len(net), 1)
    expct = [tr.d * (tr.exit_px / tr.px0 - 1.0) for tr in trades]
    wins = sum(1 for e in expct if e > 0)
    lo, hi, pn = bootstrap_sharpe_ci(net_net)
    holds = [tr.hold_d for tr in trades]
    return {
        "sr": sharpe_of(net_net), "ci": (lo, hi, pn), "ret": float(net_net.sum()),
        "dd": max_dd_of(net_net), "trades": len(trades),
        "expct": float(np.mean(expct)), "win": wins / len(expct),
        "avg_hold": float(np.mean(holds)), "mix": exits_mix, "net": net_net,
    }


def main():
    cfg = load_config(None)
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval, data_dir="saved_data_12m",
        top_n_volume=658, min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    funding = load_funding(Path("saved_data_12m") / "funding_rates.json")
    fund_cost = pd.DataFrame(
        {s: funding_rate_series(s, funding, panel.index) for s in panel.pairs}
    ).fillna(0.0)
    in_univ = universe_mask(panel, 200)
    signals = build_signals("psar", panel)
    atrp = atr4_pct(panel)
    print(f"panel {len(panel.pairs)} pairs {len(panel.index)} bars\n", flush=True)

    print("| variant | SR [90% CI] | P0 | ret | DD | trades | expct | win | avg hold | exits |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    results = {}
    for mode in ("cap7", "notime", "loss7", "trail7"):
        r = sim(panel, fund_cost, in_univ, signals, atrp, mode)
        results[mode] = r
        lo, hi, pn = r["ci"]
        days = len(r["net"]) / 24
        mix = " ".join(f"{k}:{v}" for k, v in sorted(r["mix"].items()))
        print(f"| {mode} | {r['sr']:.2f} [{lo:.2f},{hi:.2f}] | {pn:.0%} | "
              f"{r['ret']:+.1%} (~{r['ret'] * 365 / days:+.0%}/yr) | {r['dd']:.1%} | "
              f"{r['trades']} | {r['expct']:+.2%} | {r['win']:.0%} | "
              f"{r['avg_hold']:.1f}d | {mix} |", flush=True)

    # slippage stress for the best non-baseline variant
    best = max(("notime", "loss7", "trail7"), key=lambda m: results[m]["sr"])
    rb = results[best]
    half = len(rb["net"]) // 2
    h1 = rb["net"].iloc[:half]
    h2 = rb["net"].iloc[half:]
    for name, seg in (("H1", h1), ("H2", h2)):
        lo, hi, pn = bs(seg)
        print(f"{best} {name}: SR {sharpe_of(seg):.2f} [{lo:.2f},{hi:.2f}] "
              f"P0 {pn:.0%} ret {seg.sum():+.1%} dd {max_dd_of(seg):.1%}", flush=True)
    r45 = sim(panel, fund_cost, in_univ, signals, atrp, best, extra_bps=45.0)
    lo, hi, pn = r45["ci"]
    print(f"{best} @+45bps/side: SR {r45['sr']:.2f} [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} ret {r45['ret']:+.1%} dd {r45['dd']:.1%}", flush=True)


if __name__ == "__main__":
    main()
