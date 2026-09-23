"""Research v3: generalized TA signal + N hedge legs, broad point-in-time universe.

Generalizes pass 8 in three directions:
1. SHORT signals: Donchian breakdown (close < 20d low) shorts the coin, hedged
   by LONGING the N strongest coins (mirror of the long setup).
2. N hedge legs: 1 signal coin + N hedge legs (N = 0..3), dollar-neutral per
   trade, equal weight across hedge legs.
3. Broad universe: top-30 / top-100 / top-200 by trailing 30d dollar volume,
   chosen causally at every bar (point-in-time, all 657 pairs).

Unchanged bar: continuous 12 months, next-bar fills, 7 bps/side fees, real
funding where history exists (missing -> 0 carry, stated), 5% adverse stop,
7d time stop, 3 slots x 1/3 book. Reported at 1x book gross.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pair_scout.config import load_config
from pair_scout.data.funding import funding_rate_series, load_funding
from pair_scout.data.loader import load_panel
from pair_scout.research_sizing import (
    bootstrap_sharpe_ci,
    max_dd_of,
    sharpe_of,
)

logger = logging.getLogger(__name__)

FEE = 7e-4
BTC = "BTCUSDT"
SLOT = 1.0 / 3.0
MAX_HOLD_BARS = 7 * 24
STOP = 0.05
MOM_DAYS = 7


@dataclass
class Trade:
    sig_sym: str
    sig_dir: int  # +1 long signal, -1 short signal
    hedge: list = field(default_factory=list)  # [(sym, signed weight)]
    entry_i: int = 0
    exit_i: int = -1
    entry_px: float = float("nan")
    exit_px: float = float("nan")
    reason: str = ""


def donchian_signals(closes_1h: pd.DataFrame, entry_days: int = 20,
                     exit_days: int = 10) -> dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    """Per pair: (long_active, long_exit, short_active, short_exit) on 1h index."""
    out = {}
    c4 = closes_1h.resample("4h").last()
    hi = c4.rolling(entry_days * 6).max()
    lo = c4.rolling(exit_days * 6).min()
    lo_e = c4.rolling(entry_days * 6).min()
    hi_e = c4.rolling(exit_days * 6).max()
    for sym in closes_1h.columns:
        c = c4[sym].dropna()
        if len(c) < 400:
            continue
        long_act = (c > hi[sym].shift(1)).fillna(False)
        long_exit = (c < lo[sym].shift(1)).fillna(False)
        short_act = (c < lo_e[sym].shift(1)).fillna(False)
        short_exit = (c > hi_e[sym].shift(1)).fillna(False)
        for s in (long_act, long_exit, short_act, short_exit):
            pass
        out[sym] = (
            long_act.reindex(closes_1h.index, method="ffill").fillna(False).astype(bool),
            long_exit.reindex(closes_1h.index, method="ffill").fillna(False).astype(bool),
            short_act.reindex(closes_1h.index, method="ffill").fillna(False).astype(bool),
            short_exit.reindex(closes_1h.index, method="ffill").fillna(False).astype(bool),
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair_scout.research_ta_general")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="pair_scout/output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval, data_dir=cfg.data.data_dir,
        top_n_volume=658,
        min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    funding = load_funding(Path(cfg.data.data_dir) / "funding_rates.json")
    idx = panel.index
    closes = panel.closes
    rets = closes.pct_change()
    dollar_vol = pd.DataFrame(
        {s: panel.frames[s]["Volume"] * panel.frames[s]["Close"] for s in panel.pairs}
    )
    fund_cost = pd.DataFrame(
        {s: funding_rate_series(s, funding, idx) for s in panel.pairs}
    ).fillna(0.0)
    trailing_mom = closes / closes.shift(MOM_DAYS * panel.bars_per_day) - 1.0
    logger.info("all-pair panel: %d pairs, %d bars", len(panel.pairs), len(idx))

    start_i = 30 * panel.bars_per_day
    results = []
    signals = donchian_signals(closes)
    for universe_top in (30, 100, 200):
        univ_rank = dollar_vol.rolling(30 * 24).mean().rank(axis=1, ascending=False)
        in_universe = (univ_rank <= universe_top) & univ_rank.notna()
        for direction in ("long", "short"):
            for n_hedge in (0, 1, 2, 3):
                open_trades: list[Trade] = []
                trades: list[Trade] = []
                net = pd.Series(0.0, index=idx[start_i:])
                pos_of = {t: k for k, t in enumerate(net.index)}
                for i in range(start_i, len(idx)):
                    t = idx[i]
                    # exits
                    still = []
                    for tr in open_trades:
                        c_s = signals.get(tr.sig_sym)
                        if c_s is None:
                            still.append(tr)
                            continue
                        l_act, l_exit, s_act, s_exit = c_s
                        px = closes[tr.sig_sym].iloc[i]
                        if tr.sig_dir > 0:
                            reason = ("signal" if l_exit.iloc[i]
                                      else "time" if i - tr.entry_i >= MAX_HOLD_BARS
                                      else "stop" if px / tr.entry_px - 1 <= -STOP else "")
                        else:
                            reason = ("signal" if s_exit.iloc[i]
                                      else "time" if i - tr.entry_i >= MAX_HOLD_BARS
                                      else "stop" if px / tr.entry_px - 1 >= STOP else "")
                        if reason:
                            tr.exit_i, tr.exit_px, tr.reason = i, px, reason
                            trades.append(tr)
                        else:
                            still.append(tr)
                    open_trades = still
                    # entries
                    if len(open_trades) < 3:
                        held = {tr.sig_sym for tr in open_trades}
                        for tr in open_trades:
                            held |= {h[0] for h in tr.hedge}
                        for sym in signals:
                            if len(open_trades) >= 3:
                                break
                            if sym in held or sym == BTC:
                                continue
                            if sym not in in_universe.columns or not bool(
                                in_universe.at[t, sym]
                            ):
                                continue
                            l_act, _, s_act, _ = signals[sym]
                            want_dir = 0
                            if direction in ("long", "both") and (
                                l_act.iloc[i] and not l_act.iloc[i - 1]
                            ):
                                want_dir = +1
                            elif direction in ("short", "both") and (
                                s_act.iloc[i] and not s_act.iloc[i - 1]
                            ):
                                want_dir = -1
                            if want_dir == 0:
                                continue
                            mom_row = trailing_mom.iloc[i - 1].dropna()
                            mom_row = mom_row[mom_row.index.isin(
                                set(in_universe.columns[in_universe.loc[t]])
                            )]
                            mom_row = mom_row.drop([sym] + list(held), errors="ignore")
                            if len(mom_row) < n_hedge:
                                continue
                            if want_dir > 0:
                                hedge_syms = list(mom_row.sort_values().index[:n_hedge])
                                hedge_w = -0.5 / max(n_hedge, 1)
                            else:
                                hedge_syms = list(
                                    mom_row.sort_values(ascending=False).index[:n_hedge]
                                )
                                hedge_w = +0.5 / max(n_hedge, 1)
                            tr = Trade(
                                sig_sym=sym, sig_dir=want_dir, entry_i=i,
                                entry_px=float(closes[sym].iloc[i]),
                                hedge=[(h, hedge_w) for h in hedge_syms],
                            )
                            open_trades.append(tr)
                            held.add(sym)
                    # accrual
                    bar = 0.0
                    for tr in open_trades:
                        w_sig = SLOT * tr.sig_dir
                        bar += SLOT * tr.sig_dir * rets[tr.sig_sym].iloc[i]
                        bar -= fund_cost[tr.sig_sym].iloc[i] * w_sig
                        for h_sym, h_w in tr.hedge:
                            w = SLOT * h_w
                            bar += w * rets[h_sym].iloc[i]
                            bar -= fund_cost[h_sym].iloc[i] * w
                    net.iloc[pos_of[t]] = bar
                for tr in open_trades:
                    tr.exit_i = len(idx) - 1
                    tr.exit_px = float(closes[tr.sig_sym].iloc[-1])
                    trades.append(tr)
                fees = sum(
                    SLOT * (1.0 + len(tr.hedge)) * 2 * FEE for tr in trades
                )
                net_net = net - fees / max(len(net), 1)
                # per-trade expectancy at 1x trade notional (signal+hedges)
                expct = []
                for tr in trades:
                    tr_ret = (
                        tr.sig_dir * (tr.exit_px / tr.entry_px - 1.0) * 0.5
                        + sum(
                            (-np.sign(h[1]))
                            * (closes[h[0]].iloc[min(tr.exit_i, len(closes) - 1)]
                               / closes[h[0]].iloc[tr.entry_i] - 1.0)
                            * (abs(h[1]) / 0.5) * 0.5
                            for h in tr.hedge
                        )
                    )
                    expct.append(tr_ret)
                wins = [e for e in expct if e > 0]
                lo, hi_, pn = bootstrap_sharpe_ci(net_net)
                results.append({
                    "label": f"top{universe_top} {direction} {n_hedge}h",
                    "sr": sharpe_of(net_net), "ci": (lo, hi_, pn),
                    "ret": float(net_net.sum()), "dd": max_dd_of(net_net),
                    "trades": len(trades),
                    "expectancy": float(np.mean(expct)) if expct else float("nan"),
                    "win_rate": len(wins) / len(expct) if expct else float("nan"),
                    "fees": fees,
                    "signal_leg_pnl": float(
                        sum(
                            t.sig_dir * (t.exit_px / t.entry_px - 1.0) * 0.5
                            for t in trades
                        )
                    ),
                })
                r = results[-1]
                logger.info(
                    "%s: SR %.2f P0 %.0f%% ret %+.1f%% trades %d expct %.2f%%",
                    r["label"], r["sr"], r["ci"][2] * 100, r["ret"] * 100,
                    r["trades"], r["expectancy"] * 100,
                )

    results.sort(key=lambda r: -(r["sr"] if np.isfinite(r["sr"]) else -9))
    lines = [
        "# Generalized TA signals + N hedges, broad point-in-time universe (v3)",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        "Donchian 20d/10d. Long signal -> short N weakest; short signal (20d low",
        "breakdown) -> long N strongest. Dollar-neutral per trade, 3 slots x 1/3",
        "book, <=7d hold, 5% stop, 7bps/side, real funding (missing history = 0).",
        "",
        "| Variant | Port SR [90% CI] | P(SR<=0) | Ret (1x) | maxDD | Trades | Expectancy | Win | Fees |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lo, hi_, pn = r["ci"]
        lines.append(
            f"| {r['label']} | {r['sr']:.2f} [{lo:.2f},{hi_:.2f}] | {pn:.0%} | "
            f"{r['ret']:+.1%} | {r['dd']:.1%} | {r['trades']} | "
            f"{r['expectancy']:+.2%} | {r['win_rate']:.0%} | {r['fees']:.1%} |"
        )
    lines += [
        "",
        "Signal-leg PnL (0.5x per trade, unhedged component) by universe:",
    ]
    for r in sorted(results, key=lambda x: x["label"]):
        lines.append(f"- {r['label']}: signal legs {r['signal_leg_pnl']:+.1%}")
    report = "\n".join(lines)
    print(report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"ta_general_{stamp}.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
