"""Research v2: price-action entries + hedge, POINT-IN-TIME universe.

Fixes the survivorship bias of v1: the panel used to be the top-30 pairs by
trailing volume at the END of the sample — i.e. last year's winners (ZEC
32x, BR 14x). A breakout strategy "catching" those is circular. Here the
universe at any bar is the top-30 pairs by trailing 30-day dollar volume,
computed causally from all 657 pairs.

Signals (price action, `ta` package, 4h bars):
  donchian: close > prior 20d high (entry), close < prior 10d low (exit)
  ema:      EMA5d/EMA20d cross up (entry), cross down (exit)
  rsi:      RSI14 exits oversold (entry), RSI > 70 (exit)
Portfolio: 3 slots x 1/3 book, max hold 7d, 5% adverse stop. Hedges:
  weak = short the 7d-momentum laggard in-universe; btc = short BTC;
  none = long-only. Continuous 12 months, next-bar fills, 7 bps/side,
  real funding. Reports per-trade expectancy at 1x trade notional (the
  cleanest TA-quality metric) plus portfolio equity at 1x gross.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

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
UNIV_TOP = 30
UNIV_VOL_DAYS = 30
MOM_DAYS = 7


@dataclass
class Trade:
    long_sym: str
    short_sym: str
    entry_i: int
    exit_i: int = -1
    entry_px: float = float("nan")
    exit_px: float = float("nan")
    beta: float = 1.0
    reason: str = ""


def build_signals(closes_1h: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Boolean 'long active' frames (1h index) for every pair; events are
    cross-ups (False->True)."""
    out = {}
    c4 = closes_1h.resample("4h").last()
    for sym in closes_1h.columns:
        c = c4[sym].dropna()
        if len(c) < 400:
            continue
        if strategy == "donchian":
            hi_n = c.rolling(20 * 6).max()
            lo_exit = c.rolling(10 * 6).min()
            active = c > hi_n.shift(1)
            inactive = c < lo_exit.shift(1)
        elif strategy == "ema":
            fast = EMAIndicator(c, window=30).ema_indicator()
            slow = EMAIndicator(c, window=120).ema_indicator()
            above = (fast > slow).fillna(False)
            active = above
            inactive = ~above
        elif strategy == "rsi":
            rsi = RSIIndicator(c, window=14).rsi()
            was_os = (rsi < 30).rolling(10).max().fillna(0).astype(bool)
            active = (was_os & (rsi >= 30)).fillna(False)
            inactive = (rsi > 70).fillna(False)
        else:
            raise ValueError(strategy)
        s = active.reindex(closes_1h.index, method="ffill").fillna(False).astype(bool)
        x = inactive.reindex(closes_1h.index, method="ffill").fillna(False).astype(bool)
        out[sym] = (s, x)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair_scout.research_ta_pairs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="pair_scout/output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval, data_dir=cfg.data.data_dir,
        top_n_volume=658,  # everything: point-in-time universe is chosen later
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
    univ_rank = dollar_vol.rolling(UNIV_VOL_DAYS * 24).mean().rank(
        axis=1, ascending=False
    )
    in_universe = (univ_rank <= UNIV_TOP) & univ_rank.notna()
    trailing_mom = closes / closes.shift(int(MOM_DAYS * panel.bars_per_day)) - 1.0
    fund_cost = pd.DataFrame(
        {s: funding_rate_series(s, funding, idx) for s in panel.pairs}
    ).fillna(0.0)
    btc_ret = rets[BTC] if BTC in rets.columns else rets.iloc[:, 0]
    logger.info("all-pair panel: %d pairs, %d bars", len(panel.pairs), len(idx))

    start_i = 30 * panel.bars_per_day
    results = []
    for strategy in ("donchian", "ema", "rsi"):
        signals = build_signals(closes, strategy)
        for hedge in ("weak", "btc", "none"):
            open_trades: list[Trade] = []
            trades: list[Trade] = []
            net = pd.Series(0.0, index=idx[start_i:])
            pos_of = {t: k for k, t in enumerate(net.index)}
            for i in range(start_i, len(idx)):
                t = idx[i]
                # exits
                still = []
                for tr in open_trades:
                    s_act, x_act = signals[tr.long_sym]
                    px = closes[tr.long_sym].iloc[i]
                    reason = ""
                    if x_act.iloc[i]:
                        reason = "signal"
                    elif i - tr.entry_i >= MAX_HOLD_BARS:
                        reason = "time"
                    elif px / tr.entry_px - 1.0 <= -STOP:
                        reason = "stop"
                    if reason:
                        tr.exit_i = i
                        tr.exit_px = px
                        tr.reason = reason
                        trades.append(tr)
                    else:
                        still.append(tr)
                open_trades = still
                # entries
                if len(open_trades) < 3:
                    held = {tr.long_sym for tr in open_trades} | {
                        tr.short_sym for tr in open_trades if tr.short_sym != tr.long_sym
                    }
                    for sym in signals:
                        if len(open_trades) >= 3:
                            break
                        if sym in held or sym == BTC or not in_universe.at[t, sym]:
                            continue
                        s_act, _ = signals[sym]
                        if not (s_act.iloc[i] and not s_act.iloc[i - 1]):
                            continue
                        if hedge == "btc":
                            short_sym, beta = BTC, None
                        elif hedge == "none":
                            short_sym, beta = None, 0.0
                        else:
                            cand = trailing_mom.iloc[i - 1].dropna()
                            cand = cand[
                                cand.index.isin(
                                    set(in_universe.columns[in_universe.loc[t]])
                                )
                            ]
                            cand = cand.drop(
                                [sym] + list(held), errors="ignore"
                            )
                            if cand.empty:
                                continue
                            short_sym = cand.idxmin()
                            beta = None
                        if short_sym is None:
                            tr = Trade(long_sym=sym, short_sym=sym, entry_i=i,
                                       entry_px=float(closes[sym].iloc[i]))
                        else:
                            h = slice(i - 14 * panel.bars_per_day, i)
                            ref = rets[short_sym]
                            cov = rets[sym].iloc[h].cov(ref.iloc[h])
                            var = ref.iloc[h].var()
                            beta = float(np.clip(cov / var, 0.2, 3.0)) if var and var > 0 else 1.0
                            tr = Trade(long_sym=sym, short_sym=short_sym, entry_i=i,
                                       entry_px=float(closes[sym].iloc[i]), beta=beta)
                        open_trades.append(tr)
                        held.add(sym)
                # accrual
                bar = 0.0
                for tr in open_trades:
                    hedged = tr.short_sym != tr.long_sym
                    w_l = SLOT if hedged else SLOT
                    w_s = -SLOT * tr.beta if hedged else 0.0
                    bar += SLOT * rets[tr.long_sym].iloc[i]
                    if hedged:
                        bar += w_s * rets[tr.short_sym].iloc[i]
                        bar -= fund_cost[tr.short_sym].iloc[i] * (-w_s)
                    bar -= fund_cost[tr.long_sym].iloc[i] * w_l
                net.iloc[pos_of[t]] = bar
            for tr in open_trades:
                tr.exit_i = len(idx) - 1
                tr.exit_px = float(closes[tr.long_sym].iloc[-1])
                trades.append(tr)
            # fees spread per bar
            total_fees = sum(
                (1.0 + (abs(tr.beta) if tr.short_sym != tr.long_sym else 0.0))
                * FEE
                for tr in trades
            )
            net_net = net - total_fees / max(len(net), 1)
            # per-trade expectancy at 1x trade notional
            expct = [
                (tr.exit_px / tr.entry_px - 1.0) * (1.0 if True else 1.0)
                for tr in trades
            ]
            wins = [e for e in expct if e > 0]
            hold_days = [
                (tr.exit_i - tr.entry_i) / panel.bars_per_day for tr in trades
            ]
            lo, hi, pn = bootstrap_sharpe_ci(net_net)
            results.append({
                "label": f"{strategy} + {hedge} hedge",
                "sr": sharpe_of(net_net), "ci": (lo, hi, pn),
                "ret": float(net_net.sum()), "dd": max_dd_of(net_net),
                "trades": len(trades),
                "expectancy": float(np.mean(expct)) if expct else float("nan"),
                "win_rate": len(wins) / len(expct) if expct else float("nan"),
                "avg_hold": float(np.mean(hold_days)) if hold_days else float("nan"),
                "fees": total_fees,
            })
            logger.info(
                "%s: SR %.2f P0 %.0f%% ret %+.1f%% expectancy %.2f%% trades %d",
                results[-1]["label"], r["sr"] if (r := results[-1]) else 0,
                r["ci"][2] * 100 if (r := results[-1]) else 0,
                r["ret"] * 100 if (r := results[-1]) else 0,
                (r["expectancy"] * 100) if (r := results[-1]) else 0,
                r["trades"],
            )

    lines = [
        "# Price-action entries + hedge, POINT-IN-TIME universe (v2)",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        "Universe at each bar: top-30 by trailing 30d dollar volume (causal, from",
        "all 657 pairs). Slots: 3 x 1/3 book, <=7d hold, 5% stop. Fees 7bps/side,",
        "real funding. Expectancy = mean per-trade return at 1x trade notional.",
        "",
        "| Variant | Port SR [90% CI] | P(SR<=0) | Ret (1x book) | maxDD | Trades | Expectancy/trade | Win rate | Avg hold (d) | Fees |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lo, hi, pn = r["ci"]
        lines.append(
            f"| {r['label']} | {r['sr']:.2f} [{lo:.2f},{hi:.2f}] | {pn:.0%} | "
            f"{r['ret']:+.1%} | {r['dd']:.1%} | {r['trades']} | "
            f"{r['expectancy']:+.2%} | {r['win_rate']:.0%} | {r['avg_hold']:.1f} | "
            f"{r['fees']:.1%} |"
        )
    lines += [
        "",
        "Reading: 'none' isolates TA alpha (long-only). 'weak' vs 'btc' isolates",
        "the hedge choice. Expectancy is the signal-quality metric; portfolio",
        "return is at 1x gross (3 slots x 1/3).",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"ta_pairs_v2_{stamp}.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
