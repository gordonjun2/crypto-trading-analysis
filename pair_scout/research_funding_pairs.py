"""Research: funding-squeeze pair strategy (long extreme-negative funding,
short extreme-positive funding, beta-neutral, hold <= 1 week).

Thesis (documented in practitioner + academic literature): funding extremes
mark crowded one-sided positioning; crowded longs pay escalating carry and are
squeeze-prone (down), crowded shorts are squeeze-prone (up). The book collects
the funding differential from BOTH legs every settlement while betting the
positioning extreme mean-reverts within days.

Causality: the funding differential driving entries uses settlements strictly
before the bar's day; execution is next-bar; PnL includes real funding on both
legs, 5 bps fee + 2 bps slippage per side per leg.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pair_scout.analysis import beta as beta_mod
from pair_scout.backtest.spread import simulate_divergence
from pair_scout.config import load_config
from pair_scout.data.funding import funding_rate_series, load_funding
from pair_scout.data.loader import Panel, load_panel, slice_days
from pair_scout.research_sizing import (
    bootstrap_sharpe_ci,
    max_dd_of,
    sharpe_of,
)

logger = logging.getLogger(__name__)
BARS_PER_YEAR = 8760
BTC = "BTCUSDT"


@dataclass
class BookSim:
    key: str
    score: float  # funding differential (higher = better)
    net: pd.Series
    funding_pnl: float
    price_pnl: float
    n_trades: int
    exposure: float


def daily_funding_diff(
    funding: dict[str, pd.Series], short_sym: str, long_sym: str,
    lookback_days: int, days_index: pd.DatetimeIndex,
) -> pd.Series:
    """Per-UTC-day funding differential (short leg minus long leg), causal.

    Day D's value = mean rate over settlements in [D - lookback, D-1] — no
    same-day information. Missing history -> NaN (no trade).
    """
    def daily_mean(sym: str) -> pd.Series:
        s = funding.get(sym)
        if s is None or s.empty:
            return pd.Series(np.nan, index=days_index)
        d = s.groupby(s.index.floor("D")).mean().sort_index()
        return d.reindex(days_index)

    f_short = daily_mean(short_sym)
    f_long = daily_mean(long_sym)
    diff = f_short - f_long
    # day D uses settlements up to D-1: shift the daily series by one day
    return diff.shift(1).rolling(lookback_days, min_periods=lookback_days).mean()


def build_books(
    fold_no: int,
    panel_train_close: pd.Timestamp,
    panel: Panel,
    funding: dict[str, pd.Series],
    lookback_days: int,
    threshold: float,
    top_k: int,
    rets=None,
    bench=None,
    beta_min: float = 0.15,
) -> list[tuple[str, str, float]]:
    """Pick top-K funding-squeeze books from the train window (decision-safe)."""
    rows = []
    for sym in panel.pairs:
        s = funding.get(sym)
        if s is None or len(s) < 60:
            continue
        past = s.loc[:panel_train_close - pd.Timedelta(nanoseconds=1)]
        window = past.tail(lookback_days * 3)
        if len(window) < max(lookback_days, 6):
            continue
        if rets is not None:
            b = beta_mod.beta_vs_benchmark(
                rets[sym].loc[:panel_train_close], bench.loc[:panel_train_close]
            )
            if not np.isfinite(b) or b < beta_min:
                continue  # unhedgeable leg: equal-weight fallback would leak beta
        rows.append((sym, float(window.mean())))
    if len(rows) < 4:
        return []
    rows.sort(key=lambda t: t[1])
    books = []
    # 1-1 matching: most negative funding (long) with most positive (short)
    lows = [r for r in rows if r[1] <= 0]
    highs = [r for r in rows if r[1] > 0]
    for (long_sym, f_l), (short_sym, f_s) in zip(lows, reversed(highs)):
        diff = f_s - f_l
        if diff >= threshold:
            books.append((long_sym, short_sym, diff))
        if len(books) >= top_k:
            break
    return books


def simulate_book(
    book: tuple[str, str, float],
    ext,  # extended panel: train tail + test (Close-only frames)
    test_start: pd.Timestamp,
    funding: dict[str, pd.Series],
    lookback_days: int,
    threshold: float,
    hold_days: int,
    beta_long: float,
    beta_short: float,
    exit_frac: float = 0.0,
    stop_pct: float = 0.0,
    last_settlement: bool = False,
) -> BookSim | None:
    long_sym, short_sym, diff_at_entry = book

    log_long_full = np.log(ext.frames[long_sym]["Close"])
    log_short_full = np.log(ext.frames[short_sym]["Close"])

    days_index = pd.DatetimeIndex(sorted(set(ext.index.floor("D"))))
    if last_settlement:
        # most recent settled differential (per settlement, expanded to bars);
        # causal: next-bar execution means the entry bar closes after the print
        def settle_series(sym: str) -> pd.Series:
            s = funding.get(sym)
            if s is None or s.empty:
                return pd.Series(np.nan, index=ext.index)
            return s.reindex(ext.index).ffill()

        signal = settle_series(short_sym) - settle_series(long_sym)
    else:
        diff_daily = daily_funding_diff(
            funding, short_sym, long_sym, lookback_days, days_index
        )
        keys = pd.Series(ext.index.floor("D"), index=ext.index)
        signal = keys.map(diff_daily).astype(float)

    f_long = funding_rate_series(long_sym, funding, ext.index)
    f_short = funding_rate_series(short_sym, funding, ext.index)

    result = simulate_divergence(
        log_long_full,
        log_short_full,
        beta_long,
        beta_short,
        sizing="beta_balanced",
        entry_mom_per_bar=threshold,
        exit_mom_per_bar=threshold * exit_frac,
        fee_bps=5.0,
        slippage_bps=2.0,
        bars_per_year=BARS_PER_YEAR,
        max_hold_bars=int(hold_days * ext.bars_per_day),
        entry_mode="absolute",
        trade_start=test_start,
        funding_long=f_long,
        funding_short=f_short,
        signal_override=signal,
        spread_stop_pct=stop_pct,
    )
    test_mask = result.bar_returns.index >= test_start
    net = result.bar_returns[test_mask]
    if len(net) == 0:
        return None
    fund = (
        result.funding_returns.reindex(net.index).fillna(0.0)
        if result.funding_returns is not None
        else pd.Series(0.0, index=net.index)
    )
    cost = (
        result.cost_returns.reindex(net.index).fillna(0.0)
        if result.cost_returns is not None
        else pd.Series(0.0, index=net.index)
    )
    carry_received = -float(fund.sum())  # engine stores funding as a COST
    fees = float(cost.sum())
    price_gross = float(net.sum()) - fees + float(fund.sum())
    positions = (
        result.positions.reindex(net.index).fillna(0).astype(int)
        if result.positions is not None
        else None
    )
    from pair_scout.backtest.spread import _segment_trades

    trades = _segment_trades(net, positions) if positions is not None else []
    return BookSim(
        key=f"LONG {long_sym} / SHORT {short_sym}",
        score=diff_at_entry,
        net=net,
        funding_pnl=carry_received,
        price_pnl=price_gross,
        n_trades=len(trades),
        exposure=1.0,
    )


def run_fold(
    fold_no: int, t0: int, t1: int, panel: Panel, cfg,
    funding: dict[str, pd.Series], lookback_days: int, threshold: float,
    hold_days: int, top_k: int, warmup_bars: int,
    exit_frac: float = 0.0, stop_pct: float = 0.0, last_settlement: bool = False,
) -> list[BookSim]:
    train = slice_days(panel, 0, t0)
    test = slice_days(panel, t0, t1)
    if not test.frames:
        return []
    # extended panel: train tail + test (for signal lookback continuity)
    tail = {p: train.frames[p]["Close"].tail(warmup_bars) for p in train.pairs}

    class ExtPanel:
        pass

    ext = ExtPanel()
    ext.bars_per_day = panel.bars_per_day
    ext.index = panel.index[(t0 * panel.bars_per_day - warmup_bars): (t1 * panel.bars_per_day)]
    test_start = test.index[0]
    ext.frames = {}
    for p in test.frames:
        ext.frames[p] = pd.DataFrame({
            "Close": pd.concat([tail[p], test.frames[p]["Close"]])
        })

    rets = beta_mod.log_returns(panel.closes)
    bench = rets[BTC] if BTC in rets.columns else rets.iloc[:, 0]
    books = build_books(
        fold_no, train.index[-1], panel, funding,
        lookback_days, threshold, top_k, rets=rets, bench=bench,
    )

    sims = []
    for long_sym, short_sym, diff in books:
        if long_sym not in ext.frames or short_sym not in ext.frames:
            continue
        bl = beta_mod.beta_vs_benchmark(rets[long_sym].loc[:train.index[-1]], bench.loc[:train.index[-1]])
        bs = beta_mod.beta_vs_benchmark(rets[short_sym].loc[:train.index[-1]], bench.loc[:train.index[-1]])
        sim = simulate_book(
            (long_sym, short_sym, diff), ext, test_start, funding,
            lookback_days, threshold, hold_days, bl, bs,
            exit_frac=exit_frac, stop_pct=stop_pct,
            last_settlement=last_settlement,
        )
        if sim is not None and len(sim.net):
            sims.append(sim)
    return sims


def evaluate(
    label: str, folds_sims: list[list[BookSim]]
) -> dict:
    sharpes, precs, ports, trades, fund_pnls, price_pnls = [], [], [], 0, [], []
    fold_profile = []
    for sims in folds_sims:
        sims = sorted(sims, key=lambda s: s.score, reverse=True)[:2]
        if not sims:
            sharpes.append(float("nan"))
            fold_profile.append(float("nan"))
            continue
        ss, books = [], []
        pos = 0
        for s in sims:
            ss.append(sharpe_of(s.net))
            books.append(s.net)
            trades += s.n_trades
            fund_pnls.append(s.funding_pnl)
            price_pnls.append(s.price_pnl)
            if float(s.net.sum()) > 0:
                pos += 1
        sharpes.append(float(np.mean(ss)))
        fold_profile.append(float(np.mean(ss)))
        precs.append(pos / len(sims))
        port = pd.concat(books, axis=1).mean(axis=1).fillna(0.0)
        ports.append(port)
    pooled = float(np.nanmean([s for s in sharpes if np.isfinite(s)])) if any(
        np.isfinite(s) for s in sharpes
    ) else float("nan")
    port_all = pd.concat(ports).sort_index() if ports else pd.Series(dtype=float)
    return {
        "label": label,
        "pooled": pooled,
        "port_sharpe": sharpe_of(port_all) if len(port_all) else float("nan"),
        "ci": bootstrap_sharpe_ci(port_all) if len(port_all) else (float("nan"),) * 3,
        "ret": float(port_all.sum()) if len(port_all) else float("nan"),
        "dd": max_dd_of(port_all) if len(port_all) else float("nan"),
        "p_at_2": float(np.nanmean(precs)) if precs else float("nan"),
        "trades": trades,
        "funding_pnl": float(np.sum(fund_pnls)),
        "price_pnl": float(np.sum(price_pnls)),
        "fold_profile": fold_profile,
        "portfolio_returns": port_all,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair_scout.research_funding_pairs")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="pair_scout/output")
    parser.add_argument("--grid", action="store_true", help="run the full small grid")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval, data_dir=cfg.data.data_dir,
        top_n_volume=cfg.data.top_n_volume, min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    funding = load_funding(Path(cfg.data.data_dir) / "funding_rates.json")
    logger.info("panel %d days | funding %d symbols", panel.days, len(funding))

    folds = []
    t0 = cfg.eval.train_days
    while t0 + cfg.eval.test_days <= panel.days:
        folds.append((1 + len(folds), t0, t0 + cfg.eval.test_days))
        t0 += cfg.eval.roll_days
    logger.info("folds: %d", len(folds))

    warmup_bars = 10 * panel.bars_per_day  # 10 days of signal lookback
    combos = (
        [(lb, th, hd) for lb in (1, 3, 7) for th in (5e-5, 1e-4) for hd in (1, 2, 4, 7)]
        if args.grid
        else [(3, 1e-4, 4)]
    )
    results = []
    for lb, th, hd in combos:
        folds_sims = [
            run_fold(no, a, b, panel, cfg, funding, lb, th, hd, 2, warmup_bars)
            for no, a, b in folds
        ]
        r = evaluate(f"lb{lb}d thr{th * 1e4:.1f}bp hold{hd}d", folds_sims)
        results.append(r)
        logger.info(
            "%s -> pooled %.2f portSR %.2f ret %+.1f%%",
            r["label"], r["pooled"], r["port_sharpe"], r["ret"] * 100,
        )

    results.sort(key=lambda r: -(r["port_sharpe"] if np.isfinite(r["port_sharpe"]) else -9))
    lines = [
        "# Funding-squeeze pairs — walk-forward (net of fees + real funding)",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        "Long the most-negative-funding coin / short the most-positive-funding",
        "coin (top-2 books by differential), beta-neutral, causal daily signal,",
        "next-bar execution, 5+2 bps costs, real funding PnL on both legs.",
        "",
        "| Variant | Pooled SR | Port SR [90% CI] | P(SR<=0) | Ret | maxDD | P@2 | Trades | Carry received | Price (gross) | Fold SRs |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lo, hi, pn = r["ci"]
        fs = "/".join(f"{s:.0f}" for s in r["fold_profile"])
        lines.append(
            f"| {r['label']} | {r['pooled']:.2f} | {r['port_sharpe']:.2f} "
            f"[{lo:.2f},{hi:.2f}] | {pn:.0%} | {r['ret']:+.1%} | {r['dd']:.1%} | "
            f"{r['p_at_2']:.2f} | {r['trades']} | {r['funding_pnl']:+.2%} | "
            f"{r['price_pnl']:+.2%} | {fs} |"
        )
    lines += [
        "",
        "Decomposition: net = price(gross) - fees + carry(received). Carry is the",
        "hypothesis; price(gross) includes fees only in `net`. engine funding",
        "series is a cost, so carry received = -sum(funding).",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"funding_pairs_{stamp}.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
