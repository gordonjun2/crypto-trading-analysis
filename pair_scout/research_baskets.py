"""Research: cross-sectional basket strategies (N longs vs M shorts).

Tests the documented crypto cross-sectional effects on OUR universe (top-30
liquid Binance perps) with 2:1-style unbalanced baskets allowed:

- momentum factor: long trailing winners / short losers (Liu-Tsyvinski-Wu 2022:
  1-4 week formation, 1 week hold)
- short-term reversal: long losers / short winners (documented for small/
  illiquid coins; the liquid-coin literature expects the opposite sign here —
  a falsification check)

Mechanics: rebalance every R days at 00:00 UTC using strictly-prior returns,
dollar-neutral 50/50 gross, equal weight within each side, fees = 7 bps per
side on turnover, REAL funding on every leg, walk-forward 22 folds, bootstrap
CIs. Everything decision-time safe.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pair_scout.config import load_config
from pair_scout.data.funding import funding_rate_series, load_funding
from pair_scout.data.loader import Panel, load_panel
from pair_scout.research_sizing import (
    bootstrap_sharpe_ci,
    max_dd_of,
    sharpe_of,
)

logger = logging.getLogger(__name__)

FEE = 7e-4  # 5 bps + 2 bps slippage, per side, on traded notional


def simulate_basket(
    panel: Panel,
    funding: dict[str, pd.Series],
    fold_start_i: int,
    fold_end_i: int,
    formation_days: int,
    rebalance_days: int,
    n_long: int,
    n_short: int,
    sign: int = +1,  # +1 momentum (long winners), -1 reversal (long losers)
    beta_match: bool = True,
    vol_adjust: bool = False,
    buffer_mult: int | None = None,
    regime: str | None = None,  # None | "btc90": trade only when BTC 90d return > 0
) -> tuple[pd.Series, float, float, int]:
    """Simulate one fold; returns (hourly net returns, fees, carry, n_rebalances)."""
    idx = panel.index
    closes = panel.closes
    log_c = np.log(closes)
    test_idx = idx[fold_start_i:fold_end_i]
    bpd = panel.bars_per_day

    # hourly simple returns for PnL; per-leg beta vs BTC recomputed at each
    # rebalance over a trailing 14d window (causal)
    rets = closes.pct_change()
    btc_ret = rets[BTC] if BTC in rets.columns else rets.iloc[:, 0]

    # regime conditioner: BTC trailing 90d return, causal
    btc_ret90 = None
    if regime == "btc90":
        btc_ret90 = (
            closes[BTC] if BTC in closes.columns else closes.iloc[:, 0]
        ).pct_change(90 * bpd)

    reb_stamps = list(range(fold_start_i, fold_end_i, rebalance_days * bpd))
    weights = pd.Series(0.0, index=panel.pairs)
    net = pd.Series(0.0, index=test_idx)
    fees_total = 0.0
    carry_total = 0.0
    n_reb = 0
    prev_w = weights.copy()
    prev_longs: list[str] = []
    prev_shorts: list[str] = []

    # precompute per-leg funding cost series on the test window
    fund_cost = pd.DataFrame(
        {sym: funding_rate_series(sym, funding, test_idx) for sym in panel.pairs}
    ).fillna(0.0)

    for pos, start_i in enumerate(reb_stamps):
        end_i = min(
            fold_end_i,
            reb_stamps[pos + 1] if pos + 1 < len(reb_stamps) else fold_end_i,
        )
        block = idx[start_i:end_i]
        if len(block) < 2:
            continue
        # regime gate: flat while BTC 90d return is negative (prior data only)
        if btc_ret90 is not None:
            r90 = btc_ret90.iloc[start_i - 1]
            if not np.isfinite(r90) or r90 <= 0:
                turnover = float(prev_w.abs().sum())
                fee = turnover * FEE
                fees_total += fee
                if len(net.loc[block]) > 0:
                    net.iloc[net.index.get_loc(block[0])] -= fee
                prev_w = pd.Series(0.0, index=panel.pairs)
                prev_longs, prev_shorts = [], []
                continue
        # ranks from strictly-prior data (last completed formation window)
        hist_end = start_i  # bars strictly before this stamp
        hist_start = hist_end - formation_days * bpd
        if hist_start < 0:
            continue
        mom = closes.iloc[hist_end - 1] / closes.iloc[hist_start] - 1.0
        if vol_adjust:
            vol = rets.iloc[hist_start:hist_end].std() * np.sqrt(bpd)
            mom = mom / vol.replace(0.0, np.nan)
        mom = mom.dropna()
        if len(mom) < n_long + n_short:
            continue
        ranked = mom.sort_values(ascending=False)

        def pick(prev, side):
            """Rank picks with optional hysteresis: keep prior names while they
            stay inside a wider band (buffer_mult x top), fill the rest with the
            best new names."""
            long_side_wanted = (side == "long") == (sign > 0)
            keep_n = n_long if side == "long" else n_short
            if buffer_mult is None:
                band_n = keep_n
            else:
                band_n = keep_n * buffer_mult
            band = list(ranked.index[:band_n]) if long_side_wanted else list(
                ranked.index[-band_n:]
            )
            kept = [s for s in prev if s in band]
            for s in band:
                if len(kept) >= keep_n:
                    break
                if s not in kept:
                    kept.append(s)
            return kept[:keep_n]

        longs = pick(prev_longs, "long")
        shorts = pick(prev_shorts, "short")
        prev_longs, prev_shorts = longs, shorts

        # beta-match the two sides against BTC (scale short side to long beta)
        w_long = pd.Series(0.5 / max(len(longs), 1), index=longs)
        w_short = pd.Series(-0.5 / max(len(shorts), 1), index=shorts)
        if beta_match:
            h = slice(max(hist_end - 14 * bpd, 0), hist_end)
            def port_beta(syms, w):
                b = 0.0
                for s in syms:
                    cov = rets[s].iloc[h].cov(btc_ret.iloc[h])
                    var = btc_ret.iloc[h].var()
                    beta = cov / var if var and var > 0 else 0.0
                    b += w[s] * beta
                return b
            bl = port_beta(longs, w_long.abs())
            bs = port_beta(shorts, w_short.abs())
            if np.isfinite(bl) and np.isfinite(bs) and bs != 0:
                scale = abs(bl / bs)
                scale = float(np.clip(scale, 0.5, 2.0))  # sanity cap
                w_short = w_short * scale

        w_new = pd.Series(0.0, index=panel.pairs)
        w_new[longs] = w_long
        w_new[shorts] = w_short

        # hourly PnL over the block at target weights; fees on weight changes
        block_rets = rets.loc[block, panel.pairs].fillna(0.0)
        gross = (block_rets * w_new).sum(axis=1)
        carry_block = -(fund_cost.loc[block] * w_new.reindex(fund_cost.columns)).sum(axis=1)
        turnover = float((w_new - prev_w).abs().sum())
        fee = turnover * FEE
        fees_total += fee
        carry_total += float(carry_block.sum())
        net.loc[block] = gross + carry_block
        if len(net.loc[block]) > 0:
            net.iloc[net.index.get_loc(block[0])] -= fee
        prev_w = w_new
        n_reb += 1

    return net, fees_total, carry_total, n_reb


BTC = "BTCUSDT"


def run_config(cfg, panel: Panel, funding, formation_days, rebalance_days,
               n_long, n_short, sign, beta_match=True, vol_adjust=False,
               buffer_mult=None, regime=None):
    folds = []
    t0 = cfg.eval.train_days
    while t0 + cfg.eval.test_days <= panel.days:
        folds.append((t0 * panel.bars_per_day, (t0 + cfg.eval.test_days) * panel.bars_per_day))
        t0 += cfg.eval.roll_days

    ports, fees_all, carry_all, reb = [], 0.0, 0.0, 0
    folds_positive = 0
    fold_profile = []
    for fs, fe in folds:
        net, fees, carry, n_reb = simulate_basket(
            panel, funding, fs, fe, formation_days, rebalance_days,
            n_long, n_short, sign, beta_match, vol_adjust, buffer_mult, regime,
        )
        ports.append(net)
        fees_all += fees
        carry_all += carry
        reb += n_reb
        total = float(net.sum())
        fold_profile.append(total)
        folds_positive += 1 if total > 0 else 0
    port = pd.concat(ports).sort_index()
    lo, hi, pn = bootstrap_sharpe_ci(port)
    return {
        "label": (
            f"{'mom' if sign > 0 else 'rev'} f{formation_days}d r{rebalance_days}d "
            f"{n_long}L/{n_short}S{' bm' if beta_match else ''}"
            f"{' voladj' if vol_adjust else ''}"
            f"{' buf' + str(buffer_mult) if buffer_mult else ''}"
            f"{' btc90' if regime else ''}"
        ),
        "port_sharpe": sharpe_of(port),
        "ci": (lo, hi, pn),
        "ret": float(port.sum()),
        "dd": max_dd_of(port),
        "fees": fees_all,
        "carry": carry_all,
        "folds_positive": folds_positive,
        "n_folds": len(folds),
        "fold_profile": fold_profile,
        "rebalances": reb,
        "portfolio_returns": port,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair_scout.research_baskets")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="pair_scout/output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval, data_dir=cfg.data.data_dir,
        top_n_volume=cfg.data.top_n_volume, min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    funding = load_funding(Path(cfg.data.data_dir) / "funding_rates.json")
    logger.info("panel %d days, %d pairs | funding %d symbols",
                panel.days, len(panel.pairs), len(funding))

    shapes = [(5, 5), (3, 3), (2, 1), (3, 1), (4, 2)]
    combos = []
    # documented-effect tests
    for n_l, n_s in shapes:
        combos.append((7, 7, n_l, n_s, +1))    # weekly momentum (Liu et al.)
        combos.append((14, 7, n_l, n_s, +1))   # 2-week formation momentum
        combos.append((1, 1, n_l, n_s, -1))    # daily reversal (expect weak in liquid)

    results = []
    for form, reb, n_l, n_s, sign in combos:
        r = run_config(cfg, panel, funding, form, reb, n_l, n_s, sign)
        results.append(r)
        logger.info("%s -> portSR %.2f ret %+.1f%%", r["label"], r["port_sharpe"], r["ret"] * 100)

    results.sort(key=lambda r: -(r["port_sharpe"] if np.isfinite(r["port_sharpe"]) else -9))
    lines = [
        "# Cross-sectional basket strategies (N longs vs M shorts, dollar-neutral,",
        "# beta-matched sides, weekly/daily rebalance, fees 7bps/side, real funding)",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"Universe: top-{len(panel.pairs)} liquid perps, 12 months, 22 walk-forward folds",
        "",
        "| Variant | Port SR [90% CI] | P(SR<=0) | Ret | maxDD | Fees | Carry | Folds+ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lo, hi, pn = r["ci"]
        lines.append(
            f"| {r['label']} | {r['port_sharpe']:.2f} [{lo:.2f},{hi:.2f}] | "
            f"{pn:.0%} | {r['ret']:+.1%} | {r['dd']:.1%} | {r['fees']:.1%} | "
            f"{r['carry']:+.1%} | {r['folds_positive']}/{r['n_folds']} |"
        )
    lines += [
        "",
        "Notes: 'mom' = long winners/short losers (Liu-Tsyvinski-Wu 1-4w momentum);",
        "'rev' = long losers/short winners (short-term reversal, documented in",
        "small/illiquid coins — the liquid-coin literature predicts the opposite",
        "sign here, making this a falsification check). Gross 1x, sides",
        "beta-matched vs BTC with a [0.5, 2.0] scale cap.",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"baskets_{stamp}.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
