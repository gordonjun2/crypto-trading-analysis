"""Research: dispersion-scaled position sizing vs the fixed 1x baseline.

Walk-forward over the same 10 rolling folds as the reference evaluation. Per fold,
every screened candidate is simulated ONCE unscaled with gross/cost decomposition;
any sizing variant is then applied exactly (net = (gross - cost) * scale), so the
whole variant grid costs near-nothing beyond the one pass.

Reported per variant (top-3 rule-ranked books per fold, net of 5bps + 2bps):
- pooled mean OOS Sharpe of the top-3 books (reference scoreboard convention)
- folds positive / precision@3 / trades
- portfolio level: top-3 books equal-weighted within a fold, concatenated across
  folds -> pooled portfolio Sharpe, max drawdown, total return, mean exposure
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pair_scout.backtest.evaluate import _simulate_divergence_result
from pair_scout.config import load_config
from pair_scout.data.loader import Panel, load_panel, slice_days
from pair_scout.jev.ranker import rule_based_assessment
from pair_scout.screen import screen_candidates
from pair_scout.sizing import (
    ScaleSpec,
    conviction_scale_series,
    efficiency_scale_series,
    expand_to_bars,
    gates_scale_series,
    market_daily_metrics,
    market_scale_series,
    vol_target_scale_series,
)

logger = logging.getLogger(__name__)

BARS_PER_YEAR = 8760


def make_scale_fn(spec: ScaleSpec, scale_bars: pd.Series | None, panel: Panel,
                  z_entry: float, z_lookback_bars: int):
    """Return f(sim) -> per-bar multiplier series (or None for baseline)."""
    if spec.mode == "off":
        return None
    if spec.mode == "market":
        return lambda sim: scale_bars
    if spec.mode == "jev_regime_soft":
        return lambda sim: scale_bars
    if spec.mode == "vol_target":
        return lambda sim: expand_to_bars(
            vol_target_scale_series(sim.bar_returns, panel, spec),
            sim.bar_returns.index,
        )
    if spec.mode == "efficiency":
        return lambda sim: (
            expand_to_bars(
                efficiency_scale_series(sim.gross_returns, panel, spec),
                sim.bar_returns.index,
            )
        )
    if spec.mode == "conviction":
        def conv(sim):
            if sim.signal is None:
                return None
            raw = conviction_scale_series(sim.signal, z_entry, z_lookback_bars)
            scaled = (raw - 1.0) * spec.slope + 1.0
            return scaled.clip(spec.scale_min, spec.scale_max).fillna(1.0)
        return conv
    if spec.mode == "combo":
        def combo(sim):
            parts = []
            if scale_bars is not None:
                parts.append(scale_bars)
            daily = vol_target_scale_series(sim.bar_returns, panel, spec)
            parts.append(expand_to_bars(daily, sim.bar_returns.index))
            def _mul(acc, nxt):
                return (acc.reindex(nxt.index).fillna(1.0)
                        * nxt.reindex(acc.index).fillna(1.0)).clip(
                            spec.scale_min, spec.scale_max)
            out = parts[0]
            for p in parts[1:]:
                out = _mul(out, p)
            return out
        return combo
    raise ValueError(f"unknown mode {spec.mode}")


def weekend_entry_mask(panel: Panel) -> pd.Series:
    """True where the bar's UTC day is a weekday (entries skipped on Sat/Sun)."""
    dow = panel.index.dayofweek
    return pd.Series(dow < 5, index=panel.index)


def funding_tilt_per_fold(
    folds: list[FoldData], funding: dict[str, pd.Series], coef: float = 300.0,
    lo: float = 0.7, hi: float = 1.3,
) -> dict[int, dict[str, float]]:
    """Carry tilt per book: recent (train-window) funding of short leg minus long.

    A short leg with NEGATIVE recent funding means shorts PAY — tilt the book
    down; positive short funding (carry collected) tilts up. Constant within a
    fold (train-time information only).
    """
    tilts: dict[int, dict[str, float]] = {}
    for fd in folds:
        fold_tilts = {}
        for key, _score, _sim, c in fd.sims:
            f_s = _recent_mean_funding(funding, c.asset_short, fd.test_start, 7)
            f_l = _recent_mean_funding(funding, c.asset_long, fd.test_start, 7)
            if f_s is None or f_l is None:
                fold_tilts[key] = 1.0
            else:
                fold_tilts[key] = float(min(max(1.0 + coef * (f_s - f_l), lo), hi))
        tilts[fd.fold] = fold_tilts
    return tilts


def _recent_mean_funding(funding, symbol, asof: pd.Timestamp, days: int):
    series = funding.get(symbol)
    if series is None or series.empty:
        return None
    window = series.loc[:asof - pd.Timedelta(nanoseconds=1)].tail(days * 3)
    if window.empty:
        return None
    return float(window.mean())


def port_vol_target_scales(
    folds: list[FoldData], target: float, lo: float, hi: float,
) -> dict[int, dict[str, float]]:
    """Portfolio-level vol targeting: one multiplier per fold, causal.

    Fold t's multiplier = clip(target / annualized daily vol of the pooled
    top-3 book over all PRIOR folds' test days, lo, hi); fold 1 = 1x (no
    history yet). Sizes the whole portfolio down after volatile stretches.
    """
    scales: dict[int, dict[str, float]] = {}
    history: list[pd.Series] = []
    for fd in folds:
        if history:
            past = pd.concat(history).sort_index()
            daily = past.groupby(past.index.floor("D")).sum()
            vol = float(daily.tail(20).std() * np.sqrt(365))
            mult = float(min(max(target / vol, lo), hi)) if vol and np.isfinite(vol) and vol > 0 else 1.0
        else:
            mult = 1.0
        sims = sorted(fd.sims, key=lambda t: t[1], reverse=True)[:3]
        books = []
        for key, _score, sim, _c in sims:
            books.append(scaled_net(sim, None))
            scales.setdefault(fd.fold, {})[key] = mult
        if books:
            port = pd.concat(
                [b[b.index >= fd.test_start] for b in books], axis=1
            ).mean(axis=1).fillna(0.0)
            history.append(port)
    return scales


BOOK_QUESTION = None  # built lazily (typesafe_sdk import guarded)


def jev_book_scales(
    cfg, folds: list[FoldData], funding: dict | None, gates: dict[str, float],
    cache_path: Path, lo: float = 0.6, hi: float = 1.4,
) -> tuple[dict[int, dict[str, float]], str]:
    """SECOND JEV use: per-book short-horizon quality classifier as SOFT size.

    Different question from the (non-replicating) ranking call: at each fold's
    decision time, classify P(this specific book profits over the next 1-2
    days) on a 0-3 scale from the *current* setup (entry momentum, gap, vol,
    funding carry, regime). The score maps linearly to a size in [lo, hi] —
    it never selects or rejects, only sizes.
    """
    from typesafe_sdk import Noul, Score

    question = {
        "book_quality": Score(criteria=[
            {
                "what": "0 — Poor setup: spread momentum is fading or negative, the "
                "gap is stale, funding drag eats the edge, or the regime read is "
                "hostile. Loss over the next 1-2 days is likely.",
                "examples": ["signal momentum 0% with spread +40% already in place"],
            },
            {
                "what": "1 — Weak: momentum barely positive, elevated vol_ratio, or "
                "carry slightly negative; roughly a coin flip after costs.",
                "examples": ["signal momentum +0.2%/day, vol_ratio 2.8"],
            },
            {
                "what": "2 — Good: positive spread momentum with a fresh gap, sane "
                "vol, carry neutral-to-positive, regime mixed-to-favorable.",
                "examples": ["signal momentum +2%/day, gap +18%, regime P 0.55"],
            },
            {
                "what": "3 — Excellent: accelerating momentum, wide fresh gap, low "
                "correlation legs, positive carry, favorable regime.",
                "examples": ["signal momentum +4%/day accelerating, carry +1bp/8h, regime P 0.7"],
            },
        ]),
        "book_red_flag": Noul(criteria={
            "true": "Something here should cut the size hard: momentum flip imminent, "
            "one leg dominating risk, extreme negative short-leg funding, or a "
            "hostile regime read.",
            "false": "No red flag specific to the next 1-2 days.",
        }),
    }

    cache: dict[str, float] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    todo = []
    for fd in folds:
        regime_p = None
        for d in range(1, 4):
            regime_p = gates.get(str((fd.test_start - pd.Timedelta(days=d)).date()))
            if regime_p is not None:
                break
        for _key, _score, _sim, c in fd.sims[: cfg.jev.max_candidates]:
            ck = f"{c.asset_long}|{c.asset_short}|{c.train_end}"
            if ck not in cache:
                todo.append((fd.fold, c, regime_p, ck))

    error = None
    if todo:
        try:
            from pair_scout.jev.client import JevClient

            client = JevClient(cfg)

            def score_one(args):
                _fold, c, regime_p, ck = args
                f_s = _recent_mean_funding(funding or {}, c.asset_short,
                                           pd.Timestamp(c.train_end) + pd.Timedelta(days=1), 7)
                f_l = _recent_mean_funding(funding or {}, c.asset_long,
                                           pd.Timestamp(c.train_end) + pd.Timedelta(days=1), 7)
                state = {
                    "pair": {
                        "asset_long": c.asset_long,
                        "asset_short": c.asset_short,
                        "strategy": "long the stronger token, short the weaker, "
                        "beta-balanced to BTC; held up to 1-2 days",
                    },
                    "metrics": {
                        "momentum_spread_rank_window": round(c.momentum_spread, 4),
                        "signal_momentum_per_day": round(
                            (c.signal_momentum or 0.0) * 24, 5
                        ),
                        "return_correlation": round(c.return_correlation, 2),
                        "vol_ratio": round(c.vol_ratio, 2),
                        "combo_beta": round(c.combo_beta, 3),
                        "annualized_vol_long": round(c.annualized_vol_long, 2),
                        "annualized_vol_short": round(c.annualized_vol_short, 2),
                        "funding_carry_8h_short_minus_long_bps": round(
                            ((f_s or 0.0) - (f_l or 0.0)) * 1e4, 2
                        ),
                        "regime_p_momentum": regime_p,
                    },
                    "metric_reference": {
                        "signal_momentum_per_day": "per-day drift of the spread "
                        "(log long minus log short) over the 2-day signal window; "
                        "the trade is entered only if this is positive and elevated",
                        "momentum_spread_rank_window": "7-day return gap between the legs",
                        "funding_carry_8h_short_minus_long_bps": "perpetual funding "
                        "differential per 8h in bps; positive = the book COLLECTS "
                        "carry while held",
                        "regime_p_momentum": "model probability (0-1) that current "
                        "market conditions favor cross-sectional momentum books",
                        "vol_ratio / combo_beta / correlation": "risk structure of "
                        "the book; combo_beta ~0 means market-neutral",
                    },
                }
                response = client._client.system_one(
                    state=state, questions=question
                )
                score = float(response.answers["book_quality"].score) / 3.0
                flag = float(response.answers["book_red_flag"].noul)
                return ck, score, flag

            with ThreadPoolExecutor(max_workers=cfg.jev.max_workers) as pool:
                for ck, score, flag in pool.map(score_one, todo):
                    cache[ck] = score * (0.5 if flag >= 0.6 else 1.0)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
        except Exception as exc:  # noqa: BLE001
            error = f"JEV book classifier unavailable ({type(exc).__name__}: {exc})"
            logger.warning(error)

    scales: dict[int, dict[str, float]] = {}
    n = 0
    for fd in folds:
        for _key, _score, _sim, c in fd.sims:
            ck = f"{c.asset_long}|{c.asset_short}|{c.train_end}"
            if ck in cache:
                comp = min(max(cache[ck], 0.0), 1.0)
                scales.setdefault(fd.fold, {})[_key] = lo + (hi - lo) * comp
                n += 1
    note = error or f"JEV book sizing active for {n} books"
    return scales, note


@dataclass
class FoldData:
    fold: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    sims: list = field(default_factory=list)  # (key, rule_score, SimResult, candidate)

    @property
    def rule_scores(self) -> dict[str, float]:
        return {key: score for key, score, _, _ in self.sims}


def collect_fold_data(cfg, panel: Panel) -> list[FoldData]:
    funding: dict[str, pd.Series] | None = None
    if cfg.backtest.include_funding:
        from pair_scout.data.funding import load_funding
        from pathlib import Path as _P

        funding = load_funding(_P(cfg.data.data_dir) / "funding_rates.json")
        logger.info("funding loaded: %d symbols", len(funding or {}))
    folds = []
    t0 = cfg.eval.train_days
    fold_no = 0
    while t0 + cfg.eval.test_days <= panel.days:
        fold_no += 1
        t1 = t0 + cfg.eval.test_days
        train = slice_days(panel, 0, t0)
        test = slice_days(panel, t0, t1)
        screen = screen_candidates(train, cfg)
        data = FoldData(fold=fold_no, test_start=test.index[0], test_end=test.index[-1])
        if screen.passed:
            warmup_bars = max(
                int(cfg.backtest.z_lookback_days * panel.bars_per_day),
                int(cfg.backtest.signal_window_days * panel.bars_per_day)
                + int(cfg.backtest.momentum_shift_hours * panel.bars_per_day / 24)
                + 5,
            )
            warmup_closes = {
                p: train.frames[p]["Close"].tail(warmup_bars) for p in train.pairs
            }
            for c in screen.passed[:20]:
                sim = _simulate_divergence_result(
                    test, c, "vol_balanced", cfg, warmup_closes, funding=funding
                )
                data.sims.append(
                    (f"LONG {c.asset_long} / SHORT {c.asset_short}",
                     rule_based_assessment(c).composite, sim, c)
                )
        folds.append(data)
        logger.info(
            "fold %d: %d candidates (%s -> %s)",
            fold_no, len(data.sims), data.test_start.date(), data.test_end.date(),
        )
        t0 += cfg.eval.roll_days
    return folds


def scaled_net(sim, scale: pd.Series | None) -> pd.Series:
    net = (sim.gross_returns - sim.cost_returns).dropna()
    if getattr(sim, "funding_returns", None) is not None:
        net = net - sim.funding_returns.reindex(net.index).fillna(0.0)
    if scale is None:
        return net
    s = scale.reindex(net.index).fillna(1.0).astype(float)
    return net * s


def sharpe_of(net: pd.Series) -> float:
    std = float(net.std())
    return float(net.mean() / std * np.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0


def bootstrap_sharpe_ci(
    daily: pd.Series, n_boot: int = 2000, block_days: int = 5, seed: int = 7
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for annualized Sharpe + P(SR > 0).

    Works on daily returns of the pooled portfolio; blocks of `block_days`
    preserve the short-horizon autocorrelation that hourly sizing induces.
    """
    rng = np.random.default_rng(seed)
    x = daily.dropna().to_numpy()
    n = len(x)
    if n < block_days * 5:
        return float("nan"), float("nan"), float("nan")
    n_blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, n - block_days + 1, size=(n_boot, n_blocks))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        sample = np.concatenate(
            [x[s:s + block_days] for s in starts[b]][: n // block_days + 1]
        )[:n]
        sd = sample.std()
        sharpes[b] = sample.mean() / sd * np.sqrt(365) if sd > 0 else 0.0
    lo, hi = np.percentile(sharpes, [5.0, 95.0])
    return float(lo), float(hi), float((sharpes <= 0).mean())


def max_dd_of(net: pd.Series) -> float:
    equity = net.cumsum()
    return float((equity - equity.cummax()).min()) if len(equity) else 0.0


def evaluate_variant(
    label: str,
    scale_fn,
    folds: list[FoldData],
    jev_scores: dict[int, dict[str, float]] | None = None,
    book_scales: dict[int, dict[str, float]] | None = None,
    entry_mask: pd.Series | None = None,
    clip_bounds: tuple[float, float] | None = None,
    top_k: int = 3,
) -> dict:
    fold_sharpes, precs, trades_total, exposures = [], [], 0, []
    portfolio_parts = []
    fold_profile = []
    for fd in folds:
        scores = (jev_scores or {}).get(fd.fold, fd.rule_scores)
        sims = sorted(fd.sims, key=lambda t: scores.get(t[0], t[1]), reverse=True)[:top_k]
        if not sims:
            fold_sharpes.append(float("nan"))
            precs.append(float("nan"))
            fold_profile.append(float("nan"))
            continue
        sharpes, book_nets = [], []
        pos_count = 0
        for key, score, sim, _c in sims:
            book_scale = scale_fn(sim) if scale_fn is not None else None
            if book_scales is not None:
                b = book_scales.get(fd.fold, {}).get(key, 1.0)
                if book_scale is None:
                    book_scale = pd.Series(b, index=sim.bar_returns.index)
                else:
                    book_scale = book_scale * b
                if clip_bounds is not None:
                    book_scale = book_scale.clip(*clip_bounds)
            if entry_mask is not None:
                if book_scale is None:
                    book_scale = pd.Series(1.0, index=sim.bar_returns.index)
                mask = entry_mask.reindex(sim.bar_returns.index).fillna(False)
                book_scale = book_scale.mask(~mask.astype(bool), 0.0)
            net = scaled_net(sim, book_scale)
            sharpes.append(sharpe_of(net))
            book_nets.append(net)
            exposures.append(
                float(book_scale.reindex(net.index).fillna(1.0).mean())
                if book_scale is not None else 1.0
            )
            trades_total += len(sim.trades)
            if float(net.sum()) > 0:
                pos_count += 1
        fold_sharpes.append(float(np.mean(sharpes)))
        precs.append(pos_count / len(sims))
        fold_profile.append(float(np.mean(sharpes)))
        # portfolio: trim to test window, equal weight
        books = [b[b.index >= fd.test_start] for b in book_nets]
        port = pd.concat(books, axis=1).mean(axis=1).fillna(0.0)
        portfolio_parts.append(port)
    pooled = float(np.nanmean([s for s in fold_sharpes if np.isfinite(s)])) \
        if any(np.isfinite(s) for s in fold_sharpes) else float("nan")
    port_all = pd.concat(portfolio_parts).sort_index() if portfolio_parts else pd.Series(dtype=float)
    port_sharpe = sharpe_of(port_all) if len(port_all) else float("nan")
    port_dd = max_dd_of(port_all) if len(port_all) else float("nan")
    port_ret = float(port_all.sum()) if len(port_all) else float("nan")
    return {
        "label": label,
        "pooled_top3_sharpe": pooled,
        "portfolio_sharpe": port_sharpe,
        "portfolio_total_return": port_ret,
        "portfolio_max_dd": port_dd,
        "precision_at_3": float(np.nanmean(precs)) if precs else float("nan"),
        "folds_positive": sum(1 for s in fold_profile if np.isfinite(s) and s > 0),
        "trades": trades_total,
        "mean_exposure": float(np.mean(exposures)) if exposures else 1.0,
        "fold_sharpes": fold_profile,
        "portfolio_returns": port_all,
    }


def build_variants() -> list[ScaleSpec]:
    return [
        ScaleSpec(mode="off"),
        # market-level (user's original hypothesis + chop variant)
        ScaleSpec(mode="market", metric="disp_corr"),
        ScaleSpec(mode="market", metric="chop", scale_min=0.4, scale_max=1.6),
        ScaleSpec(mode="market", metric="chop", scale_min=0.3, scale_max=1.7),
        # book-level continuous sizing
        ScaleSpec(mode="vol_target", vol_target=0.20, scale_min=0.4, scale_max=1.6),
        ScaleSpec(mode="vol_target", vol_target=0.20, scale_min=0.3, scale_max=1.7),
        ScaleSpec(mode="vol_target", vol_target=0.30, scale_min=0.4, scale_max=1.6),
        ScaleSpec(mode="vol_target", vol_target=0.15, scale_min=0.5, scale_max=1.5),
        ScaleSpec(mode="efficiency"),
        ScaleSpec(mode="efficiency", scale_min=0.4, scale_max=1.6, slope=0.4),
        ScaleSpec(mode="conviction"),
        ScaleSpec(mode="conviction", scale_min=0.6, scale_max=1.4, slope=0.2),
        # JEV regime probability as a soft continuous size (cached daily gates)
        ScaleSpec(mode="jev_regime_soft", scale_min=0.4, scale_max=1.6),
        ScaleSpec(mode="jev_regime_soft", scale_min=0.3, scale_max=1.7),
        # combos (winner family) + param-neighborhood sensitivity
        ScaleSpec(mode="combo", vol_target=0.20, scale_min=0.4, scale_max=1.6),
        ScaleSpec(mode="combo", vol_target=0.25, scale_min=0.5, scale_max=1.5),
        ScaleSpec(mode="combo", vol_target=0.20, scale_min=0.5, scale_max=1.5),
        ScaleSpec(mode="combo", vol_target=0.25, scale_min=0.4, scale_max=1.6),
        ScaleSpec(mode="combo", vol_target=0.30, scale_min=0.5, scale_max=1.5),
        ScaleSpec(mode="combo", vol_target=0.25, scale_min=0.45, scale_max=1.55),
    ]


def diagnostics(folds: list[FoldData], metrics: pd.DataFrame, panel: Panel) -> str:
    """Correlate daily pooled baseline book PnL with the market-state metrics."""
    books = []
    for fd in folds:
        sims = sorted(fd.sims, key=lambda t: t[1], reverse=True)[:3]
        for _, _, sim, _c in sims:
            books.append(scaled_net(sim, None))
    if not books:
        return "no books"
    daily = pd.concat(books).groupby(lambda ix: ix.floor("D")).sum().sort_index()
    m = metrics.reindex(daily.index)
    z_disp = m["dispersion"]
    z_corr = -m["avg_corr"]
    z_chop = -m["chop"]
    out = [
        "## Diagnostics: daily baseline book PnL vs market state (train-lagged)",
        "",
        f"- corr(PnL, dispersion): {daily.corr(z_disp):+.3f}",
        f"- corr(PnL, -avg_corr):  {daily.corr(z_corr):+.3f}",
        f"- corr(PnL, -chop):      {daily.corr(z_chop):+.3f}",
        "",
        "Per-fold mean market metrics (test windows):",
        "",
        "| Fold | dispersion | avg_corr | chop | baseline top-3 SR |",
        "|---|---|---|---|---|",
    ]
    for fd in folds:
        msk = (metrics.index >= fd.test_start.floor("D") - pd.Timedelta(days=1)) & (
            metrics.index <= fd.test_end.floor("D") - pd.Timedelta(days=1)
        )
        mm = metrics[msk]
        sims = sorted(fd.sims, key=lambda t: t[1], reverse=True)[:3]
        sr = np.mean([sharpe_of(scaled_net(sim, None)) for _, _, sim, _ in sims]) if sims else float("nan")
        out.append(
            f"| {fd.fold} | {mm['dispersion'].mean():.3f} | "
            f"{mm['avg_corr'].mean():.2f} | {mm['chop'].mean():.1f} | {sr:+.2f} |"
        )
    return "\n".join(out)


def jev_composite_scores(
    cfg, folds: list[FoldData], cache_path: Path
) -> tuple[dict[int, dict[str, float]], str | None]:
    """Rank each fold's JEV subset (top-12 rule-ranked) by JEV composite.

    Disk-cached per (pair, train_end) so repeat runs cost nothing. Falls back to
    rule scores for any candidate the API fails on (never silent in the report).
    """
    cache: dict[str, float] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    todo = [
        (fd.fold, c)
        for fd in folds
        for _key, _score, _sim, c in fd.sims[: cfg.jev.max_candidates]
        if f"{c.asset_long}|{c.asset_short}|{c.train_end}" not in cache
    ]
    error: str | None = None
    if todo:
        try:
            from pair_scout.jev.client import JevClient
            from pair_scout.jev.ranker import jev_assessment

            client = JevClient(cfg)

            def score_one(args):
                fold, c = args
                a = jev_assessment(c, client.score_candidate(c), cfg.jev)
                return fold, c, a.composite

            with ThreadPoolExecutor(max_workers=cfg.jev.max_workers) as pool:
                for fold, c, composite in pool.map(score_one, todo):
                    cache[f"{c.asset_long}|{c.asset_short}|{c.train_end}"] = composite
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
        except Exception as exc:  # noqa: BLE001 — degrade to rule scores
            error = f"JEV ranking unavailable ({type(exc).__name__}: {exc})"
            logger.warning(error)
    scores: dict[int, dict[str, float]] = {}
    n_jev = 0
    for fd in folds:
        fold_scores = dict(fd.rule_scores)
        for _key, _score, _sim, c in fd.sims[: cfg.jev.max_candidates]:
            k = f"{c.asset_long}|{c.asset_short}|{c.train_end}"
            if k in cache:
                fold_scores[_key] = cache[k]
                n_jev += 1
        scores[fd.fold] = fold_scores
    note = error or f"JEV composite ranking active for {n_jev} candidate-folds"
    return scores, note


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair_scout.research_sizing")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="pair_scout/output")
    parser.add_argument("--jev", action="store_true",
                        help="also run the JEV-ranked arm (uses the Typesafe API)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval, data_dir=cfg.data.data_dir,
        top_n_volume=cfg.data.top_n_volume, min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    logger.info("panel: %d days", panel.days)
    metrics = market_daily_metrics(panel)
    folds = collect_fold_data(cfg, panel)

    variants = build_variants()

    # precompute the shared market-level / JEV-gate bar series
    market_bars: dict[str, pd.Series] = {}
    gate_bars: dict[str, pd.Series] = {}
    cache_path = Path("pair_scout/output/regime_cache.json")
    gates = {}
    if cache_path.exists():
        import json

        gates = json.loads(cache_path.read_text())

    results = []
    for spec in variants:
        if spec.mode == "off":
            fn, label = None, "off (baseline 1x)"
        elif spec.mode == "market":
            key = f"{spec.metric}|{spec.scale_min}|{spec.scale_max}|{spec.slope}|{spec.z_lookback_days}|{spec.disp_weight}"
            if key not in market_bars:
                daily = market_scale_series(panel, spec, metrics)
                market_bars[key] = expand_to_bars(daily, panel.index)
            fn = make_scale_fn(
                spec, market_bars[key], panel, cfg.backtest.z_entry,
                int(cfg.backtest.z_lookback_days * panel.bars_per_day),
            )
            label = spec.label
        elif spec.mode in ("jev_regime_soft", "combo") :
            key = f"{spec.scale_min}|{spec.scale_max}|{spec.vol_target}"
            if key not in gate_bars:
                gate_bars[key] = expand_to_bars(
                    gates_scale_series(gates, panel.index, spec), panel.index
                ) if gates else None
            fn = make_scale_fn(
                spec, gate_bars[key], panel, cfg.backtest.z_entry,
                int(cfg.backtest.z_lookback_days * panel.bars_per_day),
            )
            label = spec.label
        else:
            fn = make_scale_fn(
                spec, None, panel, cfg.backtest.z_entry,
                int(cfg.backtest.z_lookback_days * panel.bars_per_day),
            )
            label = spec.label
        results.append(evaluate_variant(label, fn, folds))

    jev_note = None
    if args.jev:
        jev_scores, jev_note = jev_composite_scores(
            cfg, folds, Path("pair_scout/output/jev_composite_cache.json")
        )
        for spec in (
            ScaleSpec(mode="off"),
            ScaleSpec(mode="vol_target", vol_target=0.20, scale_min=0.4, scale_max=1.6),
            ScaleSpec(mode="vol_target", vol_target=0.25, scale_min=0.5, scale_max=1.5),
            ScaleSpec(mode="combo", vol_target=0.20, scale_min=0.4, scale_max=1.6),
            ScaleSpec(mode="combo", vol_target=0.25, scale_min=0.5, scale_max=1.5),
        ):
            if spec.mode == "off":
                fn, label = None, "JEV rank + off (baseline)"
            elif spec.mode == "combo":
                key = f"{spec.scale_min}|{spec.scale_max}|{spec.vol_target}"
                fn = make_scale_fn(
                    spec, gate_bars.get(key), panel, cfg.backtest.z_entry,
                    int(cfg.backtest.z_lookback_days * panel.bars_per_day),
                )
                label = f"JEV rank + {spec.label}"
            else:
                fn = make_scale_fn(
                    spec, None, panel, cfg.backtest.z_entry,
                    int(cfg.backtest.z_lookback_days * panel.bars_per_day),
                )
                label = f"JEV rank + {spec.label}"
            results.append(evaluate_variant(label, fn, folds, jev_scores))

    # --- iteration 2: ideas grounded in the funding-adjusted baseline ---
    def combo_fn(spec: ScaleSpec):
        key = f"{spec.scale_min}|{spec.scale_max}|{spec.vol_target}"
        bars = gate_bars.get(key)
        if bars is None and gates:
            bars = expand_to_bars(gates_scale_series(gates, panel.index, spec),
                                  panel.index)
            gate_bars[key] = bars
        return make_scale_fn(
            spec, bars, panel, cfg.backtest.z_entry,
            int(cfg.backtest.z_lookback_days * panel.bars_per_day),
        )

    z_lb = int(cfg.backtest.z_lookback_days * panel.bars_per_day)
    winner = ScaleSpec(mode="combo", vol_target=0.20, scale_min=0.5, scale_max=1.5)
    winner_fn = combo_fn(winner)
    results.append(evaluate_variant("combo(jevxvt20%)[0.5,1.5] (prev winner)",
                                    winner_fn, folds))

    from pair_scout.data.funding import load_funding
    from pathlib import Path as _P

    funding = (
        load_funding(_P(cfg.data.data_dir) / "funding_rates.json")
        if cfg.backtest.include_funding else {}
    )

    # portfolio-level vol targeting (causal: fold t uses vol of folds < t)
    pvt = port_vol_target_scales(folds, 0.20, 0.5, 1.5)
    results.append(evaluate_variant("port_vt(20%)[0.5,1.5]", None, folds,
                                    book_scales=pvt))
    results.append(evaluate_variant("combo + port_vt(20%)", winner_fn, folds,
                                    book_scales=pvt, clip_bounds=(0.5, 1.5)))

    # carry tilt: recent short-leg funding minus long-leg (train window only)
    if funding:
        for coef in (150.0, 300.0):
            tilt = funding_tilt_per_fold(folds, funding, coef=coef)
            results.append(evaluate_variant(
                f"funding tilt k{coef:g} (baseline)", None, folds,
                book_scales=tilt,
            ))
            results.append(evaluate_variant(
                f"combo + funding tilt k{coef:g}", winner_fn, folds,
                book_scales=tilt, clip_bounds=(0.5, 1.5),
            ))

    # weekend-flat: no exposure Sat/Sun UTC
    results.append(evaluate_variant(
        "combo + weekend flat", winner_fn, folds,
        entry_mask=weekend_entry_mask(panel),
    ))

    # JEV per-book quality classifier as soft size (second, independent JEV use)
    global jev_book_note
    jev_book_note = None
    if args.jev and gates:
        bscales, jev_book_note = jev_book_scales(
            cfg, folds, funding, gates,
            Path("pair_scout/output/jev_book_cache.json"),
        )
        results.append(evaluate_variant("JEV book size [0.6,1.4]", None, folds,
                                        book_scales=bscales))
        results.append(evaluate_variant("combo + JEV book size", winner_fn, folds,
                                        book_scales=bscales, clip_bounds=(0.5, 1.5)))

    results.sort(
        key=lambda r: -(r["portfolio_sharpe"] if np.isfinite(r["portfolio_sharpe"]) else -9)
    )
    lines = [
        "# Dispersion-scaled sizing research (10 walk-forward folds, net of fees)"
        + (" + funding" if cfg.backtest.include_funding else ""),
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        "Top-3 rule-ranked books per fold; sizing applied continuously (no gates).",
        "Portfolio = top-3 books equal-weighted within a fold, chained across folds.",
        "",
        "| Variant | Pooled top-3 SR | Port SR [90% CI] | P(SR<=0) | Port Ret | Port maxDD | P@3 | Folds+ | Trades | Exp | Fold SRs |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        fs = "/".join(f"{s:.1f}" for s in r["fold_sharpes"])
        lo, hi, p_neg = bootstrap_sharpe_ci(r["portfolio_returns"])
        ci = f"[{lo:.2f},{hi:.2f}]" if np.isfinite(lo) else "—"
        lines.append(
            f"| {r['label']} | {r['pooled_top3_sharpe']:.2f} | "
            f"{r['portfolio_sharpe']:.2f} {ci} | {p_neg:.0%} | "
            f"{r['portfolio_total_return']:+.1%} | "
            f"{r['portfolio_max_dd']:.1%} | {r['precision_at_3']:.2f} | "
            f"{r['folds_positive']}/10 | {r['trades']} | {r['mean_exposure']:.2f}x "
            f"| {fs} |"
        )
    lines += ["", diagnostics(folds, metrics, panel), ""]
    if jev_note:
        lines += [f"JEV ranking note: {jev_note}", ""]
    if globals().get("jev_book_note"):
        lines += [f"JEV book-classifier note: {jev_book_note}", ""]
    lines += [
        "Idea-count honesty: this run evaluates ~30 sizing/idea variants; the",
        "best variant's edge should be discounted for multiple comparisons",
        "(deflated-Sharpe logic). Neighborhood robustness is reported alongside.",
        "",
    ]
    report = "\n".join(lines)
    print(report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"sizing_research_{stamp}.md"
    out_file.write_text(report, encoding="utf-8")
    logger.info("report written to %s", out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
