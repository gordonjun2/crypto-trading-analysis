"""Candidate generation: all combinations, analyses, features, hard filters (plan §3)."""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from pair_scout.analysis import beta as beta_mod
from pair_scout.analysis.cointegration import EGResult, engle_granger
from pair_scout.analysis.correlation import divergence_direction
from pair_scout.backtest.spread import combo_beta as combo_beta_fn, leg_weights
from pair_scout.analysis.volatility import (
    long_notional_share,
    vol_metrics,
    vol_ratio as vol_ratio_fn,
)
from pair_scout.config import AppConfig
from pair_scout.data.loader import Panel
from pair_scout.features import PairCandidate
from pair_scout.filters import CODE_LABELS, apply_filters, is_watch_tier
from pair_scout.jev.ranker import rule_based_score

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK = "BTCUSDT"


@dataclass
class ScreenOutput:
    candidates: list[PairCandidate] = field(default_factory=list)  # all combos
    passed: list[PairCandidate] = field(default_factory=list)  # filters passed
    watch: list[PairCandidate] = field(default_factory=list)  # near-miss tier
    rejected: list[PairCandidate] = field(default_factory=list)
    universe: list[str] = field(default_factory=list)

    @property
    def rejected_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.rejected:
            for code in c.filter_codes:
                label = CODE_LABELS.get(code, code)
                counts[label] = counts.get(label, 0) + 1
        return counts


def _shared_metrics(
    panel: Panel,
    cfg: AppConfig,
    long_asset: str,
    short_asset: str,
    vols: dict,
    betas: dict,
    vold: dict,
    history_bars: int,
    train_end: str,
    sizing: str = "beta_balanced",
) -> PairCandidate:
    """Build the candidate skeleton with vol/beta/sizing metrics for a given direction."""
    vol_long, vol_short = vols[long_asset], vols[short_asset]
    w_long, w_short = leg_weights(
        vol_long.annualized_vol,
        vol_short.annualized_vol,
        sizing,
        beta_y=betas.get(long_asset, float("nan")),
        beta_x=betas.get(short_asset, float("nan")),
    )
    beta_l = betas.get(long_asset, float("nan"))
    beta_s = betas.get(short_asset, float("nan"))
    div = divergence_direction(
        long_asset,
        short_asset,
        panel.frames[long_asset]["Close"],
        panel.frames[short_asset]["Close"],
        window_bars=int(cfg.screen.corr_lookback_days * panel.bars_per_day),
        momentum_lookback_bars=int(cfg.screen.momentum_lookback_days * panel.bars_per_day),
    )
    return PairCandidate(
        asset_long=long_asset,
        asset_short=short_asset,
        direction_basis="",
        strategy="cointegration",
        return_correlation=div.return_correlation,
        momentum_long=div.momentum_long_asset,
        momentum_short=div.momentum_short_asset,
        avg_daily_volume_long=vold[long_asset],
        avg_daily_volume_short=vold[short_asset],
        annualized_vol_long=vol_long.annualized_vol,
        annualized_vol_short=vol_short.annualized_vol,
        atr_pct_long=vol_long.atr_pct,
        atr_pct_short=vol_short.atr_pct,
        vol_ratio=vol_ratio_fn(vol_long, vol_short),
        skewness_long=vol_long.skewness,
        skewness_short=vol_short.skewness,
        long_notional_share=long_notional_share(vol_long, vol_short),
        choppiness_long=vol_long.choppiness,
        choppiness_short=vol_short.choppiness,
        beta_long=beta_l,
        beta_short=beta_s,
        weight_long=w_long,
        combo_beta=combo_beta_fn(beta_l, beta_s, w_long, w_short),
        history_bars=history_bars,
        history_days=history_bars // panel.bars_per_day,
        train_end=train_end,
    )


def _rank_scores(panel, cfg, closes) -> dict[str, float]:
    """Good-vs-bad ranking score per token over the rank window."""
    rank_bars = max(int(cfg.screen.momentum_rank_days * panel.bars_per_day), 2)
    scores: dict[str, float] = {}
    for p in panel.pairs:
        c = closes[p]
        mom = float(c.iloc[-1] / c.iloc[-(rank_bars + 1)] - 1.0)
        if cfg.screen.rank_score == "risk_adjusted":
            vol = float(np.log(c).diff().tail(rank_bars).std() * np.sqrt(panel.bars_per_day * 365))
            scores[p] = mom / vol if vol > 0 else 0.0
        else:
            scores[p] = mom
    return scores


def _divergence_candidates(
    panel, cfg, closes, vols, betas, vold, history_bars, train_end
):
    """Long the stronger token, short the weaker one (user goal), beta-balanced."""
    rank_bars = max(int(cfg.screen.momentum_rank_days * panel.bars_per_day), 2)
    signal_bars = max(int(cfg.backtest.signal_window_days * panel.bars_per_day), 2)
    scores = _rank_scores(panel, cfg, closes)
    mom = {}
    for p in panel.pairs:
        c = closes[p]
        mom[p] = float(c.iloc[-1] / c.iloc[-(rank_bars + 1)] - 1.0)

    book_pairs: list[tuple[str, str]] = []
    if cfg.screen.pairing == "matched":
        ranked = sorted(panel.pairs, key=lambda p: scores[p])
        depth = min(cfg.screen.matched_depth, len(ranked) // 2)
        weak_side = ranked[:depth]          # worst performers
        strong_side = ranked[-depth:][::-1]  # best performers, strongest first
        # greedy 1-1: strongest token with the weakest remaining token
        for strong, weak in zip(strong_side, weak_side):
            book_pairs.append((strong, weak))
    else:
        book_pairs = list(itertools.combinations(panel.pairs, 2))

    for strong, weak in book_pairs:
        if strong == weak:
            continue
        c = _shared_metrics(
            panel, cfg, strong, weak, vols, betas, vold, history_bars, train_end
        )
        c.strategy = "divergence"
        c.momentum_long = mom[strong]
        c.momentum_short = mom[weak]
        c.momentum_spread = mom[strong] - mom[weak]
        ll = np.log(closes[strong])
        ls = np.log(closes[weak])
        spread = ll - ls
        c.signal_momentum = float(
            (spread.iloc[-1] - spread.iloc[-(signal_bars + 1)]) / signal_bars
        )
        c.hedge_ratio = 1.0
        c.direction_basis = (
            f"ranked {cfg.screen.momentum_rank_days:.0f}d momentum "
            f"({c.momentum_long:+.1%} vs {c.momentum_short:+.1%}, "
            f"{cfg.screen.rank_score}); weights beta-balanced "
            f"(combo beta {c.combo_beta:+.2f})"
        )
        yield c


def _cointegration_candidates(
    panel, cfg, closes, vols, betas, vold, history_bars, train_end
):
    """EG-based mean-reversion candidates (original mode)."""
    for a, b in itertools.combinations(panel.pairs, 2):
        try:
            eg = engle_granger(
                a, b, closes[a], closes[b], zscore_window=cfg.screen.zscore_window_bars
            )
        except ValueError as exc:
            logger.debug("skipping %s/%s: %s", a, b, exc)
            continue
        z = eg.zscore
        if not math.isfinite(z):
            continue
        # spread = log(Y) - beta*log(X); z>0 -> Y rich -> short Y long X; z<0 -> mirror.
        if z > 0:
            long_asset, short_asset = eg.asset_x, eg.asset_y
        else:
            long_asset, short_asset = eg.asset_y, eg.asset_x
        c = _shared_metrics(
            panel, cfg, long_asset, short_asset, vols, betas, vold, history_bars, train_end
        )
        c.pvalue = eg.pvalue
        c.adf_stat = eg.adf_stat
        c.hedge_ratio = eg.hedge_ratio
        c.spread_zscore = z
        c.half_life_bars = eg.half_life_bars
        c.half_life_days = eg.half_life_bars / panel.bars_per_day
        c.hurst = eg.hurst
        c.spread_y_asset = eg.asset_y
        c.spread_x_asset = eg.asset_x
        c.direction_basis = (
            f"sign of rolling spread z-score; spread = log({eg.asset_y}) - "
            f"{eg.hedge_ratio:.4f}*log({eg.asset_x}) (train window)"
        )
        yield c


def screen_candidates(
    panel: Panel, cfg: AppConfig, benchmark: str = DEFAULT_BENCHMARK
) -> ScreenOutput:
    """Run all analyses over all C(n,2) combinations; return filtered candidates."""
    pairs = panel.pairs
    closes = panel.closes
    bars_per_year = panel.bars_per_day * 365

    vols: dict[str, object] = {}
    vold: dict[str, float] = {}
    for p in pairs:
        df = panel.frames[p]
        vols[p] = vol_metrics(df, bars_per_year)
        window = cfg.data.trailing_volume_days * panel.bars_per_day
        vold[p] = float(df["Volume"].tail(window).mean())
    betas = {}
    if benchmark in closes.columns:
        rets = beta_mod.log_returns(closes)
        bench_rets = rets[benchmark]
        betas = {p: beta_mod.beta_vs_benchmark(rets[p], bench_rets) for p in pairs}

    out = ScreenOutput(universe=pairs)
    history_bars = len(panel.index)
    train_end = str(panel.index[-1].date())

    if cfg.screen.mode == "divergence":
        generator = _divergence_candidates
    else:
        generator = _cointegration_candidates

    for candidate in generator(panel, cfg, closes, vols, betas, vold, history_bars, train_end):
        outcome = apply_filters(candidate, cfg.screen)
        candidate.filter_reasons = outcome.reasons
        candidate.filter_codes = outcome.codes
        out.candidates.append(candidate)
        if outcome.passed:
            out.passed.append(candidate)
        elif is_watch_tier(candidate, cfg.screen):
            candidate.watch_tier = True
            out.watch.append(candidate)
        else:
            out.rejected.append(candidate)

    out.passed.sort(key=lambda c: rule_based_score(c), reverse=True)
    out.watch.sort(key=lambda c: rule_based_score(c), reverse=True)
    logger.info(
        "screening: %d combos scanned, %d passed filters, %d watch-tier",
        len(out.candidates), len(out.passed), len(out.watch),
    )
    return out
