"""Candidate generation: all combinations, analyses, features, hard filters (plan §3)."""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field

from pair_scout.analysis import beta as beta_mod
from pair_scout.analysis.cointegration import EGResult, engle_granger
from pair_scout.analysis.correlation import divergence_direction
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


def _eg_to_candidate(
    panel: Panel,
    cfg: AppConfig,
    eg: EGResult,
    vols: dict,
    betas: dict,
    vold: dict,
    history_bars: int,
    train_end: str,
) -> PairCandidate | None:
    """Map an EG orientation to long/short assets via the single direction rule (B12)."""
    z = eg.zscore
    if not math.isfinite(z):
        return None
    # spread = log(Y) - beta*log(X); z>0 -> Y rich -> short Y long X; z<0 -> mirror.
    if z > 0:
        long_asset, short_asset = eg.asset_x, eg.asset_y
    else:
        long_asset, short_asset = eg.asset_y, eg.asset_x

    vol_long = vols[long_asset]
    vol_short = vols[short_asset]
    div = divergence_direction(
        long_asset,
        short_asset,
        panel.frames[long_asset]["Close"],
        panel.frames[short_asset]["Close"],
        window_bars=int(cfg.screen.corr_lookback_days * panel.bars_per_day),
        momentum_lookback_bars=int(cfg.screen.momentum_lookback_days * panel.bars_per_day),
    )
    half_life_days = eg.half_life_bars / panel.bars_per_day
    c = PairCandidate(
        asset_long=long_asset,
        asset_short=short_asset,
        direction_basis=(
            f"sign of rolling spread z-score; spread = log({eg.asset_y}) - "
            f"{eg.hedge_ratio:.4f}*log({eg.asset_x}) (train window)"
        ),
        spread_y_asset=eg.asset_y,
        spread_x_asset=eg.asset_x,
        pvalue=eg.pvalue,
        adf_stat=eg.adf_stat,
        hedge_ratio=eg.hedge_ratio,
        spread_zscore=z,
        half_life_bars=eg.half_life_bars,
        half_life_days=half_life_days,
        hurst=eg.hurst,
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
        beta_long=betas.get(long_asset, float("nan")),
        beta_short=betas.get(short_asset, float("nan")),
        history_bars=history_bars,
        history_days=history_bars // panel.bars_per_day,
        train_end=train_end,
    )
    return c


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

    for a, b in itertools.combinations(pairs, 2):
        try:
            eg = engle_granger(
                a, b, closes[a], closes[b], zscore_window=cfg.screen.zscore_window_bars
            )
        except ValueError as exc:
            logger.debug("skipping %s/%s: %s", a, b, exc)
            continue
        candidate = _eg_to_candidate(
            panel, cfg, eg, vols, betas, vold, history_bars, train_end
        )
        if candidate is None:
            continue
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
