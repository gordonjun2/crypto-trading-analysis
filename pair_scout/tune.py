"""Walk-forward parameter sweep for the divergence strategy (short-horizon mandate).

Grid (kept deliberately small to limit overfitting): momentum rank window, signal
window, entry momentum, and max holding period. Selection is by pooled OOS top-K
Sharpe across folds, net of fees — exactly the statistic the strategy lives or
dies by. All combos are reported so the sensitivity is visible.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field, replace as dc_replace
from datetime import datetime, timezone

import numpy as np

from pair_scout.backtest.evaluate import PairOutcome, _simulate_candidate
from pair_scout.config import AppConfig
from pair_scout.data.loader import Panel, slice_days
from pair_scout.jev.ranker import rule_based_assessment
from pair_scout.jev.regime import entry_allowed_series
from pair_scout.screen import screen_candidates

logger = logging.getLogger(__name__)

GRID = {
    "rank_days": (3.0, 5.0, 7.0),
    "signal_days": (2.0, 3.0),
    "entry_mom": (0.01, 0.02),  # absolute floor; entries are z-gated on top
    "hold_days": (1.0, 2.0, 3.0),
}


@dataclass
class TuneResult:
    params: dict
    pooled_topk_sharpe: float
    mean_precision_at_k: float
    total_trades: int
    fold_sharpes: list[float] = field(default_factory=list)


def _rolling_folds(cfg: AppConfig, panel: Panel) -> list[tuple[int, int]]:
    """(train_end, test_end) day pairs rolling forward until data runs out."""
    folds = []
    t0 = cfg.eval.train_days
    while t0 + cfg.eval.test_days <= panel.days:
        folds.append((t0, t0 + cfg.eval.test_days))
        t0 += cfg.eval.roll_days
    return folds


def _eval_combo(
    base_cfg: AppConfig,
    panel: Panel,
    folds: list[tuple[int, int]],
    rank_days: float,
    signal_days: float,
    entry_mom: float,
    hold_days: float,
    top_k: int,
    regime_gates: dict[str, float] | None = None,
) -> TuneResult:
    cfg = base_cfg
    cfg = dc_replace(
        cfg,
        screen=dc_replace(cfg.screen, momentum_rank_days=rank_days),
        backtest=dc_replace(
            cfg.backtest,
            signal_window_days=signal_days,
            divergence_entry_mom=entry_mom,
            max_hold_days=hold_days,
        ),
    )
    folds = _rolling_folds(cfg, panel)
    fold_sharpes: list[float] = []
    precisions: list[float] = []
    total_trades = 0
    for fold_no, (t0, t1) in enumerate(folds, 1):
        train = slice_days(panel, 0, t0)
        test = slice_days(panel, t0, t1)
        screen = screen_candidates(train, cfg)
        passed = screen.passed[:20]  # cap sims per combo/fold
        if not passed:
            fold_sharpes.append(float("nan"))
            precisions.append(float("nan"))
            continue
        warmup_bars = max(
            int(cfg.backtest.z_lookback_days * panel.bars_per_day),
            int(cfg.backtest.signal_window_days * panel.bars_per_day)
            + int(cfg.backtest.momentum_shift_hours * panel.bars_per_day / 24)
            + 5,
        )
        warmup_closes = {
            p: train.frames[p]["Close"].tail(warmup_bars) for p in train.pairs
        }
        outcomes: list[PairOutcome] = []
        allowed = (
            entry_allowed_series(test, regime_gates, cfg.jev.regime_gate_min)
            if regime_gates
            else None
        )
        for c in passed:
            outcomes.append(
                _simulate_candidate(
                    test, c, "beta_balanced", cfg, "tune",
                    rule_based_assessment(c).composite,
                    warmup_closes=warmup_closes,
                    entry_allowed=allowed,
                )
            )
        outcomes.sort(key=lambda o: o.score, reverse=True)
        top = outcomes[:top_k]
        sharpes = [o.test_sharpe for o in top if np.isfinite(o.test_sharpe)]
        fold_sharpes.append(float(np.mean(sharpes)) if sharpes else float("nan"))
        pos = [1.0 if o.test_return > 0 else 0.0 for o in top]
        precisions.append(float(np.mean(pos)) if pos else float("nan"))
        total_trades += sum(o.n_trades for o in top)
    finite = [s for s in fold_sharpes if np.isfinite(s)]
    pooled = float(np.mean(finite)) if finite else float("nan")
    precs = [p for p in precisions if np.isfinite(p)]
    mean_prec = float(np.mean(precs)) if precs else float("nan")
    return TuneResult(
        params=dict(rank_days=rank_days, signal_days=signal_days,
                    entry_mom=entry_mom, hold_days=hold_days),
        pooled_topk_sharpe=pooled,
        mean_precision_at_k=mean_prec,
        total_trades=total_trades,
        fold_sharpes=fold_sharpes,
    )


def run_tune(cfg: AppConfig, panel: Panel) -> list[TuneResult]:
    folds = _rolling_folds(cfg, panel)
    logger.info("tuning over %d folds: %s (panel has %d days)", len(folds), folds, panel.days)

    results: list[TuneResult] = []
    combos = list(itertools.product(
        GRID["rank_days"], GRID["signal_days"], GRID["entry_mom"], GRID["hold_days"]
    ))
    for i, (rd, sd, em, hd) in enumerate(combos, 1):
        result = _eval_combo(cfg, panel, folds, rd, sd, em, hd, cfg.eval.top_k)
        results.append(result)
        logger.info(
            "combo %2d/%d rank=%.0fd signal=%.0fd entry=%.0f%% hold=%.0fd -> "
            "pooled SR %.2f, precision %.2f, trades %d",
            i, len(combos), rd, sd, em * 100, hd,
            result.pooled_topk_sharpe, result.mean_precision_at_k, result.total_trades,
        )
    results.sort(key=lambda r: (-(r.pooled_topk_sharpe if np.isfinite(r.pooled_topk_sharpe) else -9),
                                -(r.mean_precision_at_k if np.isfinite(r.mean_precision_at_k) else 0)))
    return results


def results_to_markdown(results: list[TuneResult], generated_at: datetime) -> str:
    lines = [
        "# PairScout divergence parameter sweep (walk-forward, net of fees)",
        "",
        f"Generated: {generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        "Ranking metric: pooled mean OOS Sharpe of the top-K rule-ranked candidates "
        "across folds. Small grid by design; treat the winner as a starting point, "
        "not an optimum (multiple-comparison risk).",
        "",
        "| Rank | rank_d | signal_d | entry | hold_d | Pooled SR | Precision@K | Trades | Fold SRs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        p = r.params
        folds_str = "/".join(f"{s:.1f}" for s in r.fold_sharpes)
        lines.append(
            f"| {i} | {p['rank_days']:.0f} | {p['signal_days']:.0f} | "
            f"{p['entry_mom']:.0%} | {p['hold_days']:.0f} | "
            f"{r.pooled_topk_sharpe:.2f} | {r.mean_precision_at_k:.2f} | "
            f"{r.total_trades} | {folds_str} |"
        )
    return "\n".join(lines)
