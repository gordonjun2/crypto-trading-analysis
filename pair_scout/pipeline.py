"""Pipeline orchestration: load → screen → JEV rank → report → deliver (plan §3)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

from pair_scout.backtest.evaluate import run_evaluation
from pair_scout.config import AppConfig
from pair_scout.data.loader import Panel, load_panel
from pair_scout.jev.client import JevClient, JevError
from pair_scout.jev.ranker import Assessment, jev_assessment, rank, rule_based_assessment
from pair_scout.report import build_report
from pair_scout.screen import DEFAULT_BENCHMARK, ScreenOutput, screen_candidates

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    panel: Panel
    screen: ScreenOutput
    ranked: list[Assessment]
    jev_used: bool
    jev_degraded_reason: str | None
    report_chunks: list[str]
    evaluation: object | None = None  # EvaluationReport when evaluate() ran


def _jev_score_subset(
    client: JevClient, cfg: AppConfig, candidates: list
) -> list[Assessment]:
    subset = candidates[: cfg.jev.max_candidates]
    logger.info("JEV scoring %d candidate(s), %d worker(s)", len(subset), cfg.jev.max_workers)

    def score_one(c):
        return jev_assessment(c, client.score_candidate(c), cfg.jev)

    with ThreadPoolExecutor(max_workers=cfg.jev.max_workers) as pool:
        scored = list(pool.map(score_one, subset))
    if len(subset) < len(candidates):
        logger.info(
            "%d candidate(s) beyond max_candidates=%d not JEV-scored (rule-ranked)",
            len(candidates) - len(subset), cfg.jev.max_candidates,
        )
        scored.extend(rule_based_assessment(c) for c in candidates[len(subset):])
    return scored


def run_pipeline(cfg: AppConfig, use_jev: bool = True) -> RunResult:
    panel = load_panel(
        cex=cfg.data.cex,
        interval=cfg.data.interval,
        data_dir=cfg.data.data_dir,
        top_n_volume=cfg.data.top_n_volume,
        min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    screen = screen_candidates(panel, cfg, benchmark=DEFAULT_BENCHMARK)
    pool = screen.passed + screen.watch

    jev_used = False
    degraded: str | None = None
    if use_jev and cfg.jev.enabled:
        try:
            client = JevClient(cfg)
            assessments = _jev_score_subset(client, cfg, pool)
            jev_used = True
        except JevError as exc:
            logger.warning("JEV unavailable (%s) — degraded rule-based mode", exc)
            degraded = str(exc)
            assessments = [rule_based_assessment(c) for c in pool]
    else:
        assessments = [rule_based_assessment(c) for c in pool]

    ranked = rank(assessments, top_k=cfg.jev.top_k)
    chunks = build_report(
        now=datetime.now(timezone.utc),
        universe_count=len(panel.pairs),
        scanned_count=len(screen.candidates),
        passed_count=len(screen.passed),
        scored_count=sum(1 for a in ranked if a.jev_used),
        ranked=ranked,
        rejected_summary=screen.rejected_summary,
        jev_used=jev_used,
        watch_count=len(screen.watch),
        jev_degraded_reason=degraded,
        entry_z=cfg.backtest.entry_z,
        stop_z=cfg.backtest.stop_z,
    )
    # surface the watch tier count in the log; report lines cover the details
    logger.info(
        "pipeline: %d passed, %d watch-tier, %d ENTER/WATCH listed",
        len(screen.passed), len(screen.watch),
        sum(1 for a in ranked if a.action in ("ENTER", "WATCH")),
    )
    return RunResult(
        panel=panel,
        screen=screen,
        ranked=ranked,
        jev_used=jev_used,
        jev_degraded_reason=degraded,
        report_chunks=chunks,
    )


def run_evaluation_pipeline(cfg: AppConfig, use_jev: bool = True):
    panel = load_panel(
        cex=cfg.data.cex,
        interval=cfg.data.interval,
        data_dir=cfg.data.data_dir,
        top_n_volume=cfg.data.top_n_volume,
        min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    return run_evaluation(cfg, panel, use_jev=use_jev)
