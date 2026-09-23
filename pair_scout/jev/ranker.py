"""Ranking: normalize JEV scores -> weighted composite -> gates -> actions (plan §4.4).

Also provides the transparent rule-based fallback score used in degraded mode
and in the no-JEV evaluation arm.
"""

from __future__ import annotations

from dataclasses import dataclass

from pair_scout.config import JevConfig
from pair_scout.features import PairCandidate
from pair_scout.jev.client import JevAnswerSet

MAX_LEVELS = {"reversion_quality": 3, "entry_attractiveness": 3, "executability": 2}


@dataclass(frozen=True)
class Assessment:
    candidate: PairCandidate
    action: str  # ENTER | WATCH
    composite: float  # JEV composite, or rule-based score in degraded/no-JEV mode
    confidence: float
    reason: str
    jev_used: bool


def normalize(score: float, top_level: int) -> float:
    """Divide by the rubric's top level (docs: /patterns/composite-scoring)."""
    if top_level <= 0:
        raise ValueError("top_level must be positive")
    return max(0.0, min(1.0, score / top_level))


def composite_score(answers: JevAnswerSet, cfg: JevConfig) -> float:
    rev = normalize(answers.reversion.score, MAX_LEVELS["reversion_quality"])
    entry = normalize(answers.entry.score, MAX_LEVELS["entry_attractiveness"])
    execut = normalize(answers.executability.score, MAX_LEVELS["executability"])
    return (
        cfg.weight_reversion * rev
        + cfg.weight_entry * entry
        + cfg.weight_executability * execut
    )


def _clip01(v: float) -> float:
    if v != v or v < 0:
        return 0.0
    return min(v, 1.0)


def rule_based_score(c: PairCandidate) -> float:
    """Transparent statistical ranking used when JEV is unavailable or disabled."""
    exec_score = _clip01(1.0 - (c.vol_ratio - 1.0) / 3.0)
    if c.strategy == "divergence":
        mom_score = _clip01(c.momentum_spread / 0.25)
        beta_score = _clip01(1.0 - abs(c.combo_beta) / 0.15)
        corr_score = _clip01(1.0 - c.return_correlation) if c.return_correlation == c.return_correlation else 0.0
        return 0.45 * mom_score + 0.20 * beta_score + 0.15 * corr_score + 0.20 * exec_score
    p_score = max(0.0, 1.0 - min(c.pvalue / 0.10, 1.0))
    hl = c.half_life_days
    if hl != hl or hl <= 0:  # NaN or non-positive
        hl_score = 0.0
    else:
        hl_score = max(0.0, 1.0 - abs(hl - 5.0) / 5.0)  # peak at 5 days, 0 at 0/10+
    z_score = _clip01((abs(c.spread_zscore) - 1.5) / 1.5)
    return 0.40 * p_score + 0.25 * hl_score + 0.20 * z_score + 0.15 * exec_score


def _gate_failures(a: JevAnswerSet, composite: float, cfg: JevConfig) -> list[str]:
    failures: list[str] = []
    if composite < cfg.min_composite:
        failures.append(f"composite {composite:.2f} < {cfg.min_composite}")
    if a.reversion.confidence < cfg.min_confidence:
        failures.append(f"confidence {a.reversion.confidence:.2f} < {cfg.min_confidence}")
    if a.direction_noul < cfg.min_direction_noul:
        failures.append(f"direction noul {a.direction_noul:.2f} < {cfg.min_direction_noul}")
    if a.red_flag_noul >= cfg.max_red_flag_noul:
        failures.append(f"red flag noul {a.red_flag_noul:.2f} >= {cfg.max_red_flag_noul}")
    return failures


def _top_dimension(a: JevAnswerSet) -> str:
    dims = {
        "reversion quality": normalize(a.reversion.score, MAX_LEVELS["reversion_quality"]),
        "entry": normalize(a.entry.score, MAX_LEVELS["entry_attractiveness"]),
        "executability": normalize(a.executability.score, MAX_LEVELS["executability"]),
    }
    return max(dims, key=dims.get)


def jev_assessment(c: PairCandidate, answers: JevAnswerSet, cfg: JevConfig) -> Assessment:
    composite = composite_score(answers, cfg)
    failures = _gate_failures(answers, composite, cfg)
    if failures:
        action = "WATCH"
        reason = "passes filters; JEV gate: " + "; ".join(failures)
    else:
        action = "ENTER"
        reason = f"strong {_top_dimension(answers)} + filters passed"
    if c.watch_tier:
        action = "WATCH"  # near-miss tier never auto-ENTERs (no actionable setup yet)
        reason = "near-miss: " + "; ".join(c.filter_reasons)
    return Assessment(
        candidate=c,
        action=action,
        composite=composite,
        confidence=answers.reversion.confidence,
        reason=reason,
        jev_used=True,
    )


def rule_based_assessment(c: PairCandidate, note: str | None = None) -> Assessment:
    if c.watch_tier:
        reason = "near-miss: " + "; ".join(c.filter_reasons)
    else:
        reason = note or "rule-based statistical score (JEV unavailable/disabled)"
    return Assessment(
        candidate=c,
        action="WATCH",  # rule-based scores never auto-ENTER
        composite=rule_based_score(c),
        confidence=0.0,
        reason=reason,
        jev_used=False,
    )


def rank(
    assessments: list[Assessment], top_k: int
) -> list[Assessment]:
    """ENTER block first (composite desc, capped at top_k), then WATCH."""
    ordered = sorted(
        assessments, key=lambda a: (0 if a.action == "ENTER" else 1, -a.composite, a.candidate.key)
    )
    enter_seen = 0
    result: list[Assessment] = []
    for a in ordered:
        if a.action == "ENTER":
            enter_seen += 1
            if enter_seen > top_k:
                result.append(
                    Assessment(
                        candidate=a.candidate,
                        action="WATCH",
                        composite=a.composite,
                        confidence=a.confidence,
                        reason="clears gates but outside top-K",
                        jev_used=a.jev_used,
                    )
                )
                continue
        result.append(a)
    return result
