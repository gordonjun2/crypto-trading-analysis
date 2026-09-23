"""JEV question definitions (plan §4.3) — one request per candidate, questions parallel.

Rubric levels are concrete per the Score docs (structured criteria lift confidence
when levels overlap). Weights/gates live in config, not here.
"""

from __future__ import annotations

from typesafe_sdk import Noul, Score

REVERSION_RUBRIC = [
    {
        "what": "0 — No reliable relationship: p-value near or above threshold, "
        "half-life noise-like or infinite, Hurst >= 0.5 suggests the spread trends.",
        "examples": ["p-value 0.09 with half-life 45 days and Hurst 0.6"],
    },
    {
        "what": "1 — Weak relationship: p-value passes but is marginal (0.03-0.05), "
        "half-life or Hurst are borderline, mixed evidence across metrics.",
        "examples": ["p-value 0.04, half-life 25 days, Hurst 0.49"],
    },
    {
        "what": "2 — Solid relationship: p-value clearly below threshold (<=0.02), "
        "half-life in the tradeable range (0.5-15 days), Hurst below 0.5.",
        "examples": ["p-value 0.01, half-life 8 days, Hurst 0.38"],
    },
    {
        "what": "3 — Strong relationship: p-value <= 0.01, half-life 1-10 days, "
        "Hurst well below 0.5, sane positive hedge ratio, long history backing it.",
        "examples": ["p-value 0.002, half-life 5 days, Hurst 0.31, 60+ days history"],
    },
]

ENTRY_RUBRIC = [
    {
        "what": "0 — No edge: spread z-score near the mean (|z| < 1.5); entering now "
        "pays costs for no expected reversion.",
        "examples": ["z = 0.6"],
    },
    {
        "what": "1 — Marginal entry: z just past the threshold (|z| 1.5-2.0); some "
        "expected reversion, thin margin over costs.",
        "examples": ["z = 1.7"],
    },
    {
        "what": "2 — Good entry: |z| 2.0-3.0 with reversion evidence; the spread is "
        "well outside its normal range.",
        "examples": ["z = 2.4 with half-life ~1 week"],
    },
    {
        "what": "3 — Excellent entry: |z| > 3.0 approaching stop territory but with "
        "strong stationarity evidence, or extreme dislocation on liquid legs.",
        "examples": ["z = 3.3, p-value 0.004, both legs highly liquid"],
    },
]

EXECUTABILITY_RUBRIC = [
    {
        "what": "0 — Hard to execute: illiquid legs, extreme vol_ratio or ATR%, or "
        "crash-prone skew makes fills/slippage unacceptable.",
        "examples": ["avg daily volume $2M, vol_ratio 5.5, short-leg skew -2.8"],
    },
    {
        "what": "1 — Executable with care: moderate liquidity or somewhat unbalanced "
        "leg risks; size limits and wider stops needed.",
        "examples": ["avg daily volume $40M, vol_ratio 2.4"],
    },
    {
        "what": "2 — Clean execution: both legs liquid, balanced risk, sane skew and "
        "volatility.",
        "examples": ["avg daily volume $300M+, vol_ratio 1.3, mild skews"],
    },
]

DIRECTION_NOUL = Noul(
    criteria={
        "true": "Given the metrics, the proposed long/short assignment is the "
        "consistent one (mean reversion: sign of z-score vs the spread convention "
        "log(short-leg) - hedge_ratio*log(long-leg); divergence: the long leg has "
        "the stronger momentum and the beta-balanced weights are coherent).",
        "false": "The proposed direction fights the data or cannot be determined.",
    }
)

RED_FLAG_NOUL = Noul(
    criteria={
        "true": "Something in the metrics should veto trading this pair: evidence "
        "barely below threshold, extreme volatility or vol_ratio, crash-prone skew, "
        "very short history, degenerate hedge or beta, momentum that is pure market "
        "beta instead of idiosyncratic divergence.",
        "false": "No metric alone looks disqualifying.",
    }
)

COINTEGRATION_QUESTIONS: dict[str, Score | Noul] = {
    "reversion_quality": Score(criteria=REVERSION_RUBRIC),
    "entry_attractiveness": Score(criteria=ENTRY_RUBRIC),
    "executability": Score(criteria=EXECUTABILITY_RUBRIC),
    "direction_consistent": DIRECTION_NOUL,
    "red_flag": RED_FLAG_NOUL,
}

DIVERGENCE_QUESTIONS: dict[str, Score | Noul] = {
    "reversion_quality": Score(
        criteria=[
            {
                "what": "0 — No tradeable divergence: the momentum gap is tiny or "
                "fully explained by market beta; the pair is essentially one bet on "
                "bitcoin.",
                "examples": ["momentum spread 3%, return correlation 0.97"],
            },
            {
                "what": "1 — Weak divergence: gap exists but modest, legs highly "
                "correlated, or one leg's beta is too small to balance.",
                "examples": ["momentum spread 6%, correlation 0.9"],
            },
            {
                "what": "2 — Solid divergence: clear idiosyncratic gap over the "
                "ranking window, moderate correlation, both legs hedgeable.",
                "examples": ["momentum spread 15%, correlation 0.6"],
            },
            {
                "what": "3 — Strong divergence: large gap, fresh signal momentum, "
                "low correlation between legs, clean beta balancing, liquid legs.",
                "examples": ["momentum spread 30%, correlation 0.3, signal rising"],
            },
        ]
    ),
    "entry_attractiveness": Score(
        criteria=[
            {
                "what": "0 — Late entry: the divergence already happened and is "
                "fading; signal momentum near zero or negative.",
                "examples": ["7d spread momentum ~0 after a 40% gap"],
            },
            {
                "what": "1 — Marginal: gap barely above the minimum, signal "
                "momentum flat.",
                "examples": ["spread +5.5%, 7d momentum +0.5%"],
            },
            {
                "what": "2 — Good: healthy gap with still-positive spread momentum.",
                "examples": ["spread +12%, 7d momentum +3%"],
            },
            {
                "what": "3 — Excellent: large gap, strong and accelerating signal "
                "momentum, no exhaustion signs.",
                "examples": ["spread +25%, 7d momentum +6%, accelerating"],
            },
        ]
    ),
    "executability": Score(criteria=EXECUTABILITY_RUBRIC),
    "direction_consistent": DIRECTION_NOUL,
    "red_flag": RED_FLAG_NOUL,
}

_QUESTIONS_BY_STRATEGY = {
    "cointegration": COINTEGRATION_QUESTIONS,
    "divergence": DIVERGENCE_QUESTIONS,
}

# Default kept for backward compatibility with the original mean-reversion flow.
QUESTIONS = COINTEGRATION_QUESTIONS


def build_questions(strategy: str) -> dict[str, Score | Noul]:
    """Strategy-aware question set (same IDs and types, strategy-specific rubrics)."""
    return _QUESTIONS_BY_STRATEGY.get(strategy, COINTEGRATION_QUESTIONS)
