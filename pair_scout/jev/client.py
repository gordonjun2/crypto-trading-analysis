"""TypeSafe JEV client wrapper — typed answers, explicit error taxonomy, in-memory cache."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from typesafe_sdk import (
    NoulAnswer,
    RetryPolicy,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeAPIError,
    TypeSafeClient,
    TypeSafeError,
)

from pair_scout.config import AppConfig
from pair_scout.features import PairCandidate, build_jev_state
from pair_scout.jev.questions import QUESTIONS

logger = logging.getLogger(__name__)


class JevError(Exception):
    """Base class for JEV failures."""


class JevUnavailable(JevError):
    """The API is unreachable / failing after retries — triggers degraded mode."""


class JevBadAnswer(JevError):
    """The response did not contain the expected answers."""


@dataclass(frozen=True)
class JevAnswerSet:
    reversion: ScoreAnswer
    entry: ScoreAnswer
    executability: ScoreAnswer
    direction_noul: float
    red_flag_noul: float
    model: str


class JevClient:
    """Thin wrapper: builds state, sends ONE batched request per candidate, caches."""

    def __init__(self, cfg: AppConfig) -> None:
        if not cfg.typesafe_api_key:
            raise JevUnavailable(
                "TYPESAFE_API_KEY is not set — add it to .env or export it"
            )
        self._cfg = cfg
        self._client = TypeSafeClient(
            api_key=cfg.typesafe_api_key,
            model=cfg.jev.model,
            retry=RetryPolicy(max_retries=cfg.jev.max_retries),
            timeout=cfg.jev.timeout_seconds,
        )
        self._cache: dict[tuple[str, str, str], JevAnswerSet] = {}

    def score_candidate(self, c: PairCandidate) -> JevAnswerSet:
        key = (c.asset_long, c.asset_short, c.train_end)
        if key in self._cache:
            logger.debug("JEV cache hit for %s", c.key)
            return self._cache[key]
        state = build_jev_state(c)
        try:
            response: SystemOneResponse = self._client.system_one(
                state=state, questions=QUESTIONS
            )
        except TypeSafeAPIError as exc:  # rate limit/timeout/auth after SDK retries
            raise JevUnavailable(f"{type(exc).__name__}: {exc}") from exc
        except TypeSafeError as exc:
            raise JevUnavailable(f"{type(exc).__name__}: {exc}") from exc
        answers = JevClient._parse(response)
        self._cache[key] = answers
        return answers

    @staticmethod
    def _parse(response: SystemOneResponse) -> JevAnswerSet:
        a = response.answers
        try:
            rev, entry, exec_ = (
                a["reversion_quality"],
                a["entry_attractiveness"],
                a["executability"],
            )
            direction, red_flag = a["direction_consistent"], a["red_flag"]
        except KeyError as exc:
            raise JevBadAnswer(f"missing answer for {exc}") from exc
        for name, ans in (("reversion", rev), ("entry", entry), ("executability", exec_)):
            if not isinstance(ans, ScoreAnswer):
                raise JevBadAnswer(f"{name} is not a score answer")
        for name, ans in (("direction", direction), ("red_flag", red_flag)):
            if not isinstance(ans, NoulAnswer):
                raise JevBadAnswer(f"{name} is not a noul answer")
        return JevAnswerSet(
            reversion=rev,
            entry=entry,
            executability=exec_,
            direction_noul=float(direction.noul),
            red_flag_noul=float(red_flag.noul),
            model=response.model,
        )
