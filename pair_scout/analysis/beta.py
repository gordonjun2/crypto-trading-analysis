"""Benchmark betas and an optional CVXPY beta-neutral basket optimizer.

From the beta-neutral notebooks (plan §1.3/§1.4). In v1 this module provides
secondary per-leg metrics (betas shown in JEV state / report); the basket
optimizer is available but not part of pair ranking (plan §10.3).

Fixes: explicit exception types instead of bare ``except:`` (B6), no input
mutation (B7), ``math.isclose`` weight validation (B8), convergence status
reported (M6), rolling-covariance pattern reused for OOS beta monitoring.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff().dropna(how="all")


def beta_vs_benchmark(returns: pd.Series, bench_returns: pd.Series) -> float:
    joined = pd.concat([returns, bench_returns], axis=1, keys=["r", "b"]).dropna()
    if len(joined) < 30:
        return float("nan")
    var = float(joined["b"].var())
    if var <= 0:
        return float("nan")
    cov = float(joined["r"].cov(joined["b"]))
    return cov / var


def rolling_betas(
    close: pd.DataFrame, benchmark: str, window_bars: int = 168
) -> pd.DataFrame:
    """Rolling betas of every asset vs the benchmark (rolling-covariance pattern)."""
    rets = log_returns(close)
    if benchmark not in rets.columns:
        raise KeyError(f"benchmark {benchmark} missing from panel")
    bench = rets[benchmark]
    cov = rets.rolling(window_bars).cov(bench)
    var = bench.rolling(window_bars).var()
    return cov.div(var, axis=0)


@dataclass(frozen=True)
class BasketSolution:
    weights: pd.Series
    portfolio_beta: float
    converged: bool
    status: str


def beta_neutral_weights(
    close: pd.DataFrame,
    benchmark: str,
    min_abs_weight: float = 0.05,
    max_attempts: int = 10,
    beta_tolerance: float = 1e-4,
) -> BasketSolution:
    """Min-variance beta-neutral basket (CVXPY), iterative min-weight rescaling.

    Mirrors beta-neutral-pair-trading-auto.ipynb but reports convergence status and
    uses explicit exception types (B6). Raises nothing on solver failure — the
    status is returned to the caller (M6).
    """
    try:
        import cvxpy as cp
    except ImportError as exc:  # explicit, not bare
        raise RuntimeError("cvxpy is required for the beta-neutral basket module") from exc

    rets = log_returns(close)
    if benchmark not in rets.columns:
        raise KeyError(f"benchmark {benchmark} missing from panel")
    assets = [c for c in rets.columns if c != benchmark]
    rets = rets[assets]
    sigma = rets.cov().to_numpy()
    bench_var = float(close[benchmark].pct_change().var())
    if bench_var <= 0:
        raise ValueError("degenerate benchmark variance")
    betas = np.array(
        [float(rets[a].cov(close[benchmark].pct_change())) / bench_var for a in assets]
    )

    lower = 0.0
    for attempt in range(1, max_attempts + 1):
        w = cp.Variable(len(assets))
        constraints = [
            cp.abs(w @ betas) <= beta_tolerance,
            cp.sum(w) == 1.0,
            w >= lower,
        ]
        problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), constraints)
        try:
            problem.solve()
        except cp.error.SolverError as exc:
            logger.warning("solver error on attempt %d: %s", attempt, exc)
            continue
        if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            weights = pd.Series(np.asarray(w.value).ravel(), index=assets)
            beta = float(weights @ betas)
            if not math.isclose(float(weights.abs().sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                weights = weights / weights.abs().sum()
            return BasketSolution(
                weights=weights,
                portfolio_beta=beta,
                converged=True,
                status=str(problem.status),
            )
        lower = min_abs_weight * 0.5 ** (attempt - 1)
    return BasketSolution(
        weights=pd.Series(dtype=float), portfolio_beta=float("nan"), converged=False,
        status=f"no solution after {max_attempts} attempts",
    )
