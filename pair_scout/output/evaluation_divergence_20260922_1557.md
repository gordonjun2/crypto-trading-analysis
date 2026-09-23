# PairScout walk-forward evaluation

Generated: 2026-09-22 15:57 UTC

## Per-fold results

| Fold | Mode | Arm | Sizing | Tested | Mean Sharpe (top-K) | Precision@K | Best pick |
|---|---|---|---|---|---|---|---|
| 1 | divergence | 1-notebook | equal | 6 | 3.51 | 1.00 | LONG UNIUSDT / SHORT LTCUSDT (SR 2.66) |
| 1 | divergence | 2-consolidated | vol_balanced | 114 | -1.26 | 0.00 | LONG LINKUSDT / SHORT MKRUSDT (SR -0.72) |
| 1 | divergence | 2-consolidated | equal | 114 | -1.26 | 0.00 | LONG LINKUSDT / SHORT MKRUSDT (SR -0.72) |
| 1 | divergence | 3-consolidated+jev | vol_balanced | 12 | -2.39 | 0.00 | LONG LINKUSDT / SHORT BCHUSDT (SR -2.77) |
| 2 | divergence | 1-notebook | equal | 6 | 2.93 | 1.00 | LONG CRVUSDT / SHORT LINKUSDT (SR 2.96) |
| 2 | divergence | 2-consolidated | vol_balanced | 205 | -5.01 | 0.00 | LONG SKLUSDT / SHORT CRVUSDT (SR -4.55) |
| 2 | divergence | 2-consolidated | equal | 205 | -5.01 | 0.00 | LONG SKLUSDT / SHORT CRVUSDT (SR -4.55) |
| 2 | divergence | 3-consolidated+jev | vol_balanced | 12 | -5.01 | 0.00 | LONG SKLUSDT / SHORT CRVUSDT (SR -4.55) |

## Ranking quality (pooled across folds)

- Spearman(score, test Sharpe) — 1-notebook [divergence]: 0.238
- Spearman(score, test Sharpe) — 2-consolidated [divergence]: -0.283
- Spearman(score, test Sharpe) — 3-consolidated+jev [divergence]: -0.699

## Limitation

~62 days, one regime, one exchange, funding rates of the short perp leg not modeled, small candidate count per fold. Results are directional evidence about ranking usefulness, **not** proof of profitability.
