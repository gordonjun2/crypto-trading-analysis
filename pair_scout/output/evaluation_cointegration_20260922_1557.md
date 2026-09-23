# PairScout walk-forward evaluation

Generated: 2026-09-22 15:57 UTC

## Per-fold results

| Fold | Mode | Arm | Sizing | Tested | Mean Sharpe (top-K) | Precision@K | Best pick |
|---|---|---|---|---|---|---|---|
| 1 | cointegration | 1-notebook | equal | 6 | 3.51 | 1.00 | LONG UNIUSDT / SHORT LTCUSDT (SR 2.66) |
| 1 | cointegration | 2-consolidated | vol_balanced | 3 | -5.13 | 0.00 | LONG ATOMUSDT / SHORT FILUSDT (SR -4.84) |
| 1 | cointegration | 2-consolidated | equal | 3 | -4.83 | 0.00 | LONG ATOMUSDT / SHORT FILUSDT (SR -4.82) |
| 1 | cointegration | 3-consolidated+jev | vol_balanced | 3 | -5.13 | 0.00 | LONG XLMUSDT / SHORT BTCUSDT (SR -7.35) |
| 2 | cointegration | 1-notebook | equal | 6 | 2.93 | 1.00 | LONG CRVUSDT / SHORT LINKUSDT (SR 2.96) |
| 2 | cointegration | 2-consolidated | vol_balanced | 2 | -4.96 | 0.00 | LONG ETCUSDT / SHORT SOLUSDT (SR -5.53) |
| 2 | cointegration | 2-consolidated | equal | 2 | -6.90 | 0.00 | LONG ETCUSDT / SHORT SOLUSDT (SR -6.07) |
| 2 | cointegration | 3-consolidated+jev | vol_balanced | 2 | -4.96 | 0.00 | LONG UNIUSDT / SHORT SOLUSDT (SR -4.39) |

## Ranking quality (pooled across folds)

- Spearman(score, test Sharpe) — 1-notebook [cointegration]: 0.238
- Spearman(score, test Sharpe) — 2-consolidated [cointegration]: 0.200
- Spearman(score, test Sharpe) — 3-consolidated+jev [cointegration]: -0.300
- Cointegration persistence (train p<0.05 → test p<0.05): 3% of 33

## Limitation

~62 days, one regime, one exchange, funding rates of the short perp leg not modeled, small candidate count per fold. Results are directional evidence about ranking usefulness, **not** proof of profitability.
