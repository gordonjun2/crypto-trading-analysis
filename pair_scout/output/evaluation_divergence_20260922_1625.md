# PairScout walk-forward evaluation

Generated: 2026-09-22 16:25 UTC

## Per-fold results

| Fold | Mode | Arm | Sizing | Tested | Mean Sharpe (top-K) | Precision@K | Best pick |
|---|---|---|---|---|---|---|---|
| 1 | divergence | 1-notebook | equal | 6 | 4.81 | 1.00 | LONG ONDOUSDT / SHORT 1000PEPEUSDT (SR 6.66) |
| 1 | divergence | 2-consolidated | vol_balanced | 120 | -2.18 | 0.33 | LONG ENAUSDT / SHORT WLDUSDT (SR 0.39) |
| 1 | divergence | 2-consolidated | equal | 120 | -1.59 | 0.67 | LONG ENAUSDT / SHORT WLDUSDT (SR 1.33) |
| 2 | divergence | 1-notebook | equal | 6 | 5.19 | 1.00 | LONG ADAUSDT / SHORT BRUSDT (SR 5.95) |
| 2 | divergence | 2-consolidated | vol_balanced | 192 | 6.14 | 1.00 | LONG ZECUSDT / SHORT ENAUSDT (SR 9.69) |
| 2 | divergence | 2-consolidated | equal | 192 | 6.32 | 1.00 | LONG ZECUSDT / SHORT ENAUSDT (SR 10.04) |
| 3 | divergence | 1-notebook | equal | 5 | 3.07 | 1.00 | LONG ADAUSDT / SHORT 1000PEPEUSDT (SR 1.25) |
| 3 | divergence | 2-consolidated | vol_balanced | 204 | -2.52 | 0.00 | LONG NEARUSDT / SHORT HYPEUSDT (SR -4.19) |
| 3 | divergence | 2-consolidated | equal | 204 | -2.47 | 0.00 | LONG NEARUSDT / SHORT HYPEUSDT (SR -4.02) |
| 4 | divergence | 1-notebook | equal | 6 | 3.01 | 1.00 | LONG BNBUSDT / SHORT PUMPUSDT (SR 1.63) |
| 4 | divergence | 2-consolidated | vol_balanced | 201 | -3.58 | 0.00 | LONG WLDUSDT / SHORT ZECUSDT (SR -0.05) |
| 4 | divergence | 2-consolidated | equal | 201 | -3.63 | 0.00 | LONG WLDUSDT / SHORT ZECUSDT (SR -0.16) |
| 5 | divergence | 1-notebook | equal | 6 | 8.05 | 1.00 | LONG 龙虾USDT / SHORT GUSDT (SR 9.07) |
| 5 | divergence | 2-consolidated | vol_balanced | 156 | 0.39 | 0.33 | LONG WLDUSDT / SHORT ONDOUSDT (SR -0.51) |
| 5 | divergence | 2-consolidated | equal | 156 | 0.43 | 0.33 | LONG WLDUSDT / SHORT ONDOUSDT (SR -0.50) |
| 6 | divergence | 1-notebook | equal | 6 | 4.91 | 1.00 | LONG PUMPUSDT / SHORT 龙虾USDT (SR 7.68) |
| 6 | divergence | 2-consolidated | vol_balanced | 206 | nan | 0.33 | LONG ZECUSDT / SHORT WLDUSDT (SR -1.10) |
| 6 | divergence | 2-consolidated | equal | 206 | nan | 0.33 | LONG ZECUSDT / SHORT WLDUSDT (SR -1.16) |
| 7 | divergence | 1-notebook | equal | 6 | 1.25 | 0.67 | LONG DOGEUSDT / SHORT WLDUSDT (SR -0.87) |
| 7 | divergence | 2-consolidated | vol_balanced | 195 | 0.19 | 0.67 | LONG PUMPUSDT / SHORT ZECUSDT (SR -2.49) |
| 7 | divergence | 2-consolidated | equal | 195 | 0.54 | 0.67 | LONG PUMPUSDT / SHORT ZECUSDT (SR -2.17) |
| 8 | divergence | 1-notebook | equal | 6 | 5.66 | 1.00 | LONG BNBUSDT / SHORT NEARUSDT (SR 7.83) |
| 8 | divergence | 2-consolidated | vol_balanced | 141 | 1.83 | 0.67 | LONG PUMPUSDT / SHORT ONDOUSDT (SR 7.78) |
| 8 | divergence | 2-consolidated | equal | 141 | 2.83 | 0.67 | LONG PUMPUSDT / SHORT ONDOUSDT (SR 8.39) |
| 9 | divergence | 1-notebook | equal | 6 | 4.15 | 1.00 | LONG ONDOUSDT / SHORT GUSDT (SR 2.18) |
| 9 | divergence | 2-consolidated | vol_balanced | 217 | nan | 0.00 | LONG PUMPUSDT / SHORT ZECUSDT (SR nan) |
| 9 | divergence | 2-consolidated | equal | 217 | nan | 0.00 | LONG PUMPUSDT / SHORT ZECUSDT (SR nan) |
| 10 | divergence | 1-notebook | equal | 6 | 4.21 | 1.00 | LONG ONEUSDT / SHORT NEARUSDT (SR 4.83) |
| 10 | divergence | 2-consolidated | vol_balanced | 247 | -0.85 | 0.67 | LONG 龙虾USDT / SHORT AKEUSDT (SR -8.27) |
| 10 | divergence | 2-consolidated | equal | 247 | -0.38 | 0.67 | LONG 龙虾USDT / SHORT AKEUSDT (SR -6.12) |

## Ranking quality (pooled across folds)

- Spearman(score, test Sharpe) — 1-notebook [divergence]: 0.445
- Spearman(score, test Sharpe) — 2-consolidated [divergence]: 0.073

## Limitation

~62 days, one regime, one exchange, funding rates of the short perp leg not modeled, small candidate count per fold. Results are directional evidence about ranking usefulness, **not** proof of profitability.
