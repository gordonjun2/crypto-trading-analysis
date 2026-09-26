# TA family scan — research conclusion (2026-09-23, pass 11)

Mandate: scan OTHER directional price-action TA families (beyond Donchian) for
≤1-week holds on Binance perps, same methodology bar as pass 8–10, and see if
anything beats the sitting champion (dual-direction Donchian 20d/10d, top-200
point-in-time, 1x gross: hourly SR 5.77–6.10, daily-block CI [0.91, 1.61],
+535%/yr, DD −31%).

## Answer: YES — Parabolic SAR flips, at slot breadth

**New champion: dual-direction PSAR (af 0.02→0.2, 4h bars), 6–8 concurrent
slots × 1/slot book (1x gross), top-200 point-in-time universe, ≤7d hold,
5% stop, flip-exit.**

| Config (12m continuous, fees 7bps/side + real funding, next-bar fills) | Hourly SR | Daily-block 90% CI | P(SR≤0) | Ret | maxDD |
|---|---|---|---|---|---|
| PSAR 6 slots @1x | 14.46 | [2.65, 3.26] | 0% | +769% | −6.5% |
| **PSAR 8 slots @1x** | **15.71** | **[2.90, 3.53]** | **0%** | **+835%** | **−7.9%** |
| PSAR 10 slots @1x | 16.71 | [3.09, 3.72] | 0% | +832% | −6.6% |
| PSAR 8 slots @1x, +45bps/side slippage | 10.49 | [1.83, 2.47] | 0% | +558% | −8.3% |
| PSAR 4 slots @1x (conservative) | 12.89 | [2.31, 2.93] | 0% | +744% | −9.2% |

vs old champion Donchian: daily-CI SR ~2.6→3.2 (≈2x), DD −31%→−8%, return
+535%→+835%. Trade quality transformed: PSAR top-10 trades = 22% of PnL
(Donchian pass-9: 81%); median trade positive; win rate 54%; **12/12 months
positive** (min +36%).

## The slot-breadth finding (fair 1x comparison, correct per-slot fees)

At 1x gross with per-slot notional 1/N (fees scaled correctly):

| Slots | SR | Ret | maxDD | Trades/yr |
|---|---|---|---|---|
| 3 | 11.91 | +763% | −12.3% | 891 |
| 4 | 12.89 | +744% | −9.2% | 1321 |
| 6 | 14.46 | +769% | −6.5% | 1918 |
| 8 | 15.71 | +835% | −7.9% | 2465 |
| 10 | 16.71 | +832% | −6.6% | 3104 |

Monotone: more concurrent small bets dominate fewer big ones. This is the same
"breadth scales" law as the pass-10 universe finding, now at the portfolio
level. 10 slots costs 8.5 alerts/day live (execution burden) — 6–8 is the
practical sweet spot.

## Full validation battery on the champion (8 slots @1x) — ALL PASS

- **Parameter neighborhood** (af0 × afmax, 6 points): 12.97–15.99 — flat
- **Fee stress**: ×2 fees 12.09, ×3 fees 11.29 (slots4); +45bps/side extra at
  8 slots: 10.49 P0 0%
- **Halves**: H1 14.42 / H2 17.11; quartiles 11–17 — no dead regime
- **Universe**: top-100 13.04 / top-200 12.89 / top-300 12.22 (slots4) — flat
- **Stops**: 3%/5%/8% flat (13.7/12.9/12.6 slots4); hold 5/7/10d flat
  (10d never binds — the SAR flip IS the exit)
- **Signal-lag falsification**: enter 1 bar late → 9.54 (graceful decay, no
  exact-bar timing artifact); PSAR verified on synthetic series (trend rides,
  flip timing correct; NOTE: the installed `ta` package's PSAR returns a
  corrupt-length series — do not use as oracle)
- **Direction split**: long-only 11.25 / short-only 9.85 (slots4) — both work
- **Fragility**: drop-5-random-coins → ret range +958%..+1055% (slots4)
- **Entry priority**: ranking candidates by 24h momentum REJECTED (DD −46%;
  FIFO by symbol kept)

## Round-1 family scan (top-200, both directions, 3 slots, 12m continuous)

| Family | SR | Ret | maxDD | Verdict |
|---|---|---|---|---|
| psar | 11.91 | +763% | −12.3% | **champion** |
| keltner e10 m2 | 7.45 | +712% | −20.3% | strong runner-up |
| rsi_regime 60/55 | 8.65 | +1198% | −23.1% | strong, higher DD |
| supertrend (fixed) | 8.09 | +663% | −21.2% | strong, h2-weaker |
| range_expansion | 6.51 | +800% | −42.5% | DD too fat |
| donchian_40_20 | 6.31 | +585% | −28.7% | slower Donchian |
| bollinger | 5.85 | +513% | −25.4% | ~ Donchian |
| donchian_20_10 (anchor) | 5.77 | +532% | −31.0% | reproduced ✓ |
| adx_donchian | 5.75 | +569% | −29.8% | gate ≈ neutral |
| donchian_10_5 | 5.03 | +406% | −22.4% | too fast |
| ema_cross_4_16 | 4.22 | +408% | −25.6% | weak |
| ema_cross_8_32 | 3.63 | +281% | −29.0% | weak |
| macd | 3.37 | +225% | −25.1% | weak |

Notes: donchian_vol and donchian_trend gates were non-binding on entry edges
(breakouts already carry volume and sit above the 50d EMA). Ensembles
(PSAR∪RSI, PSAR∪RSI∪Keltner) did NOT beat pure PSAR (union-exit semantics
drag). Keltner/RSI/supertrend remain credible alternates but lose to PSAR on
the joint (SR, DD) criterion.

## JEV verdict (user-sanctioned test)

JEV daily P(momentum) as a soft size on the champion: 13.14 vs 12.89 base
(+0.25, inside the neighborhood noise band 12.2–13.9); inverted sizing is
negative (12.32). **Third independent negative for JEV at ≤1w horizons**
(after ranking, after gating). Not adopted; cache retained for reporting.

## Caveats (stated, not hidden)

1. **Delisted-pair survivorship** — optimistic, unquantifiable (unchanged).
2. **One exceptional trend year** — 12/12 positive months is partly the
   regime; PSAR whipsaw in chop is bounded by stops but a range-year would
   compress returns.
3. **Churn/slippage is the make-or-break**: 2465 trades/yr at 8 slots ≈ 55%
   of book in round-trip fees per year (survives ×3 fees and +45bps stress,
   but live small-cap fills on flip bars must be measured).
4. FIFO-by-symbol entry selection is arbitrary-but-neutral (momentum-ranked
   selection is worse; tested).
5. Multiple-testing: ~40 configurations this pass, but the champion wins on
   family-level evidence (6-point flat param neighborhood, slot sweep
   monotone, both directions, all halves) — not a lone spike.

## Verdict

**NEW BEST RESULT OF THE PROJECT — replace the Donchian probe.** Dual-direction
PSAR flips, 6–8 slots × 1/N, top-200 point-in-time, ≤7d, 5% stop: daily-block
CI [2.90, 3.53] at 1x gross, +835%/yr, DD −7.9%, 12/12 positive months,
survives the full battery with headroom. `ta-probe` now defaults to
`--family psar --universe-top 200 --slots 8` (donchian kept as `--family
donchian`). Paper-trade it; measure live flip-bar slippage before capital.
