# Price-action TA + hedged entries — research conclusion (2026-09-23, pass 8)

Mandate: use technical analysis (Python `ta` package) to find good single-coin
entries, hedge each with a weak coin (user example: long SUI / short AVAX),
≤1-week holds. Same validation bar as all prior passes.

## Method upgrades forced by this pass

1. **Point-in-time universe.** The old panel was the top-30 by trailing volume
   at the END of the sample — i.e. the year's winners (ZEC 33x, BR 14.5x). A
   breakout strategy "catching" those is circular. Now the universe at every
   bar is the top-30 by trailing 30-day dollar volume, computed causally from
   all 657 pairs (the top-30 membership churns ~100% across the year — 88
   distinct pairs, zero stable core — so this matters enormously).
2. **Continuous simulation** (no fold resets — the pass-7 Sharpe-inflation fix).
3. **Exposure normalization**: 3 slots × 1/3 book ≈ ≤1x gross (average ~0.73x
   invested).
4. **Per-trade expectancy** as the signal-quality metric (independent of
   sizing/leverage).

## Signals tested (canonical price action, `ta`, 4h bars)

- **Donchian 20d/10d**: long close > prior 20-day high; exit < prior 10-day low
- **EMA 5d/20d cross**: long while fast above slow
- **RSI-14 pullback**: long when RSI leaves oversold; exit > 70
- All: ≤7d time stop, 5% adverse stop, max 3 concurrent trades

## Results (12 months continuous, fees 7bps/side, real funding, 1x book)

| Variant (best of hedge modes) | Port SR [90% CI] | P(SR≤0) | Ret | maxDD | Expectancy/trade | Win rate |
|---|---|---|---|---|---|---|
| Donchian + BTC hedge | 5.25 [0.75, 1.40] | 0% | +302% | −19.9% | +4.5% | 48% |
| EMA + none (long-only) | 2.55 [0.18, 0.86] | 1% | +114% | −14.7% | +1.7% | 38% |
| RSI + BTC hedge | 2.75 [0.21, 0.89] | 0% | +95% | −19.7% | +1.4% | 50% |

Robustness battery (Donchian 20d/10d long-only):
- **Neighborhood**: breakout 10d SR 5.70 / 20d 5.06 / 55d 3.71 — monotone
  degradation, not an island.
- **Halves**: H1 4.17 (P0 0%), H2 6.00 (P0 0%) — both halves strongly positive.
- **Fee stress**: ×2 → 4.91, ×3 → 4.76. Slippage +10bps/side → survives
  (analogue checks on EMA/RSI: 2.19–2.20, P0 1–2%).
- **Trade shape**: classic trend-following — 48% win rate, median trade −0.25%,
  expectancy +4.5%; top-10 of 135 trades carry 81% of profit. Real but
  lottery-skewed: it lives off a handful of altcoin moons (ZEC 33x, BR 14.5x
  were in this sample).

## The hedge answer (user's specific question)

The weak-coin hedge (short the 7d laggard) is roughly a WASH vs a plain BTC
hedge: donchian +302% (BTC) vs +297% (weak); rsi +95% vs +84%; DDs similar.
The alpha is in the TA long; the hedge's job is only to shave drawdown
(−19% → −16..−20% depending on signal), and any hedge does that. Shorting the
"weak coin" adds idiosyncratic risk without extra return — shorting BTC is
simpler and no worse.

## Caveats (stated, not hidden)

1. **Delisted-pair survivorship**: coins delisted before today are absent from
   the exchange list, so their (often losing) late-sample trades are missing.
   Direction of bias: optimistic. Unquantifiable with this dataset.
2. **One year of data, exceptional alt-trend sample** (several 5–33x moves).
   A choppy year would produce far fewer monster trades; the stops and exits
   are what protect the downside, but the upside depends on alt-seasons
   occurring.
3. **Breakout execution**: entries fill on the first 1h bar after a 4h close
   crosses the channel — real breakout slippage is the main live risk. Stress
   tests at +10bps/side extra still survive, but live fills on fast breakouts
   can be worse than any fixed bps.
4. Multiple-testing discount: ~20 cells tested this pass, but the result is
   family-level (3 independent signal types, both halves, smooth neighborhoods)
   — materially stronger evidence than any previous pass.

## Verdict

**First strategy to survive the full battery: PROMISING, paper-trade it.**
Donchian 20d/10d breakout, long-only or BTC-hedged, 3 × 1/3 slots, ≤7d hold,
5% stop. Do not size capital before live paper-trading confirms the expectancy
(breakout slippage is the make-or-break variable). Recommended next step: wire
the Donchian scanner into the daily Telegram run as a live paper-trade probe.
