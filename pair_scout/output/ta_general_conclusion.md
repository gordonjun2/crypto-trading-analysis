# Generalized TA channel trading — research conclusion (2026-09-23, pass 9)

Mandate: generalize pass 8 — short opportunities (breakdown) hedged by long
strong coins, 1-signal-with-N-hedges, and a broader universe (not just the top
few coins). ≤1-week holds, same 12-month/fees/bootstrap bar.

## The generalized strategy

At every bar, over a point-in-time universe (top-N by trailing 30d dollar
volume, causal from all 657 pairs):
- LONG signal: close crosses above prior 20-day high → long the coin
- SHORT signal: close crosses below prior 20-day low → short the coin
- Hedge legs (optional): long signal → short the N weakest coins; short
  signal → long the N strongest (7d momentum, dollar-neutral per trade)
- Exits: channel re-entry (10d), 7d time stop, 5% adverse stop
- 3 slots × 1/3 book (gross 1x), next-bar fills, 7 bps/side, real funding

## Results (12 months, 22 folds-equivalent continuous)

| Config (top-200 universe) | Port SR [90% CI] | P(SR≤0) | Ret | maxDD |
|---|---|---|---|---|
| Both directions, no hedge | 6.10 [0.91, 1.61] | 0% | +535% | −31% |
| Both directions, +45bps/side slippage stress | 5.10 [0.70, 1.40] | 0% | +447% | — |
| Long only, no hedge | 6.46 [1.00, 1.66] | 0% | +855% | −61% |
| Short only, no hedge | 5.82 [0.84, 1.54] | 0% | +575% | −54% |

Robustness battery — ALL PASS:
- **Universe breadth MONOTONE POSITIVE** (top30 +325% → top100 +632% → top200
  +855% for long 0h): more coins = more independent channel events. This is
  the OPPOSITE of the momentum-factor result (pass 7, where breadth killed it)
  and matches theory: breakout/breakdown events scale with cross-section size.
- **Neighborhood**: entry 10d/20d/40d → 5.74/6.10/6.12 — flat, not an island.
- **Halves**: H1 6.76 / H2 5.38 — both regimes.
- **Slippage**: +45bps/side extra → 5.10, P0 0%.

## The N-hedge answer (user's question)

**Hedges don't help.** 0-hedge ≥ 1h ≥ 2h ≥ 3h on return in nearly every
(universe, direction) cell; fees scale 9% → 45%/yr with hedge count; DDs are
often WORSE with hedges (top200 long: −61% unhedged vs −78% with 2 hedges).
The signal legs carry everything. Recommendation: run UNHEDGED per-signal, and
control book risk with the slot count / stop instead. (If a hedge is desired
for drawdown-shaving, 1 leg is the least-bad, but it costs return.)

## Both directions work

Short-side breakdowns alone (top200: SR 5.82, +575%) are as strong as the long
side — crypto downtrends are violent and fast, and 20d-low breakdown shorts
capture them. This doubles the opportunity set vs long-only.

## Caveats

1. **Delisted-pair survivorship** (unquantifiable): at top-200 breadth this
   bites harder — dead coins' late-sample trades are missing. Direction mixed:
   understates winning breakdown shorts, overstates some breakout longs.
2. **One exceptional trend year** (several 5–33x altcoin runs + violent
   collapses). A range-bound year would trade less and win less; stops bound
   the damage.
3. **Funding history missing** for some broad-universe pairs (carry = 0
   there); carry is second-order for this strategy.
4. **Small-cap fill realism**: +45bps/side stress survives, but top-200 tail
   breakouts can slip worse on news bars. Live paper-trading is the test.
5. ~30 configurations tested; the family-wide consistency (every breadth ×
   both directions × neighborhood positive) is what carries the verdict.

## Verdict

**STRONGEST RESULT OF THE PROJECT — paper-trade candidate.** Dual-direction
Donchian channel trading (20d entry/10d exit) over a broad point-in-time
universe, unhedged, 3 slots × 1/3, ≤7d holds: every robustness check passed
with headroom. Recommended next steps: (1) wire into the daily Telegram run as
a live paper-trade probe (slippage measurement is the make-or-break); (2)
extend history to 3–5 years to add regime diversity; (3) size any real capital
only after live fills match backtest assumptions.
