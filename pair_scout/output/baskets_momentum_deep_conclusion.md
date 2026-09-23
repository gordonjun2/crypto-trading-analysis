# Basket momentum — deep exploration conclusion (2026-09-23, pass 7)

Mandate: explore basket momentum further ("2 long 1 short allowed"), iterate
with research, same 12-month/fees/bootstrap bar.

## What was tried (~50 configurations total across the momentum family)

| Lever | Result |
|---|---|
| Breadth 30 → 50 → 100 pairs | top30 f7/r7 5v5 = 2.01 P0 4% **collapses to 0.25–0.30 (P0 25–45%) at top50/top100.** If the factor were real, breadth (less noise, more names — the documented condition) should have helped. It destroyed it. |
| Vol-adjusted ranks (mom/vol) | Best fold-chunked: 2.81 P0 1%, passes BOTH halves (H1 2.01, H2 3.51), passes 2x fees, passes universe re-anchor (30d volume window → 3.49) — the strongest candidate yet. |
| Vol-adjusted neighborhood | Healthy where non-adjusted was an island: ALL r7 formations positive (f5 2.10 / f7 2.81 / f10 2.82 / f14 1.87, P0 0–5%); r5 dies (fold-reset artifact, see below). |
| Continuous simulation (no fold resets) | **SR 1.12 [−0.11, 0.61], P0 14%, ret +41%/yr, DD −57%.** The fold-chunked simulation inflated Sharpe ~2.5x (forced liquidation at fold boundaries + rebalance-timing luck). |
| Name robustness (drop 5 random pairs, 3 seeds) | SR 0.49 / 0.94 / 3.07 — result dominated by a handful of names. |
| Name count | 3v3 DD −91%; 10v10 SR 0.84 P0 21% — no free diversification inside 28 coins. |
| Regime gate (BTC 90d > 0) | "Fixes" H2 only by switching H1 off (H1 0.00 flat) — conditioner chosen after seeing H2 fail; rejected as self-deception despite Daniel–Moskowitz priors. |
| Buffers/hysteresis | Mild improvement (1.62 P0 8%), does not survive continuous simulation check. |

## Verdict

**Not tradeable — but the closest thing to a real effect found so far.** The
continuous, honest version of vol-adjusted weekly momentum earns +41%/yr at 1x
with a daily-level Sharpe of ~0.4–0.5 and P(SR≤0) = 14% — suggestive, not
significant, with −57% drawdown and heavy dependence on which coins are in the
universe. It fails the bar this project set (P0 < 5% across robustness checks).

## Methodological finding that affects ALL earlier passes

**Fold-chunked simulation (positions force-closed at fold boundaries) inflates
Sharpe ~2–2.5x vs a continuous run.** The 2.81 fold-chunked became 1.12
continuous. Every fold-based number reported in earlier passes shares this bias
direction; the pass-4 verdicts (which rejected everything anyway) are
conservative and stand, but future research must use continuous simulation for
rule-based factors.

## Why it keeps failing (one paragraph)

The 12-month sample has a single regime break (~March 2026). Before it:
trending, cross-sectional effects pay. After it: choppy, they don't. With one
regime transition, no ≤1-week-horizon cross-sectional strategy can be validated
to significance — there is effectively ONE regime observation per side. This is
a data limitation, not (only) a strategy limitation. The effects documented in
the literature live in 1800-coin universes with years of data spanning many
regimes.

## If you want to keep pulling this thread

1. **More history**: extend to 3–5 years of 1h data (Binance has it) → many
   regime transitions → the vol-adjusted weekly momentum factor becomes
   testable properly. This is the single highest-value next step, and the
   harness runs as-is on a bigger panel.
2. **Broader universe**: 300–500 pairs (infrastructure exists; funding data is
   the bottleneck — paginated fetcher is written).
3. **Risk structuring**: the vol-adjusted variant's −57% DD needs portfolio-level
   vol targeting (tested shape exists in `research_sizing.py`) before any
   capital conversation.
