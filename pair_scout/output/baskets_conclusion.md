# Cross-sectional basket strategies — research conclusion (2026-09-23, pass 6)

Mandate: explore pair/basket structures beyond 1-1 books ("2 long 1 short
allowed"), ≤1-week horizon, same validation bar (12 months, walk-forward, real
funding, fees, bootstrap, neighborhood + halves robustness).

## The idea (literature-grounded)

Cross-sectional factors on our top-30 liquid perp universe:
- **Weekly momentum** (Liu–Tsyvinski–Wu 2022: 1–4 week formation, 1 week hold,
  winner-minus-loser) — the documented flagship crypto factor;
- **Short-term reversal** (long losers / short winners) — documented in
  small/illiquid coins; the liquid-coin literature predicts the OPPOSITE sign
  here, making it a built-in falsification check;
- **Unbalanced baskets** (2L/1S, 3L/1S, 4L/2S) — the user's suggested shape.

Harness: dollar-neutral 50/50 gross, equal weight per name, sides beta-matched
vs BTC (scale-capped), rebalance every R days on strictly-prior returns, fees
7 bps/side on turnover, REAL funding on every leg, 22 walk-forward folds.

## Results (12 months, ~25 variants)

| Family (best of family) | Port SR [90% CI] | P(SR≤0) | Ret | Verdict |
|---|---|---|---|---|
| Weekly momentum f7/r7 **5v5** | **2.01** [0.03, 0.76] | 4% | +68% | **REJECTED** — see below |
| Weekly momentum f7/r7 3v3 | 1.00 [−0.17, 0.54] | 18% | +46% | weak |
| Momentum f5/r7 5v5 | 1.31 [−0.10, 0.63] | 12% | +45% | weak |
| Momentum f3/r1 5v5 (daily) | 1.16 [−0.17, 0.60] | 17% | +44% | fees 22%/yr; halves unstable |
| Daily reversal f1/r1 (all shapes) | −1.39 to −2.02 | 91–97% | −52% | falsification check PASSED (sign matches liquid-coin literature) |
| Unbalanced 2L/1S, 3L/1S (all) | −0.53 to −1.35 | 67–90% | — | concentration risk: 1 short name at 0.5x notional = idiosyncratic blowups (DD −100..−160%) |

## Why the headline 2.01 is rejected

1. **Neighborhood**: only f7/r7 works. Its 8 neighbors: f5/r5 −0.81, f5/r7
   +1.31, f5/r10 +0.17, f7/r5 −0.01, f7/r10 −0.14, f10/r5 −1.36, f10/r7 +1.11,
   f10/r10 −0.14. A real factor degrades smoothly; this is an island.
2. **Halves**: H1 (Oct 25–Mar 26) SR 2.37; **H2 (Mar–Sep 26) SR −0.13**
   (P0 56%). The entire "edge" is one regime.
3. **Multiple comparisons**: max of ~25 correlated variants; the daily-level
   bootstrap CI [0.03, 0.76] already overlaps zero.

## What the falsification check bought us

The reversal family's uniform, strongly negative result (long losers loses big
on liquid coins) is exactly what the liquid-coin literature predicts — the
harness detects documented effects in the documented direction. That makes the
negative verdict on momentum *meaningful*: it's not a broken backtest, it's a
broken edge (on this universe, in this regime sample, net of 7 bps/side).

## Structural conclusion (three passes converging)

Momentum divergence (1-1 books), funding-squeeze pairs, and cross-sectional
baskets all die the same death on this 12-month sample: gross cross-sectional
effects exist directionally, but (a) they concentrate in H1 (Oct 25–Mar 26),
(b) at ≤1-week horizons, fees and funding asymmetries consume the remainder,
and (c) unbalanced/concentrated shapes add idiosyncratic blowup risk. With one
regime break in the sample, NOTHING cross-sectional validates robustly at
≤1-week horizons net of costs in top-30 perps.

The 2L/1S idea specifically: in a dollar-neutral frame, fewer shorts = heavier
per-name short = concentrated blowup risk. If you ever run unbalanced books,
cap per-name weight (more names per side) and beta-match — the concentration,
not the asymmetry, is what kills it.

## Byproducts

- `pair_scout/research_baskets.py`: general N-vs-M basket backtester
  (beta-matched sides, real funding, causal ranks, bootstrap reporting).
- Harness falsification validated against the liquid-coin literature.

## The honest menu now

(a) longer horizons (2–8 weeks) where documented momentum can clear costs;
(b) broader universes (the documented effects live in the long tail of 1800
coins — needs more data infrastructure); (c) cross-exchange or spot-perp basis
structures; (d) event-driven pairs. All need the same 12-month + bootstrap bar
this project now enforces by default.
