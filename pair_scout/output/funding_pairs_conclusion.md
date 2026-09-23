# Funding-squeeze pairs — research conclusion (2026-09-23, pass 5)

Mandate: a NEW pair-trading idea, creative but evidence-driven, ≤1-week holds,
web-researched priors, JEV allowed freely, tested at the standard set by this
project (12 months, walk-forward, real funding, fees, bootstrap, neighborhood
robustness).

## The idea

**Funding-squeeze pairs**: long the coin with the most extreme negative funding,
short the most extreme positive funding, beta-neutral. The book COLLECTS the
funding differential from both legs every settlement while betting the crowded
positioning mean-reverts within days. Priors confirmed via research: funding
extremes mark crowded positioning and precede squeezes (multiple practitioner
and academic sources, incl. a 2025 ScienceDirect study on funding strategies).

Data: full 12 months of real Binance funding for the universe (1098
settlements/pair, 2025-09-22 → 2026-09-23), 22 walk-forward folds, next-bar
execution, 5 bps + 2 bps slippage per side per leg.

## Results (~35 configurations: lookbacks 3/7d, thresholds 0.5–2bp/8h, holds
1–7d, trailing-mean vs last-settlement signals, early exits, price stops,
top-2 concentrated vs full cross-section baskets)

| Shape (best of family) | Port SR [90% CI] | P(SR≤0) | Ret | maxDD | Carry | Price |
|---|---|---|---|---|---|---|
| Top-2 concentrated, mean signal | 0.85 [−0.19, 0.55] | 22% | +52% | −51% | +51% | +156%* |
| Top-2, last-settlement signal + stop | 0.83 [−0.22, 0.56] | 25% | +43% | −41% | +49% | +13% |
| Full cross-section basket | 0.23 [−0.34, 0.48] | 45% | +10% | −27% | +109% | −76% |

*first-run decomposition had inverted sign semantics; corrected in later runs.

## The economics (the actual finding)

1. **The carry is real and large**: +45% to +110%/yr collected across every
   configuration. The funding-differential harvest works as advertised.
2. **But funding extremes are momentum markers.** The hot coin keeps rising and
   the crashed coin keeps falling over ≤1-week horizons: the price side loses
   −22% to −82%/yr depending on construction. Carry minus price drag ≈ noise
   (P(SR ≤ 0) = 22–50%).
3. **Fee math**: liquid-universe funding extremes are ~1–3bp/8h and revert
   within hours-to-a-day; a round trip costs ~14bp/leg-pair. Harvestable carry
   per trade ≈ fees per trade. At top-30 liquidity and ≤1-week horizons, the
   trade cannot clear costs after price risk.
4. Diversifying (basket of all qualifying pairs) removes most idiosyncratic
   risk (DD −51% → −27%) but also most signal — the price drag scales with the
   carry, leaving ~0 net.

## Verdict

**Negative — do not trade at this timeframe.** Funding-carry harvest needs
multi-week horizons (where short-horizon momentum decays and carry compounds)
or delta-hedged spot-perp structures (collect perp funding, hold spot — no
price direction) — both outside the ≤1-week perp-pair mandate. JEV was not used
for selection (proven harmful in pass 3); the regime read remains available for
context/sizing if a viable core signal is ever found.

## Positive byproducts

- Full 12-month funding dataset (fixed a pagination subtlety: the fundingRate
  endpoint returns the OLDEST 1000 from startTime — paginate forward).
- `simulate_divergence` now accepts external signals (`signal_override`) — any
  per-bar series can drive the book (funding, basis, spreads, ML outputs).
- `pair_scout/research_funding_pairs.py` harness with causal daily/settlement
  signals and PnL decomposition (net = price − fees + carry).

## Where this leaves pair trading at ≤1 week in top-30 perps

Two structural families tested and exhausted on this data: momentum divergence
(passes 1–4, dead on 12m) and funding-squeeze mean reversion (pass 5, carry
real but price-dominated). Both die the same way: gross effects exist, fees +
short-horizon price risk consume them. The honest menu from here: (a) accept
longer horizons (2–8 weeks) where carry/momentum economics can clear costs;
(b) trade more liquid pairs on other exchanges (cross-exchange basis — needs
new data); (c) event-driven pairs (listings, unlocks, delistings) — needs event
data. Each is a new project with the same validation bar.
