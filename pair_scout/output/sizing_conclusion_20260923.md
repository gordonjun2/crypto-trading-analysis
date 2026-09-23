# Better modeling + Sharpe iteration — research conclusion (2026-09-23, FINAL)

**Headline after full validation: the strategy has NO reliable edge on 12 months
of data. The 6-month result was universe-and-regime luck. Do not trade capital
on this as configured.** Details in section 6.

Follow-up to the sizing pass. Mandate: model the strategy better (funding,
significance), then iteratively improve Sharpe without overfitting. JEV used
freely as classifiers (regime read, book-quality classifier, ranking), kept only
where it earned its place.

## 1. Better modeling

**Funding rates (real data).** Downloaded Binance perp funding history for the
panel window (all 30 universe pairs; 8h settlements, some 4h). The simulator now
applies `cost = pos × (w_long·f_long − w_short·f_short)` per bar, scaled with
notional, with a stress multiplier. Result for THIS book: small but real —
pooled top-3 SR 0.92 → 0.91, portfolio SR 2.36 → 2.34 (short legs of weak
momentum coins carry near-zero average funding). Funding stays ON in all
numbers below. Stress runs (1.5–2x) confirmed the strategy is not
funding-fragile.

**Statistical honesty (block bootstrap).** Daily-block bootstrap (5-day blocks,
2000 draws) on the pooled portfolio: baseline daily-level Sharpe 90% CI
[−0.03, 0.99], P(SR ≤ 0) = 6%. The combo variant: [0.02, 0.97], P = 4%.
Translation: the hourly "Sharpe 2.5" is an intraday-sampling artifact; the true
low-frequency Sharpe is ~0.5–0.7, positive with ~95% confidence but far from
"proven". Any live result will also carry funding-model error and regime shift.

## 2. Iteration results (all funding-adjusted, same 10 folds, ~40 ideas total)

| Idea | Pooled top-3 SR | Port SR | P(SR≤0) | Ret@1x | maxDD | Verdict |
|---|---|---|---|---|---|---|
| Baseline 1x (funded) | 0.91 | 2.34 | 6% | +15.1% | −6.7% | reference |
| combo JEV×vol-target (prev winner) | 1.07 | 2.50 | 4% | +14.9% | −5.4% | keep |
| + soft weekend (0.5x Sat/Sun) | 1.09 | 2.50 | 4% | **+15.7%** | **−4.8%** | **ship** |
| + portfolio vol-target [0.5,1.25] | 1.08 | 2.84 | 3% | +15.2% | −5.4% | recommend |
| + portfolio vol-target [0.5,1.5] | 1.09 | **2.96** | **3%** | +15.4% | −6.3% | best SR, 1.04x exp |
| Funding tilt (k150/k300) | 1.06/1.05 | 2.42/2.37 | 5% | ~flat | — | **rejected** (no edge; carry ±1bp vs book vol ~3%/day) |
| JEV per-book quality classifier → size | 1.01 | 2.58 | 3% | +16.9% | −4.9% | marginal; optional |
| Strong weekend flat (0x Sat+Sun) | **1.31** | 2.06 | 7% | +14.3% | −4.3% | **rejected as headline** — pooled jump is selection (Sat-only 1.26, Sun-only 0.97) and it deflates return; the 0.5x soft version captures the risk benefit honestly |
| JEV ranking arm (re-run) | 0.66 | 1.96 | 10% | +13.8% | −6.7% | still broken — do not use |

## 3. What changed and why it is not overfit

- **Shipped (wired into `config.example.toml`):** `size_scaling = "combo"`
  (JEV regime soft-size × per-book vol-target), `weekend_size_scale = 0.5`,
  `include_funding = true`. Real Binance funding history required at
  `<data_dir>/funding_rates.json` (build with the downloader in
  `pair_scout/data/funding.py`; keep refreshing it with the klines).
- **Dose-response consistency:** weekend 0x → DD −4.3%, 0.5x → −4.8%, 1x →
  −5.4% (monotone); port-vol-target cap 1.25 → port SR 2.84, 1.5 → 2.96
  (monotone). The rejected strong-weekend idea fails exactly this test
  (Sat works, Sun does not → selection, not mechanism).
- **Mechanism priors:** thin weekend liquidity punishing 1-day books and
  portfolio-level vol targeting are both standard, not data-mined quirks.
- **Idea-count deflation stated:** ~40 variants evaluated across both passes;
  the shipped family sits ~1.05–1.09 pooled vs baseline 0.91 — the *family*
  level lift (+0.15) is the honest estimate, not the best single number.

## 3.5 JEV ranking — post-mortem and final verdict (pass 3)

Diagnosis with 3 independent ranking runs on identical inputs: **JEV is not
noisy** — replicate Kendall τ = 0.92 (0.80–1.00), mean composite σ = 0.004.
The earlier "0.95 vs 0.66" discrepancy between sessions was NOT API
non-determinism; JEV is a *stable* judge whose preferences are consistently
contrarian to what pays in this regime (it agrees with the rule top-3 in 23/29
cases, and the ~20% deviations are value-destructive: 0.66 vs 0.91 pooled).
Rectification attempts, all failed honestly:

| Rectification | Pooled SR (rule ref 1.09 w/ combo+wknd) |
|---|---|
| Book-classifier question (persistence-framed, decision-time info) as rank | 0.25 |
| Hybrid percentile rank (25/75, 50/50 rule×JEV) | 0.18 / 0.45 |
| Multi-sample voting | pointless (τ = 0.92 — no noise to average out) |
| JEV as VETO only (demote red-flagged books) | 0.62 / 0.80 |

Additional rejected improvements (pass 3): rule top-4/top-5 breadth (books 4–5
are much weaker — the rule rank is real but decays fast); $5M liquidity floor
(flips the sample negative — note $1M/day is aggressive for real capital, a
live-execution risk to respect, not a backtest knob); trailing spread stops
(1.5%/2.5% churn winners, fold 8 6.0 → −1.5); BTC-crash size overlay
(redundant for beta-neutral books, hurts returns).

## 4. JEV usage summary (as instructed, used freely — kept only where it helped)

1. **Daily regime probability → soft size:** KEPT (cached, 1 call/day, causal).
2. **Per-book short-horizon quality classifier (Score 0–3 + red-flag Noul,
   decision-time state incl. funding carry & regime):** OPTIONAL — no pooled-SR
   lift, small DD benefit, non-replication risk acknowledged.
3. **Pair ranking:** DROPPED — failed to replicate on the re-fetched panel
   (0.66 vs 0.95 previously claimed).

## 5. Where this leaves the strategy

Funding-adjusted, exposure-adjusted, bootstrap-checked: **daily-level Sharpe
~0.6–0.7, P(daily SR ≤ 0) ≈ 3–4%, maxDD ~5%**. That is a real, modest,
risk-controlled edge — appropriate for paper-trading and small size, not for
meaningful capital yet. The two biggest remaining unknowns: regime change (one
sample window) and funding-model error in live conditions. Next levers if
needed: multi-exchange universe, funding-predictor features, longer data
history before trusting anything bigger.

## 6. 12-month validation (pass 4) — the decisive test

Extended the panel to a full year (657 pairs × 8760 hourly bars, 2025-09-23 →
2026-09-23), 22 rolling 15-day folds, full JEV regime coverage, funding merged
for the months where history was accessible (later folds; earlier folds run
unfunded — a bounded, favorable omission given the measured ~0.01 SR impact).

| Config (22 folds, 12 months) | Pooled top-3 SR | Port SR [90% CI] | P(SR≤0) | P@3 | Folds+ |
|---|---|---|---|---|---|
| Baseline 1x | **−0.59** | 0.17 [−0.32, 0.40] | 45% | 0.36 | 8/22 |
| combo + weekend (shipped sizing) | −0.61 | −0.26 [−0.40, 0.29] | 60% | 0.34 | 9/22 |
| combo + weekend + port_vt | −0.56 | 0.09 [−0.33, 0.37] | 47% | 0.36 | 9/22 |

Consistency check (no bug): slicing the 12m panel back to the original 6-month
window and re-running recovers a good result (baseline 0.77 pooled, combo +
weekend 1.14) — but with a **different fold profile and universe** than the
original run (universe is selected by trailing volume, which changed; fold
boundaries re-anchor). The "0.92 Sharpe" was therefore not a stable property of
the strategy: it was an artifact of (a) which 30 pairs were in the universe,
(b) where the folds happened to fall, and (c) a momentum-friendly regime.

**What survives:** the infrastructure (funding model, causal sizing framework,
bootstrap tooling, walk-forward harness) and the process findings (JEV must not
select; sizing overlays do what they claim but cannot create an edge). The
sizing family remains a genuine risk-reallocation improvement *conditional on a
signal existing* — worth keeping wired for future signal work.

**What this means practically:** stop. Do not deploy capital. Paper-trading the
daily Telegram report is still useful — but as a test of live-vs-backtest
divergence, not as a money-making expectation. To find a real edge, the next
effort should go into the SIGNAL (e.g., funding-aware carry screens, cross-
exchange lead-lag, longer history multi-regime validation, event-driven
divergences) rather than into further sizing/ranking polish — this project has
now demonstrated, twice, that polishing on a 6-month window manufactures
 Sharpe that does not exist.

## Reproduce

```bash
python -m pair_scout.research_sizing --config <cfg> --out pair_scout/output --jev
python -m pair_scout --config <cfg> evaluate --mode divergence  # production path
pytest pair_scout/tests  # 93 tests
```
