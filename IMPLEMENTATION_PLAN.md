# Implementation Plan — Consolidated Pair-Trading Screener with JEV Ranking + Telegram Delivery

Status: **Implemented** — see `pair_scout/` (package + tests), `config.example.toml`,
`README.md` (PairScout section), `pair_scout/output/` (evaluation reports).
Date: 2026-09-22
Scope: Consolidate `cointegration-pair-trading.ipynb`, `correlation-pair-trading.ipynb`,
`beta-neutral-pair-trading-auto.ipynb`, `beta-neutral-pair-trading-manual.ipynb`, and
`volatility-trading.ipynb` into a single maintainable Python application that screens
candidate pairs, ranks them with the JEV classifier (TypeSafe System One), and delivers
a report to Telegram. **Analysis and recommendations only — no order placement.**

---

## 1. What each notebook actually implements (facts from the code)

### 1.1 `cointegration-pair-trading.ipynb` ("Mean Reversion")

**Pipeline (cells 8–33):**
1. Loads Binance **4h** close prices for all available perpetual-futures pairs from
   `saved_data/` via `process_data('mean_reversion', ...)` (shared `data_manager.py`);
   drops pairs with >10% NaN, keeps top 100 by trailing (last-bar) volume; slices
   2024-10-08 → 2025-02-07; linear interpolation + ffill/bfill.
2. Runs `statsmodels.tsa.stattools.coint` (Engle-Granger ADF) on **close-price levels**
   for all C(n,2) combinations; keeps pairs with p < 0.01; heatmap + bar chart.
3. Manually selects 4 ticker pairs; MinMax-scaled price plots with 15-bar rolling mean.
4. Plots the price ratio `ticker2/ticker1` with **full-sample** z-score against ±1
   thresholds, annotated with long/short directions.
5. Backtest: `signals_zscore_evolution` — rolling 15-bar z-score of the ratio; enter
   when z ≤ −1 / exit-ish at z ≥ 1 (and a mirrored `first_ticker=False` variant for the
   other leg); `utils.calculate_profit` accumulates per-leg price-unit profit; the two
   legs' cumulative profits are summed and plotted.

**Verdict:** the only notebook with a complete (if crude) signal → backtest loop for
classic mean-reversion pair trading. Reusable: the pair-scan structure, rolling z-score
signal concept. The backtest itself is unreliable (see §2).

### 1.2 `correlation-pair-trading.ipynb` ("Negative / Low Correlation")

**Pipeline (cells 7–22):**
1. Loads Binance **1d** closes, same `process_data`/`sanitize_data` path (2024-12-01 →
   2025-01-19).
2. Computes Pearson correlation of **price levels** for all pairs; selects pairs with
   corr < 0.1; heatmap.
3. Manually picks one pair (XRPUSDT, PENDLEUSDT) and plots MinMax-scaled smoothed
   prices.
4. **No signal generation, no backtest, no exit rules.** The strategy described in the
   markdown (long the stronger asset, short the weaker, e.g. around token unlocks) is
   never implemented in code — direction is chosen by the human.

**Verdict:** screening idea only. The levels-correlation calculation is statistically
unsound; the momentum/divergence logic exists only as prose. Little is reusable beyond
the (broken) screening concept, which will be re-implemented on returns.

### 1.3 `beta-neutral-pair-trading-auto.ipynb` (CVXPY optimizer)

**Pipeline (cells 8–48):**
1. Loads **1d** closes for a **manually specified** dict of pairs with desired
   directions (1 long / −1 short / 0 any) plus a benchmark (BTCUSDT). No discovery —
   the universe is hand-picked.
2. Computes return covariance Σ; betas vs benchmark as `Cov(r_i, r_m)/Var(r_m)`.
3. CVXPY problem: minimize `w'Σw` s.t. `|w'β| ≤ 1e-4` (beta neutrality), `|Σw − 1| ≤
   1e-4`, plus per-asset minimum long/short weight constraints enforced through an
   iterative constraint-rescaling loop (up to 10 attempts, `dynamic_scaling` is the
   identity function).
4. Rolling variant: 7-day rolling covariance, re-optimized per date; on solver failure
   the previous weights are reused; aborts after 5% consecutive unsolvable dates.
5. Backtest: `portfolio_ret = (weights.shift(1) * returns).sum(axis=1)` — correctly
   shifts weights by one bar (no same-bar look-ahead); plotted vs BTC cumulative return.

**Verdict:** a market-neutral **basket** optimizer, not a pair finder. The CVXPY
formulation and the `shift(1)` execution convention are sound and reusable; the
candidate-universe step is entirely manual.

### 1.4 `beta-neutral-pair-trading-manual.ipynb`

Same data pipeline; the user provides **fixed weights** with `Σ|w| = 1` (e.g. SOL 0.7 /
ETH −0.3); rolling betas of the fixed portfolio vs BTC are computed and backtested with
`shift(1)` vs BTC. As saved, the notebook is **broken**: `start_date = '2024-0-01'`
(month 0), so `sanitize_data` rejects the range and returns empty data.

### 1.5 `volatility-trading.ipynb`

**Pipeline (cells 7–18):**
1. Loads Binance **1h** candles keeping **High/Low/Close** (`process_data('volatility',
   ...)` retains those columns), all pairs, top 100 by trailing volume; slices
   2025-02-01 → 2025-02-09 (~8 days, `window_length = 168` bars = 1 week).
2. Per asset, computes seven metrics over the window: std of simple returns, ATR
   relative to price (last value), average (High−Low)/last close ×100, skewness of
   returns, a "Choppiness Index", cumulative rolling-vol sum, and log-return std.
3. Normalizes all metrics with `MinMaxScaler` across assets (skewness negated so
   "lower = more volatile"), sums them into a **Total Volatility Score**, and ranks
   assets by it.

**Verdict:** purely descriptive risk dashboard — **no pair logic, no signals, no
backtest**. It cannot select pairs by itself, but it is genuinely useful to pair
trading as a **risk/leg-quality layer**: (a) per-leg volatility enables risk-balanced
leg sizing, (b) extreme ATR/skew flags dangerous legs, (c) the vol ratio between legs
detects one-leg-dominated pairs, (d) vol-regime context informs entry timing. Reused
in that role (see §3); the composite score itself is not ported (see M8).

### 1.6 Shared infrastructure

- `data_manager.py` — downloads Binance/OKX/Bybit klines to per-pair pkl files;
  `process_data` merges into one panel (NaN threshold, top-N volume filter);
  `sanitize_data` slices dates, interpolates, ffill/bfill.
- `utils.py` — `calculate_profit` (buy/sell pairing used by the cointegration notebook)
  and `plot_strategy`.
- `cex_api/` — key-less kline downloaders.
- Existing local data: `saved_data/binance/1h/*.pkl` — **44 pairs, 1h candles,
  2025-07-02 → 2025-09-02 (1500 bars each)**. This is the only in-repo dataset usable
  for evaluation.

---

## 2. Issues found and corrections

### Bugs

| # | Location | Issue | Fix |
|---|----------|-------|-----|
| B1 | `beta-neutral-...-manual.ipynb` cell 13 | `start_date='2024-0-01'` invalid month; notebook yields no data | Config validated with `pd.to_datetime(..., errors='raise')` at load |
| B2 | `utils.calculate_profit` | `profit[-1] = prices[-1] - prices[bi]` uses `-1` as a **label** on a DatetimeIndex Series (KeyError in pandas 2.x); profit in absolute price units, not %; only models one leg | Replaced by a proper spread-level backtest (§5); legacy function kept in `utils.py`, not used |
| B3 | `correlation` notebook cell 16 | Pearson correlation computed on **price levels** (trending, non-stationary) — spurious | Correlation computed on returns; divergence pairs selected on return correlation + momentum spread |
| B4 | `cointegration` notebook cell 29 | Z-score chart uses full-sample mean/std — look-ahead in the presented "signal" | All z-scores computed on rolling windows only |
| B5 | `cointegration` notebook cell 31 | Trade executes on the same bar's close that generated the signal | Execution shifted to next bar (weight/signal `shift(1)`, matching the beta notebooks' convention) |
| B6 | `beta-neutral-auto` rolling loop | Bare `except:` swallows *all* errors (data bugs disguised as "unsolvable") | Explicit exception types; solver failures logged with reason; typed error taxonomy |
| B7 | Both beta notebooks cell 9 | `selected_pairs[benchmark_token] = 0` mutates the user's input dict | Pure functions; no input mutation |
| B8 | `manual` notebook cell 9 | Float equality `portfolio_weights_sum != 1` | `math.isclose` tolerance |
| B9 | `cointegration` notebook cell 17 | Engle-Granger run in one arbitrary direction (EG is asymmetric: `coint(A,B) ≠ coint(B,A)`) | Test both orientations, keep the better (lower-p / higher-|ADF|) one; hedge ratio from OLS |
| B10 | `volatility` notebook cell 16 | "Choppiness Index" formula (`100·Σ(H−L)/global_range`) is not the standard CHOP (`100·log10(ΣTR/range)/log10(n)`); unbounded and scale-dependent | Use the standard CHOP formula, or drop it; not load-bearing in the pipeline |
| B11 | `volatility` notebook cell 16 | "Cumulative Volatility" sums 168-bar rolling stds over the whole series — scale depends on sample length, not comparable across windows | Not ported; per-window annualized vol used instead |
| B12 | `cointegration` notebook cells 29 vs 31 | Direction conventions are inconsistent: the chart builds the ratio as `t2/t1` and annotates **"Long t2, Short t1" at high z** (inverted — a high ratio means t2 is expensive, mean reversion wants you short t2), while the backtest builds `t1/t2` with opposite signal mapping | Single consistent direction rule: hedge-ratio OLS sign + spread z-score orientation, defined once in `analysis/cointegration.py` and unit-tested |

### Methodological / backtesting problems

- **M1 — In-sample selection bias (cointegration & correlation notebooks):** pairs are
  selected with statistics computed on the same window that is then traded. Fix: strict
  train/test (walk-forward) split; all screening statistics computed on the train
  window only; performance measured out-of-sample.
- **M2 — No transaction costs anywhere**, while the cointegration notebook's own
  conclusion admits trade frequency would be cost-dominated. Fix: configurable
  per-leg fee (default 5 bps taker) + slippage in the backtest.
- **M3 — Universe selection uses future data:** `process_data` ranks volume by the
  *last* bar's trailing window; `sanitize_data` **backfills** leading NaNs with future
  prices. Acceptable for a live screener, leakage for a historical backtest. Fix:
  volume ranked on data available at decision time; forward-fill only for live
  alignment; leading-NaN pairs dropped for backtests.
- **M4 — No stationarity quality checks** beyond raw EG p-value: no half-life, no Hurst
  exponent, no spread stationarity re-check. Fix: feature set extended (half-life via
  OU fit, Hurst on spread, spread z-score) with hard filter ranges.
- **M5 — The "low correlation" strategy has no implemented direction rule.** Fix:
  direction derived from realized momentum (long stronger / short weaker over lookback,
  as the markdown describes), recorded as part of the candidate.
- **M6 — Beta notebooks**: min-weight constraint rescaling loop may not converge
  (capped at 10 tries, silently returns last solution); reported portfolio beta uses
  same-window betas (in-sample). Fix: report convergence status; betas evaluated
  out-of-sample in evaluation mode.
- **M7 — Duplicate imports, `warnings.filterwarnings("ignore")`, plotting mixed with
  logic, no logging, no persistence, no error handling.** Fix: modules with type hints,
  `logging`, pure functions, tests.
- **M8 — `volatility` notebook's "Total Volatility Score" is not a meaningful
  composite:** five of the seven metrics measure the same latent quantity (volatility),
  so the equal-weight MinMax sum double-counts it ~5×, and `MinMaxScaler` collapses
  every other asset toward 0 in the presence of one extreme outlier. Fix: keep the raw
  per-asset metrics; use them individually in filters/JEV state, and derive pair-level
  risk features from ratios (not summed scores).

---

## 3. Consolidated workflow (PairScout)

```
config.toml + .env
        │
        ▼
┌─────────────┐   ┌──────────────┐   ┌───────────────────────┐
│ data layer  │──▶│ analysis     │──▶│ candidate generation  │
│ (pkl panel) │   │ modules      │   │ + hard filters        │
└─────────────┘   └──────────────┘   └───────────┬───────────┘
                                                 ▼
                                       ┌───────────────────┐
                                       │ JEV ranking       │──▶ ranked candidates
                                       │ (composite score, │
                                       │  confidence gate) │
                                       └─────────┬─────────┘
                                                 ▼
                                       ┌───────────────────┐
                                       │ report builder    │──▶ Telegram / dry-run
                                       └───────────────────┘
```

Steps mapped to the request:
1. **Load data** — reuse existing `saved_data/**.pkl` panels (loader kept compatible
   with `data_manager.py` artifacts) or trigger a fresh download via the existing
   `data_manager.py` CLI (unchanged).
2. **Analyses (modular, independently testable):**
   - `analysis/cointegration.py` — Engle-Granger both orientations, OLS hedge ratio,
     spread half-life (OU), Hurst exponent, rolling spread z-score. *(from notebook 1)*
   - `analysis/correlation.py` — return correlation, momentum divergence direction.
     *(from notebook 2, corrected)*
   - `analysis/beta.py` — betas vs benchmark + optional CVXPY beta-neutral basket
     scoring. *(from notebooks 3/4; also reuses their pandas rolling-covariance
     `rolling(window).cov()` pattern for out-of-sample beta monitoring, and keeps the
     notebooks' symmetric-solution presentation in mind — for pairs, direction is fixed
     by the hedge ratio, so no mirrored second portfolio is needed)*
   - `analysis/volatility.py` — per-asset risk metrics on High/Low/Close: return std,
     annualized vol, ATR%, price-range%, return skewness, standard CHOP.
     *(from notebook 5, corrected — raw metrics only, no composite score)*
3. **Candidate generation** — all (A,B) combinations from the top-N liquid universe;
   each strategy module scores every combination.
4. **Feature calculation** — one `PairCandidate` dataclass with all metrics, including
   pair-level risk features derived from the volatility module: `vol_ratio`
   (higher-vol leg ÷ lower-vol leg), per-leg `atr_pct` and skewness, and a suggested
   **vol-balanced leg notional ratio** (σ-based, informational only).
5. **Hard filters (rule-based, retained regardless of JEV):** minimum history, p-value
   ≤ 0.05 (train window), half-life within [4 bars, 30 days], |z| ≥ entry threshold for
   an actionable setup, minimum dollar volume, both legs individually sane
   (ATR% within bounds, |skew| not extreme, `vol_ratio` ≤ max — a pair dominated by one
   wild leg is rejected). Failures recorded with reasons (shown in report).
6. **JEV scoring** — §4.
7. **Ranking** — composite score, confidence-gated.
8. **Final list** — ranked recommendations with direction (long/short per leg),
   entry z-score context, vol-balanced leg sizing hint, invalidation levels; or an
   explicit **NO-TRADE** result.

Package layout (new; notebooks untouched):

```
pair_scout/
  __init__.py
  config.py            # TOML + env loading, validated dataclasses
  log.py               # structured logging setup
  data/loader.py       # pkl panel loader (typed), sanitize (leakage-safe flags)
  analysis/
    cointegration.py   # EG scan, hedge ratio, half-life, hurst, spread z-score
    correlation.py     # return corr, momentum divergence
    beta.py            # betas, optional beta-neutral basket (cvxpy)
    volatility.py      # per-asset vol metrics, pair-level risk features
  features.py          # PairCandidate dataclass + metric explanations
  filters.py           # hard rule filters w/ reasons
  jev/
    client.py          # TypeSafeClient wrapper: RetryPolicy, error taxonomy, cache
    questions.py       # question definitions (documented rubric)
    ranker.py          # normalize → weight → gates → rank
  backtest/spread.py   # walk-forward z-score spread strategy w/ fees
  backtest/evaluate.py # Sharpe, maxDD, hit rate, precision@K, Spearman
  report.py            # build Telegram-safe report (HTML + plain fallback)
  telegram.py          # python-telegram-bot sender, chunking, dry-run
  pipeline.py          # orchestration
  cli.py               # argparse: run / evaluate / --dry-run / --no-jev
tests/                 # pytest: features, filters, ranker, backtest math, report
config.example.toml
.env.example
requirements.txt       # appended: python-telegram-bot, typesafe-sdk, pytest, python-dotenv
```

Dependencies added (only what's needed): `python-telegram-bot`, `typesafe-sdk`,
`python-dotenv`, `pytest`. Everything else (pandas, numpy, scipy, statsmodels, cvxpy)
is already installed in `venv/` (Python 3.11.9). Note: root `requirements.txt` pins
`urllib3==1.26.15` — new deps use `httpx`, no conflict.

---

## 4. JEV integration (from the docs, not assumptions)

### 4.1 Relevant documented capabilities

| Capability (docs) | Use here |
|---|---|
| **Score** primitive (`/primitives/score`) — ordered, *descriptive* levels 0–9; answer = probability-weighted position (fractional), `probabilities`, `confidence` | The core ranking signal: rate each candidate on concrete dimensions |
| **Noul** (`/primitives/noul`) — P(yes), no separate confidence | Blocking checks: "is there a red flag that should veto this trade?" |
| **Choice** (`/primitives/choice`) — probabilities over options (up to 255); the *skill-suggestion cookbook* ranks a whole catalog from one Choice's probability distribution | Not needed for ranking pairs (state differs per pair → one request per candidate); kept in mind if we ever rank within one request |
| **Composite scoring pattern** (`/patterns/composite-scoring`) — split a judgment into atomic Scores, normalize each by `len(criteria)−1`, weight **in code**, rank | Exactly the ranking architecture; weights live in `config.toml` |
| **Confidence** (`/confidence`) — derived from probability spread; three-range gating (act / caution / escalate); thresholds scale with stakes | Candidates below a confidence floor are demoted to "watch", never auto-recommended |
| **Parallel questions in one request** (`/primitives`, parallel-questions cookbook) — all questions on one state are evaluated in parallel; batching is ~10× cheaper/faster | All questions for one candidate go in **one request** |
| **State as structured JSON** (`/concepts/state`) — named fields; reference via backticked paths; text only, English primary | Candidate metrics sent as named JSON fields with plain-language context |
| **Structured criteria** (objects with `what`/`examples`) — lifts confidence when levels overlap | Score levels written as concrete situations with examples |
| **Auth / errors / rate limits** (`/api`) — `Authorization: Bearer $TYPESAFE_API_KEY`; 401, 422, 429, 529; retry 429/5xx with exponential backoff | SDK default `RetryPolicy` (429/408/5xx, backoff 0.5→5 s, `Retry-After` aware) tuned to `max_retries=4`, `timeout=60` |
| **Python SDK** (`/sdk/python`) — `pip install typesafe-sdk`, `TypeSafeClient()` reads `TYPESAFE_API_KEY`; sync `client.system_one(state=..., questions=...)`; answers typed (`response.scores[...]`, `.nouls[...]`) | Used directly |
| **Training / fine-tuning** | **Not offered.** Customization = question/criteria design + state composition + code-side weights. Labeled outcomes can feed classical ML (feature-discovery cookbook), out of scope for v1 |

### 4.2 Input contract (what we send JEV)

One HTTP request per candidate pair that survives hard filters.

```jsonc
// state (JSON object)
{
  "pair": {
    "asset_long":  "BNBUSDT",          // proposed long leg
    "asset_short": "ENAUSDT",          // proposed short leg
    "direction_basis": "hedge ratio sign from OLS of short on long (train window)"
  },
  "metrics": {                          // computed on TRAIN window only
    "cointegration_pvalue": 0.004,
    "hedge_ratio": 1.83,
    "spread_half_life_days": 6.2,
    "hurst_exponent": 0.38,
    "current_spread_zscore": 2.1,       // rolling window, last closed bar
    "return_correlation_90d": 0.71,
    "momentum_30d_long": 0.041,         // for divergence candidates
    "momentum_30d_short": -0.12,
    "avg_daily_volume_usd": 184_000_000,
    "annualized_vol_long": 0.62,
    "annualized_vol_short": 1.10,
    "atr_pct_long": 2.4,                // ATR / price % (per-leg risk)
    "atr_pct_short": 4.9,
    "vol_ratio": 1.77,                  // higher-vol leg / lower-vol leg
    "skewness_long": -0.31,             // tail-risk asymmetry per leg
    "skewness_short": -0.88,
    "history_days": 62
  },
  "metric_reference": {
    // Plain-language domain context JEV lacks: what each metric means and
    // typical good ranges for statistical pair trading.
    "cointegration_pvalue": "Engle-Granger p-value; lower = stronger evidence the spread is stationary and mean-reverts. <0.01 strong, <0.05 acceptable, >0.10 weak.",
    "spread_half_life_days": "Days for a spread shock to decay half way. 2–15 tradeable; >30 too slow; <1 mostly noise/fees.",
    "hurst_exponent": "<0.5 mean-reverting spread, >0.5 trending. Lower is better here.",
    "current_spread_zscore": "How many std devs the spread is from its rolling mean. |z|≥1.5 is the typical entry zone; entry near 0 has no edge.",
    "atr_pct / vol_ratio": "Per-leg volatility and the ratio between legs. A vol_ratio far above 1 means one leg dominates risk; very high ATR% means wider stops and higher execution cost.",
    "skewness": "Negative skew = fatter left tail (crash risk) on that leg; large negative skew on the short leg is a warning sign.",
    "...": "..."
  }
}
```

**Why each field:** JEV has no domain knowledge of our statistics — the docs say to
give the model "the material you would present to a panel of experts." Raw numbers
alone are meaningless to it; each metric is paired with an interpretation in
`metric_reference`, and each question's rubric restates what good/bad looks like.
Direction fields let JEV check consistency; volume/volatility let it weigh
executability; history length lets it discount thin evidence.

### 4.3 Questions (all in the same request, evaluated in parallel)

| ID | Type | Rubric (levels are concrete, per the Score docs) |
|----|------|--------------------------------------------------|
| `reversion_quality` | Score 0–3 | 0 = metrics indicate no reliable mean-reverting relationship … 3 = strong stationary relationship, sensible hedge ratio and half-life |
| `entry_attractiveness` | Score 0–3 | 0 = spread near mean / no edge … 3 = spread far outside normal range with reversion evidence |
| `executability` | Score 0–2 | 0 = illiquid legs, extreme `vol_ratio`/ATR% or crash-prone skew … 2 = both legs liquid, balanced risk, sane behavior |
| `direction_consistent` | Noul | "Given the data, is the proposed long/short direction consistent with the relationship?" |
| `red_flag` | Noul | "Is there anything in the metrics that should veto trading this pair (e.g. p-value barely below threshold, extreme volatility, very short history)?" |

### 4.4 Output contract and ranking

```python
normalized = answers["reversion_quality"].score / 3          # 0..1 (docs: divide by top level)
entry      = answers["entry_attractiveness"].score / 3
execut     = answers["executability"].score / 2

composite  = 0.45*normalized + 0.35*entry + 0.20*execut       # weights in config.toml
```

Recommendation requires **all** of (JEV score is never the only safeguard):
1. All hard filters passed (§3 step 5);
2. `composite ≥ min_composite` (default 0.60, tunable);
3. `confidence` on `reversion_quality` ≥ `min_confidence` (default 0.40 — conservative
   starting gate per the confidence docs; to be calibrated during evaluation);
4. `direction_consistent.noul ≥ 0.60`;
5. `red_flag.noul < 0.50`.

Rank order = `composite` descending. Actions: `ENTER` (all gates pass, top-K),
`WATCH` (passes filters but fails a JEV gate), `REJECT` (with recorded reason).
No-Trade report emitted when the ENTER list is empty.

**Degraded mode:** if the JEV API is unavailable/fails after retries, the pipeline
falls back to a transparent rule-based statistical score, and the report is explicitly
marked "JEV unavailable — rule-based ranking only". Failures are never silent.

### 4.5 Limitations of JEV for this task (documented, not hidden)

- No fine-tuning; domain knowledge must be injected via `metric_reference` and rubric
  text — quality depends on how well metrics are translated to language.
- Jev evaluates *language*, not raw numerical optimality; thresholding of numeric
  metrics stays in code (filters), JEV adds a semantic second opinion.
- No confidence on Noul answers; gates rely on the probabilities themselves.
- Cost/latency scale with candidate count (one request per candidate; all questions per
  candidate batched in one call). Concurrency capped (default 4 workers) + result cache
  keyed by (pair, train-window end).
- Model judgments are calibrated but not omniscient — hence hard filters, confidence
  gates, and out-of-sample validation (§6) remain mandatory.

---

## 5. Backtest / evaluation design

**Data reality:** 44 pairs × 1h bars × 2025-07-02→2025-09-02 (~62 days) — the only
in-repo dataset. Walk-forward, leakage-safe:

```
Fold 1:  train = day 0–40        test = day 41–62
Fold 2:  train = day 0–50 (roll) test = day 51–62        (if data permits)
```

All screening stats (p-value, half-life, Hurst, correlations, betas) computed on train
only. Trading simulated on test only, with:
- signal from rolling z-score (window 15 bars as in notebook 1), entry |z|≥1.5,
  exit at |z|≤0.5, stop at |z|≥3.5 (the notebook's looser ±1 entry is retained as a
  documented config alternative — it trades far more and fee sensitivity is evaluated
  in §5);
- execution at next bar close (`shift(1)`), both legs;
- fees 5 bps/side/leg + 2 bps slippage (configurable);
- leg sizing: **vol-balanced** notionals (inverse-σ weighting from the volatility
  module — the volatility notebook's useful contribution), with equal-notional as a
  configurable fallback; both sizing modes are compared in the evaluation to measure
  whether the volatility layer actually helps.

**Arms compared on identical test windows:**
1. **Original notebook logic** (reimplemented faithfully: in-sample selection, levels
   correlation for the correlation strategy, same-bar execution, no fees) —
   demonstrates the selection/look-ahead bias quantitatively.
2. **Consolidated, no JEV** — ranked by the rule-based statistical score.
3. **Consolidated + JEV ranking** — full pipeline.

**Metrics (chosen because they answer distinct questions):**
- *Strategy quality:* out-of-sample annualized Sharpe, max drawdown, per-trade hit rate
  (net of fees) — standard, fee-aware, comparable across arms.
- *Ranking quality:* Spearman correlation between each arm's rank score (train) and
  test Sharpe; precision@3 (share of top-3 picks with positive net OOS PnL) — measures
  whether the *ranking* (JEV's actual job) adds information, not just raw returns.
- *Stability:* cointegration persistence (train p<0.05 → test p<0.05 rate).

**Stated limitation:** ~62 days, one regime, one exchange, funding rates of the short
perp leg not modeled, and a small candidate count per fold. Results are directional
evidence about ranking usefulness, **not** proof of profitability. This will be stated
verbatim in the final report and README rather than papered over.

---

## 6. Telegram delivery

- `python-telegram-bot` v21+ (async), triggered via `asyncio.run` from the CLI.
- Env: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_GROUP_ID` (+ `TYPESAFE_API_KEY`); loaded from
  `.env` via `python-dotenv`; `.env` already gitignored.
- **Format:** HTML parse mode; all dynamic strings escaped with
  `telegram.helpers.escape_html`; hard 4096-char limit with chunk-on-boundary
  splitting; retry with backoff on `TimedOut`/`NetworkError`.
- **Message contents (from actually available data):** analysis timestamp, universe +
  candidate counts, per-candidate: rank, pair + long/short legs, JEV composite +
  confidence, key stats (p-value, half-life, z), recommended action (ENTER/WATCH),
  one-line reason (top contributing JEV dimension + filter summary), invalidation
  (z-score stop, half-life re-check), and a clear "NO TRADE — no candidate passed
  filters/gates" block when applicable.
- **Dry-run default:** `--dry-run` prints the exact payload to stdout/logs; sending
  requires explicit flag + credentials.

---

## 7. Tests (pytest)

| Test | Covers |
|------|--------|
| `test_cointegration.py` | synthetic cointegrated vs random-walk series: p-value ordering, hedge-ratio recovery, half-life on an AR(1) with known φ |
| `test_features.py` | metric computation edge cases (NaN windows, short series → graceful skip) |
| `test_filters.py` | each hard rule passes/blocks with recorded reason |
| `test_ranker.py` | normalization, weighting, all 5 gates, ranking order, degraded-mode fallback |
| `test_volatility.py` | ATR/price on a crafted series, standard CHOP formula, vol_ratio, vol-balanced sizing ratio |
| `test_backtest.py` | shift(1) execution (no same-bar fill), fee math, exit/stop logic on crafted spreads |
| `test_report.py` | HTML escaping, 4096 chunking, NO-TRADE block, message determinism |
| `test_config.py` | env precedence, invalid config rejection |

## 8. Setup & execution (planned UX)

All credentials live **only** in `.env` — `TYPESAFE_API_KEY` (already set on this
machine), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_GROUP_ID` (placeholders appended, values to
be filled). `config.toml` holds tunables, never secrets. `.env.example` documents the
required keys.

Local / interactive:

```bash
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill TELEGRAM_BOT_TOKEN, TELEGRAM_GROUP_ID (TYPESAFE_API_KEY already set)
cp config.example.toml config.toml

python -m pair_scout run --dry-run          # full pipeline, report to stdout
python -m pair_scout run                    # send to Telegram
python -m pair_scout run --no-jev           # rule-based ranking only
python -m pair_scout evaluate               # walk-forward comparison (3 arms), writes report
pytest                                      # tests
```

Server deployment notes (the user will run this on a server):

- **Headless by design** — the pipeline produces reports, not figures; no matplotlib
  display backend needed. Notebook plotting helpers are not imported.
- **Scheduling** — designed for cron, e.g. daily after the 00:00 UTC candle closes:
  `15 0 * * * cd /path/to/crypto-trading-analysis && ./venv/bin/python -m pair_scout run >> pair_scout.log 2>&1`
- **Logging** — structured logs to stdout/file (`pair_scout.log`); timestamps in UTC;
  one line per pipeline stage; JEV request failures logged with status and retry count.
- **Safety defaults on a server** — `--dry-run` is the default of the `run` command
  unless `--send` is passed explicitly; missing/blank `TELEGRAM_*` values make the
  sender fail fast with a clear error instead of silently skipping delivery.
- **Fresh data** — `python data_manager.py -c binance -i 1h -l 1500` can precede a run
  (cron step or `--refresh-data` flag wrapping the existing CLI) so the screener always
  sees current candles.
- **Config precedence** — CLI flags > `config.toml` > env vars (non-secret) > `.env`
  (secrets).

## 9. Example Telegram report (target shape)

```
📊 PairScout — 22 Sep 2026 01:15 UTC
Universe: 44 pairs · candidates scanned: 946 · passed filters: 7 · JEV scored: 7

1) 🟢 ENTER  LONG BNBUSDT / SHORT ENAUSDT
   JEV 0.74 (conf 0.81) · p=0.004 · HL=6.2d · z=2.1 · vol ratio 1.8
   Sizing hint: 63% notional long / 37% short (vol-balanced)
   Reason: strong reversion quality + attractive entry z-score.
   Invalidate: |z|>3.5 stop; re-check p-value weekly.
2) 🟡 WATCH  LONG LTCUSDT / SHORT XLMAUSDT
   JEV 0.58 (conf 0.35) · p=0.031 · HL=11d · z=1.6
   Reason: passes filters; JEV confidence below gate.

NO TRADE above threshold? — 1 candidate(s) cleared all gates.
⚠️ Not financial advice. Analysis only — no orders are placed.
```

## 10. Unresolved decisions (defaults chosen, easily changed)

1. **Default dataset** for the live screener: existing `binance/1h` pkl data
   (only complete local dataset). Config-overridable; `data_manager.py` can refresh.
2. **Weights** in the composite score (0.45/0.35/0.20) and gates (0.60 / 0.40 / 0.60 /
   0.50) are starting values to be calibrated on the evaluation fold — the docs
   explicitly recommend thresholding against your own data.
3. **Whether the beta-neutral basket module participates in v1 ranking** or ships as a
   scored-but-optional analysis (it produces baskets, not pairs). Default: include as
   secondary metrics (betas shown per leg), not a separate candidate type.
4. **Funding-rate modeling** for perpetual shorts is out of scope v1 (no funding data
   in repo); noted in every report.
5. **Token unlock/emission calendar** (the correlation notebook's qualitative driver
   for "which asset is weak") requires an external data source not present in the repo
   — out of scope v1; momentum proxies stand in for it. Recorded as a future feature,
   not a silent omission.
