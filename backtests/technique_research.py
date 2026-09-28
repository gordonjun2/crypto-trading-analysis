"""Technique research harness: single-technique sweeps on the 8-slot champion.

FINAL VERDICT (2026-09-27, round 16 — EVALUATION PROTOCOL). All past-data
results are IN-SAMPLE (family selection + params + eval on the same 24m).
Robustness checks used: daily-block bootstrap CIs, H1/H2 splits, parameter
plateaus, 12m cross-dataset (overlapping window — composition sensitivity
only), a-priori mechanisms. TRUE out-of-sample = forward data only.
(a) WALK-FORWARD (round 16, iterate14 grid k{5,6} x vol{3,4} x l1{0.5,0.75},
monthly roll, trailing-12m estimation): OOS SR 6.76 P0 0% vs fixed params
6.83 on the same months — parameter-instability risk is LOW (flat plateau);
the estimation never degrades the book.
(b) USER SPLIT (train 2024-09..2025-12, pseudo-OOS 2026-01..09 — window
contaminated by prior inspection): FADE v5 train 4.95 / test 6.98; squeeze
0.88 / 1.76; pair 4.20 / 6.69. No out-of-window degradation; the 2026
regime is simply more active (fade-friendly).
(c) GOING-FORWARD RULE: production configs FROZEN; new ideas adopted only
if they improve WALK-FORWARD OOS (not full-sample SR); real validation =
forward paper-trading via the live journal (4-8 weeks minimum).

FINAL VERDICT (2026-09-28, round 18 — PRODUCTION v6). HONEST FILLS +
COOLDOWN: (i) tranche/origin exits modeled as RESTING LIMITS filled on
intrabar wick touch (iterate17 fade_sim3 touch=True) — the close-based
model filled tranches at overshoot prices and overstated return by ~14pp/yr
(+185% -> +171%/yr) at identical SR; (ii) 12h same-symbol re-entry cooldown
(plateau 12-24h flat, binds on ~11/848 trades, improves BOTH halves:
H1 4.30->4.81, train 4.86->5.22, full 5.77->6.00) — churn re-entries after
a completed fade are systematically bad. FADE v6: SR 6.00, +176%/yr,
DD -13.7% (12m cross-check 7.12). Pair 50/50: SR 5.21, +105%/yr,
DD -9.7%, worst month -0.2%. Vol-targeted pair rejected (+0.2 SR, worse
DD); inverse-vol weights rejected (4.77).

FINAL VERDICT (2026-09-28, round 17 — post-v5 battery). ALL FAIL:
squeeze trigger conditioning (compression duration >= 6/18 bars 0.92/0.63,
tightness pct <= 30% 1.04, volume dry-up 0.56 vs 1.24 production); squeeze
stop/cap sweep (stop 4/6% flat 1.24/1.30; cap 10/14d RAISES DD to
-40/-49% — mid-band exit IS the risk control); failed-breakout fade NEW
FAMILY dead (-0.78 P0 85%); fade new-high re-entry (kills half the trades,
3.54); hour-of-day windows (00-12 UTC carries, 12-24 weak — no adoptable
filter); slots 4/12 worse than 8 (5.53/5.64).

FINAL VERDICT (2026-09-28, rounds 21-24 — exhaustion sweep). ALL REJECTED,
production v6 + 60/40 stands: squeeze 2h/1h state grids degenerate (23/0
trades); retest entries (0.88); fade trailing-last tranche (5.72, origin
ride stays); k fine grid on v6 base — 4.0-5.0 is ONE flat plateau
(k4.25 6.26 / k4.75 6.22 / k4.5 6.16 / k5 6.00; DD rises as k falls;
user-split test window prefers k5) -> no adoption, k5 sits mid-plateau
(robustness positive); vol 2.75/3.25 noise; liquidity tiers HURT at every
cut (top-50 1.87, top-100 3.39, rank101-200 4.41, full 6.00) — the fade
edge lives in mid/low-liquidity pumps and universe DIVERSITY is part of
the edge; second-bar exhaustion too rare (47 trades); weekend/weekday
sub-books both weaker than combined (3.91/4.59 vs 6.00). Pattern across
17-24: adoptions were execution-honesty + one weight tilt; every
conditioning/filter idea reduced the book.

FINAL VERDICT (2026-09-28, round 20 — squeeze structure + pair weights).
ADOPTED 60/40 PAIR TILT: fade/squeeze 60/40 dominates 50/50 on every split
(full 5.73 vs 5.21, H1 4.26/3.78, H2 7.16/6.62, train 4.75/4.20,
test 7.21/6.69; DD -9.3% vs -9.7%; worst month -0.5%) — books are
uncorrelated and fade is the stronger book; inverse-vol a-priori agrees.
REJECTED: squeeze 50d-SMA trend alignment (1.20 flat); trigger volume
>= 1.5x (0.39 — no volume signal exists in squeeze breakouts: 3x family
fail, <= 0.8x dry-up fail, 1.5x fail). 12h state grid GENUINELY FAILS
(-0.12, DD -63.7%, 826 trades — shift(1) in _states_to_signals is in
grid-bar units so mapping stays causal; longer 12h holds bleed via the
5% stop).

FINAL VERDICT (2026-09-28, round 19 — FADE microstructure round 2). ALL
FAIL: fatigue filter (skip 2+/3+ closes in 48h) — identical book, cooldown
12h already subsumes it; time-based tranches 25%@18h/12h (5.54/5.44, worse
DD — retrace-based exits already optimal); pump-bar shape (close-low)/range
>= 0.7 hurts (5.22), <= 0.3 has only 2 trades in 24m (99.7% of 5xATR pump
bars close in the top 30% of their range); staggered entry flat (5.97,
lower ret).

FINAL VERDICT (2026-09-27, round 15 — PRODUCTION v5). 3-TRANCHE EXIT:
FADE v5 = spike >= 5xATR + vol >= 3x, top-200 alt -> FADE SHORT; realize
50% at 50% retracement, 25% at 75%, 25% rides to origin; cap 42h; ivol.
24m: SR 5.77, CI [0.91,1.44] P0 0%, +185%/yr, DD -14.7%; halves 4.53/6.93
both P0 0%; 21/23 months, worst -3.1%. Tranche plateau 40-60%/70-80% all
SR 5.5-5.9; cap 36-48h flat; cross-checked on 12m (5.84 -> 7.20).
Rejected this round: alternative splits (front-loaded 50%@50% is the
driver), adaptive level by vol bucket (4.51), funding conditioning (data
coverage too thin: 3 trades), BTC-quiet (fewer trades, no gain), w_cap
(noise), limit+tranche (worse). Squeeze refinements: 1h momentum
confirmation REJECTED (0.55 P0 23%); short-only ties both-sides (keep
both); overlays confirmed. PAIR 50/50: SR 5.16, P0 0%, +109%/yr,
DD -9.7%, 21/23 months, worst month -0.9%.

FINAL VERDICT (2026-09-27, round 14 — PRODUCTION v4). Scale-out exit
BREAKTHROUGH on the fade book (backtests/iterate14.py):
(a) HOURLY: FADE v4 = spike >= 5x 24h-ATR + volume >= 3x, top-200 alt ->
FADE SHORT; REALIZE HALF the position when price retraces 75% of the flash
move; remainder exits at origin (price = pre-flash close), cap 42h; 8 slots,
ivol. Rationale: many flashes retrace part-way then stall above origin to
the time-cap; banking half early converts that loss tail into wins.
SR 4.72, CI [0.70,1.24] P0 0%, +156%/yr, DD -15.0% (24m); halves 3.69/5.67
BOTH P0 0% (scale-out fixed the H1 weakness); 20/23 months, worst -4.8%.
Plateaus: scale 25-90% all P0 0%; cap 36-48h all SR 4.5-4.7; cross-checked
on the independent 12m dataset (3.14 -> 5.84). Limit entries, next-bar
execution, k/vol plateau tweaks: no further improvement.
(b) DAILY: SQUEEZE v2 - BB(20d) inside Keltner(14d,2xATR) compression,
close outside BB -> trade breakout direction, TRUE mid-band-cross exit
(state persists until c4 crosses mid; FS.sim exits on state deactivation,
so the exit must be embedded in the state frames). SR 1.24, CI [0.01,0.49]
P0 4%, +34%/yr, DD -17.5%; exit plateau mid>time>opp all positive; shorts
carry the edge (short-only 1.27 vs long-only -0.04); KC2.0/BB20 is the
plateau center (KC1.5 trade-starved, KC2.5 diluted).
(c) PAIR 50/50: SR 4.39, P0 0%, +95%/yr, DD -10.4%, 19/23 months, worst
month -1.8%; corr(fade,squeeze) -0.01.
Round 14c new-family hunt: funding_mom, vol_breakout (WITH volume
explosions), xs_momentum (7d cross-sectional), btc_lag — ALL FAIL at the
daily cadence over 24m (P0 28-90%). Only against-the-volume trades work.

FINAL VERDICT (2026-09-26, round 13b — PRODUCTION REVISION). PSAR RETIRED
(fails 24m: every config negative). New production pair, both 24m-validated:
(a) HOURLY: FADE v3 (unchanged): 1h close > +5x 24h-ATR, volume >= 3x,
top-200 alt -> FADE SHORT; exit at pre-flash close (cap 24h); 8 slots, ivol.
SR 1.53, CI [0.07,0.57] P0 2%, +45%/yr, DD -20.1%, 16/23 months positive.
(b) DAILY (NEW): SQUEEZE-BREAKOUT - BB(20d) inside Keltner(14d,2xATR)
compression, then close outside the BB -> trade the breakout direction,
exit at mid-band cross; 8 slots, ivol+voltarget+riskoff (no blowoff needed).
 dy/nogate over 24m: SR 1.04, CI [-0.01,0.42] P0 6%, +13%/yr, DD -14.8%,
halves +7%/+19% (both positive), worst month -3.0% (remarkably shallow).
~1 signal/day; actionable at the 10:00 SGT check. The ONLY family of 19
with P0 < 10% at the daily cadence over 24 months (all trend families
negative: psar/donchian/bollinger/keltner/supertrend/ema_cross/macd/...).
(c) PAIR 50/50 squeeze+fade: corr -0.00 (independent), SR 1.82,
CI [0.13,0.62] P0 1%, +29%/yr, DD -10.2%; split plateau 30/70..70/30 all
P0 <= 1% (SR 1.70-1.85). Compare retired-PSAR pair on the same 24m sample:
SR 1.19, DD -14.7% - the new pair beats it on every axis with honest data.
Round 9-12 combo numbers (SR 2.23-3.87) described the favorable last-12m
window only; magnitude is regime-dependent, sign is not.
ROUND 13 (24-MONTH BACKTEST). Data extended: saved_data_24m (657 pairs,
2024-09..2026-09, 17.5k hourly bars, funding fetched for 657 symbols;
PS_DATA_DIR env). FADE v3 SURVIVES the full 24 months (SR 1.53 P0 2%);
PSAR daily FAILS over 24 months (SR -0.82 P0 89% long-only gated; H1
significantly negative). Round 9-12 daily-cell results are regime-conditional.
ROUND 12 (12m window): daily LONG-only discovery: combo SR 3.87, DD -12%.
ROUND 11 (volume discovery): volume >= 3x filter: fade 2.77->3.14, combo
3.28->3.60, DD -15->-13.5%. Rejected: hour filters (16-24h UTC best),
funding filter, extremity sizing, combo riskoff.
ROUND 10 (fade v2 discovery): spike-only + revert-to-origin exit:
fade SR 2.77 (from 1.58), combo 3.28. Dump-fades avg -0.08% vs
spike-fades +1.36% - upward liquidation spirals revert, crashes do not.
Decision-hour structural: 00/06h ~0.95, 14/20h negative.
ROUND 9 (two-book discovery): fade book found (1h |ret| > 5x ATR24 -> fade,
hold 12h): SR 1.58 vs champion 1.44, corr -0.08, 50/50 combo SR 2.23.
Drift curve: gated 4h-flip candidates have NEGATIVE mean drift at every
horizon (+0.06% 1h -> -0.51% 24h) - flip alpha cannot be traded hourly.
ROUND 8 (hedge retest + price action): EVERY hedge worsens the champion
under causal accounting (BTC short 0.3x: SR 1.44->1.14, DD -19.6->-26.3%;
mom-basket short catastrophic) - iteration-10's no-hedge rule CONFIRMED
honestly. Six price-action families (sweep reversal, inside bar, outside
bar, prior-day-high breakout, strong close, 3-bar momentum): none beats
the champion on any variant; best was pa_strong_close hr/nogate +48%/yr
P0 12% but its daily variants are weak/negative (inconsistent -> noise).
ROUND 7 (hourly rescue attempts): execution timing (market/delay 2-12h/
pullback limits 1-3%/stops off+wide), a JEV-style walk-forward classifier
(TypeSafe Jev emulation: causal features -> p(win) -> filter; selects WORSE
trades, -38%/yr), an hourly-native intraday-reversion family (fees +
adverse selection destroy it), and a hybrid book (daily entries + hourly
exits: SR 1.46 -> 0.65). Every intraday action on 4h-grain information
loses; the edge exists only at the once-daily 02:00 UTC decision boundary.
ROUND 6 (risk overlays, post-causal-fix): riskoff (halve new entries while
book DD > 20%; plateau verified at 15/20/25) + voltarget cut maxDD on ALL
variants (dy/gated -28.6->-19.1%, dy/nogate -33.9->-25.5%, hr -44->-22%)
at modest SR cost; dy/nogate improves to P0 5%. Entry-delay rescue of
hourly REJECTED (dly4 hr/gated spike +39%/yr P0 12% but neighbors dly2/dly8
worse - fragile spike, not a mechanism).
ROUND 5 — CAUSAL SIGNAL FIX (dominates everything above): 4h states were
mapped to hourly bars from their label-bar START (up to 4h lookahead). After
the fix (research_ta_families._states_to_signals): HOURLY cadence has NO edge
for any family (artifact); DAILY survives and blowoff,ivol still helps
(dy/nogate P0 28% bare -> 7% with techniques). Score gate rejects counter-
momentum entries, so reversal families (bb_rev, ema_cross, rsi2) cannot run
in the gated book; their bare nogate edge does not survive the overlay.
Failed ideas (documented so they are not re-tested):
  round 1: regime (kills hourly), htf (no-op under gate, hurts nogate),
           atrstop (helps hourly, hurts daily nogate), cooldown, recency,
           voltarget (no effect)
  round 2: adx>=20 (halves trades, ADX lags fresh flips), chandelier 3xATR
           (exits hourly winners before flip), fundx (helps nogate only),
           corr 0.85 cap (near no-op), gross cap 1.25 (never binds)
  round 3: nomaj10/nomaj20 (excludes majors alpha), longonly/shortonly
           (both directions contribute - L/S is best)
  round 4: htfpsar (4h PSAR structurally incompatible with fresh 1h flips:
           at flip time the 4h SAR is still on the other side -> 0 trades)
  sensitivity: blowoff 0.20/0.30 and ivol wide/unclipped all worse-or-equal
           on >=2 variants, no cliffs -> a-priori params sit on a plateau.
Monthly consistency of final config: hr/nogate 12/13 positive months (worst
-1.0%), hr/gated 11/13, dy/nogate 11/13, dy/gated 8/13.

Techniques (all parameters a priori, no fitting):
  regime   : BTC 7d momentum must align with trade direction (LONG if >0)
  htf      : coin's own 7d momentum must align with trade direction
  ivol     : position weight = cross-sectional-median-vol / coin 7d vol,
             /8, clipped to [0.5/8, 2/8] (risk parity-ish, causal)
  atrstop  : stop = 2x 1-day ATR (adaptive) instead of fixed 5%
  blowoff  : skip entries with |24h move| > 25% (exhausted thrusts)
  cooldown : no re-entry on a sym within 48 bars (2d) of any exit
  recency  : flip must be within last 12 bars at decision (mainly daily)
  voltarget: gross scaled by min(1, expanding-median / BTC 30d vol)

Round 2 (on top of blowoff,ivol — the current strategy):
  adx      : require ADX(24h, Wilder) >= 20 at entry (chop filter)
  chand    : chandelier exit - trail 3x 1-day ATR from best price since entry
  fundx    : skip entry if last known funding against position > 0.1%/8h
  corr     : skip candidate if |30d corr| > 0.85 with any open position
  gross    : skip entry if ivol book gross would exceed 1.25x

Run from repo root:  ./venv/bin/python backtests/technique_research.py
Data: saved_data_12m
"""
import os
import sys
import logging
from collections import defaultdict

sys.path.insert(0, "/root/crypto-trading-analysis")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from pathlib import Path

from pair_scout.config import load_config
from pair_scout.data.funding import funding_rate_series, load_funding
from pair_scout.data.loader import load_panel
from pair_scout.research_ta_families import (
    _psar_states, psar_signals_panel, universe_mask,
)
from pair_scout.research_sizing import (
    bootstrap_sharpe_ci, max_dd_of, sharpe_of,
)

FEE = 7e-4
SLIP_BPS = 2.0
BTC = "BTCUSDT"
SLOTS = 8
BASE_W = 1.0 / SLOTS
STOP_FIX = 0.05
CAP_BARS = 7 * 24
UNIV = 200
DECISION_HOUR_UTC = 2
ATR_K = 2.0          # atrstop: stop distance = ATR_K x 1-day ATR
BLOWOFF_MAX = 0.25   # blowoff: skip |24h move| above this
COOLDOWN = 48        # cooldown bars
RECENCY = 12         # recency bars
VOL_WIN = 7 * 24     # realized vol window (7d of hourly bars)
ADX_WIN = 24         # ADX window (1 day of hourly bars)
ADX_MIN = 20.0       # adx: entry requires ADX above this (literature: 20-25)
CHAND_K = 3.0        # chandelier: trail = 3x 1-day ATR from best px (classic)
FUND_MAX_8H = 0.001  # fundx: skip if funding against position > 0.1%/8h (~110%/yr)
CORR_WIN = 30 * 24   # corr: lookback window
CORR_MAX = 0.85      # corr: skip candidate above this |corr| vs book
GROSS_CAP = 1.25     # gross: max book gross multiple

cfg = load_config(None)
panel = load_panel(
    cex=cfg.data.cex, interval=cfg.data.interval,
    data_dir=os.environ.get("PS_DATA_DIR", "saved_data_12m"),
    top_n_volume=658, min_history_bars=cfg.data.min_history_bars,
    max_nan_fraction=cfg.data.max_nan_fraction,
    trailing_volume_days=cfg.data.trailing_volume_days,
)
idx = panel.index
closes = panel.closes
bpd = panel.bars_per_day
funding = load_funding(
    Path(os.environ.get("PS_DATA_DIR", "saved_data_12m")) / "funding_rates.json")
fund_cost = pd.DataFrame(
    {s: funding_rate_series(s, funding, idx) for s in panel.pairs}
).fillna(0.0)
in_u_all = universe_mask(panel, UNIV)
sig = psar_signals_panel(panel)

dollar = pd.DataFrame(
    {s: panel.frames[s]["Volume"] * panel.frames[s]["Close"] for s in panel.pairs})
rank_df = dollar.rolling(30 * bpd).mean().rank(axis=1, ascending=False)
mom7_df = closes / closes.shift(7 * bpd) - 1.0
ret1d_df = closes / closes.shift(bpd) - 1.0

atr_np, rank_np, mom_np, r1_np, vol_np = {}, {}, {}, {}, {}
for s in panel.pairs:
    f = panel.frames[s]
    p = f["Close"].shift(1)
    tr = pd.concat([f["High"] - f["Low"], (f["High"] - p).abs(),
                    (f["Low"] - p).abs()], axis=1).max(axis=1)
    atr_np[s] = (tr.rolling(bpd).mean() / f["Close"]).values
    rank_np[s] = rank_df[s].values
    mom_np[s] = mom7_df[s].values
    r1_np[s] = ret1d_df[s].values
    vol_np[s] = (closes[s] / closes[s].shift(1) - 1.0).rolling(VOL_WIN).std().values

# cross-sectional median vol per bar (causal denominator for ivol weights)
vol_mat = np.column_stack([vol_np[s] for s in panel.pairs])
med_vol_bar = np.nanmedian(vol_mat, axis=1)

# ADX(ADX_WIN, Wilder) per sym — chop/whipsaw filter
adx_np = {}
for s in panel.pairs:
    f = panel.frames[s]
    h, l, c = f["High"], f["Low"], f["Close"]
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=f.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=f.index)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    n = ADX_WIN
    atrw = tr.ewm(alpha=1 / n, min_periods=n).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, min_periods=n).mean() / atrw
    mdi = 100 * mdm.ewm(alpha=1 / n, min_periods=n).mean() / atrw
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    adx_np[s] = dx.ewm(alpha=1 / n, min_periods=n).mean().values

# last known funding per sym (causal ffilled 8h rates) + return matrix for corr
last_fund = fund_cost.replace(0.0, np.nan).ffill()
R_np = {s: (closes[s] / closes[s].shift(1) - 1.0).values for s in panel.pairs}

# 4h-resampled PSAR state, mapped causally to hourly bars (shift 1 = last
# CLOSED 4h bar) — the literature-standard higher-TF alignment filter
htf_long, htf_short = {}, {}
for s in panel.pairs:
    f = panel.frames[s]
    h4 = f["High"].resample("4h").max().dropna()
    l4 = f["Low"].resample("4h").min().dropna()
    c4 = f["Close"].resample("4h").last().dropna()
    la4, sa4 = _psar_states(h4, l4, c4)
    st = (la4.astype(int) - sa4.astype(int)).shift(1)
    st_h = st.reindex(idx).ffill().fillna(0).values
    htf_long[s] = st_h > 0
    htf_short[s] = st_h < 0


def book_max_corr(symv, i, held):
    if not held:
        return 0.0
    lo = max(0, i - CORR_WIN + 1)
    x = R_np[symv][lo:i + 1].astype(float)
    if not np.isfinite(x).all():
        return 0.0
    x = x - x.mean()
    sxx = float(np.dot(x, x))
    if sxx <= 0:
        return 0.0
    mx = 0.0
    for os in held:
        y = R_np[os][lo:i + 1].astype(float)
        if not np.isfinite(y).all():
            continue
        y = y - y.mean()
        syy = float(np.dot(y, y))
        if syy <= 0:
            continue
        c = abs(float(np.dot(x, y)) / np.sqrt(sxx * syy))
        if c > mx:
            mx = c
    return mx

# BTC regime + vol target inputs
btc_mom = mom_np[BTC]
btc_r = (closes[BTC] / closes[BTC].shift(1) - 1.0).values
btc_vol30 = pd.Series(btc_r).rolling(30 * bpd).std().values
btc_med = pd.Series(btc_vol30).expanding(min_periods=30 * bpd).median().values

sig_arr = {s: (la.values.astype(bool), sa.values.astype(bool))
           for s, (la, _le, sa, _se) in sig.items()}

flips_at = {}
last_flip = {}
FLIPS_AT_BAR = defaultdict(list)
for symv, (la_v, sa_v) in sig_arr.items():
    la_chg = np.flatnonzero(la_v[1:] != la_v[:-1]) + 1
    la_last = np.zeros(len(la_v), dtype=np.int64)
    la_last[la_chg] = la_chg
    la_last = np.maximum.accumulate(la_last)
    sa_chg = np.flatnonzero(sa_v[1:] != sa_v[:-1]) + 1
    sa_last = np.zeros(len(sa_v), dtype=np.int64)
    sa_last[sa_chg] = sa_chg
    sa_last = np.maximum.accumulate(sa_last)
    last_flip[symv] = (la_last, sa_last)
    la_set = {int(i) for i in la_chg}
    for i in la_chg:
        FLIPS_AT_BAR[int(i)].append((symv, "LONG" if la_v[i] else "SHORT"))
    for i in sa_chg:
        if int(i) in la_set:
            continue
        FLIPS_AT_BAR[int(i)].append((symv, "SHORT" if sa_v[i] else "LONG"))

pc_np = {s: (closes[s] / closes[s].shift(1) - 1.0).fillna(0.0).values
         for s in panel.pairs}
fund_np = {s: fund_cost[s].values for s in panel.pairs}


def score_at(sym, i, direction):
    if not bool(np.isfinite(rank_np[sym][i])) or i < 7 * bpd:
        return None
    m = mom_np[sym][i]
    if not np.isfinite(m):
        return None
    aligned = (direction == "LONG") == (m > 0)
    trend = 40.0 * min(abs(m) / 0.30, 1.0) if aligned else 0.0
    a = atr_np[sym][i]
    if not np.isfinite(a) or a <= 0:
        return None
    r1 = r1_np[sym][i]
    if not np.isfinite(r1):
        return None
    thrust = 30.0 * min(abs(r1) / (2.0 * a), 1.0)
    rk = rank_np[sym][i]
    liq = 30.0 * max(0.0, 1.0 - (float(rk) - 1.0) / 300.0)
    return round(trend + thrust + liq)


def sim(hourly, gate, tech=None):
    tech = tech or set()
    dly = next((int(t[3:]) for t in tech if t.startswith("dly")), 0)
    hybrid = "hybrid" in tech  # daily entries, hourly exits
    dec = ([i for i in range(30 * bpd, len(idx))]
           if hourly else
           [i for i in range(30 * bpd, len(idx)) if idx[i].hour == DECISION_HOUR_UTC])
    dec_entry = (dec if not hybrid else
                 [i for i in dec if idx[i].hour == DECISION_HOUR_UTC])
    entry_set = set(dec_entry)
    if hybrid:
        dec = list(range(30 * bpd, len(idx)))
    last_entry_i = 30 * bpd - 24
    open_tr, trades = [], []
    last_exit_bar = {}
    net_arr = np.zeros(len(idx))
    fee_cost = FEE + SLIP_BPS * 1e-4
    last_acc = 30 * bpd - 1
    net_run, eq_max = 0.0, 0.0
    pending = []  # (enter_bar, sym, direction) — delayed entries
    risk_scale = 1.0

    def accrue_to(i):
        nonlocal last_acc, net_run
        if i > last_acc:
            for t in open_tr:
                s = t["sym"]
                sgn = 1.0 if t["dir"] == "LONG" else -1.0
                w = t.get("w", BASE_W)
                net_arr[last_acc + 1:i + 1] += (
                    w * sgn * (pc_np[s][last_acc + 1:i + 1]
                               - fund_np[s][last_acc + 1:i + 1]))
            net_run += float(net_arr[last_acc + 1:i + 1].sum())
            last_acc = i

    for k, i in enumerate(dec):
        accrue_to(i)
        if "riskoff" in tech or "riskoff15" in tech or "riskoff25" in tech:
            thr = (0.15 if "riskoff15" in tech else
                   0.25 if "riskoff25" in tech else 0.20)
            eq_max = max(eq_max, net_run)
            dd = 1.0 - net_run / eq_max if eq_max > 1e-12 else 0.0
            risk_scale = 0.5 if dd > thr else 1.0
        still = []
        for tr in open_tr:
            d = tr["dir"]
            la_v, sa_v = sig_arr[tr["sym"]]
            flipped = (not la_v[i]) if d == "LONG" else (not sa_v[i])
            c_i = float(closes[tr["sym"]].iloc[i])
            if "atrstop" in tech:
                a_i = atr_np[tr["sym"]][i]
                stop_d = ATR_K * a_i if (np.isfinite(a_i) and a_i > 0) else STOP_FIX
            else:
                stop_d = STOP_FIX
            adverse = ((c_i / tr["px0"] - 1.0 <= -stop_d) if d == "LONG"
                       else (c_i / tr["px0"] - 1.0 >= stop_d))
            expired = (i - tr["i0"]) >= CAP_BARS
            trail = False
            if "chand" in tech and not (flipped or adverse or expired):
                a_i = atr_np[tr["sym"]][i]
                if np.isfinite(a_i) and a_i > 0:
                    if d == "LONG":
                        tr["best"] = max(tr.get("best", tr["px0"]), c_i)
                        trail = c_i < tr["best"] * (1.0 - CHAND_K * a_i)
                    else:
                        tr["best"] = min(tr.get("best", tr["px0"]), c_i)
                        trail = c_i > tr["best"] * (1.0 + CHAND_K * a_i)
            if flipped or adverse or expired or trail:
                reason = ("flip" if flipped else ("stop" if adverse
                          else ("time" if expired else "trail")))
                tr.update(j0=i, exit_px=c_i, reason=reason)
                trades.append(tr)
                last_exit_bar[tr["sym"]] = i
            else:
                still.append(tr)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        cands = []
        if i not in entry_set:
            cands = None  # hybrid: exits only outside entry bars
        elif hourly:
            if dly:
                for symv, direction in FLIPS_AT_BAR.get(i, ()):
                    pending.append((i + dly, symv, direction))
                due = [p for p in pending if p[0] == i]
                if due:
                    pending = [p for p in pending if p[0] > i]
                for _ei, symv, direction in due:
                    la_v, sa_v = sig_arr[symv]
                    if not (la_v[i] if direction == "LONG" else sa_v[i]):
                        continue
                    if symv in held:
                        continue
                    if "recency" in tech and i - last_flip[symv][0 if direction == "LONG" else 1][i] > RECENCY:
                        continue
                    cands.append((symv, direction))
            else:
                src = FLIPS_AT_BAR.get(i, ())
                for symv, direction in src:
                    if symv in held:
                        continue
                    if "recency" in tech and i - last_flip[symv][0 if direction == "LONG" else 1][i] > RECENCY:
                        continue
                    cands.append((symv, direction))
        else:
            if dly and pending:
                due = [p for p in pending if p[0] == i]
                if due:
                    pending = [p for p in pending if p[0] > i]
                for _ei, symv, direction in due:
                    la_v, sa_v = sig_arr[symv]
                    if not (la_v[i] if direction == "LONG" else sa_v[i]):
                        continue
                    if symv in held or not bool(in_u_all[symv].iloc[i]):
                        continue
                    cands.append((symv, direction))
            lookback = last_entry_i
            for symv, (la_v, sa_v) in sig_arr.items():
                if symv == BTC or symv in held or symv not in closes.columns:
                    continue
                if not bool(in_u_all[symv].iloc[i]):
                    continue
                if la_v[i] != la_v[lookback]:
                    direction = "LONG" if la_v[i] else "SHORT"
                elif sa_v[i] != sa_v[lookback]:
                    direction = "SHORT" if sa_v[i] else "LONG"
                else:
                    continue
                if "recency" in tech:
                    lf = last_flip[symv][0 if direction == "LONG" else 1][i]
                    if lf == 0 or i - lf > RECENCY:
                        continue
                if dly:
                    if k + dly < len(dec):
                        pending.append((dec[k + dly], symv, direction))
                else:
                    cands.append((symv, direction))
        filt = []
        if cands is None:
            cands = []
        bo = BLOWOFF_MAX if "blowoff" in tech else None
        for t in tech:
            if t.startswith("bo") and len(t) > 2 and t[2:].isdigit():
                bo = int(t[2:]) / 100.0
        for symv, direction in cands:
            if symv == BTC:
                continue
            if "regime" in tech:
                bm = btc_mom[i]
                if not np.isfinite(bm):
                    continue
                if (direction == "LONG") != (bm > 0):
                    continue
            if "htf" in tech:
                m = mom_np[symv][i]
                if not np.isfinite(m) or (direction == "LONG") != (m > 0):
                    continue
            if "blowoff" in tech or bo is not None:
                r1 = r1_np[symv][i]
                if np.isfinite(r1) and abs(r1) > bo:
                    continue
            if "cooldown" in tech and symv in last_exit_bar \
                    and i - last_exit_bar[symv] < COOLDOWN:
                continue
            if "adx" in tech:
                a_ = adx_np[symv][i]
                if not np.isfinite(a_) or a_ < ADX_MIN:
                    continue
            if "fundx" in tech and symv in last_fund.columns:
                lf = last_fund[symv].iloc[i]
                if np.isfinite(lf) and ((direction == "LONG" and lf > FUND_MAX_8H)
                                        or (direction == "SHORT"
                                            and lf < -FUND_MAX_8H)):
                    continue
            maj_n = next((int(t[5:]) for t in tech if t.startswith("nomaj")), None)
            if maj_n is not None:
                rk_ = rank_np[symv][i]
                if np.isfinite(rk_) and rk_ <= maj_n:
                    continue
            if "htfpsar" in tech:
                if direction == "LONG" and not htf_long[symv][i]:
                    continue
                if direction == "SHORT" and not htf_short[symv][i]:
                    continue
            if "longonly" in tech and direction == "SHORT":
                continue
            if "shortonly" in tech and direction == "LONG":
                continue
            sc = score_at(symv, i, direction)
            if sc is None or (gate is not None and sc <= gate):
                continue
            if "corr" in tech and held and book_max_corr(symv, i, held) > CORR_MAX:
                continue
            filt.append((-sc, symv, direction, sc))
        filt.sort()
        for _neg, symv, direction, sc in filt:
            if len(open_tr) >= SLOTS:
                break
            if "ivol" in tech or "ivolw" in tech or "ivoln" in tech:
                v = vol_np[symv][i]
                mv = med_vol_bar[i]
                if np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0:
                    clo, chi = (1.0 / 3.0, 3.0) if "ivolw" in tech else (0.5, 2.0)
                    if "ivoln" in tech:
                        clo, chi = 0.0, float("inf")
                    w = float(np.clip((mv / v) * BASE_W, clo * BASE_W, chi * BASE_W))
                else:
                    w = BASE_W
            else:
                w = BASE_W
            if "voltarget" in tech:
                bv, bm_ = btc_vol30[i], btc_med[i]
                if np.isfinite(bv) and np.isfinite(bm_) and bv > 0:
                    w *= float(min(1.0, bm_ / bv))
            if "gross" in tech:
                book_g = sum(t.get("w", BASE_W) for t in open_tr)
                if book_g + w > GROSS_CAP:
                    continue
            w *= risk_scale
            net_arr[i] -= w * 2 * fee_cost
            open_tr.append({"sym": symv, "dir": direction, "i0": i, "j0": None,
                            "px0": float(closes[symv].iloc[i]), "score": sc,
                            "exit_px": None, "reason": "open", "w": w})
        if i in entry_set:
            last_entry_i = i
    accrue_to(len(idx) - 1)
    for t in open_tr:
        t["j0"], t["exit_px"], t["reason"] = len(idx) - 1, float(closes[t["sym"]].iloc[-1]), "open"
        trades.append(t)

    net_net = pd.Series(net_arr, index=idx)
    expct = [(t["exit_px"] / t["px0"] - 1.0 if t["dir"] == "LONG"
              else t["px0"] / max(t["exit_px"], 1e-12) - 1.0) for t in trades]
    wins = sum(1 for e in expct if e > 0)
    holds = [(t["j0"] - t["i0"]) / bpd for t in trades]
    return {"net": net_net, "trades": trades, "n": len(trades),
            "expct": float(np.mean(expct)) if expct else 0.0,
            "win": wins / len(trades) if trades else 0.0,
            "avg_hold": float(np.mean(holds)) if holds else 0.0}


TECHS = {
    "base": None,
    "regime": {"regime"},
    "htf": {"htf"},
    "ivol": {"ivol"},
    "atrstop": {"atrstop"},
    "blowoff": {"blowoff"},
    "cooldown": {"cooldown"},
    "recency": {"recency"},
    "voltarget": {"voltarget"},
}


def report(label, r):
    lo, hi, pn = bootstrap_sharpe_ci(r["net"])
    days = len(r["net"]) / 24
    half = len(r["net"]) // 2
    h1, h2 = sharpe_of(r["net"].iloc[:half]), sharpe_of(r["net"].iloc[half:])
    print(f"| {label} | {r['n']} | {sharpe_of(r['net']):.2f} | "
          f"[{lo:.2f},{hi:.2f}] P0 {pn:.0%} | "
          f"{r['net'].sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(r['net']):.1%} | {r['win']:.0%} | "
          f"{r['expct']:+.2%} | {h1:.2f} | {h2:.2f} |", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    print("| tech | cadence/gate | trades | SR(bar) | daily-block CI | ret | maxDD | win | expct | H1 SR | H2 SR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    if which == "sweep":
        for name, tech in TECHS.items():
            for cadence, gate, lab in ((True, 80, "hr/gated"), (False, 80, "dy/gated")):
                r = sim(cadence, gate, tech)
                report(f"{name} | {lab}", r)
    elif which == "sweep3":
        BASE = {"blowoff", "ivol"}
        for name, extra in (("blowoff,ivol", None), ("+nomaj10", {"nomaj10"}),
                            ("+nomaj20", {"nomaj20"}),
                            ("+longonly", {"longonly"}),
                            ("+shortonly", {"shortonly"})):
            tech = BASE | (extra or set())
            for cadence, gate, lab in ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                                       (False, 80, "dy/gated"), (False, None, "dy/nogate")):
                r = sim(cadence, gate, tech)
                report(f"{name} | {lab}", r)
    elif which == "sweep4":
        BASE = {"blowoff", "ivol"}
        for name, extra in (("blowoff,ivol", None), ("+htfpsar", {"htfpsar"})):
            tech = BASE | (extra or set())
            for cadence, gate, lab in ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                                       (False, 80, "dy/gated"), (False, None, "dy/nogate")):
                r = sim(cadence, gate, tech)
                report(f"{name} | {lab}", r)
    elif which == "sweep5":
        for name, tech in (("base", {"blowoff", "ivol"}),
                           ("bo20,ivol", {"bo20", "ivol"}),
                           ("bo30,ivol", {"bo30", "ivol"}),
                           ("blowoff,ivolw", {"blowoff", "ivolw"}),
                           ("blowoff,ivoln", {"blowoff", "ivoln"})):
            for cadence, gate, lab in ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                                       (False, 80, "dy/gated"), (False, None, "dy/nogate")):
                r = sim(cadence, gate, tech)
                report(f"{name} | {lab}", r)
    elif which == "final":
        tech = {"blowoff", "ivol"}
        for cadence, gate, lab in ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                                   (False, 80, "dy/gated"), (False, None, "dy/nogate")):
            r = sim(cadence, gate, tech)
            report(f"blowoff,ivol | {lab}", r)
            m = r["net"].resample("MS").sum()
            pos = int((m > 0).sum())
            print(f"  months: {pos}/{len(m)} positive | worst {m.min():+.1%} | "
                  f"best {m.max():+.1%} | median {m.median():+.1%}", flush=True)
    elif which == "sweep6":
        for name, tech, vsel in (
                ("base", {"blowoff", "ivol"}, "all"),
                ("dly2", {"blowoff", "ivol", "dly2"}, "hr"),
                ("dly4", {"blowoff", "ivol", "dly4"}, "hr"),
                ("dly8", {"blowoff", "ivol", "dly8"}, "hr"),
                ("dly12", {"blowoff", "ivol", "dly12"}, "hr"),
                ("dly1", {"blowoff", "ivol", "dly1"}, "dy"),
                ("voltarget", {"blowoff", "ivol", "voltarget"}, "dy"),
                ("riskoff", {"blowoff", "ivol", "riskoff"}, "dy"),
                ("riskoff", {"blowoff", "ivol", "riskoff"}, "hr_g")):
            if vsel == "hr":
                vs = ((True, 80, "hr/gated"), (True, None, "hr/nogate"))
            elif vsel == "dy":
                vs = ((False, 80, "dy/gated"), (False, None, "dy/nogate"))
            elif vsel == "hr_g":
                vs = ((True, 80, "hr/gated"),)
            else:
                vs = ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                      (False, 80, "dy/gated"), (False, None, "dy/nogate"))
            for cadence, gate, lab in vs:
                r = sim(cadence, gate, tech)
                report(f"{name} | {lab}", r)
    elif which == "sweep7":
        for name, tech, vsel in (
                ("riskoff15", {"blowoff", "ivol", "riskoff15"}, "dy"),
                ("riskoff25", {"blowoff", "ivol", "riskoff25"}, "dy"),
                ("vt+riskoff", {"blowoff", "ivol", "voltarget", "riskoff"}, "all"),
                ("riskoff", {"blowoff", "ivol", "riskoff"}, "hr_n")):
            if vsel == "dy":
                vs = ((False, 80, "dy/gated"), (False, None, "dy/nogate"))
            elif vsel == "hr_n":
                vs = ((True, None, "hr/nogate"),)
            else:
                vs = ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                      (False, 80, "dy/gated"), (False, None, "dy/nogate"))
            for cadence, gate, lab in vs:
                r = sim(cadence, gate, tech)
                report(f"{name} | {lab}", r)
    elif which == "sweep8":
        for name, tech in (("base", {"blowoff", "ivol"}),
                           ("hybrid d-entry/x-hourly",
                            {"blowoff", "ivol", "hybrid"})):
            for cadence, gate, lab in ((False, 80, "dy/gated"),
                                       (False, None, "dy/nogate")):
                r = sim(cadence, gate, tech)
                report(f"{name} | {lab}", r)
    elif which == "sweep2":
        BASE = {"blowoff", "ivol"}
        for name, extra in (("blowoff,ivol", None), ("+adx", {"adx"}),
                            ("+chand", {"chand"}), ("+fundx", {"fundx"}),
                            ("+corr", {"corr"}), ("+gross", {"gross"})):
            tech = BASE | (extra or set())
            for cadence, gate, lab in ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                                       (False, 80, "dy/gated"), (False, None, "dy/nogate")):
                r = sim(cadence, gate, tech)
                report(f"{name} | {lab}", r)
    else:  # combo: name -> comma list of techniques, e.g. combo:regime,ivol
        name = which.split(":", 1)[1]
        tech = set(name.split(",")) if name else None
        for cadence, gate, lab in ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                                   (False, 80, "dy/gated"), (False, None, "dy/nogate")):
            r = sim(cadence, gate, tech)
            report(f"{name or 'base'} | {lab}", r)
