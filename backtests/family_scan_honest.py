"""Honest 8-slot family scan: re-rank ALL signal families under truthful
accrual (the iteration-11 scan used entry-bar credit), plus 4 NEW families.

Existing (build_signals): psar, donchian, donchian_10_5, donchian_40_20, macd,
bollinger, keltner, supertrend, rsi_regime, adx_donchian, donchian_vol,
donchian_trend, range_expansion, ema_cross_20_10.
New here:
  rsi2_rev : RSI(2) mean reversion (long RSI2<5 -> exit >60; short mirror)
  bb_rev   : Bollinger fade (close below 20d-2sigma band -> exit at mid)
  squeeze  : BB-in-Keltner squeeze then close outside BB (exit at mid)
  fund_fade: funding-extreme reversal (last funding <= -0.1%/8h -> long until
             funding > 0; short mirror)

Sim: identical mechanics to backtests/champion_honest.py / technique_research.py
- 8 slots x 1/8, entry at flip-bar close, accrual from next bar
- exits: state DEACTIVATION (the exit-state frames passed in sig are NOT
  consulted — embed the exit rule in the state frames, see
  backtests/iterate14.py squeeze2) / 5% stop / 7d cap
- 7bps+2bps costs + real funding; optional overlay: blowoff/ivol/voltarget/
  riskoff
Run:  ./venv/bin/python backtests/family_scan_honest.py dailyscan # 24m daily scan
      ./venv/bin/python backtests/family_scan_honest.py scan      # bare, nogate
      ./venv/bin/python backtests/family_scan_honest.py full psar # 4 vars + overlay
Data: PS_DATA_DIR (default saved_data_12m)
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
    _atr, _ema, _resample_ohlcv, _rsi, _states_to_signals, build_signals,
    universe_mask,
)
from pair_scout.research_sizing import (
    bootstrap_sharpe_ci, max_dd_of, sharpe_of,
)

FEE = 7e-4
SLIP_BPS = 2.0
BTC = "BTCUSDT"
SLOTS = 8
SLOT_FRAC = 1.0 / SLOTS
STOP = 0.05
CAP_BARS = 7 * 24
UNIV = 200
DECISION_HOUR_UTC = 2
BLOWOFF_MAX = 0.25
VOL_WIN = 7 * 24

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

dollar = pd.DataFrame(
    {s: panel.frames[s]["Volume"] * panel.frames[s]["Close"] for s in panel.pairs})
rank_df = dollar.rolling(30 * bpd).mean().rank(axis=1, ascending=False)
mom7_df = closes / closes.shift(7 * bpd) - 1.0
ret1d_df = closes / closes.shift(bpd) - 1.0
atr_np, rank_np, mom_np, r1_np = {}, {}, {}, {}
for s in panel.pairs:
    f = panel.frames[s]
    p = f["Close"].shift(1)
    tr = pd.concat([f["High"] - f["Low"], (f["High"] - p).abs(),
                    (f["Low"] - p).abs()], axis=1).max(axis=1)
    atr_np[s] = (tr.rolling(bpd).mean() / f["Close"]).values
    rank_np[s] = rank_df[s].values
    mom_np[s] = mom7_df[s].values
    r1_np[s] = ret1d_df[s].values
vol_mat = np.column_stack([
    (closes[s] / closes[s].shift(1) - 1.0).rolling(VOL_WIN).std().values
    for s in panel.pairs])
med_vol_bar = np.nanmedian(vol_mat, axis=1)
PAIR_IX = {s: j for j, s in enumerate(panel.pairs)}

last_fund = fund_cost.replace(0.0, np.nan).ffill()


def custom_signals(name):
    h4, l4, c4, _v4 = _resample_ohlcv(panel)
    if name == "rsi2_rev":
        r2 = _rsi(c4, 2)
        la, le = r2 < 5, r2 > 60
        sa, se = r2 > 95, r2 < 40
    elif name == "bb_rev":
        mid = c4.rolling(20 * 6).mean()
        sd = c4.rolling(20 * 6).std(ddof=0)
        la, le = c4 < mid - 2 * sd, c4 > mid
        sa, se = c4 > mid + 2 * sd, c4 < mid
    elif name == "squeeze":
        mid = _ema(c4, 20 * 6)
        atr = _atr(h4, l4, c4, 14 * 6)
        up, dn = mid + 2 * atr, mid - 2 * atr
        sd = c4.rolling(20 * 6).std(ddof=0)
        bup, bdn = mid + 2 * sd, mid - 2 * sd
        sq = (bup < up) & (bdn > dn)
        la = (c4 > bup) & sq.shift(1).fillna(False)
        le = c4 < mid
        sa = (c4 < bdn) & sq.shift(1).fillna(False)
        se = c4 > mid
    elif name == "fund_fade":
        lf4 = last_fund.reindex(c4.index, method="ffill")
        la, le = lf4 <= -0.001, lf4 > 0
        sa, se = lf4 >= 0.001, lf4 < 0
    if name == "intraday_rev":
        # hourly-native mean reversion: buy a 1h dip > 2x daily ATR, hold 6h
        out = {}
        for s in panel.pairs:
            f = panel.frames[s]
            c = f["Close"]
            p = c.shift(1)
            tr1 = pd.concat([f["High"] - f["Low"], (f["High"] - p).abs(),
                             (f["Low"] - p).abs()], axis=1).max(axis=1)
            atr1 = (tr1.rolling(24).mean() / c).fillna(0.0)
            r1 = (c / p - 1.0).fillna(0.0)
            dip = (r1 < -2.0 * atr1).astype(bool)
            spk = (r1 > 2.0 * atr1).astype(bool)
            la = dip.rolling(6).max().fillna(0).astype(bool)
            sa = spk.rolling(6).max().fillna(0).astype(bool)
            out[s] = (la, ~la, sa, ~sa)
        return out
    if name in ("pa_sweep", "pa_inside", "pa_outside", "pa_pdh",
                "pa_strong_close", "pa_3bar"):
        # price-action families on 4h bars (no Open column: use range/close-pos)
        h, l, c = h4.fillna(np.nan), l4.fillna(np.nan), c4
        rng = h - l
        cpos = ((c - l) / rng.replace(0.0, np.nan)).clip(0, 1)
        H12 = 12  # hold 12 x 4h bars = 2 days
        H6 = 6
        if name == "pa_sweep":
            swept_lo = l < l.rolling(12).min().shift(1)
            swept_hi = h > h.rolling(12).max().shift(1)
            la = (swept_lo & (cpos > 0.7))
            sa = (swept_hi & (cpos < 0.3))
            return _states_to_signals(
                la.rolling(H12).max().fillna(0).astype(bool),
                ~(la.rolling(H12).max().fillna(0).astype(bool)),
                sa.rolling(H12).max().fillna(0).astype(bool),
                ~(sa.rolling(H12).max().fillna(0).astype(bool)),
                idx, c4=c4)
        if name == "pa_inside":
            inside = (h <= h.shift(1)) & (l >= l.shift(1))
            la = (inside.shift(1).fillna(False) & (c > h.shift(1)))
            sa = (inside.shift(1).fillna(False) & (c < l.shift(1)))
            return _states_to_signals(
                la.rolling(H12).max().fillna(0).astype(bool),
                ~(la.rolling(H12).max().fillna(0).astype(bool)),
                sa.rolling(H12).max().fillna(0).astype(bool),
                ~(sa.rolling(H12).max().fillna(0).astype(bool)),
                idx, c4=c4)
        if name == "pa_outside":
            outside = (h > h.shift(1)) & (l < l.shift(1))
            la = outside & (cpos > 0.6)
            sa = outside & (cpos < 0.4)
            return _states_to_signals(
                la.rolling(H12).max().fillna(0).astype(bool),
                ~(la.rolling(H12).max().fillna(0).astype(bool)),
                sa.rolling(H12).max().fillna(0).astype(bool),
                ~(sa.rolling(H12).max().fillna(0).astype(bool)),
                idx, c4=c4)
        if name == "pa_pdh":
            pdh = c.rolling(6).max().shift(6)  # prior day's high (4h grid)
            pdl = c.rolling(6).min().shift(6)
            la = (c > pdh)
            sa = (c < pdl)
            return _states_to_signals(
                la.rolling(H12).max().fillna(0).astype(bool),
                ~(la.rolling(H12).max().fillna(0).astype(bool)),
                sa.rolling(H12).max().fillna(0).astype(bool),
                ~(sa.rolling(H12).max().fillna(0).astype(bool)),
                idx, c4=c4)
        if name == "pa_strong_close":
            la = (cpos > 0.85)
            sa = (cpos < 0.15)
            return _states_to_signals(
                la.rolling(H6).max().fillna(0).astype(bool),
                ~(la.rolling(H6).max().fillna(0).astype(bool)),
                sa.rolling(H6).max().fillna(0).astype(bool),
                ~(sa.rolling(H6).max().fillna(0).astype(bool)),
                idx, c4=c4)
        if name == "pa_3bar":
            up3 = (c > c.shift(1)) & (c.shift(1) > c.shift(2)) & \
                  (c.shift(2) > c.shift(3))
            dn3 = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & \
                  (c.shift(2) < c.shift(3))
            la = up3
            sa = dn3
            return _states_to_signals(
                la.rolling(H6).max().fillna(0).astype(bool),
                ~(la.rolling(H6).max().fillna(0).astype(bool)),
                sa.rolling(H6).max().fillna(0).astype(bool),
                ~(sa.rolling(H6).max().fillna(0).astype(bool)),
                idx, c4=c4)
        raise ValueError(name)
    return _states_to_signals(la, le, sa, se, idx, c4=c4)


CUSTOM = {"rsi2_rev", "bb_rev", "squeeze", "fund_fade", "intraday_rev",
          "pa_sweep", "pa_inside", "pa_outside", "pa_pdh", "pa_strong_close",
          "pa_3bar"}


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


def sim(sig, hourly, gate, tech=frozenset()):
    la_arr = {}
    for s, (la, _le, sa, _se) in sig.items():
        la_arr[s] = (la.values.astype(bool), sa.values.astype(bool))
    dec = ([i for i in range(30 * bpd, len(idx))]
           if hourly else
           [i for i in range(30 * bpd, len(idx)) if idx[i].hour == DECISION_HOUR_UTC])
    open_tr, trades = [], []
    for k, i in enumerate(dec):
        still = []
        for tr in open_tr:
            d = tr["dir"]
            la_v, sa_v = la_arr[tr["sym"]]
            flipped = (not la_v[i]) if d == "LONG" else (not sa_v[i])
            c_i = float(closes[tr["sym"]].iloc[i])
            adverse = ((c_i / tr["px0"] - 1.0 <= -STOP) if d == "LONG"
                       else (c_i / tr["px0"] - 1.0 >= STOP))
            expired = (i - tr["i0"]) >= CAP_BARS
            if flipped or adverse or expired:
                reason = "flip" if flipped else ("stop" if adverse else "time")
                tr.update(j0=i, exit_px=c_i, reason=reason)
                trades.append(tr)
            else:
                still.append(tr)
        open_tr = still
        held = {t["sym"] for t in open_tr}
        cands = []
        if hourly:
            lookback = i - 1
        else:
            lookback = dec[k - 1] if k else i - 24
        for symv, (la_v, sa_v) in la_arr.items():
            if symv == BTC or symv in held or symv not in PAIR_IX:
                continue
            if not bool(in_u_all[symv].iloc[i]):
                continue
            # entries only on fresh ACTIVATION (state families go active ->
            # neutral, so a deactivation is an exit, not an opposite entry)
            long_flip = bool(la_v[i]) and not bool(la_v[lookback])
            short_flip = bool(sa_v[i]) and not bool(sa_v[lookback])
            if long_flip:
                direction = "LONG"
            elif short_flip:
                direction = "SHORT"
            else:
                continue
            if "blowoff" in tech:
                r1 = r1_np[symv][i]
                if np.isfinite(r1) and abs(r1) > BLOWOFF_MAX:
                    continue
            sc = score_at(symv, i, direction)
            if sc is None or (gate is not None and sc <= gate):
                continue
            cands.append((-sc, symv, direction, sc))
        cands.sort()
        for _neg, symv, direction, sc in cands:
            if len(open_tr) >= SLOTS:
                break
            if "ivol" in tech:
                v, mv = vol_mat[i, PAIR_IX[symv]], med_vol_bar[i]
                w = (float(np.clip((mv / v) * SLOT_FRAC,
                                   0.5 * SLOT_FRAC, 2.0 * SLOT_FRAC))
                     if (np.isfinite(v) and v > 0 and np.isfinite(mv) and mv > 0)
                     else SLOT_FRAC)
            else:
                w = SLOT_FRAC
            open_tr.append({"sym": symv, "dir": direction, "i0": i, "j0": None,
                            "px0": float(closes[symv].iloc[i]), "score": sc,
                            "exit_px": None, "reason": "open", "w": w})
    for t in open_tr:
        t["j0"], t["exit_px"], t["reason"] = len(idx) - 1, float(closes[t["sym"]].iloc[-1]), "open"
        trades.append(t)

    net = pd.Series(0.0, index=idx)
    pc = closes.pct_change()
    tot_fee = 0.0
    for t in trades:
        i0, j0 = t["i0"] + 1, t["j0"]
        w = t.get("w", SLOT_FRAC)
        tot_fee += w * 2 * (FEE + SLIP_BPS * 1e-4)
        if j0 <= t["i0"]:
            continue
        sgn = 1.0 if t["dir"] == "LONG" else -1.0
        net.iloc[i0:j0 + 1] += (w * sgn
                                * pc[t["sym"]].iloc[i0:j0 + 1].fillna(0).values)
        net.iloc[i0:j0 + 1] -= (w * sgn
                                * fund_cost[t["sym"]].iloc[i0:j0 + 1].values)
    net_net = net - tot_fee / max(len(net), 1)
    expct = [(t["exit_px"] / t["px0"] - 1.0 if t["dir"] == "LONG"
              else t["px0"] / max(t["exit_px"], 1e-12) - 1.0) for t in trades]
    wins = sum(1 for e in expct if e > 0)
    return {"net": net_net, "n": len(trades), "trades": trades,
            "expct": float(np.mean(expct)) if expct else 0.0,
            "win": wins / len(trades) if trades else 0.0}


def report(label, r):
    if r["n"] == 0:
        print(f"| {label} | 0 | - | - | - | - | - |", flush=True)
        return
    lo, hi, pn = bootstrap_sharpe_ci(r["net"])
    days = len(r["net"]) / 24
    print(f"| {label} | {r['n']} | {sharpe_of(r['net']):.2f} | "
          f"[{lo:.2f},{hi:.2f}] P0 {pn:.0%} | "
          f"{r['net'].sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(r['net']):.1%} | {r['expct']:+.2%} |", flush=True)


ALL_FAMILIES = ["psar", "donchian", "donchian_10_5", "donchian_40_20", "macd",
                "bollinger", "keltner", "supertrend", "rsi_regime",
                "adx_donchian", "donchian_vol", "donchian_trend",
                "range_expansion", "ema_cross_20_10",
                "rsi2_rev", "bb_rev", "squeeze", "fund_fade", "intraday_rev"]

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    print("| family | variant | trades | SR(bar) | daily-block CI | ret | maxDD | expct |")
    print("|---|---|---|---|---|---|---|---|")
    if mode == "dailyscan":
        for fam in ALL_FAMILIES:
            try:
                sig = (build_signals(fam, panel) if fam not in CUSTOM
                       else custom_signals(fam))
            except Exception as e:
                print(f"| {fam} | BUILD FAILED: {e} | | | | | | |", flush=True)
                continue
            for gate, lab in ((None, "dy/nogate"), (80, "dy/gated")):
                r = sim(sig, False, gate)
                report(f"{fam} | {lab}", r)
    elif mode == "scan":
        for fam in ALL_FAMILIES:
            try:
                sig = (build_signals(fam, panel) if fam not in CUSTOM
                       else custom_signals(fam))
            except Exception as e:
                print(f"| {fam} | BUILD FAILED: {e} | | | | | | |", flush=True)
                continue
            for hourly, gate, lab in ((True, None, "hr/nogate"),
                                      (False, None, "dy/nogate")):
                r = sim(sig, hourly, gate)
                report(f"{fam} | {lab}", r)
    elif mode == "full":
        fam = sys.argv[2]
        sig = build_signals(fam, panel) if fam not in CUSTOM else custom_signals(fam)
        for tech, tlab in ((frozenset(), "bare"),
                           (frozenset(("blowoff", "ivol")), "+blowoff,ivol")):
            for hourly, gate, lab in ((True, 80, "hr/gated"), (True, None, "hr/nogate"),
                                      (False, 80, "dy/gated"), (False, None, "dy/nogate")):
                r = sim(sig, hourly, gate, tech)
                report(f"{fam} | {tlab} | {lab}", r)
