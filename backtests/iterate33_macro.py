"""Round 33 (Track A): CRYPTO-LINKED MACRO — prediction-market odds as
regime gates for the existing v8/v9 books. SEPARATE from Track B (weather).
VERDICT (technique_research round 33): ALL GATES REJECTED — recession
level/slope gates degrade both books (blunt cuts strip PnL; the
"weak at odds<10.5%" gradient does not survive H1->H2), FOMC windows
are PROFITABLE (vetoes remove pnl). Macro odds = risk-off dashboard
only; forward archive accumulates for event studies.

Series (Polymarket CLOB, interval=all&fidelity=1440 = FULL daily history,
unlike the ~30d intraday pruning):
  - US recession by end of 2026 (token 10037920855962615102): 343d daily
  - Fed decision markets (Oct/Dec 2026): P(cut) build-up ~100d per event
Causality: daily close of market date D enters gates only from D+1 00:00
(shift(1)) — value known a full day, no leakage. FOMC decision dates are
the public schedule (federalreserve.gov), known years ahead.

Batteries:
  33a  12m-window baselines (anchor vs best4 xcheck SR 5.16 levered)
  33b  recession LEVEL: trade-outcome quartile gradient + gates
       (cuts fit on H1, evaluated on H2 + full-window book control)
  33c  recession SLOPE d20d: risk-off veto, same H1-fit/H2-test protocol
  33d  FOMC-window entry vetoes (full 24m window, public schedule)
  33e  Fed P(cut) level gradient (short series — gradient only)
Run:  PS_DATA_DIR=saved_data_24m ./venv/bin/python backtests/iterate33_macro.py [a..f]
"""
import sys

sys.path.insert(0, "/root/crypto-trading-analysis")
sys.path.insert(0, "/root/crypto-trading-analysis/backtests")

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import technique_research as TR
from iterate10 import idx
from iterate14 import squeeze2, SQ_OVER
from iterate25 import load_flow
from iterate31 import fade_sim9, cont_sim8, sq_stop_hybrid, gross_series
from pair_scout.research_sizing import bootstrap_sharpe_ci, max_dd_of, sharpe_of

CACHE = "saved_data_live/prediction"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def fetch_token_daily(tok, tries=3):
    for k in range(tries):
        try:
            h = _get(f"https://clob.polymarket.com/prices-history?market={tok}"
                     "&interval=all&fidelity=1440")
            pts = h.get("history", [])
            if pts:
                return pd.Series({pd.Timestamp(p["t"], unit="s").normalize(): p["p"]
                                  for p in pts}).sort_index()
        except Exception:
            pass
        time.sleep(2 * (k + 1))
    return pd.Series(dtype=float)


RECESSION_Q = "US recession by end of 2026"


def _find_token(question):
    q = urllib.parse.quote(question)
    evs = _get("https://gamma-api.polymarket.com/public-search"
               f"?q={q}&limit_per_type=4&events_status=all").get("events", [])
    for ev in evs:
        for m in ev.get("markets", []):
            if question.strip("? ").lower() in \
                    (m.get("question") or "").lower():
                toks = m.get("clobTokenIds")
                if toks:
                    return json.loads(toks)[0]
    return None


def load_recession(refresh=False):
    path = f"{CACHE}/recession26_daily.json"
    if os.path.exists(path) and not refresh:
        d = json.load(open(path))
        s = pd.Series({pd.Timestamp(k): v for k, v in d.items()})
    else:
        tok = _find_token(RECESSION_Q)
        if tok is None:
            return pd.Series(dtype=float)
        s = fetch_token_daily(tok)
        json.dump({str(k.date()): float(v) for k, v in s.items()},
                  open(path, "w"))
    return s.sort_index()


def load_fed_cut(refresh=False):
    """P(FOMC cut) from Oct 2026 decision markets (cut25 + cut50 YES)."""
    path = f"{CACHE}/fedcut_oct26_daily.json"
    if os.path.exists(path) and not refresh:
        d = json.load(open(path))
        return pd.Series({pd.Timestamp(k): v for k, v in d.items()})
    q = urllib.parse.quote("Fed decision in October")
    evs = _get("https://gamma-api.polymarket.com/public-search"
               f"?q={q}&limit_per_type=8&events_status=active").get("events", [])
    cuts = []
    for ev in evs:
        for m in ev.get("markets", []):
            t = (m.get("question") or "").lower()
            if "decrease" in t and ("25 bps" in t or "50" in t):
                toks = m.get("clobTokenIds")
                if toks:
                    cuts.append(json.loads(toks)[0])
    if not cuts:
        return None
    s = None
    for tok in cuts[:3]:
        h = fetch_token_daily(tok)
        s = h if s is None else s.add(h, fill_value=0).clip(0, 1)
        time.sleep(0.3)
    s = s.sort_index()
    json.dump({str(k.date()): float(v) for k, v in s.items()},
              open(path, "w"))
    return s


FOMC_DAYS = [  # public schedule, decision day = day 2 of meeting
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16",
]

WIN = pd.Timedelta(days=365)


def window(net):
    t0 = idx[-1] - WIN
    return net[net.index >= t0]


def line(label, net, n=None):
    lo, hi, pn = bootstrap_sharpe_ci(net)
    days = len(net) / 24
    nn = f" [{n}]" if n is not None else ""
    print(f"| {label}{nn} | {sharpe_of(net):.2f} | [{lo:.2f},{hi:.2f}] "
          f"P0 {pn:.0%} | {net.sum() * 365 / days:+.0%}/yr | "
          f"{max_dd_of(net):.1%} |", flush=True)


def hourly_from_daily(s):
    """Daily close of day D -> hourly values, usable from D+1 00:00 UTC."""
    return s.shift(1).reindex(pd.Series(idx).dt.normalize().values).values


# cached sim runs -----------------------------------------------------------
def sims(flow):
    f9 = fade_sim9(tf=flow, cluster_gate=3)
    c9 = cont_sim8(tf=flow, stop_pct=0.06)
    s9 = sq_stop_hybrid(0.05, lev=2.0)
    return f9, c9, s9


# ---------------------------------------------------------------- battery a
def battery_a(flow):
    print("## Round 33a — 12m-window baselines (levered v9; anchor 5.16)")
    print("| variant | SR(bar) | CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    f9, c9, s9 = sims(flow)
    tri = 0.60 * f9[0] + 0.25 * s9[0] + 0.15 * c9[0]
    line("FADE v9 @5x", window(f9[0]))
    line("SQUEEZE v9 @2x", window(s9[0]))
    line("CLIM v9 @5x", window(c9[0]))
    line("TRIO levered", window(tri))
    tri1 = 0.60 * f9[0] / 5 + 0.25 * s9[0] / 2 + 0.15 * c9[0] / 5
    line("TRIO 1x", window(tri1))
    return f9, c9, s9


# ---------------------------------------------------------------- battery b
def battery_b(flow, f9, c9):
    rec = load_recession()
    print(f"\n## Round 33b — recession-26 LEVEL ({rec.index[0].date()}.."
          f"{rec.index[-1].date()}, {len(rec)}d, last {rec.iloc[-1]:.3f})")
    fg = hourly_from_daily(rec)
    f_net, f_tr, f_rc = f9
    c_net, c_tr = c9[0], c9[1]

    def trade_gradient(trades, name):
        rows = []
        for t in trades:
            if "exit_px" not in t:
                continue
            d = pd.Timestamp(idx[t["i0"]]).normalize()
            v = rec.get(d - pd.Timedelta(days=1))
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            sgn = -1.0 if t.get("dir") == "SHORT" else 1.0
            ret = sgn * (t["exit_px"] / t["px0"] - 1)
            rows.append((v, ret))
        if len(rows) < 40:
            print(f"  {name}: {len(rows)} dated trades — too few")
            return
        q = pd.DataFrame(rows, columns=["p", "ret"])
        qs = q["p"].quantile([0.25, 0.5, 0.75]).values
        labs = [f"p<{qs[0]:.3f}", f"{qs[0]:.3f}..{qs[1]:.3f}",
                f"{qs[1]:.3f}..{qs[2]:.3f}", f">{qs[2]:.3f}"]
        print(f"  {name} trade outcome by recession-odds quartile "
              f"(quartiles from sample):")
        for i, lab in enumerate(labs):
            m = (q["p"] < qs[0]) if i == 0 else \
                (q["p"] >= qs[0]) & (q["p"] < qs[1]) if i == 1 else \
                (q["p"] >= qs[1]) & (q["p"] < qs[2]) if i == 2 else \
                (q["p"] >= qs[2])
            r = q["ret"][m.values]
            print(f"    {lab}: n={len(r)} avg {r.mean():+.2%}/trade "
                  f"med {r.median():+.2%}")

    trade_gradient(f_tr, "FADE")
    trade_gradient(c_tr, "CLIM")

    # gates: cuts fit on H1 of the recession series, book control on 12m
    half = rec.index[0] + (rec.index[-1] - rec.index[0]) / 2
    h1 = rec[rec.index < half]
    lo_c, hi_c = h1.quantile(0.30), h1.quantile(0.70)
    print(f"  H1-fit cuts: lo {lo_c:.3f} hi {hi_c:.3f} (H1 median "
          f"{h1.median():.3f})")
    print("| gate | SR | CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    line("FADE base", window(f_net))
    line("FADE p<hi (risk-on)", window(fade_sim9(
        tf=flow, cluster_gate=3, fg=fg, fg_hi=hi_c)[0]))
    line("FADE p>lo (risk-off on)", window(fade_sim9(
        tf=flow, cluster_gate=3, fg=fg, fg_lo=lo_c)[0]))
    line("CLIM base", window(c_net))
    line("CLIM p<hi", window(cont_sim8(
        tf=flow, stop_pct=0.06, fg=fg, fg_hi=hi_c)[0]))
    line("CLIM p>lo", window(cont_sim8(
        tf=flow, stop_pct=0.06, fg=fg, fg_lo=lo_c)[0]))


# ---------------------------------------------------------------- battery c
def battery_c(flow, f9, c9):
    rec = load_recession()
    d20 = (rec - rec.shift(20)).dropna()
    print(f"\n## Round 33c — recession SLOPE d20d ({d20.index[0].date()}.."
          f"{d20.index[-1].date()}, last {d20.iloc[-1]:+.3f})")
    half = d20.index[0] + (d20.index[-1] - d20.index[0]) / 2
    tau = d20[d20.index < half].median()
    print(f"  H1-fit veto threshold: d20 > {tau:+.4f}")
    fg = hourly_from_daily(d20.rename_axis("date"))
    print("| gate | SR | CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    line("FADE base", window(f9[0]))
    line("FADE veto d20>tau", window(fade_sim9(
        tf=flow, cluster_gate=3, fg=fg, fg_hi=tau)[0]))
    line("CLIM base", window(c9[0]))
    line("CLIM veto d20>tau", window(cont_sim8(
        tf=flow, stop_pct=0.06, fg=fg, fg_hi=tau)[0]))
    tri_v = (0.60 * fade_sim9(tf=flow, cluster_gate=3, fg=fg,
                              fg_hi=tau)[0]
             + 0.25 * sq_stop_hybrid(0.05, lev=2.0)[0]
             + 0.15 * cont_sim8(tf=flow, stop_pct=0.06, fg=fg,
                                fg_hi=tau)[0])
    line("TRIO veto", window(tri_v))


# ---------------------------------------------------------------- battery d
def battery_d(flow, f9, c9, s9):
    foms = pd.to_datetime(FOMC_DAYS)
    masks = {}
    for lab, pre, post in (("dec-12h..dec+24h", 12, 24), ("day0..+36h", 0, 36)):
        m = np.zeros(len(idx), bool)
        for d in foms:
            t0 = d + pd.Timedelta(hours=18) - pd.Timedelta(hours=pre)
            t1 = d + pd.Timedelta(hours=18) + pd.Timedelta(hours=post)
            m |= (idx >= t0) & (idx <= t1)
        masks[lab] = m
    print("\n## Round 33d — FOMC-window entry vetoes (full window)")
    for lab, m in masks.items():
        print(f"  {lab}: masks {m.mean():.1%} of hours")

    def veto(net, m):
        out = net.copy()
        out[m] = 0.0
        return out

    print("| variant | SR | CI | ret | maxDD |")
    print("|---|---|---|---|---|")
    tri = 0.60 * f9[0] + 0.25 * s9[0] + 0.15 * c9[0]
    line("TRIO base (24m)", tri)
    for lab, m in masks.items():
        line(f"TRIO veto {lab}",
             veto(f9[0], m) * 0.60 + veto(s9[0], m) * 0.25
             + veto(c9[0], m) * 0.15)
        line(f"  FADE veto {lab}", veto(f9[0], m))
        line(f"  CLIM veto {lab}", veto(c9[0], m))
        line(f"  SQUEEZE veto {lab}", veto(s9[0], m))
    # where does FOMC-hour pnl sit? distribution of book ret in dec window
    m = masks["dec-12h..dec+24h"]
    inw = tri[m]
    print(f"  TRIO pnl inside dec-12h..+24h windows: sum {inw.sum():+.2%} "
          f"over {int(m.sum())}h (mean {inw.mean():+.4%}/h)")


# ---------------------------------------------------------------- battery e
def battery_e():
    fc = load_fed_cut()
    if fc is None or len(fc) < 40:
        print("\n## Round 33e — Fed P(cut): insufficient data")
        return
    print(f"\n## Round 33e — Fed Oct-26 P(cut) level ({fc.index[0].date()}.."
          f"{fc.index[-1].date()}, {len(fc)}d, last {fc.iloc[-1]:.3f})")
    d5 = (fc - fc.shift(5)).dropna()
    print(f"  d5d: last {d5.iloc[-1]:+.3f} | sd {d5.std():.3f} | "
          f"min {d5.min():+.3f} max {d5.max():+.3f}")
    print("  (gradient vs trade outcomes skipped: series shorter than "
          "reliable evaluation window — forward archive will accumulate)")


# ---------------------------------------------------------------- battery f
def battery_f():
    print("\n## Round 33f — verdict: see technique_research round 33a")


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if a in "abcdef"] or list("abcdef")
    flow = load_flow()
    f9 = c9 = s9 = None
    for w in which:
        if w == "a":
            f9, c9, s9 = battery_a(flow)
        elif w == "b":
            if f9 is None:
                f9, c9, s9 = sims(flow)
            battery_b(flow, f9, c9)
        elif w == "c":
            if f9 is None:
                f9, c9, s9 = sims(flow)
            battery_c(flow, f9, c9)
        elif w == "d":
            if f9 is None:
                f9, c9, s9 = sims(flow)
            battery_d(flow, f9, c9, s9)
        elif w == "e":
            battery_e()
        elif w == "f":
            battery_f()
