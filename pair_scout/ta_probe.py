"""TA channel probe: scan the point-in-time universe for Donchian entries and
deliver them to Telegram.

This is the live paper-trade probe for the pass-9/10 research (dual-direction
Donchian 20d/10d over a causal top-30 volume universe, <=7d hold, 5% stop,
unhedged). It is an ALERT system only — no orders are placed.

UX (one message per run, phone-first):
  - every record is 2 short lines (<= ~35 chars), no tables/monospace:
    NEW:    "🟢 LONG SYM · Score N" + hedge line
    OPEN:   "📈 SYM +pnl%" + "🛡 side coin · net +pnl%" (⏰ = past 7d)
    CLOSED: "📈 SYM net +pnl% · reason"
  - top-10 NEW calls by Score; Book summary strip up top; legend at the foot
  - hedge is display-only (iter-10 verdict: run unhedged); net = main + hedge 50/50
  - heartbeat when there are no signals (suppress with --only-signals)

Run daily after the 00:00 UTC candle close. `--send` delivers; default is
dry-run (prints the exact payload).
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pair_scout.config import AppConfig
from pair_scout.data.funding import load_funding  # noqa: F401 (parity with research)
from pair_scout.data.loader import load_panel
from pair_scout.research_ta_families import psar_signals_panel
from pair_scout.research_ta_general import donchian_signals

logger = logging.getLogger(__name__)

BPD = 24
MAX_HOLD_BARS = 7 * BPD
STOP = 0.05
UNIV_TOP = 30
UNIV_VOL_DAYS = 30
BTC = "BTCUSDT"
HEDGE_MOM_DAYS = 7  # mirror of the pass-10 hedge selection


@dataclass
class ProbeRow:
    sym: str
    direction: str  # LONG | SHORT
    state: str  # NEW | ONGOING
    entry_px: float
    last_px: float
    pnl: float | None
    entry_date: datetime | None
    exit_date: datetime | None = None
    exit_reason: str | None = None
    channel_hi: float | None = None
    channel_lo: float | None = None
    exit_level: float | None = None
    confidence: int | None = None
    hedge_sym: str | None = None
    hedge_dir: str | None = None  # LONG | SHORT (opposite side of the signal)
    hedge_mom: float | None = None
    mom24: float | None = None  # last-24h move at signal time (labeled "24h")


def _atr_pct(panel, sym: str, last_i: int, bpd: int) -> float | None:
    """Trailing 1-day ATR as a fraction of price, at `last_i` (causal)."""
    f = panel.frames[sym]
    c = f["Close"]
    if last_i < bpd:
        return None
    pc = c.shift(1)
    tr = pd.concat(
        [f["High"] - f["Low"], (f["High"] - pc).abs(), (f["Low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    atr = float(tr.iloc[last_i - bpd + 1: last_i + 1].mean())
    px = float(c.iloc[last_i])
    return atr / px if px > 0 and np.isfinite(atr) else None


def _confidence(closes, mom_now, vol_rank_now, panel, sym, direction,
                last_i: int, bpd: int, universe_top: int) -> int | None:
    """0-100 heuristic strength score, calibrated to spread across calls:
    trend 0-40 (full only at >=30% 7d move in-signal-direction),
    thrust 0-30 (full at >=2x daily ATR 24h move, linear),
    liquidity 0-30 (volume rank 1 -> 30, rank 300 -> 0)."""
    if last_i < HEDGE_MOM_DAYS * bpd:
        return None
    c = closes[sym]
    px = float(c.iloc[last_i])
    if px <= 0:
        return None
    mom = mom_now.get(sym)
    mom = float(mom) if mom is not None and np.isfinite(mom) else None
    aligned = mom is not None and ((direction == "LONG") == (mom > 0))
    trend_pts = 40.0 * min(abs(mom) / 0.30, 1.0) if aligned and mom else 0.0
    atrp = _atr_pct(panel, sym, last_i, bpd)
    if not atrp:
        return None
    ret1d = px / float(c.iloc[last_i - bpd]) - 1.0
    thrust_pts = 30.0 * min(abs(ret1d) / (2.0 * atrp), 1.0)
    rank = vol_rank_now.get(sym)
    liq_pts = (
        30.0 * max(0.0, 1.0 - (float(rank) - 1.0) / 300.0)
        if rank is not None and np.isfinite(rank) else 0.0
    )
    return int(round(min(100.0, max(0.0, trend_pts + thrust_pts + liq_pts))))


def _hedge_idea(mom_now, in_u_now, sym: str, direction: str):
    """Pass-10 hedge construction, N=1: short signal -> LONG the strongest 7d
    momentum coin in-universe; long signal -> SHORT the weakest. Display only."""
    if in_u_now is None:
        return None
    cand = mom_now[in_u_now.reindex(mom_now.index).fillna(False)].dropna()
    cand = cand.drop(labels=[sym, BTC], errors="ignore")
    if cand.empty:
        return None
    if direction == "SHORT":
        h = cand.idxmax()
        return h, "LONG", float(cand.max())
    h = cand.idxmin()
    return h, "SHORT", float(cand.min())


def _scan(cfg: AppConfig, data_dir: str | None = None, universe_top: int = UNIV_TOP,
          family: str = "psar"):
    """Load data and derive per-pair Donchian state over the last days."""
    panel = load_panel(
        cex=cfg.data.cex, interval=cfg.data.interval,
        data_dir=data_dir or cfg.data.data_dir,
        top_n_volume=658,
        min_history_bars=1000,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    idx = panel.index
    closes = panel.closes
    dollar_vol = pd.DataFrame(
        {s: panel.frames[s]["Volume"] * panel.frames[s]["Close"] for s in panel.pairs}
    )
    univ_rank = dollar_vol.rolling(UNIV_VOL_DAYS * BPD).mean().rank(
        axis=1, ascending=False
    )
    in_universe = (univ_rank <= universe_top) & univ_rank.notna()
    if family == "psar":
        signals = psar_signals_panel(panel)
    else:
        signals = donchian_signals(closes)
    return panel, idx, closes, in_universe, signals, univ_rank


@dataclass
class JournalTrade:
    """One alerted call, tracked main-leg + hedge-leg until it exits."""

    sym: str
    direction: str  # LONG | SHORT
    entry_date: datetime
    entry_px: float
    hedge_sym: str | None = None
    hedge_dir: str | None = None  # opposite side of the main leg
    hedge_entry_px: float | None = None
    conf: int | None = None
    state: str = "open"  # open | closed
    exit_date: datetime | None = None
    exit_px: float | None = None
    hedge_exit_px: float | None = None
    exit_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "sym": self.sym, "direction": self.direction,
            "entry_date": self.entry_date.isoformat(),
            "entry_px": self.entry_px,
            "hedge_sym": self.hedge_sym, "hedge_dir": self.hedge_dir,
            "hedge_entry_px": self.hedge_entry_px, "conf": self.conf,
            "state": self.state,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "exit_px": self.exit_px, "hedge_exit_px": self.hedge_exit_px,
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JournalTrade":
        return cls(
            sym=d["sym"], direction=d["direction"],
            entry_date=datetime.fromisoformat(d["entry_date"]),
            entry_px=float(d["entry_px"]),
            hedge_sym=d.get("hedge_sym"), hedge_dir=d.get("hedge_dir"),
            hedge_entry_px=(float(d["hedge_entry_px"])
                            if d.get("hedge_entry_px") is not None else None),
            conf=d.get("conf"), state=d.get("state", "open"),
            exit_date=(datetime.fromisoformat(d["exit_date"])
                       if d.get("exit_date") else None),
            exit_px=(float(d["exit_px"]) if d.get("exit_px") is not None else None),
            hedge_exit_px=(float(d["hedge_exit_px"])
                           if d.get("hedge_exit_px") is not None else None),
            exit_reason=d.get("exit_reason"),
        )

    @staticmethod
    def _leg_ret(direction: str, entry_px: float, px: float) -> float:
        if direction == "LONG":
            return px / entry_px - 1.0
        return entry_px / max(px, 1e-12) - 1.0

    def net_ret(self, main_px: float, hedge_px: float | None) -> float:
        """Dollar-neutral combined return: 0.5 * main + 0.5 * hedge."""
        main = self._leg_ret(self.direction, self.entry_px, main_px)
        hedge = 0.0
        if (self.hedge_sym and self.hedge_entry_px and self.hedge_dir
                and hedge_px is not None and np.isfinite(hedge_px)
                and self.hedge_entry_px > 0):
            hedge = self._leg_ret(self.hedge_dir, self.hedge_entry_px, hedge_px)
        return 0.5 * main + 0.5 * hedge


def _journal_path(data_dir: str | None, cfg: AppConfig) -> Path:
    return Path(data_dir or cfg.data.data_dir) / "ta_probe_trades.json"


def _load_journal(path: Path) -> list[JournalTrade] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return [JournalTrade.from_dict(d) for d in raw.get("trades", [])]
    except Exception as exc:  # noqa: BLE001 — corrupt journal must not kill a run
        logger.warning("unreadable trade journal %s (%s); starting fresh", path, exc)
        return []


def _save_journal(path: Path, trades: list[JournalTrade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"version": 1, "trades": [t.to_dict() for t in trades]}, indent=1
    ))


def _exit_reason(tr: JournalTrade, signals, closes, idx, last_i: int,
                 bpd: int) -> str | None:
    """First of: SAR/channel flip, adverse stop, 7d time stop."""
    sig = signals.get(tr.sym)
    px = float(closes[tr.sym].iloc[last_i])
    if sig is not None:
        la, _, sa, _ = sig
        if tr.direction == "LONG" and not bool(la.iloc[last_i]):
            return "flip"
        if tr.direction == "SHORT" and not bool(sa.iloc[last_i]):
            return "flip"
    if tr.direction == "LONG" and px / tr.entry_px - 1.0 <= -STOP:
        return "stop"
    if tr.direction == "SHORT" and px / tr.entry_px - 1.0 >= STOP:
        return "stop"
    if (idx[last_i] - pd.Timestamp(tr.entry_date)) >= pd.Timedelta(
            days=MAX_HOLD_DAYS):
        return "time"
    return None


def build_probe_rows(cfg: AppConfig, data_dir: str | None = None,
                     universe_top: int = UNIV_TOP, family: str = "psar",
                     max_slots: int = 8):
    panel, idx, closes, in_universe, signals, univ_rank = _scan(
        cfg, data_dir, universe_top, family)
    bpd = panel.bars_per_day
    last_bar_i = len(idx) - 1
    day_start_i = max(last_bar_i - bpd + 1, 1)  # last 24h window

    mom_now = (closes / closes.shift(HEDGE_MOM_DAYS * bpd) - 1.0).iloc[last_bar_i]
    vol_rank_now = univ_rank.iloc[last_bar_i] if last_bar_i < len(univ_rank) else None
    in_u_now = (
        in_universe.iloc[last_bar_i].astype(bool)
        if last_bar_i < len(in_universe) else None
    )

    # fresh flips in the last 24h (these are the calls the alert shows)
    new_rows: list[ProbeRow] = []
    still_active: list[ProbeRow] = []  # signal-active runs older than 24h
    for sym, (l_act, _l_exit, s_act, _s_exit) in signals.items():
        if sym not in in_universe.columns:
            continue
        in_u_series = in_universe[sym]
        for direction, series in (
            ("LONG", l_act & in_u_series.reindex(idx).fillna(False)),
            ("SHORT", s_act & in_u_series.reindex(idx).fillna(False)),
        ):
            if not series.iloc[last_bar_i]:
                continue
            k = last_bar_i
            while k > 0 and series.iloc[k - 1]:
                k -= 1
            row = ProbeRow(
                sym=sym, direction=direction,
                state="NEW" if k >= day_start_i else "ONGOING",
                entry_px=float(closes[sym].iloc[k]),
                last_px=float(closes[sym].iloc[last_bar_i]), pnl=None,
                entry_date=idx[k].to_pydatetime(),
                mom24=(float(closes[sym].iloc[last_bar_i]
                             / closes[sym].iloc[last_bar_i - bpd] - 1.0)
                       if last_bar_i >= bpd else None),
            )
            if row.state == "NEW":
                new_rows.append(row)
            else:
                still_active.append(row)

    new_rows.sort(key=lambda r: (-_conf_of(r), r.sym))
    for r in new_rows:
        r.confidence = _confidence(closes, mom_now, vol_rank_now, panel,
                                   r.sym, r.direction, last_bar_i, bpd,
                                   universe_top)
        hedge = _hedge_idea(mom_now, in_u_now, r.sym, r.direction)
        if hedge:
            r.hedge_sym, r.hedge_dir, r.hedge_mom = hedge

    # ---- trade journal: track each alerted call main + hedge ----
    jpath = _journal_path(data_dir, cfg)
    trades = _load_journal(jpath)
    if trades is None:
        # first run: adopt still-active positions so history is not lost
        # (hedge PnL starts at 0 at the current price — approximation)
        trades = []
        for row in still_active:
            hedge = _hedge_idea(mom_now, in_u_now, row.sym, row.direction)
            h_sym, h_dir, _mom = hedge if hedge else (None, None, None)
            trades.append(JournalTrade(
                sym=row.sym, direction=row.direction,
                entry_date=row.entry_date, entry_px=row.entry_px,
                hedge_sym=h_sym, hedge_dir=h_dir,
                hedge_entry_px=(float(closes[h_sym].iloc[last_bar_i])
                                if h_sym else None),
            ))
        logger.info("seeded trade journal with %d still-active calls",
                    len(trades))

    now_ts = idx[last_bar_i].to_pydatetime()
    for tr in trades:
        if tr.state != "open":
            continue
        reason = _exit_reason(tr, signals, closes, idx, last_bar_i, bpd)
        if reason:
            tr.state = "closed"
            tr.exit_reason = reason
            tr.exit_date = now_ts
            tr.exit_px = float(closes[tr.sym].iloc[last_bar_i])
            if tr.hedge_sym and tr.hedge_sym in closes.columns:
                tr.hedge_exit_px = float(closes[tr.hedge_sym].iloc[last_bar_i])

    open_syms = {t.sym for t in trades if t.state == "open"}
    # gate: only Score > SCORE_GATE calls are tracked, and the book is capped
    # at max_slots concurrent positions (champion sizing) — best Score first
    gated = sorted(
        (r for r in new_rows if (r.confidence or 0) > SCORE_GATE),
        key=lambda r: (-(r.confidence or 0), r.sym),
    )
    opened_syms: set[str] = set()
    skipped_slots = 0
    for r in gated:
        if r.sym in open_syms:
            continue
        if len(open_syms) >= max_slots:
            skipped_slots += 1
            continue
        trades.append(JournalTrade(
            sym=r.sym, direction=r.direction, entry_date=now_ts,
            entry_px=float(closes[r.sym].iloc[last_bar_i]),
            hedge_sym=r.hedge_sym, hedge_dir=r.hedge_dir,
            hedge_entry_px=(float(closes[r.hedge_sym].iloc[last_bar_i])
                            if r.hedge_sym and r.hedge_sym in closes.columns
                            else None),
            conf=r.confidence,
        ))
        open_syms.add(r.sym)
        opened_syms.add(r.sym)
    _save_journal(jpath, trades)

    open_trades = [t for t in trades if t.state == "open"]
    closed_recent = [
        t for t in trades
        if t.state == "closed" and t.exit_date
        and (now_ts - t.exit_date) <= timedelta(hours=24)
    ]
    hedges = _hedge_coins(mom_now, in_u_now)
    return (new_rows, open_trades, closed_recent, hedges,
            opened_syms, skipped_slots, panel, idx)


SCORE_GATE = 80  # only calls above this Score are shown/tracked (validated: 80-90 band = 92% positive months, best consistency)
MAX_HOLD_DAYS = MAX_HOLD_BARS // BPD  # 7d time stop
HEDGE_OPTIONS = 3  # hedge candidates shown per side

# 12m backtest Sharpe (annualized, daily returns) — refresh via backtests/*.py
# hourly = 8-slot champion book, decisions every bar (theoretical best case)
# daily  = Score>80 tracker book, decisions only at 10:00 SGT (your cadence)
BACKTEST_SR_HOURLY = "~3"
BACKTEST_SR_DAILY = "0.9"
BACKTEST_SR_DAILY_8SLOT = "0.3"


def _conf_of(r) -> int:
    return r.confidence if r.confidence is not None else -1


def _reason_of(tr, family: str) -> str:
    reason = tr.exit_reason or ""
    if reason == "flip":
        reason = "SAR flip" if family == "psar" else "channel flip"
    elif reason == "stop":
        reason = "5% stop"
    elif reason == "time":
        reason = "7d cap"
    return html.escape(reason)


def _pnl_arrow(pnl: float) -> str:
    return "📈" if pnl >= 0 else "📉"


def _hedge_coins(mom_now, in_u_now) -> dict[str, list[tuple[str, str, float]]]:
    """Global hedge candidates: LONG calls -> SHORT the N weakest 7d coins,
    SHORT calls -> LONG the N strongest. {call_dir: [(side, sym, mom7d), ...]}."""
    if in_u_now is None:
        return {}
    cand = mom_now[in_u_now.reindex(mom_now.index).fillna(False)].dropna()
    cand = cand.drop(labels=[BTC], errors="ignore")
    if cand.empty:
        return {}
    weak = [(s, float(v)) for s, v in cand.nsmallest(HEDGE_OPTIONS).items()]
    strong = [(s, float(v)) for s, v in cand.nlargest(HEDGE_OPTIONS).items()]
    return {
        "LONG": [("SHORT", s, v) for s, v in weak],
        "SHORT": [("LONG", s, v) for s, v in strong],
    }


def _hedge_section(hedges: dict[str, list[tuple[str, str, float]]]) -> str:
    lines = ["🛡 <b>Hedge ideas</b> — optional, 7d-momentum extremes"]
    for call_dir in ("LONG", "SHORT"):
        opts = hedges.get(call_dir) or []
        if not opts:
            continue
        primary = opts[0]
        lines.append(f"{call_dir} calls → {primary[0]} "
                     f"<b>{html.escape(primary[1])}</b> (7d {primary[2]:+.0%})")
        alts = " · ".join(f"{html.escape(s)} {v:+.0%}" for _, s, v in opts[1:])
        if alts:
            lines.append(f"   alts: {alts}")
    return "\n".join(lines)


def _new_section(top, skipped: int, n_signals: int) -> str:
    head = (f"🆕 <b>NEW — {len(top)} opened (Score >{SCORE_GATE}, "
            f"of {n_signals} signals)</b>")
    lines = []
    for r in top:
        dot = "🟢" if r.direction == "LONG" else "🔴"
        score = f" · Score {r.confidence}" if r.confidence is not None else ""
        move = f" · 24h {r.mom24:+.1%}" if r.mom24 is not None else ""
        lines.append(f"{dot} {r.direction} <b>{html.escape(r.sym)}</b>"
                     f"{score}{move}")
    if skipped:
        lines.append(f"· {skipped} gated calls skipped — 8 slots full")
    return head + "\n" + "\n".join(lines)


def _summary_head(pnls: list[float], label: str) -> str:
    """`⏳ OPEN — 39 · 38W/1L · avg +24.3%` style section header."""
    avg = sum(pnls) / len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return f"{label} — {len(pnls)} · {wins}W/{len(pnls) - wins}L · avg {avg:+.1%}"


def _pos_section(open_trades, last_px: dict, last_ts) -> str:
    n_long = sum(1 for t in open_trades if t.direction == "LONG")
    net_of = lambda t: t.net_ret(last_px.get(t.sym), last_px.get(t.hedge_sym))
    head = _summary_head([net_of(t) for t in open_trades],
                         f"⏳ <b>OPEN</b> ({n_long}▲/"
                         f"{len(open_trades) - n_long}▼) pnl since entry")
    lines = []
    fresh = 0  # just-opened trades still at ~0%: collapse to one line
    for t in sorted(open_trades, key=lambda t: -net_of(t)):
        main = JournalTrade._leg_ret(t.direction, t.entry_px,
                                     last_px.get(t.sym))
        net = net_of(t)
        age = last_ts - pd.Timestamp(t.entry_date)
        days = age.days
        if abs(net) < 0.0005 and days < 1:
            fresh += 1
            continue
        due = " · ⏰ >7d" if age >= pd.Timedelta(days=MAX_HOLD_DAYS) else ""
        lines.append(f"{_pnl_arrow(net)} <b>{html.escape(t.sym)}</b> "
                     f"{main:+.1%} · {days}d · net {net:+.1%}{due}")
    if fresh:
        lines.append(f"· {fresh} opened today at ±0%")
    return head + "\n" + "\n".join(lines)


def _closed_net_pnls(closed_trades, last_px: dict) -> list[float]:
    pnls = []
    for t in closed_trades:
        main_px = t.exit_px if t.exit_px is not None else last_px.get(t.sym, t.entry_px)
        hedge_px = (t.hedge_exit_px if t.hedge_exit_px is not None
                    else last_px.get(t.hedge_sym))
        pnls.append(t.net_ret(main_px, hedge_px))
    return pnls


def _exit_section(closed_trades, family: str, last_px: dict) -> str:
    pnls = _closed_net_pnls(closed_trades, last_px)
    head = _summary_head(pnls, "✅ <b>CLOSED (24h)</b> net")
    lines = []
    for t, pnl in sorted(zip(closed_trades, pnls), key=lambda x: -x[1]):
        lines.append(f"{_pnl_arrow(pnl)} <b>{html.escape(t.sym)}</b> "
                     f"net {pnl:+.1%} · {_reason_of(t, family)}")
    return head + "\n" + "\n".join(lines)


def build_message(new_rows, open_trades, closed_trades, hedges, opened_syms,
                  skipped_slots, panel, now: datetime,
                  universe_top: int = UNIV_TOP,
                  family: str = "psar", max_slots: int = 8) -> list[str]:
    sig_desc = ("PSAR flips" if family == "psar" else "20d/10d crosses")
    last_px = {sym: float(px) for sym, px in panel.closes.iloc[-1].items()
               if np.isfinite(px)}
    new_all = sorted(new_rows, key=lambda r: (-_conf_of(r), r.sym))

    # book summary across everything the probe tracks
    open_pnls = [t.net_ret(last_px.get(t.sym), last_px.get(t.hedge_sym))
                 for t in open_trades]
    closed_pnls = _closed_net_pnls(closed_trades, last_px)
    book_parts = [f"{len(open_trades)} open"]
    book_line2 = ""
    if open_pnls:
        best = max(open_trades, key=lambda t: t.net_ret(
            last_px.get(t.sym), last_px.get(t.hedge_sym)))
        book_line2 = (f"best {best.sym} {max(open_pnls):+.1%}")
    if closed_trades:
        closed_s = (f"{len(closed_trades)} closed 24h "
                    f"({sum(closed_pnls):+.1%})")
        book_line2 = f"{book_line2} · {closed_s}" if book_line2 else closed_s

    blocks = [
        f"🎯 <b>TA Trend Probe</b> — {now:%d %b %Y %H:%M} UTC\n"
        f"{sig_desc} · top-{universe_top} of {len(panel.pairs)} · "
        f"open {len(open_trades)} (cap {max_slots})",
    ]
    if open_pnls:
        book_parts.append(f"avg net {sum(open_pnls) / len(open_pnls):+.1%}")
        blocks.append("📋 <b>Book</b> — " + " · ".join(book_parts)
                      + ("\n" + book_line2 if book_line2 else ""))
    blocks.append(
        f"📊 <b>Backtest SR</b> (12m): {BACKTEST_SR_HOURLY} 8-slot book "
        f"(hourly) · {BACKTEST_SR_DAILY} tracker @10:00 · "
        f"{BACKTEST_SR_DAILY_8SLOT} 8-slot @10:00"
    )
    gated = [r for r in new_all if (r.confidence or 0) > SCORE_GATE]
    opened = [r for r in new_all if r.sym in opened_syms]
    if gated:
        if opened:
            blocks.append(_new_section(opened, skipped_slots, len(new_all)))
        else:
            blocks.append(
                f"🆕 <b>NEW</b> — {len(gated)} gated calls but all "
                f"{max_slots} slots full — none opened")
    elif new_all:
        blocks.append(f"🆕 <b>NEW</b> — none above Score {SCORE_GATE} "
                      f"({len(new_all)} signals)")
    if hedges:
        blocks.append(_hedge_section(hedges))
    if open_trades:
        # lifecycle: today's calls live in NEW only; OPEN starts the next day
        day_start = panel.index[-1] - pd.Timedelta(hours=24)
        older = [t for t in open_trades
                 if pd.Timestamp(t.entry_date) <= day_start]
        fresh_n = len(open_trades) - len(older)
        if older:
            blocks.append(_pos_section(older, last_px, panel.index[-1]))
        if fresh_n:
            blocks.append(f"· {fresh_n} opened today — in OPEN from tomorrow")
    if closed_trades:
        blocks.append(_exit_section(closed_trades, family, last_px))
    if not new_all and not open_trades and not closed_trades:
        blocks.append("😴 No signals in the last 24h.")
    blocks.append(
        "🟢 LONG · 🔴 SHORT · ⏰ >7d · 📈 gain · 📉 loss\n"
        "% = PnL since entry (24h = last-day move) · net = main + hedge 50/50\n"
        "⚠️ Analysis only — no orders · only Score >80 tracked · "
        "exits: flip / 5% stop / 7d cap · size 1/3 per slot"
    )

    # chunk by whole block so <pre> tables never split across messages
    chunks: list[str] = []
    cur = ""
    for b in blocks:
        candidate = f"{cur}\n\n{b}" if cur else b
        if len(candidate) <= 3900:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
        while len(b) > 3900:
            chunks.append(b[:3900])
            b = b[3900:]
        cur = b
    if cur:
        chunks.append(cur)
    return chunks


def run_ta_probe(cfg: AppConfig, send: bool = False, data_dir: str | None = None,
                 only_signals: bool = False,
                 universe_top: int = UNIV_TOP, family: str = "psar",
                 max_slots: int = 8) -> list[str]:
    new_rows, open_trades, closed_trades, hedges, opened_syms, skipped, panel, _ = \
        build_probe_rows(cfg, data_dir, universe_top, family, max_slots)
    now = datetime.now(timezone.utc)
    if only_signals and not new_rows:
        logger.info("no new signals; suppressing message (--only-signals)")
        return []
    chunks = build_message(new_rows, open_trades, closed_trades, hedges,
                           opened_syms, skipped, panel, now,
                           universe_top, family, max_slots)
    if send:
        from pair_scout.telegram import send_report_or_fail

        send_report_or_fail(cfg.telegram_bot_token, cfg.telegram_group_id, chunks)
        logger.info("probe sent to Telegram")
    else:
        logger.info("dry-run: probe printed, not sent (use --send to deliver)")
    return chunks
