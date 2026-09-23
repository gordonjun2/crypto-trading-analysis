"""CLI: run (screen + report + Telegram) / evaluate (walk-forward) / refresh-data.

`run` is dry-run by default; sending requires --send (server safety, plan §8).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from pair_scout.config import ConfigError, load_config
from pair_scout.log import setup_logging
from pair_scout.pipeline import run_evaluation_pipeline, run_pipeline
from pair_scout.telegram import send_report_or_fail

logger = logging.getLogger("pair_scout")


def _print_chunks(chunks: list[str]) -> None:
    for i, chunk in enumerate(chunks):
        if i:
            print("\n" + "-" * 60)
        print(chunk)


def cmd_run(args: argparse.Namespace) -> int:
    from dataclasses import replace as dc_replace

    cfg = load_config(args.config).with_jev(not args.no_jev)
    if args.mode != "default":
        cfg = dc_replace(cfg, screen=dc_replace(cfg.screen, mode=args.mode))
    result = run_pipeline(cfg, use_jev=not args.no_jev)
    _print_chunks(result.report_chunks)
    if args.send:
        send_report_or_fail(
            cfg.telegram_bot_token, cfg.telegram_group_id, result.report_chunks
        )
        logger.info("report sent to Telegram")
    else:
        logger.info("dry-run: report printed, not sent (use --send to deliver)")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from dataclasses import replace as dc_replace

    base_cfg = load_config(args.config).with_jev(not args.no_jev)
    modes = ["divergence", "cointegration"] if args.mode == "both" else [args.mode]
    markdown_parts = []
    written = []
    from pathlib import Path as _Path

    out_dir = _Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        cfg = dc_replace(base_cfg, screen=dc_replace(base_cfg.screen, mode=mode))
        report = run_evaluation_pipeline(cfg, use_jev=not args.no_jev)
        md = report.to_markdown(cfg)
        markdown_parts.append(md)
        stamp = report.generated_at.strftime("%Y%m%d_%H%M")
        out_file = out_dir / f"evaluation_{mode}_{stamp}.md"
        out_file.write_text(md, encoding="utf-8")
        written.append(str(out_file))
        logger.info("evaluation report written to %s", out_file)
    print("\n\n".join(markdown_parts))
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from dataclasses import replace as dc_replace
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    from pair_scout.data.loader import load_panel
    from pair_scout.tune import results_to_markdown, run_tune

    cfg = load_config(args.config).with_jev(enabled=False)
    if args.mode != "divergence":
        cfg = dc_replace(cfg, screen=dc_replace(cfg.screen, mode=args.mode))
    panel = load_panel(
        cex=cfg.data.cex,
        interval=cfg.data.interval,
        data_dir=cfg.data.data_dir,
        top_n_volume=cfg.data.top_n_volume,
        min_history_bars=cfg.data.min_history_bars,
        max_nan_fraction=cfg.data.max_nan_fraction,
        trailing_volume_days=cfg.data.trailing_volume_days,
    )
    results = run_tune(cfg, panel)
    md = results_to_markdown(results, datetime.now(timezone.utc))
    print(md)
    out_dir = _Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"tune_{cfg.screen.mode}_{stamp}.md"
    out_file.write_text(md, encoding="utf-8")
    logger.info("tune report written to %s", out_file)
    return 0


def cmd_refresh_data(args: argparse.Namespace) -> int:
    if args.legacy or args.cex != "binance":
        root = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable,
            str(root / "data_manager.py"),
            "-c", args.cex,
            "-i", args.interval,
            "-l", str(args.limit),
        ]
        if args.end:
            cmd += ["-e", args.end]
        logger.info("refreshing data via data_manager.py: %s", " ".join(cmd))
        completed = subprocess.run(cmd, cwd=str(root))
        return completed.returncode
    from pair_scout.data.refresh import refresh_all

    end_ms = int(args.end) if args.end else 0
    results = refresh_all(args.cex, args.interval, args.limit, end_ms=end_ms)
    ok = sum(1 for n in results.values() if n > 0)
    logger.info("refreshed %d/%d pairs", ok, len(results))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pair_scout",
        description="Crypto pair-trading screener with JEV ranking + Telegram delivery.",
    )
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--log-file", default=None, help="also log to this file")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="screen pairs and produce/send the report")
    run.add_argument("--send", action="store_true", help="send to Telegram (default: dry-run)")
    run.add_argument("--no-jev", action="store_true", help="skip JEV; rule-based ranking")
    run.add_argument("--mode", default="default",
                     choices=["default", "divergence", "cointegration"],
                     help="strategy mode (default: config, which defaults to divergence)")

    ev = sub.add_parser("evaluate", help="walk-forward 3-arm evaluation")
    ev.add_argument("--no-jev", action="store_true", help="skip the JEV arm")
    ev.add_argument("--mode", default="both", choices=["both", "divergence", "cointegration"],
                    help="which strategy mode(s) to evaluate")
    ev.add_argument("--out", default="pair_scout/output", help="output directory")

    rd = sub.add_parser("refresh-data", help="download fresh klines (paginated, binance)")
    rd.add_argument("-c", "--cex", default="binance")
    rd.add_argument("-i", "--interval", default="1h")
    rd.add_argument("-l", "--limit", type=int, default=4320,
                    help="number of candlesticks per pair (paginated beyond 1500)")
    rd.add_argument("-e", "--end", default="")
    rd.add_argument("--legacy", action="store_true",
                    help="delegate to data_manager.py instead of the paginated fetcher")

    tn = sub.add_parser("tune", help="walk-forward parameter sweep")
    tn.add_argument("--mode", default="divergence", choices=["divergence", "cointegration"])
    tn.add_argument("--out", default="pair_scout/output", help="output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(log_file=args.log_file)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "evaluate":
            return cmd_evaluate(args)
        if args.command == "refresh-data":
            return cmd_refresh_data(args)
        if args.command == "tune":
            return cmd_tune(args)
        parser.error(f"unknown command {args.command}")
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except Exception:  # noqa: BLE001 — top-level: log with traceback, non-zero exit
        logger.exception("unhandled error")
        return 1
    return 0
