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
    cfg = load_config(args.config).with_jev(not args.no_jev)
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
    cfg = load_config(args.config).with_jev(not args.no_jev)
    report = run_evaluation_pipeline(cfg, use_jev=not args.no_jev)
    markdown = report.to_markdown(cfg)
    print(markdown)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%d_%H%M")
    out_file = out_dir / f"evaluation_{stamp}.md"
    out_file.write_text(markdown, encoding="utf-8")
    logger.info("evaluation report written to %s", out_file)
    return 0


def cmd_refresh_data(args: argparse.Namespace) -> int:
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
    logger.info("refreshing data: %s", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(root))
    return completed.returncode


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

    ev = sub.add_parser("evaluate", help="walk-forward 3-arm evaluation")
    ev.add_argument("--no-jev", action="store_true", help="skip the JEV arm")
    ev.add_argument("--out", default="pair_scout/output", help="output directory")

    rd = sub.add_parser("refresh-data", help="download fresh klines via data_manager.py")
    rd.add_argument("-c", "--cex", default="binance")
    rd.add_argument("-i", "--interval", default="1h")
    rd.add_argument("-l", "--limit", type=int, default=1500)
    rd.add_argument("-e", "--end", default="")
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
        parser.error(f"unknown command {args.command}")
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except Exception:  # noqa: BLE001 — top-level: log with traceback, non-zero exit
        logger.exception("unhandled error")
        return 1
    return 0
