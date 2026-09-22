"""Telegram delivery via python-telegram-bot (async), retry on transient errors.

Fail-fast policy: blank/missing TELEGRAM_* values raise before any send attempt
(server safety, plan §8).
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import NetworkError, TelegramError, TimedOut

logger = logging.getLogger(__name__)


class TelegramConfigError(RuntimeError):
    """Raised when credentials are missing/blank."""


def validate_credentials(token: str | None, chat_id: str | None) -> None:
    if not token or not token.strip():
        raise TelegramConfigError(
            "TELEGRAM_BOT_TOKEN is blank — set it in .env before sending"
        )
    if not chat_id or not chat_id.strip():
        raise TelegramConfigError(
            "TELEGRAM_GROUP_ID is blank — set it in .env before sending"
        )


async def _send_all(token: str, chat_id: str, chunks: list[str]) -> None:
    bot = Bot(token=token)
    async with bot:
        for i, chunk in enumerate(chunks, 1):
            for attempt in range(1, 4):
                try:
                    await bot.send_message(
                        chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML
                    )
                    logger.info("sent chunk %d/%d", i, len(chunks))
                    break
                except (TimedOut, NetworkError) as exc:
                    if attempt == 3:
                        raise
                    wait = 2.0 * attempt
                    logger.warning(
                        "telegram send attempt %d failed (%s); retrying in %.0fs",
                        attempt, exc, wait,
                    )
                    await asyncio.sleep(wait)


def send_report(token: str, chat_id: str, chunks: list[str]) -> None:
    validate_credentials(token, chat_id)
    asyncio.run(_send_all(token.strip(), chat_id.strip(), chunks))


def send_report_or_fail(token: str | None, chat_id: str | None, chunks: list[str]) -> None:
    validate_credentials(token, chat_id)
    try:
        send_report(token, chat_id, chunks)
    except TelegramError as exc:
        raise RuntimeError(f"telegram delivery failed: {exc}") from exc
