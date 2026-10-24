"""Discord notifications through an incoming webhook (one HTTPS POST, no bot, no library).

Design rules:
  * The webhook URL is a secret (anyone holding it can post to the channel): environment only,
    never logged, never in the repr.
  * `allowed_mentions` is emptied so that untrusted text (news, tweets) can never ping
    @everyone / @here / a role.
  * A notification failure must never break trading logic: `send` swallows errors and returns False.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from urllib.parse import urlparse

log = logging.getLogger(__name__)

Transport = Callable[[str, dict], "tuple[int, Mapping[str, str]]"]   # (url, payload) -> (status, headers)

_ALLOWED_HOSTS = {"discord.com", "discordapp.com", "ptb.discord.com", "canary.discord.com"}
_COLORS = {"info": 0x3498DB, "success": 0x2ECC71, "warning": 0xF1C40F, "error": 0xE74C3C}


def _urllib_transport(url: str, payload: dict) -> tuple[int, Mapping[str, str]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "quantlab/0.1 (notifier)"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:   # noqa: S310 - host validated in __init__
            return resp.status, dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {})


class NullNotifier:
    """Used when no webhook is configured: same interface, does nothing."""

    def send(self, title: str, body: str = "", *, fields=None, level: str = "info") -> bool:
        return False


class DiscordNotifier:
    def __init__(self, webhook_url: str, transport: Transport | None = None, username: str = "quantlab",
                 max_retries: int = 2, sleep: Callable[[float], None] = time.sleep):
        parsed = urlparse(webhook_url)
        if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS or "/api/webhooks/" not in parsed.path:
            raise ValueError("not a Discord webhook URL (expected https://discord.com/api/webhooks/...)")
        self._url = webhook_url
        self._transport = transport or _urllib_transport
        self.username, self.max_retries, self._sleep = username, max_retries, sleep

    def __repr__(self) -> str:
        return "DiscordNotifier(webhook=<redacted>)"

    def send(self, title: str, body: str = "", *, fields: list[tuple[str, str]] | None = None,
             level: str = "info") -> bool:
        embed = {
            "title": title[:256],
            "description": body[:4000],
            "color": _COLORS.get(level, _COLORS["info"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if fields:
            embed["fields"] = [
                {"name": str(n)[:256], "value": str(v)[:1024] or "-", "inline": True} for n, v in fields[:25]
            ]
        payload = {"username": self.username, "embeds": [embed], "allowed_mentions": {"parse": []}}
        try:
            for attempt in range(self.max_retries + 1):
                status, headers = self._transport(self._url, payload)
                if 200 <= status < 300:
                    return True
                if status == 429 and attempt < self.max_retries:
                    retry = float({k.lower(): v for k, v in headers.items()}.get("retry-after", 1))
                    self._sleep(min(max(retry, 0.0), 30.0))
                    continue
                log.warning("discord webhook returned HTTP %s", status)
                return False
        except Exception as exc:                       # never let a notification break the caller
            log.warning("discord notification failed: %s", type(exc).__name__)
        return False


def notifier_from_env(var: str = "DISCORD_WEBHOOK_URL", enabled: bool = True, **kw):
    url = os.environ.get(var)
    if not enabled or not url:
        return NullNotifier()
    return DiscordNotifier(url, **kw)
