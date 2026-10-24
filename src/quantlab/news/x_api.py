"""X (Twitter) API v2 recent-search provider.

NOT tested against the real API from the development sandbox (no network, no token): the
parsing is tested on a response shaped like the v2 documentation. Access is paid (pay-per-use
credits since 2026, according to third-party summaries: verify on X's pricing page), so the
provider enforces a hard per-run cap on posts requested.
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlencode

from .models import NewsItem
from .sanitize import sanitize_text

BASE = "https://api.x.com/2/tweets/search/recent"


def _default_fetch(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers={**headers, "User-Agent": "quantlab/0.1"})
    with urllib.request.urlopen(req, timeout=15) as r:                      # noqa: S310 - fixed https host
        return json.loads(r.read(2_000_000))


class XProvider:
    def __init__(self, bearer_token: str | None = None, queries: list[str] | None = None,
                 max_results: int = 20, max_posts_per_run: int = 40,
                 fetch: Callable[[str, dict], dict] | None = None):
        self._token = bearer_token or os.environ.get("X_BEARER_TOKEN")
        if not self._token:
            raise ValueError("X_BEARER_TOKEN is not set")
        self.queries = queries or []
        if not 10 <= max_results <= 100:
            raise ValueError("max_results must be between 10 and 100 (API limits)")
        if len(self.queries) * max_results > max_posts_per_run:
            raise ValueError("queries x max_results exceeds max_posts_per_run (cost guard)")
        if any(len(q) > 512 for q in self.queries):
            raise ValueError("query longer than 512 characters")
        self.max_results = max_results
        self._fetch = fetch or _default_fetch

    def __repr__(self) -> str:
        return "XProvider(token=<redacted>)"

    def fetch_latest(self, since: datetime | None = None) -> list[NewsItem]:
        items: list[NewsItem] = []
        for q in self.queries:
            params = {"query": q, "max_results": self.max_results,
                      "tweet.fields": "created_at,author_id,lang", "expansions": "author_id",
                      "user.fields": "username"}
            if since:
                params["start_time"] = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            data = self._fetch(f"{BASE}?{urlencode(params)}", {"Authorization": f"Bearer {self._token}"})
            users = {u["id"]: u.get("username", "?") for u in data.get("includes", {}).get("users", [])}
            for t in data.get("data", []):
                handle = users.get(t.get("author_id"), "?")
                created = t.get("created_at")
                items.append(NewsItem(
                    source=f"x:@{handle}",
                    published=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else None,
                    title="",
                    text=sanitize_text(t.get("text"), 400),
                    url=f"https://x.com/{handle}/status/{t['id']}",
                ))
        return items
