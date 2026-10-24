"""RSS / Atom provider (free, no key). Parsing uses defusedxml: feeds are untrusted input."""
from __future__ import annotations

import logging
import urllib.request
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from defusedxml import ElementTree as ET

from .models import NewsItem
from .sanitize import sanitize_text

log = logging.getLogger(__name__)
_MAX_BYTES = 2_000_000


def _default_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "quantlab/0.1 (rss reader)"})
    with urllib.request.urlopen(req, timeout=15) as r:                     # noqa: S310 - https only, see below
        return r.read(_MAX_BYTES)


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(el, *names: str) -> str:
    for c in el:
        if _strip_ns(c.tag) in names and (c.text or "").strip():
            return c.text.strip()
    return ""


def _parse_date(s: str) -> datetime | None:
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s)                                        # RSS (RFC 822)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))            # Atom (ISO 8601)
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_feed(xml_bytes: bytes, source: str) -> list[NewsItem]:
    root = ET.fromstring(xml_bytes)
    items = []
    for el in root.iter():
        if _strip_ns(el.tag) not in ("item", "entry"):
            continue
        link = _child_text(el, "link")
        if not link:                                                         # Atom: <link href="...">
            for c in el:
                if _strip_ns(c.tag) == "link" and c.get("href"):
                    link = c.get("href")
                    break
        items.append(NewsItem(
            source=source,
            published=_parse_date(_child_text(el, "pubDate", "published", "updated", "date")),
            title=sanitize_text(_child_text(el, "title"), 200),
            text=sanitize_text(_child_text(el, "description", "summary", "content"), 400),
            url=link or None,
        ))
    return items


class RSSProvider:
    def __init__(self, feeds: Iterable[str], fetch: Callable[[str], bytes] | None = None,
                 max_items_per_feed: int = 20):
        self.feeds = list(feeds)
        for f in self.feeds:
            if urlparse(f).scheme != "https":
                raise ValueError("only https feeds are accepted")
        self._fetch = fetch or _default_fetch
        self.max_items = max_items_per_feed

    def fetch_latest(self, since: datetime | None = None) -> list[NewsItem]:
        out: list[NewsItem] = []
        for url in self.feeds:
            try:
                items = parse_feed(self._fetch(url), f"rss:{urlparse(url).hostname}")
            except Exception as exc:                                         # one bad feed must not stop the rest
                log.warning("feed skipped (%s): %s", urlparse(url).hostname, type(exc).__name__)
                continue
            out.extend(items[: self.max_items])
        seen, unique = set(), []
        for it in out:
            key = it.url or it.title
            if key in seen or (since and it.published and it.published < since):
                continue
            seen.add(key)
            unique.append(it)
        unique.sort(key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return unique
