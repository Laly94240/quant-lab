"""Everything that comes from news feeds or social media is UNTRUSTED DATA, never instructions.

Sanitising reduces noise and obvious tricks; it is not the security boundary. The boundary is
architectural: the model's output is schema-validated, restricted to a whitelist of symbols,
advisory only, and nothing in the news path can create or send an order.
"""
from __future__ import annotations

import re
import unicodedata

_TAGS = re.compile(r"<[^>]*>")
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")

_SUSPICIOUS = [re.compile(p, re.I) for p in (
    r"ignore (all |any |the |your )?(previous|prior|above|earlier)",
    r"disregard .{0,40}(instruction|rule|prompt)",
    r"(system|developer) prompt",
    r"you are (now|no longer)",
    r"new instructions?",
    r"(place|submit|execute|send|make) (an? |the )?(buy |sell )?(order|trade)",
    r"</?\s*(system|assistant|user|untrusted|instructions?)\b",
    r"(api[_ -]?key|bearer|webhook|password|secret)",
    r"\b(buy|sell|short)\b.{0,20}\b(now|immediately|asap)\b",
    r"ignore.{0,20}(instructions|consignes)",
    r"(achète|vends|passe).{0,20}(maintenant|immédiatement|un ordre)",
)]


def sanitize_text(text: str | None, max_len: int = 400) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = _TAGS.sub(" ", s)
    s = _CONTROL.sub("", s)
    s = _URL.sub("[url]", s)
    s = s.replace("@everyone", "everyone").replace("@here", "here")
    s = _WS.sub(" ", s).strip()
    return s[:max_len]


def looks_suspicious(text: str | None) -> bool:
    s = unicodedata.normalize("NFKC", text or "")
    return any(p.search(s) for p in _SUSPICIOUS)
