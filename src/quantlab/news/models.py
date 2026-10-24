from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class NewsItem:
    source: str                  # e.g. "rss:federalreserve.gov", "x:@handle"
    published: datetime | None
    title: str
    text: str
    url: str | None = None


@dataclass(frozen=True)
class RiskFlag:
    theme: str
    severity: str                # "low" | "medium" | "high"
    affected_assets: list[str]
    evidence_ids: list[int]


@dataclass(frozen=True)
class Idea:
    symbol: str
    direction: str               # "reduce" | "hold" | "increase"  (advice for a HUMAN, never an order)
    horizon: str                 # "days" | "weeks"
    rationale: str
    confidence: float
    evidence_ids: list[int]


@dataclass
class Briefing:
    regime_note: str
    risk_flags: list[RiskFlag]
    ideas: list[Idea]
    suspicious_ids: list[int] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)       # audit trail of dropped model output
    items_used: int = 0
    created_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Briefing":
        return cls(
            regime_note=d["regime_note"],
            risk_flags=[RiskFlag(**f) for f in d.get("risk_flags", [])],
            ideas=[Idea(**i) for i in d.get("ideas", [])],
            suspicious_ids=list(d.get("suspicious_ids", [])),
            rejected=list(d.get("rejected", [])),
            items_used=d.get("items_used", 0),
            created_utc=d.get("created_utc", ""),
        )
