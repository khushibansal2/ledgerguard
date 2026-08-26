"""Core datatypes. All money is integer minor units (cents) - never floats.

Float arithmetic is the #1 source of silent reconciliation drift; 0.1+0.2 != 0.3
means a penny-perfect ledger is impossible. Everything here is exact integers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any, Literal, Optional

Source = Literal["BANK", "STRIPE", "ERP_AR", "ERP_AP"]
Side = Literal["SETTLEMENT", "LEDGER"]


def money(amount_major: float) -> int:
    """Convert a human amount (12.34) to exact minor units (1234)."""
    return int(round(amount_major * 100))


def fmt(cents: int, currency: str = "USD") -> str:
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}.get(currency, currency + " ")
    return f"{sign}{sym}{c // 100:,}.{c % 100:02d}"


@dataclass
class Record:
    """One row from one system of record."""
    id: str
    source: Source
    side: Side
    txn_date: str                 # ISO date the money/document is dated
    amount: int                   # minor units, signed: +inflow / -outflow
    currency: str
    description: str
    counterparty: str = ""        # raw, un-normalised vendor/customer string
    reference: str = ""           # invoice no / remittance ref, often dirty or absent
    due_date: Optional[str] = None
    doc_type: str = ""            # INVOICE | BILL | CHARGE | TRANSFER | FEE | CREDIT_NOTE
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> date:
        return date.fromisoformat(self.txn_date)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Match:
    """A proposed reconciliation between settlement id(s) and ledger id(s)."""
    match_id: str
    settlement_ids: list[str]
    ledger_ids: list[str]
    confidence: float
    layer: Literal["L1_DETERMINISTIC", "L2_SIMILARITY", "L3_AGENTIC"]
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    residual: int = 0             # unexplained cents after the match
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Exception_:
    """A record the controller refused to match, and why. Never silently dropped."""
    record_ids: list[str]
    category: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    reason: str
    suggested_action: str
    amount: int = 0
    currency: str = "USD"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
