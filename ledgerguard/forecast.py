"""Cash position and forward forecast, built *on top of* the reconciliation.

The dependency direction is the point. A forecast built from an unreconciled
ledger silently double-counts: the invoice sits in receivables while the cash
that settled it already sits in the bank balance. Only the matched set tells
you which receivables are actually still outstanding, so this module consumes
the controller's output rather than the raw records.

Every figure carries its own certainty. Cleared cash is a fact; a receivable
due in three weeks is a probability; an unreconciled exception is neither, and
is reported as an explicit range rather than folded into a point estimate.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from .engine import Controller
from .models import Record

# Collection probability by how overdue an invoice is. These are the standard
# shape of a corporate ageing curve: most invoices pay near term, and the
# probability of collection falls off a cliff past 90 days.
AGEING_CURVE = [
    (-10 ** 9, 0, 0.93),    # not yet due
    (1, 30, 0.86),          # 1-30 days overdue
    (31, 60, 0.68),
    (61, 90, 0.45),
    (91, 10 ** 9, 0.22),    # 90+ - provision heavily
]


def collection_probability(days_overdue: int) -> float:
    for lo, hi, p in AGEING_CURVE:
        if lo <= days_overdue <= hi:
            return p
    return 0.22


class CashForecast:
    def __init__(self, ctrl: Controller, as_of: str, opening_balance: int) -> None:
        self.c = ctrl
        self.as_of = date.fromisoformat(as_of)
        self.opening = opening_balance

    # -- position ----------------------------------------------------------
    def position(self) -> dict[str, Any]:
        cleared = sum(r.amount for r in self.c.settlements
                      if r.d <= self.as_of and r.currency == "USD")
        reconciled_ids = {i for m in self.c.matches for i in m.settlement_ids}
        unreconciled = [r for r in self.c.settlements if r.id not in reconciled_ids]
        unrec_value = sum(r.amount for r in unreconciled if r.currency == "USD")

        open_ap, open_ar = self._open_items()
        return {
            "as_of": self.as_of.isoformat(),
            "opening_balance": self.opening,
            "cleared_movements": cleared,
            "closing_bank_balance": self.opening + cleared,
            "reconciled_pct": (len(reconciled_ids) / len(self.c.settlements)
                               if self.c.settlements else 0.0),
            "unreconciled_count": len(unreconciled),
            "unreconciled_value": unrec_value,
            "open_payables": sum(r.amount for r in open_ap),
            "open_receivables": sum(r.amount for r in open_ar),
            "net_working_capital": (self.opening + cleared
                                    + sum(r.amount for r in open_ar)
                                    + sum(r.amount for r in open_ap)),
        }

    def _open_items(self) -> tuple[list[Record], list[Record]]:
        open_docs = [r for r in self.c.ledger if r.id not in self.c.consumed]
        ap = [r for r in open_docs if r.amount < 0 and r.doc_type in ("BILL",)]
        ar = [r for r in open_docs if r.amount > 0 and r.doc_type == "INVOICE"]
        return ap, ar

    # -- ageing ------------------------------------------------------------
    def ageing(self) -> dict[str, list[dict]]:
        ap, ar = self._open_items()
        out: dict[str, list[dict]] = {"receivables": [], "payables": []}
        for label, items in (("receivables", ar), ("payables", ap)):
            buckets: dict[str, dict] = defaultdict(lambda: {"count": 0, "amount": 0})
            for r in items:
                due = date.fromisoformat(r.due_date) if r.due_date else r.d
                od = (self.as_of - due).days
                key = ("current" if od <= 0 else "1-30" if od <= 30
                       else "31-60" if od <= 60 else "61-90" if od <= 90 else "90+")
                buckets[key]["count"] += 1
                buckets[key]["amount"] += r.amount
            order = ["current", "1-30", "31-60", "61-90", "90+"]
            out[label] = [{"bucket": k, **buckets[k]} for k in order if k in buckets]
        return out

    # -- forward projection -------------------------------------------------
    def project(self, horizon_days: int = 30) -> dict[str, Any]:
        """Weekly buckets with an expected case and a confidence band.

        The band is not decoration. `low` assumes every receivable slips a
        bucket and every payable lands on its due date - the treasury question
        is never "what do I expect" but "what is the worst week I must fund".
        """
        ap, ar = self._open_items()
        weeks: list[dict[str, Any]] = []
        balance = self.opening + sum(r.amount for r in self.c.settlements
                                     if r.d <= self.as_of and r.currency == "USD")
        run_exp = run_low = run_high = balance

        for w in range(horizon_days // 7 + (1 if horizon_days % 7 else 0)):
            start = self.as_of + timedelta(days=w * 7 + 1)
            end = min(self.as_of + timedelta(days=(w + 1) * 7),
                      self.as_of + timedelta(days=horizon_days))
            if start > end:
                break

            inflow_exp = inflow_high = 0
            for r in ar:
                due = date.fromisoformat(r.due_date) if r.due_date else r.d
                if start <= due <= end:
                    p = collection_probability((self.as_of - due).days)
                    inflow_exp += int(r.amount * p)
                    inflow_high += r.amount
            # Outflows are near-certain: a payable you have agreed is a payable
            # you will pay. Treat AP as a hard commitment on the due date.
            outflow = sum(r.amount for r in ap
                          if r.due_date and start <= date.fromisoformat(r.due_date) <= end)

            run_exp += inflow_exp + outflow
            run_low += outflow                    # nothing collects, all bills land
            run_high += inflow_high + outflow
            weeks.append({
                "week": w + 1,
                "from": start.isoformat(), "to": end.isoformat(),
                "expected_inflow": inflow_exp, "committed_outflow": outflow,
                "expected_balance": run_exp,
                "low_balance": run_low, "high_balance": run_high,
            })

        trough = min((w["low_balance"] for w in weeks), default=run_low)
        return {
            "horizon_days": horizon_days,
            "opening_position": balance,
            "weeks": weeks,
            "worst_case_trough": trough,
            "liquidity_warning": trough < 0,
            "unreconciled_overhang": sum(
                r.amount for r in self.c.settlements
                if r.id not in {i for m in self.c.matches for i in m.settlement_ids}
                and r.currency == "USD"),
        }
