"""The three-layer reconciliation engine.

    L1  DETERMINISTIC  exact reference + exact amount. No scoring, no model.
    L2  SIMILARITY     exact amount, fuzzy entity + date decay. Still no model.
    L3  AGENTIC        amount does NOT tie out; a tool-using resolver must
                       produce an arithmetic explanation for the difference.

Two rules govern the whole design:

1. Escalating layers cost more and are trusted less. Anything L1 can settle
   never reaches a model, which is both cheaper and safer - ~60% of a real
   book is exact-reference traffic and no LLM should be anywhere near it.

2. A match at L3 must be *arithmetically closed*. The resolver may only book a
   difference it can name and compute (this FX rate, that fee schedule, this
   statutory tax rate). "These look related" is not an explanation, and the
   engine will escalate rather than accept it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .audit import AuditLog
from .models import Exception_, Match, Record, fmt
from .similarity import amount_score, counterparty_score, date_score, reference_score
from . import tools


@dataclass
class Policy:
    """Risk-weighted autonomy. Confidence alone is a bad gate: an 0.80 match on
    $80 and an 0.80 match on $80,000 carry very different downside, so the bar
    rises with materiality."""
    auto_book_min: float = 0.75
    material_amount: int = 500_000          # $5,000 in cents
    material_min_confidence: float = 0.88
    max_residual: int = 2                   # cents of unexplained difference

    def gate(self, m: Match, amount: int) -> tuple[bool, str]:
        if abs(m.residual) > self.max_residual:
            return False, f"unexplained residual {fmt(abs(m.residual))} exceeds tolerance"
        if m.confidence < self.auto_book_min:
            return False, f"confidence {m.confidence:.2f} below auto-book floor {self.auto_book_min:.2f}"
        if abs(amount) >= self.material_amount and m.confidence < self.material_min_confidence:
            return False, (f"material item {fmt(abs(amount))} at confidence "
                           f"{m.confidence:.2f} requires >= {self.material_min_confidence:.2f}")
        return True, "within policy"


class Controller:
    def __init__(self, records: list[Record], policy: Policy | None = None,
                 audit: AuditLog | None = None, planner=None) -> None:
        self.records = records
        self.by_id = {r.id: r for r in records}
        self.policy = policy or Policy()
        self.audit = audit or AuditLog()
        self.planner = planner              # optional LLM planner (see agent.py)

        # Work chronologically, ties broken by id. Layers consume documents as
        # they match, so whichever settlement is processed first gets first
        # claim on a shared candidate - meaning the order of the input file
        # would otherwise decide the outcome. A close runs oldest-first; making
        # that explicit is what lets the same book reconcile identically no
        # matter how the bank chose to sort the export.
        order = lambda r: (r.txn_date, r.id)
        self.settlements = sorted((r for r in records if r.side == "SETTLEMENT"), key=order)
        self.ledger = sorted((r for r in records if r.side == "LEDGER"), key=order)
        self.consumed: set[str] = set()

        self.matches: list[Match] = []
        self.exceptions: list[Exception_] = []
        self.escalated: list[Match] = []    # resolved, but held for human sign-off
        self.stats = {"L1": 0, "L2": 0, "L3": 0, "tool_calls": 0}
        self._mid = 0

    # -- helpers -----------------------------------------------------------
    def _next_mid(self) -> str:
        self._mid += 1
        return f"M-{self._mid:04d}"

    def _open_ledger(self) -> list[Record]:
        return [r for r in self.ledger if r.id not in self.consumed]

    def _book(self, m: Match, driver: Record) -> None:
        ok, why = self.policy.gate(m, driver.amount)
        self.audit.emit("MATCH_PROPOSED", match_id=m.match_id, layer=m.layer,
                        confidence=round(m.confidence, 4),
                        settlement=m.settlement_ids, ledger=m.ledger_ids,
                        residual=m.residual, rationale=m.rationale,
                        tools=m.tools_used, evidence=m.evidence)
        for rid in m.settlement_ids + m.ledger_ids:
            self.consumed.add(rid)
        if ok:
            self.matches.append(m)
            self.stats[m.layer.split("_")[0]] += 1
            self.audit.emit("MATCH_BOOKED", match_id=m.match_id, policy=why)
        else:
            self.escalated.append(m)
            self.exceptions.append(Exception_(
                record_ids=m.settlement_ids + m.ledger_ids,
                category="POLICY_HOLD", severity="MEDIUM",
                reason=f"Resolved at {m.layer} but held: {why}",
                suggested_action="Controller review; approve to book as proposed.",
                amount=driver.amount, currency=driver.currency,
                evidence={"proposed": m.to_dict()}))
            self.audit.emit("MATCH_HELD", match_id=m.match_id, policy=why)

    def _flag(self, exc: Exception_) -> None:
        self.exceptions.append(exc)
        self.audit.emit("EXCEPTION_RAISED", category=exc.category,
                        severity=exc.severity, records=exc.record_ids,
                        reason=exc.reason, action=exc.suggested_action)

    # -- LAYER 1 -----------------------------------------------------------
    def layer1(self) -> None:
        """Exact reference + exact amount + same currency. Proof, not inference."""
        by_ref: dict[tuple[str, int, str], list[Record]] = {}
        for r in self.ledger:
            if r.reference:
                by_ref.setdefault((r.reference.upper(), r.amount, r.currency), []).append(r)

        for s in self.settlements:
            if s.id in self.consumed or not s.reference:
                continue
            key = (s.reference.upper(), s.amount, s.currency)
            hits = [r for r in by_ref.get(key, []) if r.id not in self.consumed]
            if len(hits) == 1:
                l = hits[0]
                self._book(Match(self._next_mid(), [s.id], [l.id], 1.0,
                                 "L1_DETERMINISTIC",
                                 f"Exact reference {s.reference} with identical "
                                 f"amount {fmt(s.amount, s.currency)} and currency.",
                                 {"reference": s.reference, "amount": s.amount}), s)
            elif len(hits) > 1:
                # Two ledger docs identical on reference, amount and currency.
                # Which one is "paid" is arbitrary - they are the same document
                # twice - but the choice must not depend on the order the bank
                # file happened to arrive in, or the same book reconciles two
                # different ways on two different days. Oldest first, ties broken
                # by id: the FIFO convention an AP subledger already uses, and
                # deterministic under any permutation of the input.
                hits = sorted(hits, key=lambda r: (r.txn_date, r.id))
                self.audit.emit("AMBIGUITY_DETECTED", settlement=s.id,
                                candidates=[h.id for h in hits], key=str(key),
                                resolution="oldest-first, deterministic")
                l = hits[0]
                self._book(Match(self._next_mid(), [s.id], [l.id], 0.97,
                                 "L1_DETERMINISTIC",
                                 f"Exact reference {s.reference}; {len(hits)} identical "
                                 f"ledger documents exist - settled against the earliest, "
                                 f"remainder flagged as suspected duplicate billing.",
                                 {"reference": s.reference, "duplicates": [h.id for h in hits]}), s)

    # -- LAYER 2a ----------------------------------------------------------
    def layer2_cohorts(self) -> None:
        """Pair off closed cohorts of genuinely indistinguishable items.

        Two identical $4,820 bills from one vendor, settled by two identical
        $4,820 payments with no remittance advice, are not a matching problem -
        there is no fact that distinguishes pairing A-1/B-2 from A-2/B-1, and
        both produce an identical ledger. Escalating them to a human is theatre:
        the human has no more information than the engine does.

        So when the cohort is *closed* - equal counts on both sides, one
        counterparty, no separating reference - apply them FIFO, the same
        convention an AP subledger uses. The guard is strict: an unequal cohort
        (3 bills, 2 payments) means one bill genuinely was not paid, and picking
        which would be a fabrication, so that case falls through to the
        ambiguity guard and becomes an exception.
        """
        open_led = [l for l in self._open_ledger() if l.counterparty]
        groups: dict[tuple, list[Record]] = {}
        for l in open_led:
            groups.setdefault((l.amount, l.currency), []).append(l)

        # Iterate groups and their members in a stable order. Everything below
        # picks a representative document and pairs FIFO, so an unstable order
        # here would let the same book close two different ways.
        for (amount, currency) in sorted(groups):
            docs = sorted(groups[(amount, currency)], key=lambda r: (r.txn_date, r.id))
            if len(docs) < 2:
                continue
            # every document in the cohort must be the same counterparty
            if any(counterparty_score(docs[0].counterparty, d.counterparty) < 0.95
                   for d in docs[1:]):
                continue
            if any(d.reference for d in docs) and len({d.reference for d in docs}) > 1:
                # references separate them; let L1/L2 use that evidence instead
                if not all(d.reference for d in docs):
                    continue
            setts = [s for s in self.settlements
                     if s.id not in self.consumed and s.amount == amount
                     and s.currency == currency and not s.reference
                     and counterparty_score(s.counterparty or s.description,
                                            docs[0].counterparty) >= 0.90]
            if len(setts) != len(docs):
                continue

            self.audit.emit("COHORT_DETECTED", counterparty=docs[0].counterparty,
                            amount=amount, currency=currency, size=len(docs),
                            ledger=[d.id for d in docs],
                            settlements=[s.id for s in setts])
            for s, l in zip(sorted(setts, key=lambda r: (r.txn_date, r.id)),
                            sorted(docs, key=lambda r: (r.txn_date, r.id))):
                self._book(Match(
                    self._next_mid(), [s.id], [l.id], 0.93, "L2_SIMILARITY",
                    f"Closed cohort: {len(docs)} indistinguishable documents of "
                    f"{fmt(amount, currency)} from '{l.counterparty}' settled by "
                    f"{len(setts)} identical payments with no remittance advice. "
                    f"Applied FIFO by date - any pairing yields the same ledger, "
                    f"and the cohort as a whole reconciles exactly.",
                    {"cohort_size": len(docs), "convention": "FIFO",
                     "cohort_ledger": [d.id for d in docs],
                     "cohort_settlements": [x.id for x in setts]}), s)

    # -- LAYER 2 -----------------------------------------------------------
    def layer2(self, floor: float = 0.70) -> None:
        """Amount ties out to the cent; identity and timing must be inferred.

        Scored pairs are booked greedily best-first with mutual exclusion, so a
        strong pair claims its counterparty before a weaker one can.
        """
        scored: list[tuple[float, Record, Record, dict]] = []
        runner_up: dict[str, float] = {}
        for s in self.settlements:
            if s.id in self.consumed:
                continue
            per_settlement: list[tuple[float, Record, dict]] = []
            for l in self._open_ledger():
                if l.currency != s.currency or l.amount != s.amount:
                    continue
                cp = counterparty_score(s.counterparty or s.description, l.counterparty)
                dt = (s.d - l.d).days
                ds = date_score(dt)
                rf = reference_score(s.reference, l.reference)
                # Amount ties to the cent by construction here, so identity does
                # the discriminating; date only modulates. Weighting date heavily
                # would reject correct cross-month settlements, which is the exact
                # population a controller most needs matched.
                score = 0.70 * cp + 0.15 * ds + 0.15 * max(rf, cp * 0.6)
                per_settlement.append((score, l, {"amount_exact": True,
                                                  "counterparty_score": round(cp, 3),
                                                  "date_lag_days": dt,
                                                  "date_score": round(ds, 3),
                                                  "reference_score": round(rf, 3)}))
            per_settlement.sort(key=lambda x: (-x[0], x[1].id))
            if per_settlement:
                runner_up[s.id] = per_settlement[1][0] if len(per_settlement) > 1 else 0.0
            for score, l, ev in per_settlement:
                if score >= floor:
                    scored.append((score, s, l, ev))

        for score, s, l, ev in sorted(scored, key=lambda x: (-x[0], x[1].id, x[2].id)):
            if s.id in self.consumed or l.id in self.consumed:
                continue
            # Ambiguity guard: an identical-amount book with two equally plausible
            # counterparties is a coin flip, and a coin flip booked confidently is
            # worse than an exception. Demand daylight over the runner-up.
            second = runner_up.get(s.id, 0.0)
            if score - second < 0.12:
                self.audit.emit("AMBIGUITY_DETECTED", settlement=s.id,
                                best=l.id, best_score=round(score, 4),
                                runner_up_score=round(second, 4),
                                note="margin below 0.12 - deferred to L3/exception")
                continue
            lag = ev["date_lag_days"]
            note = (f"Amount ties to the cent ({fmt(s.amount, s.currency)}). "
                    f"Counterparty '{s.counterparty or s.description}' matched to "
                    f"'{l.counterparty}' at {ev['counterparty_score']:.2f}; "
                    f"settled {lag}d after the document date.")
            if l.txn_date[:7] != s.txn_date[:7]:
                note += " Crosses the month boundary - accrual cut-off applies."
            self._book(Match(self._next_mid(), [s.id], [l.id], round(min(score, 0.99), 4),
                             "L2_SIMILARITY", note, ev), s)

    # -- LAYER 3 -----------------------------------------------------------
    def layer3(self) -> None:
        """Amount does not tie out. Prove the difference or escalate it."""
        from .agent import Resolver
        resolver = Resolver(self, planner=self.planner)
        for s in self.settlements:
            if s.id in self.consumed:
                continue
            outcome = resolver.resolve(s)
            self.stats["tool_calls"] += outcome.tool_calls
            if outcome.match:
                self._book(outcome.match, s)
            else:
                self._flag(outcome.exception)

    # -- sweep unresolved ledger docs --------------------------------------
    def sweep(self, as_of: str) -> None:
        dupes = tools.duplicate_scan(self.records)
        self.stats["tool_calls"] += 1
        dupe_ids = {i for d in dupes for i in d["ids"]}
        dupe_by_id = {i: d for d in dupes for i in d["ids"]}

        for l in self._open_ledger():
            if l.id in dupe_ids and any(x in self.consumed for x in dupe_by_id[l.id]["ids"]):
                d = dupe_by_id[l.id]
                self._flag(Exception_(
                    [l.id], "DUPLICATE_BILLING", "HIGH",
                    f"Reference {d['reference']} for '{l.counterparty}' exists on "
                    f"{len(d['ids'])} ledger documents at {fmt(l.amount, l.currency)}, "
                    f"but only one settlement was found. Paying both would be a "
                    f"{fmt(abs(l.amount), l.currency)} overpayment.",
                    "Block for payment; confirm with vendor and void the duplicate.",
                    l.amount, l.currency, {"sibling_ids": d["ids"]}))
                continue

            open_kind = "OPEN_RECEIVABLE" if l.amount > 0 else "OPEN_PAYABLE"
            if l.doc_type == "CREDIT_NOTE":
                open_kind = "UNAPPLIED_CREDIT"
            overdue = bool(l.due_date and l.due_date < as_of)
            self._flag(Exception_(
                [l.id], open_kind, "MEDIUM" if overdue else "LOW",
                (f"No settlement found in the period. Document dated {l.txn_date}"
                 + (f", due {l.due_date}" + (" - OVERDUE." if overdue else ".")
                    if l.due_date else ".")),
                ("Chase the counterparty; ageing bucket feeds the cash forecast."
                 if not overdue else "Overdue - escalate to collections/AP run."),
                l.amount, l.currency, {"doc_type": l.doc_type, "overdue": overdue}))

    # -- orchestration -----------------------------------------------------
    def run(self, as_of: str = "2026-04-05") -> "Controller":
        self.audit.emit("RUN_START", records=len(self.records),
                        settlements=len(self.settlements), ledger=len(self.ledger),
                        policy=self.policy.__dict__, as_of=as_of)
        self.audit.emit("LAYER_START", layer="L1_DETERMINISTIC")
        self.layer1()
        self.audit.emit("LAYER_START", layer="L2_COHORT")
        self.layer2_cohorts()
        self.audit.emit("LAYER_START", layer="L2_SIMILARITY")
        self.layer2()
        self.audit.emit("LAYER_START", layer="L3_AGENTIC")
        self.layer3()
        self.audit.emit("LAYER_START", layer="SWEEP")
        self.sweep(as_of)
        self.audit.emit("RUN_END", matched=len(self.matches),
                        escalated=len(self.escalated),
                        exceptions=len(self.exceptions), stats=self.stats)
        return self
