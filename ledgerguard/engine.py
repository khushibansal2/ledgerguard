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
    """Risk-weighted autonomy: the confidence required rises with the money.

    Confidence alone is a bad gate. An 0.80 match on $80 and an 0.80 match on
    $800,000 carry the same probability of being wrong and wildly different
    consequences, so a single threshold is either too loose at the top of the
    book or too strict at the bottom - and a controller who is made to approve
    trivia stops reading the queue, which is its own failure mode.

    The ladder below is expressed the way a delegated-authority matrix already
    is in a real finance function, so it can be handed to an auditor as policy
    rather than as a tuning constant.
    """
    # (threshold in minor units, confidence required at or above it)
    # Calibrated against the confidence the evidence layers actually produce
    # (see ablate.py): each rung is set to hold roughly the least-evidenced
    # tenth of its band rather than a quarter of it. A gate that holds a
    # quarter of the material book is not prudence, it is a controller who
    # stops reading the queue by Wednesday.
    ladder: tuple[tuple[int, float], ...] = (
        (0,           0.75),   # under $1,000  - routine
        (100_000,     0.82),   # $1,000+       - reviewable
        (1_000_000,   0.90),   # $10,000+      - material
        (10_000_000,  0.96),   # $100,000+     - significant; near-certainty only
    )
    max_residual: int = 2                   # cents of unexplained difference

    def required_confidence(self, amount: int) -> float:
        req = self.ladder[0][1]
        for threshold, needed in self.ladder:
            if abs(amount) >= threshold:
                req = needed
        return req

    def band(self, amount: int) -> str:
        names = ["routine", "reviewable", "material", "significant"]
        idx = 0
        for i, (threshold, _n) in enumerate(self.ladder):
            if abs(amount) >= threshold:
                idx = i
        return names[idx]

    def gate(self, m: Match, amount: int) -> tuple[bool, str]:
        if abs(m.residual) > self.max_residual:
            return False, f"unexplained residual {fmt(abs(m.residual))} exceeds tolerance"
        req = self.required_confidence(amount)
        if m.confidence < req:
            return False, (f"{self.band(amount)} item {fmt(abs(amount))} at confidence "
                           f"{m.confidence:.2f} requires >= {req:.2f}")
        return True, (f"within policy: {self.band(amount)} item needs "
                      f"{req:.2f}, has {m.confidence:.2f}")


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
        self.stats = {"L0": 0, "L1": 0, "L2": 0, "L3": 0, "tool_calls": 0}
        self._mid = 0

    # -- helpers -----------------------------------------------------------
    def _next_mid(self) -> str:
        self._mid += 1
        return f"M-{self._mid:04d}"

    def _open_ledger(self) -> list[Record]:
        return [r for r in self.ledger if r.id not in self.consumed]

    def _open_settlements(self) -> list[Record]:
        """Unconsumed settlements that could genuinely settle a document.

        A line the rail typed as a reversal is money coming back, never a
        payment against an invoice. If L0 could not pair it, the honest outcome
        is an exception - letting it reach the similarity or resolver layers
        would close an invoice that is in fact still open.
        """
        return [r for r in self.settlements
                if r.id not in self.consumed and r.doc_type != "REVERSAL"]

    def _exposure(self, m: Match) -> int:
        """The value actually at risk in a match, not the value of one line.

        A group of instalments clearing one bill exposes the whole bill; gating
        on the first payment would authorise a $10,000 booking at the $6,000
        rung. Take the larger of the two sides, since either can be the
        complete economic event - a reversal pair has no ledger side at all.
        """
        settle = abs(sum(self.by_id[i].amount for i in m.settlement_ids
                         if i in self.by_id))
        ledger = abs(sum(self.by_id[i].amount for i in m.ledger_ids
                         if i in self.by_id))
        return max(settle, ledger)

    def _book(self, m: Match, driver: Record) -> None:
        ok, why = self.policy.gate(m, self._exposure(m))
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
                evidence={"proposed": m.to_dict()},
                missing_evidence=(
                    "None - the match is fully evidenced. This is a delegated-"
                    "authority hold, not an unresolved item: a controller "
                    "signature is the only thing required.")))
            self.audit.emit("MATCH_HELD", match_id=m.match_id, policy=why)

    def _flag(self, exc: Exception_) -> None:
        self.exceptions.append(exc)
        self.audit.emit("EXCEPTION_RAISED", category=exc.category,
                        severity=exc.severity, records=exc.record_ids,
                        reason=exc.reason, action=exc.suggested_action)

    # -- LAYER 0 -----------------------------------------------------------
    def layer0_reversals(self) -> None:
        """Cancel equal-and-opposite settlement pairs before anything else runs.

        A returned ACH, a re-presented payment and a chargeback all produce bank
        lines that look exactly like real settlements. Left in the pool they do
        damage in both directions: the reversal can be booked against an open
        bill (understating the expense), or the original can be matched while
        its reversal is left to sit in the exception queue forever.

        Neither line settles a document, so this pass books them against *each
        other* - an economically closed event with no ledger entry. It runs
        first because every later layer assumes the settlements it sees are real
        movements of money.
        """
        pool = [s for s in self.settlements if s.id not in self.consumed]
        used: set[str] = set()

        # Work from the reversal outwards. The reversal is the line the rail
        # explicitly typed, so it is the fact; the payment it cancels is the
        # thing to be inferred.
        for rev in [r for r in pool if r.doc_type == "REVERSAL"]:
            if rev.id in used:
                continue
            # Direction matters: money can only be given back after it was
            # taken. A failed payment is normally re-presented, leaving three
            # same-value lines - original, return, retry - and the return
            # offsets either negative line arithmetically. Restricting to
            # earlier lines stops the pass cancelling the retry and leaving the
            # original to be booked a second time.
            partners = [
                p for p in pool
                if p.id not in used and p.id != rev.id
                and p.doc_type != "REVERSAL"
                and p.amount == -rev.amount and p.currency == rev.currency
                and p.d <= rev.d and (rev.d - p.d).days <= 30
                and counterparty_score(p.counterparty, rev.counterparty) >= 0.90
            ]
            if not partners:
                continue
            if len(partners) > 1:
                # Two payments of the same value to the same vendor before the
                # same return. Refusing here would be worse than choosing: the
                # reversal is certainly cancelling one of them, and leaving it
                # unpaired lets a return be booked against an invoice later.
                # Take the nearest preceding payment - the convention a bank
                # applies, and economically identical when the amounts match.
                self.audit.emit("AMBIGUITY_DETECTED", settlement=rev.id,
                                candidates=[p.id for p in partners],
                                resolution="nearest preceding payment")
            partners.sort(key=lambda p: (p.txn_date, p.id), reverse=True)
            a = partners[0]
            used.update({a.id, rev.id})
            gap = (rev.d - a.d).days
            kind = ("chargeback" if "CHARGEBACK" in (a.description + rev.description).upper()
                    else "returned payment")
            self._book(Match(
                self._next_mid(), [a.id, rev.id], [], 0.98, "L0_REVERSAL",
                f"{fmt(abs(a.amount), a.currency)} {kind} for "
                f"'{a.counterparty}': the settlement on {a.txn_date} was "
                f"reversed on {rev.txn_date}, {gap} days later. The pair nets to "
                f"zero and settles no document - booking either line against a "
                f"ledger entry would misstate the period.",
                {"pair": [a.id, rev.id], "gap_days": gap, "kind": kind}), a)

    # -- LAYER 1 -----------------------------------------------------------
    def layer1(self) -> None:
        """Exact reference + exact amount + same currency. Proof, not inference."""
        by_ref: dict[tuple[str, int, str], list[Record]] = {}
        for r in self.ledger:
            if r.reference:
                by_ref.setdefault((r.reference.upper(), r.amount, r.currency), []).append(r)

        for s in self._open_settlements():
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
            setts = [s for s in self._open_settlements()
                     if s.amount == amount
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
        for s in self._open_settlements():
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
        for s in self._open_settlements():
            # The list above is a snapshot. An instalment match consumes several
            # settlements at once, so by the time the loop reaches one of them it
            # may already be booked - re-check rather than resolve it twice.
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
        # Reversals that never found the payment they cancel. They are excluded
        # from every matching layer by design, so without this they would be the
        # one class of record that vanishes silently - the exact failure the
        # exception ledger exists to prevent.
        for r in self.settlements:
            if r.id in self.consumed or r.doc_type != "REVERSAL":
                continue
            self._flag(Exception_(
                [r.id], "UNPAIRED_REVERSAL", "HIGH",
                f"{fmt(abs(r.amount), r.currency)} returned on {r.txn_date} "
                f"('{r.description}') but no matching outbound payment to "
                f"'{r.counterparty}' was found in the period. Either the original "
                f"sits outside this window or the amounts disagree.",
                "Trace the original payment before closing; an unmatched reversal "
                "means either cash was returned that was never recorded as sent, "
                "or the original is still overstating the period.",
                r.amount, r.currency, {"doc_type": r.doc_type},
                missing_evidence=(
                    "The prior-period bank statement containing the original "
                    "payment, or the rail's return-reason reference tying this "
                    "credit to the debit it reverses.")))

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
                    l.amount, l.currency, {"sibling_ids": d["ids"]},
                    missing_evidence=(
                        "Confirmation from the vendor that reference "
                        f"{d['reference']} was issued once. If both are genuine, "
                        "a second settlement should exist and does not.")))
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
                l.amount, l.currency, {"doc_type": l.doc_type, "overdue": overdue},
                missing_evidence=(
                    "None - this is an open item, not a failure. It resolves "
                    "when the counterparty pays; until then it is carried in "
                    "the forecast at its ageing-adjusted value.")))

    # -- orchestration -----------------------------------------------------
    def run(self, as_of: str = "2026-04-05") -> "Controller":
        self.audit.emit("RUN_START", records=len(self.records),
                        settlements=len(self.settlements), ledger=len(self.ledger),
                        policy=self.policy.__dict__, as_of=as_of)
        self.audit.emit("LAYER_START", layer="L0_REVERSAL")
        self.layer0_reversals()
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
