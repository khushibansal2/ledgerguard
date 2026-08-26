# LedgerGuard

An autonomous finance controller that closes one full ops loop — bank, processor
and ERP records in, a booked ledger and a forward cash position out — and
reports every item it refused to book, with the reason attached.

Zero third-party dependencies. Python 3.11+, standard library only.

```bash
python -m ledgerguard.cli                 # close one batch, full report
python -m ledgerguard.cli --seeds 200     # robustness across 200 batches
python -m ledgerguard.ablate              # does each layer earn its place?
python -m ledgerguard.report              # regenerate the HTML close report
python tests/test_controller.py           # 13 invariants, no pytest needed
python -m ledgerguard.verify_claims       # re-verify every number in this README
python -m ledgerguard.cli --csv work.csv  # exception worklist for the controller
```

Every figure below is asserted in CI by `verify_claims`, so a change that
regresses accuracy fails the build instead of quietly shipping.

## Measured, not asserted

Over **200 independent batches — 22,576 records, 9,400 settlement events**,
graded against a truth key the engine never sees:

| Metric | Result | |
|---|---|---|
| Precision | **99.93%** | 0 false positives; 6 partial matches in 8,800 links |
| Straight-through | **93.5%** | booked with no human involvement |
| Resolved correctly | **99.2%** | including items held for sign-off |
| Escalation recall | **100%** | every truly-unmatchable record refused |
| Throughput | ~1,180 rec/s | ~85 ms per batch |
| Model usage | 1.25 tool calls/event | 81% of events never reach a model |

Confidence is calibrated, not decorative — pooled over 9,400 events, every
confidence band's observed accuracy matches its label.

## The design in one idea

Three layers, each costing more and trusted less than the one above it:

| | Layer | Test | Model involved |
|---|---|---|---|
| **L1** | Deterministic | exact reference + exact amount | never |
| **L2** | Similarity | amount ties out; entity and timing inferred | never |
| **L3** | Agentic | amount does *not* tie out — prove the difference | may order hypotheses |

**A match at L3 must be arithmetically closed.** The resolver may only book a
difference it can name and compute — *this* FX rate, *that* fee schedule, *this*
statutory tax rate. "These look related" is not an explanation, and the engine
escalates rather than accept it. The LLM chooses which explanation to test; a
deterministic tool computes the answer; a policy gate decides whether it may be
booked. With no API key the built-in planner runs the identical hypothesis set,
so the pipeline is fully reproducible offline and the model is an accelerator,
never a dependency.

## Why it should be believed

**Every layer is ablated.** Same batches, same grading, one capability removed
at a time (`python -m ledgerguard.ablate`):

| Configuration | Precision | Straight-through | Mis-booked value |
|---|---|---|---|
| naive: equal amount | 90.8% | 47.4% | $373,724 |
| naive: amount + 3-day window | 88.7% | 36.4% | $321,806 |
| no L3 resolver | 100% | 48.6% | $0 |
| no cohort pass | 100% | 93.2% | $0 |
| no policy gate | 100% | 99.2% | $0 |
| **LedgerGuard (full)** | **100%** | **93.5%** | **$0** |

Both baselines are given the same chronological ordering the engine uses, so
they are beaten on merit rather than on setup.

**Record order cannot change the answer.** The generator emits each settlement
next to the document it settles, so positional adjacency would leak the answer;
every batch is shuffled before use. A test then re-shuffles each book three ways
and asserts the close is *identical* every time. Getting that test to pass
exposed four real order-dependence bugs — including one where the code claimed
to settle duplicates "against the earliest" while actually taking whichever the
file listed first. Every tie-break in the engine is now ordered on a stable key.

**The audit trail is tamper-evident.** Every decision appends a JSON line whose
hash commits to the previous one. Altering any historical entry breaks the chain
at that index and every index after it; a test forges an entry and asserts the
chain notices.

**Nothing is silently dropped.** A property test asserts that every record in
every batch ends up matched, held, or on the exception ledger. An engine that
quietly ignores what it cannot handle looks more accurate than one that reports
it — that test exists to remove the incentive.

## What it still gets wrong

- **69 unresolved** (0.78%), all one shape: an unequal cohort of identical
  amounts — three indistinguishable bills against two indistinguishable
  payments. No fact separates them, so it refuses rather than guesses.
- **4 partial matches** (0.045%): two short-payments from one vendor, each net of
  a credit note, where the resolver applied the sibling's credit note. The pair
  nets correctly; the individual application does not.
- **The policy gate is unproven on this data.** Removing it raises
  straight-through to 99.2% with no measured precision loss. It is retained as
  tail-risk insurance against errors this synthetic distribution does not
  contain — a judgement call, not a result these numbers support.
- **Synthetic data is not production data.** FX curves, fee schedules and tax
  rates are frozen tables. Real books add partial settlements, reversals,
  chargebacks and multi-currency netting this build does not model.

## The data it has to survive

`generate.py` builds 115 records across 8 adversarial classes, because clean 1:1
data proves nothing: FX bills settled in USD net of an unbooked wire fee, one
remittance covering four invoices, short-pays against open credit notes, Stripe
gross-vs-net settlement, tax-inclusive payment against tax-exclusive documents,
cross-month timing lag with the remittance reference stripped, duplicate vendor
billing, and records that are genuinely unmatchable.

Bank descriptors are dirty in the ways real ones are — `SQ *BLUE BOTTLE 4471`,
`NORDWIND TRADERS`, `CIRRUSCLOUD.IO`, `KSTRL ANALYTICS UK`. Two matching bugs
found and fixed during development came straight from this: `startswith` cannot
see an initialism (`NW` ← `NorthWind`), and token matching cannot split a
concatenated descriptor across two ledger tokens. Both fixes are in
`similarity.py`, each with the reasoning in a docstring.

## Layout

```
ledgerguard/
  models.py       integer-cents datatypes (never floats — 0.1+0.2 != 0.3)
  generate.py     synthetic batch + hidden truth key
  similarity.py   Jaro-Winkler, token-set, abbreviation, de-spaced entity matching
  engine.py       L1 / L2 / cohort / L3 orchestration + risk-weighted policy gate
  agent.py        tool-using resolver; heuristic and optional LLM planners
  tools.py        deterministic tool belt (also exposed as JSON schemas)
  audit.py        hash-chained tamper-evident log
  forecast.py     cash position, ageing, 30-day band
  evaluate.py     precision / recall / calibration against the truth key
  ablate.py       baselines and per-layer ablations
  report.py       HTML close report, generated from run output
  verify_claims.py  turns every published number into a build gate
tests/            13 property invariants
out/              results.json, aggregate.json, ablation.json, audit.jsonl, dashboard.html
```

## Notes on judgement calls

Money is integer minor units everywhere; float arithmetic is the single largest
source of silent reconciliation drift. Similarity is closed-form and
dependency-free so a controller can re-derive any score by hand during an audit.
The policy gate raises its confidence bar with materiality, because an 0.80 match
on $80 and an 0.80 match on $80,000 carry very different downside. And closed
cohorts of genuinely indistinguishable items are applied FIFO rather than
escalated, because handing a human a choice they have no more information to
make than the engine does is theatre, not control.
