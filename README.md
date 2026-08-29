# LedgerGuard

An autonomous finance controller that decides what can safely be booked, proves
why, and refuses anything it cannot prove.

```
   BANK ─┐
PROCESSOR├──► RECONCILE ──► RESOLVE ──► POLICY GATE ─┬──► BOOKED LEDGER ──► CASH POSITION
    ERP ─┘    (exact +      (tools,     (materiality) │                      today → +30d
              similarity)   arithmetic)               └──► EXCEPTION QUEUE
                                                            with the evidence
                                                            needed to close it
```

**Live demo: https://khushibansal2.github.io/ledgerguard/** — the real engine,
compiled to WebAssembly and running in your browser. No server, no API key,
nothing to keep alive. Close a batch, click any settlement, read its decision
trace.

Zero third-party dependencies. Python 3.11+, standard library only.

### Run it as a service

```bash
python -m ledgerguard.server          # http://localhost:8000
```

| Route | |
|---|---|
| `GET /` | the demo page |
| `GET /health` | liveness |
| `POST /api/close` | `{"seed": 20260827}` → the full close as JSON |
| `GET /api/trace?id=BANK-5024` | one settlement's decision path |

```bash
curl -s localhost:8000/api/close -d '{"seed":20260827}' | head -40
curl -s "localhost:8000/api/trace?id=BANK-5024"
```

`render.yaml` deploys this as a single free service — no database, no
dependencies to install, and the demo page still works while a free instance is
cold because the engine runs client-side.

```bash
python -m ledgerguard.cli                 # close one batch, full report
python -m ledgerguard.trace BANK-5024     # one settlement's full decision path
python -m ledgerguard.cli --seeds 200     # robustness across 200 batches
python -m ledgerguard.ablate              # does each layer earn its place?
python -m ledgerguard.cli --csv work.csv  # exception worklist for the controller
python -m ledgerguard.redteam             # can a hostile planner move the ledger?
python -m ledgerguard.scale               # how does it behave as the book grows?
python tests/test_controller.py           # 16 invariants, no pytest needed
python -m ledgerguard.verify_claims       # re-verify every accuracy claim
python -m ledgerguard.check_readme        # assert this README matches the run
```

Every figure below is asserted in CI: `verify_claims` re-measures the accuracy
claims, and `check_readme` asserts the numbers written here still match the
committed run output. A regression — or a stale figure left in this file —
fails the build instead of quietly shipping.

## Measured, not asserted

Over **200 independent batches — 27,218 records, 12,600 settlement events**,
graded against a truth key the engine never sees:

| Metric | Result | |
|---|---|---|
| Precision | **99.92%** | 2 wrong bookings in 12,600 settlement events |
| **Dollar-weighted precision** | **99.90%** | **$12,000 mis-booked out of $63.2M settled** |
| Straight-through | **89.6%** | cleared with no human involvement |
| Resolved correctly | **99.0%** | including items held for sign-off |
| Escalation recall | **100%** | every truly-unmatchable record refused |
| Throughput | ~1,212 rec/s | ~112 ms per batch |
| Model usage | 1.48 tool calls/event | 56% cleared before the resolver is reached |

Row counts hide what a controller is accountable for: ten thousand correct $10
matches and one wrong $500,000 match is 99.99% precision and a catastrophe. So
exposure is reported alongside frequency, and **$12,000 of $63.2M** is the
number that matters.

## The design in one idea

Four layers, each costing more and trusted less than the one above it. In the
default configuration **no model is called at all** — the built-in planner runs
the same hypothesis set. Enabling the LLM planner sends it only the settlements
that fail to tie out.

| | Layer | Test | Model |
|---|---|---|---|
| **L0** | Reversal | equal-and-opposite pairs that settle nothing | never |
| **L1** | Deterministic | exact reference + exact amount | never |
| **L2** | Similarity | amount ties out; entity and timing inferred | never |
| **L3** | Agentic | amount does *not* tie out — prove the difference | may order hypotheses |

### Running the model path (free, no paid account)

The planner speaks the OpenAI chat-completions dialect over `urllib`, so any
provider works and none is pinned in source. Three environment variables:

```bash
# Groq - free tier, no card required (console.groq.com/keys)
export LEDGERGUARD_API_BASE=https://api.groq.com/openai/v1
export LEDGERGUARD_MODEL=qwen/qwen3.8-27b
export LEDGERGUARD_API_KEY=gsk_...

# or Google AI Studio - free tier (aistudio.google.com/apikey)
export LEDGERGUARD_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai
export LEDGERGUARD_MODEL=gemini-2.0-flash

# or Ollama - entirely local, no key, no account, works offline
export LEDGERGUARD_API_BASE=http://localhost:11434/v1
export LEDGERGUARD_MODEL=llama3.1

python -m ledgerguard.cli --llm
```

Measured on one batch, deterministic planner versus a live Groq call:

| | Booked | Exceptions | Precision | Mis-booked | Tool calls |
|---|---|---|---|---|---|
| built-in planner | 48 | 21 | 100% | $0 | 106 |
| LLM planner (25 live calls, 6 fell back) | 48 | 21 | 100% | $0 | **98** |

The ledger is identical. The model earned an 8% reduction in tool calls through
better hypothesis ordering, and six of its twenty-five calls failed outright and
fell through to the built-in ordering without anyone noticing. That is the whole
claim: the model changes cost, not correctness.

Model ids move; check the provider's current list if one is rejected. Note that
reasoning-style models can spend their entire token budget thinking and return
an empty string, and that some providers sit behind a WAF that rejects urllib's
default User-Agent with a 403 that looks exactly like a bad key. Nothing
above is required — with no configuration the built-in planner runs the same
hypothesis set, which is why the hosted demo needs no key at all.

**A match at L3 must be arithmetically closed.** The resolver may only book a
difference it can name and compute — *this* FX rate, *that* fee schedule, *this*
statutory tax rate. "These look related" is not an explanation. The LLM chooses
which explanation to test; a deterministic tool computes the answer; a policy
gate decides whether it may be booked.

**And it refuses when two explanations both work.** A vendor running a
short-payment and an instalment plan at once produces two allocations that each
balance to the cent. The arithmetic cannot choose between them, so the engine
does not either — it raises `AMBIGUOUS_ALLOCATION` and names the remittance
advice that would settle it.

### Where the agent actually is

`python -m ledgerguard.trace BANK-5024` reconstructs one decision from the
hash-chained log — nothing is re-run or re-derived:

```
  LAYER PATH
  L0  reversal pairing                   no match
  L1  exact reference + amount           no match
  L2  entity + timing similarity         no match
  L3  tool-using resolver                MATCHED

  HYPOTHESES  (ordered by HeuristicPlanner)
    H1  fx           the document is in another currency
    H2  split        one payment covers several documents

  TOOL EXECUTION
    fx           CLOSED
      fx_convert   -> $3,168.25 at 1.2673 [fx_curve[GBP/USD][2026-03]]
      fee_explains -> $15.00 matches WIRE_OUT_DOMESTIC

  ARITHMETIC   2,500.00 GBP x 1.2673 = $3,168.25, + $15.00 wire fee = $3,183.25
  residual $0.00 (tolerance $0.02)

  POLICY GATE  band reviewable ($3,183.25) - requires 0.82 - confidence 0.92
  DECISION: BOOK
```

### Authority rises with the money

A single confidence threshold is either too loose at the top of the book or too
strict at the bottom. The gate is a delegated-authority ladder, calibrated
against the confidence the evidence layers actually produce:

| Band | Value | Confidence required |
|---|---|---|
| routine | under $1,000 | 0.75 |
| reviewable | $1,000+ | 0.82 |
| material | $10,000+ | 0.90 |
| significant | $100,000+ | 0.96 |

## Why it should be believed

**Every layer is ablated.** Same batches, same grading, one capability removed
at a time (`python -m ledgerguard.ablate`):

| Configuration | Precision | Straight-through | Mis-booked value |
|---|---|---|---|
| naive: equal amount | 77.4% | 39.4% | $1,608,121 |
| naive: amount + 3-day window | 72.7% | 30.2% | $1,508,155 |
| no reversal pass | 93.7% | 79.9% | $712,900 |
| no L3 resolver | 100% | 50.8% | $0 |
| no cohort pass | 99.9% | 89.2% | $6,000 |
| no policy gate | 99.9% | 99.1% | $6,000 |
| **LedgerGuard (full)** | **99.9%** | **89.5%** | **$6,000** |

Two readings worth pausing on. Dropping the reversal pass costs **$712,900** —
a returned payment looks exactly like a real one, so without it the later layers
book money coming *back* against an open invoice. And dropping L3 costs no
precision at all but halves throughput: the resolver buys coverage, not
accuracy, which is exactly the trade it should be making.

**A hostile planner cannot move the ledger.** The whole design rests on the
claim that the model only orders hypotheses while deterministic tools compute
and a policy gate decides. That is a safety claim, so it is tested rather than
argued: `redteam.py` swaps in six hostile planners — one that inverts the
ordering, one that front-runs the riskiest allocations, one returning injection
strings and invalid names, one flooding the resolver, one returning nothing, and
one that raises on every call — and asserts the invariants hold.

| Planner | Booked | Mis-booked | Tool calls |
|---|---|---|---|
| heuristic (control) | 953 | $0 | 1,805 |
| reversed order | 953 | $0 | 2,216 |
| allocation-first | 953 | $0 | 2,192 |
| garbage names | 773 | $0 | 1,132 |
| flooding | 953 | $0 | 49,151 |
| raises every call | 953 | $0 | 1,805 |

**The worst a hostile plan achieves is spending 27× the tool calls to reach the
same ledger.** Building this found two genuine defects: a planner that raised
killed the entire close, and a planner that omitted one hypothesis *silently
disabled the competing-allocation check* — the safety property was
planner-dependent, and mis-booked $21,300 until it was made mandatory. The
red-team invariants also caught a gate bypass, where a multi-line match was
authorised on its first payment rather than the whole event's exposure.

**Record order cannot change the answer.** The generator emits each settlement
next to the document it settles, so adjacency would leak the answer; every batch
is shuffled. A test then re-shuffles each book three ways and asserts the close
is *identical*. Getting it to pass exposed five real order-dependence bugs —
including code that claimed to settle duplicates "against the earliest" while
actually taking whichever the file listed first.

**The audit trail is tamper-evident.** Every decision appends a JSON line whose
hash commits to the previous one. A test forges an entry and asserts the chain
notices.

**Nothing is silently dropped.** A property test asserts every record in every
batch ends up matched, held, or on the exception ledger. An engine that quietly
ignores what it cannot handle looks more accurate than one that reports it —
that test removes the incentive.

## What happens as the book grows

`python -m ledgerguard.scale` composes many batches into one book and measures
where the curve bends. Accuracy holds; speed does not, and the README says so.

With counterparty cardinality growing alongside the book — what actually happens
as a company scales:

| Records | Events | Seconds | Precision | Mis-booked |
|---|---|---|---|---|
| 138 | 63 | 0.12 | 100.00% | $0 |
| 545 | 252 | 1.05 | 100.00% | $0 |
| 1,090 | 504 | 4.09 | 100.00% | $0 |
| 2,178 | 1,008 | 19.93 | 97.98% | $62,530 |

Growth is **~n^1.9** — still quadratic. Adding a blocking index cut the constant
by 4–9× but did not bend the curve, because L2 compares every open settlement
against every open document in its block. **Extrapolated to 100,000 records that
is roughly 6.7 hours, so this build is not production-scale as written.** The fix
is stronger blocking on a vendor-master id, which a real ERP already has and this
synthetic book does not.

Running the same test with a *fixed* ten-vendor pool — so density rather than
size grows — degrades precision to 95% at 2,178 records. That is the honest
limit of the discrimination: when hundreds of same-vendor, same-amount documents
compete for one payment, the evidence genuinely runs out.

Both numbers were hard to measure correctly. Two earlier versions of this test
were broken fixtures rather than results: namespacing record ids gave every
record in the book the same initial and collapsed every block into one, and
prefixing vendor names with a shared word made *different* vendors inside a
batch more similar to each other. Both looked like algorithmic limits.

## The exception queue answers three questions

Not just *what happened*, but *why the engine could not resolve it* and *what
evidence would*:

```
[HIGH] AMOUNT_VARIANCE_UNEXPLAINED   BANK-5042, BILL-3057   $9,880.00
  Closest fit is BILL-3057 ($9,800.00), a variance of $80.00 (0.82%).
  Tested and rejected: tax, split. No FX rate, fee schedule, statutory tax
  rate or document subset explains the difference.
  -> Do not book. Confirm against the vendor statement.
  Missing: a remittance advice, or a vendor statement showing what the extra
  $80.00 was for - a credit, a rebilled cost or an error.
```

`--csv` exports the queue severity- then value-ordered, so the largest exposure
is the first thing a controller touches.

## Closing the books changes the cash position

The dependency direction is the point. An unreconciled book counts settled
invoices twice — once as cash in the bank, again as a receivable still expected
— so a forecast built on it overstates next month's inflow by exactly the value
of everything already paid. The controller reports the delta it removed:

```
  59 documents cleared against settlements.
  Forecasting on the unreconciled ledger would have expected $146,590.80
  of receivables. After the close the true figure is $122,050.00 - an
  overstatement of $24,540.80, from counting invoices already paid.
```

The 30-day projection then carries an expected case and a worst case, because
the treasury question is never "what do I expect" but "what is the worst week I
must fund".

## What it still gets wrong

- **2 wrong bookings** in 12,600 events ($12,000 of $63.2M). Both are
  allocation choices between two arithmetically valid readings.
- **92 unresolved** (0.73%): 69 unequal cohorts of identical amounts, 18
  instalment groups, 5 short-pays. All refused rather than guessed, all on the
  exception ledger.
- **6 partial matches** (0.05%): two short-payments from one vendor where the
  resolver applied the sibling's credit note. Both groupings balance.
- **The policy gate is unproven on this data.** Removing it raises
  straight-through from 89.5% to 99.1% with no measured precision loss. It is
  retained as tail-risk insurance against errors this synthetic distribution
  does not contain — a judgement call, not a result these numbers support.
- **The `significant` rung (>$100,000) is untested.** No settlement in this
  data reaches it.
- **It is not production-scale.** ~n^1.9 growth; ~6.7 hours extrapolated to
  100,000 records. Accuracy holds, throughput does not.
- **Synthetic data is not production data.** FX curves, fee schedules and tax
  rates are frozen tables. Real books add partial-period cutoffs, multi-currency
  netting and intercompany elimination this build does not model.

## The data it has to survive

`generate.py` builds ~136 records across 11 adversarial classes, because clean
1:1 data proves nothing:

FX bills settled in USD net of an unbooked wire fee · one remittance covering
four invoices · **several payments clearing one bill** · short-pays against open
credit notes · Stripe gross-vs-net settlement · tax-inclusive payment against
tax-exclusive documents · cross-month timing lag with the reference stripped ·
duplicate vendor billing · **failed payments returned and re-presented** ·
**customer chargebacks that reopen a settled invoice** · records that are
genuinely unmatchable.

Bank descriptors are dirty in the ways real ones are — `SQ *BLUE BOTTLE 4471`,
`NORDWIND TRADERS`, `CIRRUSCLOUD.IO`, `KSTRL ANALYTICS UK`. Two matching bugs
found during development came straight from this: `startswith` cannot see an
initialism (`NW` ← `NorthWind`), and token matching cannot split a concatenated
descriptor. Both fixes are in `similarity.py` with the reasoning in a docstring.

## Layout

```
ledgerguard/
  models.py       integer-cents datatypes (never floats — 0.1+0.2 != 0.3)
  generate.py     synthetic batch + hidden truth key
  similarity.py   Jaro-Winkler, token-set, abbreviation, de-spaced matching
  engine.py       L0/L1/L2/cohort/L3 orchestration + materiality ladder
  agent.py        tool-using resolver; heuristic and optional LLM planners
  tools.py        deterministic tool belt (also exposed as JSON schemas)
  audit.py        hash-chained tamper-evident log
  forecast.py     cash position, ageing, 30-day band, reconciliation delta
  evaluate.py     precision / recall / dollar-weighting / calibration
  ablate.py       baselines and per-layer ablations
  trace.py        replay one decision from the audit trail
  redteam.py      hostile planners vs. the containment claim
  scale.py        growth curve and the density limit
  build_web.py    inlines the library into a browser demo (Pyodide)
  server.py       stdlib HTTP service: demo page + JSON API
  report.py       HTML close report, generated from run output
  verify_claims.py  turns every accuracy claim into a build gate
  check_readme.py   asserts this README matches the committed run
tests/            16 property invariants
web/              generated single-page demo, deployed to GitHub Pages
out/              results.json, aggregate.json, ablation.json, dashboard.html
```

## Notes on judgement calls

Money is integer minor units everywhere; float arithmetic is the single largest
source of silent reconciliation drift. Similarity is closed-form and
dependency-free so a controller can re-derive any score by hand during an audit.
Closed cohorts of genuinely indistinguishable items are applied FIFO rather than
escalated, because handing a human a choice they have no more information to
make than the engine does is theatre, not control. And a reversal is never
allowed to settle a document in any layer — money coming back is not a payment.
