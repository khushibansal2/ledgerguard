"""Synthetic finance batch with *adversarial* noise and a hidden ground-truth key.

The point of this module is that it is honest about difficulty. Clean 1:1 data
proves nothing, so every generated batch contains the failure modes that make
real month-end close take days: FX + wire fees, split remittances, short-pays
against credit notes, vendor aliasing, cross-month timing, gross-vs-net
processor settlement, tax-inclusive pricing, duplicate billing, and records
that are genuinely unmatchable.

The engine never imports the truth key. Only evaluate.py does.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from .models import Record, money

# --- a tiny frozen FX table, also served to the agent as a "rate API" ----------
FX: dict[tuple[str, str], dict[str, float]] = {
    ("EUR", "USD"): {"2026-03": 1.0842, "2026-04": 1.0901},
    ("GBP", "USD"): {"2026-03": 1.2673, "2026-04": 1.2588},
    ("INR", "USD"): {"2026-03": 0.01201, "2026-04": 0.01195},
}

# (canonical ERP name, dirty bank/processor descriptors)
VENDORS = [
    ("Blue Bottle Coffee Inc",  ["SQ *BLUE BOTTLE 4471", "BLUEBOTTLE COFFEE", "SQ*BLUE BOTTLE COFF"]),
    ("Acme Logistics LLC",      ["ACME LOGISTICS L.L.C.", "ACME LOGISTIC", "ACMELOGISTICS ACH"]),
    ("Northwind Traders GmbH",  ["NORDWIND TRADERS", "NORTHWIND TRADERS GMBH", "NW TRADERS EU"]),
    ("Cirrus Cloud Services",   ["CIRRUS CLOUD SVCS", "CIRRUSCLOUD.IO", "CIRRUS CLD SERVICES"]),
    ("Halberd Legal LLP",       ["HALBERD LEGAL", "HALBERD L.L.P.", "HALBERD LEGAL LLP"]),
    ("Meridian Print Co",       ["MERIDIAN PRINTING", "MERIDIAN PRINT CO.", "MERIDN PRINT"]),
    ("Kestrel Analytics Ltd",   ["KESTREL ANALYTIC", "KESTREL ANALYTICS LTD", "KSTRL ANALYTICS UK"]),
    ("Onyx Facilities Pvt Ltd", ["ONYX FACILITIES", "ONYX FACILITIES PVT", "ONYX FAC IN"]),
    ("Sable Media Group",       ["SABLE MEDIA GRP", "SABLE MEDIA GROUP", "SABLEMEDIA"]),
    ("Vantage Insurance Corp",  ["VANTAGE INS CORP", "VANTAGE INSURANCE", "VANTAGE INS"]),
]

CUSTOMERS = [
    "Redwood Health Systems", "Pallas Robotics", "Juniper Foods Ltd",
    "Atlas Freight Co", "Corvid Software Inc", "Lumen Retail Group",
]

ALIAS_OF = {v[0]: v[1] for v in VENDORS}

# --- synthetic vendor universes, for scale testing --------------------------
_HEADS = ["Cald", "Mor", "Vant", "Bris", "Tarn", "Quill", "Yarr", "Wren", "Fyfe",
          "Ober", "Lark", "Pell", "Ashd", "Ryde", "Sten", "Gorse", "Ives",
          "Nyle", "Umber", "Zeph", "Hald", "Jarn", "Kirk", "Dray", "Ember",
          "Foss"]
_TAILS = ["mont", "wick", "ford", "stead", "hollow", "gate", "bourne", "field",
          "ridge", "combe"]
_SUFFIX = ["Logistics", "Analytics", "Holdings", "Supply", "Systems",
           "Partners", "Foods", "Metals", "Freight", "Studios"]
_FORM = ["Inc", "LLC", "Ltd", "GmbH", "LLP", "Pvt Ltd"]


def vendor_universe(n: int, seed: int = 7) -> list[tuple[str, list[str]]]:
    """Build `n` distinct vendors with the same dirty-descriptor problems.

    Scale tests need counterparty cardinality to grow with the book - a company
    with 50,000 monthly settlements does not have ten suppliers. Reusing one
    small pool measures density, not size, and the two have very different
    answers.

    Each generated vendor gets the same three descriptor pathologies the fixed
    pool has (initialism, concatenation, truncation), so a larger universe is
    not an easier one.
    """
    rng = random.Random(seed)
    out: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    while len(out) < n:
        head = _HEADS[len(out) % len(_HEADS)]
        stem = head + rng.choice(_TAILS)
        canon = f"{stem} {rng.choice(_SUFFIX)} {rng.choice(_FORM)}"
        if canon in seen:
            continue
        seen.add(canon)
        word = canon.split()[1]
        aliases = [
            f"{stem.upper()} {word[:6].upper()}",                 # truncated
            f"{stem.upper()}{word.upper()}",                      # concatenated
            f"{stem[0].upper()}{stem[-2:].upper()} {word.upper()}",  # initialism
        ]
        out.append((canon, aliases))
    return out


class Batch:
    """Generated records plus the truth key (settlement ids -> ledger ids)."""

    def __init__(self) -> None:
        self.records: list[Record] = []
        self.truth: list[dict] = []
        self.unmatchable: list[dict] = []

    def add(self, r: Record) -> Record:
        self.records.append(r)
        return r

    def link(self, settlement: list[str], ledger: list[str], case: str) -> None:
        self.truth.append({"settlement": settlement, "ledger": ledger, "case": case})

    def orphan(self, ids: list[str], case: str, why: str) -> None:
        self.unmatchable.append({"ids": ids, "case": case, "why": why})


def _fx(cur: str, month: str) -> float:
    return FX[(cur, "USD")][month]


def generate(seed: int = 20260827,
             vendors: list[tuple[str, list[str]]] | None = None) -> Batch:
    rng = random.Random(seed)
    b = Batch()
    # A caller may supply its own counterparty universe (see scale.py); the
    # fixed pool stays the default so every published figure is comparable.
    pool = vendors or VENDORS
    alias_of = {v[0]: v[1] for v in pool}
    base = date(2026, 3, 2)
    ctr = {"inv": 1000, "bank": 5000, "stp": 8000, "bill": 3000}

    def nid(k: str) -> str:
        ctr[k] += 1
        return f"{k.upper()}-{ctr[k]}"

    def iso(d: date) -> str:
        return d.isoformat()

    def ref(p: str) -> str:
        return f"{p}-{rng.randint(10000, 99999)}"

    # ---- CASE 1: clean 1:1 AP settlements (the easy majority) ---------------
    for _ in range(16):
        canon, aliases = rng.choice(pool)
        amt = money(rng.choice([1250, 340.50, 8900, 2115.75, 640, 12400.25, 775.10, 3300]))
        d0 = base + timedelta(days=rng.randint(0, 20))
        bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -amt, "USD",
                            f"Bill from {canon}", canon, ref("INV"),
                            iso(d0 + timedelta(days=30)), "BILL"))
        bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT",
                          iso(d0 + timedelta(days=rng.randint(0, 2))), -amt, "USD",
                          f"ACH DEBIT {rng.choice(aliases)} {bill.reference}",
                          rng.choice(aliases), bill.reference, None, "TRANSFER"))
        b.link([bk.id], [bill.id], "clean_1to1")

    # ---- CASE 2: cross-month timing lag, remittance reference stripped ------
    for _ in range(6):
        canon, aliases = rng.choice(pool)
        amt = money(rng.choice([4820, 1990.40, 7350, 560.25]))
        d0 = date(2026, 3, 27) + timedelta(days=rng.randint(0, 3))
        bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -amt, "USD",
                            f"Bill from {canon}", canon, ref("INV"),
                            iso(d0 + timedelta(days=15)), "BILL"))
        bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT",
                          iso(d0 + timedelta(days=rng.randint(4, 6))), -amt, "USD",
                          f"ACH DEBIT {rng.choice(aliases)}",
                          rng.choice(aliases), "", None, "TRANSFER"))
        b.link([bk.id], [bill.id], "timing_lag_cross_month")

    # ---- CASE 3: FX bill settled in USD, plus an unbooked wire fee ----------
    _fx_names = [pool[i % len(pool)][0] for i in (2, 6, 2, 7, 6)]
    fx_cases = list(zip(["EUR", "GBP", "EUR", "INR", "GBP"], _fx_names))
    for cur, canon in fx_cases:
        aliases = alias_of.get(canon) or pool[0][1]
        foreign = money(88000 if cur == "INR" else rng.choice([1000, 2500, 4200, 1800]))
        d0 = base + timedelta(days=rng.randint(2, 25))
        rate = _fx(cur, f"{d0.year}-{d0.month:02d}")
        fee = money(rng.choice([15, 22.50, 30]))
        usd = int(round(foreign * rate)) + fee
        bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -foreign, cur,
                            f"Bill from {canon}", canon, ref("INV"),
                            iso(d0 + timedelta(days=30)), "BILL"))
        bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT",
                          iso(d0 + timedelta(days=rng.randint(1, 3))), -usd, "USD",
                          f"INTL WIRE {rng.choice(aliases)} FX {cur}/USD",
                          rng.choice(aliases), "", None, "TRANSFER"))
        b.link([bk.id], [bill.id], "fx_plus_wire_fee")

    # ---- CASE 4: one remittance settling N bills (split payment) ------------
    for group in range(4):
        canon, aliases = pool[group % len(pool)]
        ids: list[str] = []
        total = 0
        d0 = base + timedelta(days=rng.randint(3, 18))
        for _ in range(rng.randint(2, 4)):
            amt = money(rng.choice([1500, 2750.50, 990, 4300.25, 615]))
            total += amt
            bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER",
                                iso(d0 - timedelta(days=rng.randint(1, 9))), -amt, "USD",
                                f"Bill from {canon}", canon, ref("INV"),
                                iso(d0 + timedelta(days=20)), "BILL"))
            ids.append(bill.id)
        bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT", iso(d0), -total, "USD",
                          f"ACH BATCH PMT {rng.choice(aliases)} {len(ids)} INV",
                          rng.choice(aliases), "", None, "TRANSFER"))
        b.link([bk.id], ids, "split_remittance")

    # ---- CASE 5: Stripe gross charge vs net deposit (processor fee) ---------
    for _ in range(5):
        cust = rng.choice(CUSTOMERS)
        gross = money(rng.choice([2400, 899.99, 15000, 3250.40, 640]))
        fee = int(round(gross * 0.029)) + 30  # 2.9% + 30c
        d0 = base + timedelta(days=rng.randint(1, 24))
        inv = b.add(Record(nid("inv"), "ERP_AR", "LEDGER", iso(d0), gross, "USD",
                           f"Invoice to {cust}", cust, ref("AR"),
                           iso(d0 + timedelta(days=14)), "INVOICE"))
        stp = b.add(Record(nid("stp"), "STRIPE", "SETTLEMENT", iso(d0 + timedelta(days=2)),
                           gross - fee, "USD",
                           f"STRIPE PAYOUT net of fees ch_{rng.randint(10 ** 9, 10 ** 10)}",
                           cust, "", None, "CHARGE"))
        b.link([stp.id], [inv.id], "stripe_gross_vs_net")

    # ---- CASE 6: short-pay against an open credit note ----------------------
    for _ in range(3):
        canon, aliases = rng.choice(pool)
        amt = money(rng.choice([6200, 3400, 9100]))
        credit = money(rng.choice([200, 450.50, 175]))
        d0 = base + timedelta(days=rng.randint(4, 20))
        bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -amt, "USD",
                            f"Bill from {canon}", canon, ref("INV"),
                            iso(d0 + timedelta(days=30)), "BILL"))
        cn = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0 - timedelta(days=6)),
                          credit, "USD", f"Credit note from {canon}", canon,
                          f"CN-{rng.randint(1000, 9999)}", None, "CREDIT_NOTE"))
        bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT", iso(d0 + timedelta(days=2)),
                          -(amt - credit), "USD",
                          f"ACH DEBIT {rng.choice(aliases)} NET OF CN",
                          rng.choice(aliases), "", None, "TRANSFER"))
        b.link([bk.id], [bill.id, cn.id], "short_pay_credit_note")

    # ---- CASE 7: tax-inclusive settlement vs tax-exclusive ledger line ------
    for rate_pct, juris in [(18, "IN_GST"), (20, "UK_VAT"), (18, "IN_GST"), (19, "DE_VAT")]:
        canon, aliases = rng.choice(pool)
        net = money(rng.choice([1200, 5400, 860, 2300]))
        tax = int(round(net * rate_pct / 100))
        d0 = base + timedelta(days=rng.randint(2, 22))
        bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -net, "USD",
                            f"Bill from {canon} (net of tax)", canon, ref("INV"),
                            iso(d0 + timedelta(days=30)), "BILL",
                            {"tax_treatment": "EXCLUSIVE", "jurisdiction": juris}))
        bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT", iso(d0 + timedelta(days=1)),
                          -(net + tax), "USD", f"ACH DEBIT {rng.choice(aliases)} INCL TAX",
                          rng.choice(aliases), "", None, "TRANSFER"))
        b.link([bk.id], [bill.id], "tax_inclusive_vs_exclusive")

    # ---- CASE 8: genuinely unmatchable / must be escalated ------------------
    # 8a duplicate vendor billing, only one settlement exists
    canon, aliases = pool[5 % len(pool)]
    amt = money(4500)
    d0 = base + timedelta(days=9)
    dupref = ref("INV")
    dup1 = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -amt, "USD",
                        f"Bill from {canon}", canon, dupref,
                        iso(d0 + timedelta(days=30)), "BILL"))
    dup2 = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -amt, "USD",
                        f"Bill from {canon}", canon, dupref,
                        iso(d0 + timedelta(days=30)), "BILL"))
    bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT", iso(d0 + timedelta(days=1)),
                      -amt, "USD", f"ACH DEBIT {aliases[0]} {dupref}",
                      aliases[0], dupref, None, "TRANSFER"))
    b.link([bk.id], [dup1.id], "duplicate_billing_one_paid")
    b.orphan([dup2.id], "DUPLICATE_BILLING",
             "Identical vendor/reference/amount raised twice; only one settlement exists.")

    # 8b bank debits with no ledger document at all
    for desc, amt in [("CARD 4471 UNKNOWN MERCHANT SVC", money(318.44)),
                      ("WIRE OUT REF UNAVAILABLE", money(2750))]:
        bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT",
                          iso(base + timedelta(days=rng.randint(5, 25))), -amt, "USD",
                          desc, "", "", None, "TRANSFER"))
        b.orphan([bk.id], "UNIDENTIFIED_OUTFLOW",
                 "No ledger document exists for this debit.")

    # 8c open payables never settled in period (legitimately open)
    for _ in range(3):
        canon, _aliases = rng.choice(pool)
        amt = money(rng.choice([7700, 1450.60, 5200]))
        d0 = base + timedelta(days=rng.randint(18, 28))
        bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -amt, "USD",
                            f"Bill from {canon}", canon, ref("INV"),
                            iso(d0 + timedelta(days=45)), "BILL"))
        b.orphan([bill.id], "OPEN_PAYABLE",
                 "Not yet due; no settlement expected within the period.")

    # 8d near-miss amount with no fee/tax/FX basis - possible wrong-amount payment
    canon, aliases = pool[8 % len(pool)]
    d0 = base + timedelta(days=14)
    bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -money(9800), "USD",
                        f"Bill from {canon}", canon, ref("INV"),
                        iso(d0 + timedelta(days=30)), "BILL"))
    bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT", iso(d0 + timedelta(days=1)),
                      -money(9880), "USD", f"ACH DEBIT {aliases[0]}",
                      aliases[0], "", None, "TRANSFER"))
    b.orphan([bill.id, bk.id], "AMOUNT_VARIANCE_UNEXPLAINED",
             "Settlement exceeds bill by $80.00 with no fee, tax or FX basis.")

    # ---- CASE 9: instalments - one bill cleared by several payments ---------
    # The mirror image of a split remittance, and the harder direction: the
    # engine must recognise that no single settlement will ever tie out, and
    # that the *set* of them does.
    for _ in range(3):
        canon, aliases = rng.choice(pool)
        total = money(rng.choice([10000, 24000, 15600]))
        d0 = base + timedelta(days=rng.randint(2, 12))
        bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -total, "USD",
                            f"Bill from {canon}", canon, ref("INV"),
                            iso(d0 + timedelta(days=45)), "BILL"))
        first = int(total * rng.choice([0.6, 0.5, 0.75]))
        parts = [first, total - first]
        pay_ids = []
        for n, part in enumerate(parts):
            bk = b.add(Record(nid("bank"), "BANK", "SETTLEMENT",
                              iso(d0 + timedelta(days=6 + n * 11)), -part, "USD",
                              f"ACH DEBIT {rng.choice(aliases)} INSTALMENT {n + 1}/2",
                              rng.choice(aliases), "", None, "TRANSFER"))
            pay_ids.append(bk.id)
        b.link(pay_ids, [bill.id], "instalment_settlement")

    # ---- CASE 10: failed payment, reversed, then re-sent --------------------
    # Three bank lines for one economic event. Booking the reversal against the
    # bill would understate the expense; missing the pair double-counts it.
    for _ in range(2):
        canon, aliases = rng.choice(pool)
        amt = money(rng.choice([8000, 3450, 11200]))
        d0 = base + timedelta(days=rng.randint(4, 16))
        bill = b.add(Record(nid("bill"), "ERP_AP", "LEDGER", iso(d0), -amt, "USD",
                            f"Bill from {canon}", canon, ref("INV"),
                            iso(d0 + timedelta(days=30)), "BILL"))
        alias = rng.choice(aliases)
        out = b.add(Record(nid("bank"), "BANK", "SETTLEMENT",
                           iso(d0 + timedelta(days=1)), -amt, "USD",
                           f"ACH DEBIT {alias}", alias, "", None, "TRANSFER"))
        rev = b.add(Record(nid("bank"), "BANK", "SETTLEMENT",
                           iso(d0 + timedelta(days=3)), amt, "USD",
                           f"ACH RETURN R01 {alias} INSUFFICIENT FUNDS",
                           alias, "", None, "REVERSAL"))
        again = b.add(Record(nid("bank"), "BANK", "SETTLEMENT",
                             iso(d0 + timedelta(days=5)), -amt, "USD",
                             f"ACH DEBIT {alias} REPRESENTED", alias, "", None, "TRANSFER"))
        b.link([out.id, rev.id], [], "payment_reversal_pair")
        b.link([again.id], [bill.id], "represented_payment")

    # ---- CASE 11: customer chargeback claws a settled invoice back ----------
    # Same mechanism as a reversal, opposite side of the book: the pair nets to
    # nothing and the receivable must go *back* to open rather than stay closed.
    for _ in range(2):
        cust = rng.choice(CUSTOMERS)
        amt = money(rng.choice([5000, 12500]))
        d0 = base + timedelta(days=rng.randint(6, 18))
        inv = b.add(Record(nid("inv"), "ERP_AR", "LEDGER", iso(d0), amt, "USD",
                           f"Invoice to {cust}", cust, ref("AR"),
                           iso(d0 + timedelta(days=30)), "INVOICE", {"open": True}))
        chg = b.add(Record(nid("stp"), "STRIPE", "SETTLEMENT",
                           iso(d0 + timedelta(days=2)), amt, "USD",
                           f"STRIPE PAYMENT {cust}", cust, "", None, "CHARGE"))
        cb = b.add(Record(nid("stp"), "STRIPE", "SETTLEMENT",
                          iso(d0 + timedelta(days=9)), -amt, "USD",
                          f"STRIPE CHARGEBACK dispute_{rng.randint(10 ** 6, 10 ** 7)} {cust}",
                          cust, "", None, "REVERSAL"))
        b.link([chg.id, cb.id], [], "chargeback_pair")
        b.orphan([inv.id], "OPEN_RECEIVABLE",
                 "Settled then charged back; the receivable is open again.")

    # ---- open AR feeding the forecast (not part of reconciliation truth) ----
    for _ in range(6):
        cust = rng.choice(CUSTOMERS)
        amt = money(rng.choice([12000, 4800, 26500, 9300, 15750]))
        d0 = base + timedelta(days=rng.randint(10, 28))
        inv = b.add(Record(nid("inv"), "ERP_AR", "LEDGER", iso(d0), amt, "USD",
                           f"Invoice to {cust}", cust, ref("AR"),
                           iso(d0 + timedelta(days=rng.choice([15, 30, 45]))),
                           "INVOICE", {"open": True}))
        b.orphan([inv.id], "OPEN_RECEIVABLE",
                 "Awaiting customer payment; feeds the cash forecast.")

    # Shuffle before returning. Generation emits each settlement directly after
    # the document it settles, so positional adjacency would leak the answer -
    # a matcher that simply took the nearest equal amount would score well for
    # a reason that does not exist in a real bank file. Randomising the order
    # forces every match to be earned from the record contents alone.
    rng.shuffle(b.records)
    return b
