"""String/entity similarity primitives, pure stdlib.

Deliberately no fuzzy-matching dependency: every score here is a closed-form
function a controller can re-derive by hand during an audit. A reconciliation
you cannot explain to an auditor is a reconciliation you cannot book.
"""
from __future__ import annotations

import math
import re

# Corporate-form and payment-rail tokens carry no identifying information and
# actively hurt similarity ("Acme LLC" vs "ACME L.L.C." should score ~1.0).
NOISE = {
    "inc", "llc", "ltd", "limited", "llp", "plc", "gmbh", "bv", "nv", "sa", "ag",
    "pvt", "private", "co", "corp", "corporation", "company", "group", "grp",
    "the", "and", "of", "ach", "debit", "credit", "pmt", "payment", "wire",
    "intl", "sq", "tst", "pos", "card", "transfer", "batch", "inv", "ref", "net",
    "incl", "tax", "fx", "usd", "eur", "gbp", "inr", "payout", "fees", "stripe",
}

_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, drop trailing card/reference digits."""
    s = _ALNUM.sub(" ", s.lower())
    s = re.sub(r"\b\d{3,}\b", " ", s)          # 4471, ch_1234567 remnants
    return " ".join(s.split())


def tokens(s: str) -> list[str]:
    return [t for t in normalize(s).split() if t not in NOISE and len(t) > 1]


def jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(len(a), len(b)) // 2 - 1
    window = max(window, 0)
    a_flag = [False] * len(a)
    b_flag = [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        lo = max(0, i - window)
        hi = min(i + window + 1, len(b))
        for j in range(lo, hi):
            if not b_flag[j] and b[j] == ch:
                a_flag[i] = b_flag[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    # count transpositions
    k = trans = 0
    for i, ch in enumerate(a):
        if a_flag[i]:
            while not b_flag[k]:
                k += 1
            if ch != b[k]:
                trans += 1
            k += 1
    trans //= 2
    m = float(matches)
    return (m / len(a) + m / len(b) + (m - trans) / m) / 3.0


def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    """Jaro with a prefix bonus - vendor names agree at the head far more often
    than at the tail, which is exactly what bank descriptors truncate."""
    j = jaro(a, b)
    if j < 0.7:
        return j
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return j + prefix * p * (1 - j)


def abbreviation_score(short: str, long: str) -> float:
    """Score contractions that bank descriptors produce but prefix matching misses.

    Card networks and ACH rails truncate to fixed field widths by dropping
    vowels and interior letters, yielding initialisms and consonant skeletons:
    NW <- NorthWind, KSTRL <- Kestrel, FAC <- Facilities. None of these is a
    prefix of the source, so `startswith` sees nothing.

    The guard against matching everything is the first-letter anchor: real
    abbreviations preserve the initial character. That single constraint drops
    the obvious false positives ("EU" against "traders") while keeping the
    contraction family, and it keeps the rule explainable to an auditor -
    "same first letter, letters appear in order" is a sentence, not a model.
    """
    if len(short) < 2 or len(long) < 4 or len(short) >= len(long):
        return 0.0
    if short[0] != long[0]:
        return 0.0
    it = iter(long)
    if not all(ch in it for ch in short):          # subsequence test
        return 0.0
    # Longer evidence is stronger evidence: 'nw' is weaker proof than 'kstrl'.
    return 0.86 + 0.10 * min(len(short) / len(long), 1.0)


def token_set_ratio(a: str, b: str) -> float:
    """Order-insensitive overlap, robust to bank descriptors that reorder or
    truncate name parts. Falls back to per-token Jaro-Winkler so near-miss
    spellings (NORDWIND vs NORTHWIND) still score."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    small, large = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    total = 0.0
    used: set[int] = set()
    for t in small:
        best, best_j = 0.0, -1
        for j, u in enumerate(large):
            if j in used:
                continue
            # a truncated token ("analytic" in "analytics") counts as a hit
            if u.startswith(t) or t.startswith(u):
                s = 0.97
            else:
                s = max(jaro_winkler(t, u),
                        abbreviation_score(t, u), abbreviation_score(u, t))
            if s > best:
                best, best_j = s, j
        if best_j >= 0 and best > 0.82:
            used.add(best_j)
            total += best
    return total / len(small)


def despaced(s: str) -> str:
    """Collapse a name to its letters, dropping corporate-form noise words."""
    return "".join(tokens(s)) or normalize(s).replace(" ", "")


def counterparty_score(a: str, b: str) -> float:
    """Take the most optimistic of three views of the same pair.

    They fail on different inputs, and a bank descriptor usually breaks exactly
    one of them:
      - whole-string  loses to reordering
      - token-set     loses to concatenation, because a descriptor that ships
                      'CIRRUSCLOUD.IO' gives one token to align against two
      - despaced      loses to nothing much, but is the weakest evidence, so
                      it only ever raises a score the others already support

    Taking the max is safe here because the downstream gates are what actually
    authorise a booking; this function's job is recall of plausible candidates.
    """
    if not a or not b:
        return 0.0
    return max(jaro_winkler(normalize(a), normalize(b)),
               token_set_ratio(a, b),
               jaro_winkler(despaced(a), despaced(b)))


def reference_score(a: str, b: str) -> float:
    """Exact-ish match on an invoice/remittance reference is near-proof."""
    if not a or not b:
        return 0.0
    na, nb = re.sub(r"[^A-Z0-9]", "", a.upper()), re.sub(r"[^A-Z0-9]", "", b.upper())
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.9
    return 0.0


def date_score(delta_days: int, half_life: float = 5.0) -> float:
    """Exponential decay. Settlement lags payment instruction by rail-dependent
    days (ACH 1-3, wire 0-1, card capture 2-3), so proximity is evidence but a
    hard window would break every cross-month cutoff."""
    if delta_days < 0:
        delta_days = abs(delta_days) * 2      # ledger *after* settlement is odd
    return math.exp(-delta_days / half_life)


def amount_score(a: int, b: int, tol_bps: int = 0) -> float:
    """1.0 on the cent, decaying with relative gap. tol_bps widens the plateau
    for cases where a known fee/FX basis is already accounted for."""
    if a == b:
        return 1.0
    denom = max(abs(a), abs(b), 1)
    gap_bps = abs(a - b) * 10000 // denom
    if gap_bps <= tol_bps:
        return 0.99
    return max(0.0, 1.0 - gap_bps / 1200.0)   # ~0 once 12% apart
