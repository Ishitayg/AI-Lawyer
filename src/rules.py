"""Deterministic legal rules. The LLM never computes these.

Everything here is pure Python with a cited source. The agent may *call* these
functions, but it cannot override their output -- src/validate.py cross-checks
the final answer against whatever this module returned during the turn.

Why this exists: the statute text alone gives the WRONG answer on forum.
Section 34 of the bare Act says the District Commission's limit is one crore
rupees. That figure was superseded by the Consumer Protection (Jurisdiction of
the District Commission, the State Commission and the National Commission)
Rules, 2021, notified 30.12.2021, which cut it to fifty lakh rupees. A pure RAG
system retrieving s.34 would confidently misdirect a user to the wrong forum.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass

LAKH = 100_000
CRORE = 100 * LAKH

PECUNIARY_SOURCE = (
    "Consumer Protection (Jurisdiction of the District Commission, the State "
    "Commission and the National Commission) Rules, 2021, notified 30.12.2021 "
    "(supersedes the figures printed in ss. 34, 47 and 58 of the bare Act)"
)
LIMITATION_SOURCE = "Section 69, Consumer Protection Act, 2019"
TERRITORIAL_SOURCE = "Section 34(2), Consumer Protection Act, 2019"

# Verified against Gorantla Geosynthetics Pvt Ltd v. Akshaya Signature Homes
# (NCDRC, 23.08.2023), which records the reduction to Rs. 50,00,000 by the
# 30.12.2021 notification. Evidence kept at data/raw/evidence_pecuniary_2021_*.html
SLABS = [
    (50 * LAKH, "District Consumer Disputes Redressal Commission",
     "value of goods or services paid as consideration does not exceed Rs. 50,00,000"),
    (2 * CRORE, "State Consumer Disputes Redressal Commission",
     "exceeds Rs. 50,00,000 but does not exceed Rs. 2,00,00,000"),
    (float("inf"), "National Consumer Disputes Redressal Commission",
     "exceeds Rs. 2,00,00,000"),
]

TERRITORIAL_OPTIONS = [
    "where the opposite party ordinarily resides, carries on business, has a branch office, or personally works for gain",
    "where any one of several opposite parties so resides or works, with the Commission's permission",
    "where the cause of action arose, wholly or in part",
    "where YOU (the complainant) reside or personally work for gain",
]


@dataclass
class ForumResult:
    amount_inr: float | None
    forum: str | None
    forum_basis: str | None
    pecuniary_source: str
    territorial_options: list[str]
    territorial_source: str
    note: str


@dataclass
class LimitationResult:
    cause_of_action_date: str | None
    deadline: str | None
    days_remaining: int | None
    status: str  # within | expired | unknown
    source: str
    note: str


_AMOUNT_WORDS = {"lakh": LAKH, "lac": LAKH, "lakhs": LAKH, "crore": CRORE, "crores": CRORE, "cr": CRORE, "k": 1_000}


def parse_amount(text) -> float | None:
    """Accept 50000, '50,000', 'Rs 50000', '18k', '4 lakh', '1.5 crore'."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).lower().replace(",", "").replace("₹", " ")
    s = re.sub(r"\b(rs|inr|rupees)\b", " ", s)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr|k)\b", s)
    if m:
        return float(m.group(1)) * _AMOUNT_WORDS[m.group(2)]
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


# Indian digit grouping (2,50,00,000 = 2.5 crore) is read wrongly by models
# trained mostly on Western grouping. Observed: gpt-oss-120b recorded
# "Rs 2,50,00,000" as "2500000" -- off by 10x -- which silently moved the case
# from the National Commission to the District Commission. Amounts are too
# load-bearing to be transcribed by a language model, so we extract them from
# the user's own words in code and cross-check.
_AMOUNT_IN_TEXT_RE = re.compile(
    r"(?:\u20b9|\bRs\.?\b|\bINR\b|\brupees\b)?\s*"
    r"(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(lakhs?|lacs?|crores?|cr\b|k\b)?",
    re.I)


def extract_amounts(text: str) -> list[float]:
    """Every money figure stated in the user's own text, largest first."""
    if not text:
        return []
    out = []
    for m in _AMOUNT_IN_TEXT_RE.finditer(text):
        num, unit = m.group(1), (m.group(2) or "").lower().rstrip()
        has_marker = bool(re.search(r"\u20b9|\bRs\.?\b|\bINR\b|\brupees\b",
                                    text[max(0, m.start() - 12):m.start()], re.I))
        if not has_marker and not unit and "," not in num:
            continue  # a bare number with no currency marker is not an amount
        val = float(num.replace(",", ""))
        if unit:
            val *= _AMOUNT_WORDS[unit.rstrip(".")] if unit.rstrip(".") in _AMOUNT_WORDS else 1
        if val > 0:
            out.append(val)
    return sorted(set(out), reverse=True)


def _parse_date(text) -> dt.date | None:
    if not text:
        return None
    s = str(text).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y", "%B %Y", "%b %Y", "%Y"):
        try:
            d = dt.datetime.strptime(s, fmt).date()
            return d.replace(day=1) if fmt in ("%B %Y", "%b %Y") else (d.replace(month=1, day=1) if fmt == "%Y" else d)
        except ValueError:
            continue
    return None


def check_forum(amount_inr=None) -> dict:
    amt = parse_amount(amount_inr)
    if amt is None:
        return asdict(ForumResult(
            None, None, None, PECUNIARY_SOURCE, TERRITORIAL_OPTIONS, TERRITORIAL_SOURCE,
            "Claim value unknown, so the forum cannot be fixed. Ask the user what they paid "
            "and what compensation they want -- the two are added for this test.",
        ))
    for ceiling, forum, basis in SLABS:
        if amt <= ceiling:
            return asdict(ForumResult(
                amt, forum, basis, PECUNIARY_SOURCE, TERRITORIAL_OPTIONS, TERRITORIAL_SOURCE,
                "Pecuniary jurisdiction under the 2019 Act is tested on the consideration paid, "
                "not on the compensation claimed.",
            ))
    raise AssertionError("unreachable")


def check_limitation(cause_of_action_date=None, today=None) -> dict:
    today = today or dt.date.today()
    d = _parse_date(cause_of_action_date)
    if d is None:
        return asdict(LimitationResult(
            None, None, None, "unknown", LIMITATION_SOURCE,
            "Date the problem arose is unknown. The two-year clock runs from the date the "
            "cause of action arose, so this must be established before advising on filing.",
        ))
    try:
        deadline = d.replace(year=d.year + 2)
    except ValueError:  # 29 Feb
        deadline = d.replace(year=d.year + 2, day=28)
    remaining = (deadline - today).days
    if remaining >= 0:
        note = f"Within the two-year limit. {remaining} day(s) left to file."
        status = "within"
    else:
        note = (f"The two-year limit expired {abs(remaining)} day(s) ago, on {deadline.isoformat()}. "
                "A complaint can still be entertained only if the Commission condones the delay "
                "under the proviso to s.69(2), which requires showing sufficient cause in a "
                "recorded reasoned order. Treat this as a real obstacle, not a formality.")
        status = "expired"
    return asdict(LimitationResult(d.isoformat(), deadline.isoformat(), remaining, status, LIMITATION_SOURCE, note))


def check_forum_and_limitation(amount_inr=None, cause_of_action_date=None, today=None) -> dict:
    return {
        "forum": check_forum(amount_inr),
        "limitation": check_limitation(cause_of_action_date, today),
        "filing_fee": {
            "amount": None,
            "note": "NOT ENCODED. Fees are prescribed by the Consumer Protection (Consumer "
                    "Disputes Redressal Commissions) Rules, 2020 and were not verified against "
                    "source for this prototype. Do not state a fee figure to the user.",
        },
    }
