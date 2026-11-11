"""Deterministic case analysis. No LLM, no retrieval.

These are the reasoning steps a competent assistant performs on the facts
themselves: order events, price the claim, work out what proof is missing,
weigh what helps and what hurts, and sequence the next actions against real
deadlines. Doing them in code means the numbers, dates and gaps are correct by
construction and the model's job is to explain them, not to compute them.
"""
from __future__ import annotations

import datetime as dt
import re

from . import rules

# What a consumer complaint realistically needs, and the words that suggest
# the user already has it. Deliberately conservative: a near-miss is reported
# as "unclear" rather than silently counted as held.
EVIDENCE_ITEMS = [
    ("proof of payment", r"invoice|bill|receipt|payment|upi|bank statement|card statement|order"),
    ("proof of the defect", r"photo|video|image|service report|job sheet|technician|inspection"),
    ("warranty or terms", r"warranty|guarantee|terms|policy|contract|agreement"),
    ("record of complaints made", r"email|e-mail|whatsapp|chat|call|complaint|ticket|letter|notice"),
    ("the seller's response", r"reply|response|refus|denied|rejection|said no"),
    ("proof of loss suffered", r"expense|cost|loss|repair bill|replacement|out of pocket"),
]


# The plan the agent is working through. Derived from state rather than asked
# of the model, so it costs no tokens, cannot be hallucinated, and the user can
# see exactly where they are.
PLAN_STAGES = [
    ("facts", "Understand what happened"),
    ("law", "Find the law that applies"),
    ("forum", "Work out where and by when to file"),
    ("assess", "Weigh the case honestly"),
    ("act", "Produce the document or next step"),
]


def progress(cf, ctx=None) -> list:
    tools = {t["tool"] for t in (ctx.tool_log if ctx else [])}
    done = {
        "facts": not cf.missing_critical(),
        "law": bool(ctx and ctx.retrieved_sections),
        "forum": bool(ctx and ctx.rule_results),
        "assess": bool(tools & {"assess_strength", "evidence_checklist", "build_timeline"}),
        "act": bool(tools & {"draft_document", "plan_next_steps", "opposing_counsel"}),
    }
    return [{"key": k, "label": label, "done": done.get(k, False)} for k, label in PLAN_STAGES]


def build_timeline(cf) -> dict:
    """Chronology from the recorded facts, with gaps named."""
    events, gaps = [], []
    pairs = [
        (cf.date_of_purchase, f"Bought/availed: {cf.goods_or_service or 'the goods or service'}"
                              + (f" for Rs {cf.amount_paid}" if cf.amount_paid else "")),
        (cf.date_problem_arose, f"Problem arose: {cf.what_went_wrong or 'the defect or deficiency'}"),
    ]
    for raw, label in pairs:
        d = rules._parse_date(raw)
        if d:
            events.append({"date": d.isoformat(), "event": label})
        elif raw:
            gaps.append(f"{label} — date recorded as {raw!r}, which could not be parsed")
        else:
            gaps.append(f"{label} — no date recorded")

    if cf.steps_taken:
        events.append({"date": "", "event": f"Steps already taken: {cf.steps_taken}"})

    lim = rules.check_limitation(cf.date_problem_arose)
    if lim["deadline"]:
        events.append({"date": lim["deadline"],
                       "event": "LAST DATE to file a consumer complaint (s.69, two years)"})

    dated = sorted([e for e in events if e["date"]], key=lambda e: e["date"])
    undated = [e for e in events if not e["date"]]

    # Order-of-events sanity: a problem cannot precede the purchase.
    p, q = rules._parse_date(cf.date_of_purchase), rules._parse_date(cf.date_problem_arose)
    if p and q and q < p:
        gaps.append(f"The problem date ({q.isoformat()}) is BEFORE the purchase date "
                    f"({p.isoformat()}). One of them is wrong — ask the user.")

    return {"timeline": dated + undated, "gaps": gaps, "limitation": lim,
            "note": "Present this as a numbered chronology. Ask about any gap listed."}


def compute_claim_value(consideration=None, compensation=None, interest_rate_pct=None,
                        interest_from=None, today=None) -> dict:
    """Total claim value. Pecuniary jurisdiction turns on the consideration
    paid, not the total claimed -- those are different numbers and conflating
    them sends people to the wrong forum."""
    today = today or dt.date.today()
    base = rules.parse_amount(consideration) or 0.0
    comp = rules.parse_amount(compensation) or 0.0
    interest, days = 0.0, 0
    rate = rules.parse_amount(interest_rate_pct)
    start = rules._parse_date(interest_from)
    if rate and start and base:
        days = max(0, (today - start).days)
        interest = base * (rate / 100.0) * (days / 365.0)

    return {
        "consideration_paid": base,
        "compensation_sought": comp,
        "interest": round(interest, 2),
        "interest_basis": (f"{rate}% simple on Rs {base:,.0f} for {days} days" if interest else None),
        "total_claimed": round(base + comp + interest, 2),
        "forum_is_decided_by": "consideration_paid only",
        "forum": rules.check_forum(base)["forum"] if base else None,
        "note": "Interest is indicative arithmetic, not an entitlement. The Commission decides "
                "whether to award interest and at what rate. Pecuniary jurisdiction is tested "
                "on the consideration paid, not on the total claimed.",
    }


def evidence_checklist(cf) -> dict:
    held_blob = " ".join([cf.evidence_held or "", cf.steps_taken or ""]).lower()
    have, missing = [], []
    for item, pattern in EVIDENCE_ITEMS:
        (have if re.search(pattern, held_blob) else missing).append(item)
    return {
        "appears_to_have": have,
        "not_mentioned": missing,
        "note": "Items under not_mentioned may simply not have been discussed. Ask about them "
                "rather than telling the user they lack them. Proof of payment and a record of "
                "the complaint made are the two that matter most in practice.",
    }


def assess_strength(cf) -> dict:
    """What helps, what hurts, what is unknown. Never a prediction."""
    helps, hurts, unknown = [], [], []

    lim = rules.check_limitation(cf.date_problem_arose)
    if lim["status"] == "within":
        helps.append(f"Filed in time — {lim['days_remaining']} days left of the two-year period.")
    elif lim["status"] == "expired":
        hurts.append("The two-year limitation period has expired. Condonation of delay under the "
                     "proviso to s.69(2) must be sought and is not granted as a matter of course.")
    else:
        unknown.append("The date the problem arose is unknown, so limitation cannot be checked.")

    blob = f"{cf.evidence_held} {cf.steps_taken}".lower()
    if re.search(r"invoice|bill|receipt|payment", blob):
        helps.append("Proof of payment is available, which establishes the consumer relationship.")
    else:
        unknown.append("No proof of payment mentioned — this is the single most important document.")
    if re.search(r"email|whatsapp|chat|complaint|ticket|letter|notice", blob):
        helps.append("There is a record of complaints already made to the opposite party.")
    if re.search(r"warranty|guarantee", blob):
        helps.append("A warranty or guarantee is available to point to.")

    # Commercial purpose is the standard defence that removes 'consumer' status.
    if re.search(r"factory|business|commercial|resale|resell|shop use|industrial|office use",
                 f"{cf.goods_or_service} {cf.what_went_wrong}".lower()):
        hurts.append("The goods or services look like they may have been obtained for a COMMERCIAL "
                     "purpose. Section 2(7) excludes that from 'consumer', subject to the "
                     "self-employment exception. This is the first thing the other side will argue.")

    if not cf.opposite_party:
        unknown.append("The opposite party has not been identified, so nobody can be served.")
    if not cf.amount_paid:
        unknown.append("The amount paid is unknown, so the forum cannot be fixed.")

    return {
        "helps": helps, "hurts": hurts, "unknown": unknown,
        "note": "Present these honestly, including the weaknesses. Do NOT convert this into a "
                "prediction of success or a percentage. If `hurts` is non-empty, say so clearly "
                "before describing the remedies.",
    }


def plan_next_steps(cf) -> dict:
    lim = rules.check_limitation(cf.date_problem_arose)
    steps = []
    missing = cf.missing_critical()
    if missing:
        steps.append({"step": "Pin down the missing facts",
                      "detail": ", ".join(m.replace('_', ' ') for m in missing), "when": "now"})
    steps.append({"step": "Gather documents",
                  "detail": "Proof of payment, the warranty, and every message sent to the seller",
                  "when": "before sending anything"})
    steps.append({"step": "Send a written legal notice",
                  "detail": "Set out the facts and demand specific relief, commonly within 15 days. "
                            "Send it so that delivery is provable.",
                  "when": "next"})
    steps.append({"step": "Try the National Consumer Helpline (1915)",
                  "detail": "Free pre-litigation mediation; often faster than filing.",
                  "when": "in parallel"})
    forum = rules.check_forum(cf.amount_paid)["forum"] if cf.amount_paid else "the appropriate Commission"
    filing = {"step": f"File a complaint before the {forum}",
              "detail": "Online via e-Daakhil (edaakhil.nic.in) or in person.",
              "when": "if the notice is ignored"}
    if lim["deadline"]:
        filing["deadline"] = lim["deadline"]
        filing["when"] += f" — and in any event before {lim['deadline']}"
    steps.append(filing)
    if lim["status"] == "expired":
        steps.insert(0, {"step": "Speak to a lawyer about condonation of delay",
                         "detail": "The two-year period has run. A complaint now needs the "
                                   "Commission to condone the delay for sufficient cause.",
                         "when": "first — this changes everything"})
    return {"steps": steps, "limitation": lim,
            "note": "Give these as a short numbered list with the deadline stated plainly."}


def plan_next_steps_general(cf) -> dict:
    """Next steps for a problem outside the consumer corpus.

    Deliberately free of section numbers, limitation periods, fees and named
    forums: in general mode none of those can be verified, and a confident
    wrong deadline is worse than no deadline. What is left is still the most
    valuable advice for someone in trouble -- write things down, reply in
    writing, keep evidence, and get free legal aid.
    """
    steps = []
    missing = cf.missing_critical()
    if missing:
        steps.append({"step": "Write down the basics",
                      "detail": "Still unclear: " + ", ".join(m.replace("_", " ") for m in missing),
                      "when": "now"})
    steps += [
        {"step": "Put everything in writing from here on",
         "detail": "Stop relying on phone calls. Send an email or a written letter so there is "
                   "a dated record of what was said. Keep a copy of everything you send.",
         "when": "immediately"},
        {"step": "Preserve your evidence",
         "detail": "Agreements, receipts, payslips, messages, photographs, names and dates. "
                   "Back them up somewhere you will not lose them.",
         "when": "immediately"},
        {"step": "Reply in writing to anything you disagree with",
         "detail": "If someone has made a demand or an allegation, an unanswered letter can be "
                   "read as agreement. A short, factual, dated reply protects you.",
         "when": "before any deadline you have been given"},
        {"step": "Get free legal aid",
         "detail": "The District Legal Services Authority in your district gives free legal "
                   "advice and representation on any matter. Walk in, or call NALSA on 15100.",
         "when": "before taking any formal step"},
        {"step": "Do not sign or vacate under pressure",
         "detail": "Do not sign anything you have not read, and do not give up possession or "
                   "rights because of a verbal threat. Ask for it in writing.",
         "when": "always"},
    ]
    return {
        "steps": steps,
        "note": "Give these as a short numbered list. Do NOT add section numbers, deadlines, "
                "fees or a specific court - none of that is verified in this mode. Tell them "
                "which kind of lawyer or office handles this type of matter instead.",
    }
