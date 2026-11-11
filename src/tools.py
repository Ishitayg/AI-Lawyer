"""The agent's seven tools: provider-neutral schemas + implementations.

Two of these are RAG (search_statute, search_cases). The rest are what make
this an agent rather than a retrieval pipeline: it decides when it has enough
facts, when to compute, when to draft, when to argue back, and when to stop
and send the person to a real lawyer.

Division of labour
  LLM decides : which tool, when, and how to phrase the answer.
  Code decides: forum, limitation, whether a citation is real, escalation.
"""
from __future__ import annotations

import json
import re
import textwrap

from . import analysis, retrieve, rules
from .state import TurnContext

# Trimmed to fit free-tier token budgets (Groq: 8k tokens/minute). The agent
# gets enough of each provision/passage to reason and quote; the full text
# stays one click away via the source URL shown to the user.
MAX_SNIPPET = 700


# ----------------------------------------------------------------- implementations
def _search_statute(ctx: TurnContext, query: str, top_k: int = 3):
    hits = retrieve.search_statute(query, top_k=int(top_k))
    for h in hits:
        ctx.retrieved_sections.add(h.meta["section_no"])
    if not hits:
        return {"results": [], "note": "No statute index found or no match. Do not cite any section."}
    return {
        "results": [
            {"citation": h.meta["citation"], "section_no": h.meta["section_no"],
             "heading": h.meta["heading"], "text": h.text[:MAX_SNIPPET], "url": h.url}
            for h in hits
        ],
        "note": "You may cite ONLY the section numbers listed above. Quote the Act's wording where it matters.",
    }


def _search_cases(ctx: TurnContext, facts_summary: str, top_k: int = 3):
    hits = retrieve.search_cases(facts_summary, top_k=int(top_k))
    for h in hits:
        ctx.retrieved_case_ids.add(str(h.ref))
        ctx.retrieved_case_titles[str(h.ref)] = h.title
    if not hits:
        return {"results": [], "note": "No comparable decision found. Say so plainly; do not invent a case name."}
    return {
        "results": [
            {"doc_id": h.ref, "title": h.title, "court": h.meta.get("court", ""),
             "date": h.meta.get("date", ""), "url": h.url, "passage": h.text[:MAX_SNIPPET]}
            for h in hits
        ],
        "note": ("These are illustrative decisions of consumer commissions, not binding precedent "
                 "on identical facts. Describe what the commission did and how the facts differ. "
                 "Cite ONLY the doc_ids above."),
    }


def _update_case_file(ctx: TurnContext, **kw):
    applied, ignored = ctx.case_file.update(**kw)
    missing = ctx.case_file.missing_critical()
    return {
        "recorded": applied,
        "corrected_field_names": getattr(ctx.case_file, "last_remapped", {}) or {},
        "ignored_unknown_fields": ignored,
        "still_missing_critical": missing,
        "note": ("Ask the user for the missing facts, at most three questions at a time, in plain language. "
                 "Do NOT guess or fill them in yourself."
                 if missing else
                 "All critical facts captured. You may now compute forum and limitation."),
    }


def _check_forum_and_limitation(ctx: TurnContext, amount_inr=None, cause_of_action_date=None):
    amount_inr = amount_inr if amount_inr not in (None, "") else ctx.case_file.amount_paid
    cause_of_action_date = cause_of_action_date if cause_of_action_date not in (None, "") else ctx.case_file.date_problem_arose
    res = rules.check_forum_and_limitation(amount_inr, cause_of_action_date)
    ctx.rule_results.append(res)
    res["note"] = ("These values are computed in code from cited sources and OVERRIDE anything the "
                   "statute text or a judgment appears to say about pecuniary limits. The bare Act "
                   "still prints the pre-2022 figures. Restate these numbers exactly.")
    return res


def _draft_document(ctx: TurnContext, kind: str, extra_instructions: str = ""):
    cf = ctx.case_file
    missing = cf.missing_critical()
    if missing:
        return {"error": "insufficient_facts", "still_missing_critical": missing,
                "note": "Collect these facts before drafting. A notice with blanks invented by you is worse than none."}
    if kind not in ("legal_notice", "consumer_complaint", "reply"):
        return {"error": f"unknown kind {kind!r}", "allowed": ["legal_notice", "consumer_complaint", "reply"]}
    forum = rules.check_forum(cf.amount_paid)
    return {
        "kind": kind,
        "case_file": cf.known(),
        "forum": forum["forum"],
        "skeleton": _SKELETONS[kind],
        "extra_instructions": extra_instructions,
        "note": ("Now write the document in full, using ONLY facts from case_file above. Where a fact is "
                 "genuinely unknown, leave a visible [IN SQUARE BRACKETS] placeholder for the user to fill. "
                 "Never invent names, dates, amounts or addresses. Cite only sections already retrieved."),
    }


_SKELETONS = {
    "legal_notice": textwrap.dedent("""\
        LEGAL NOTICE (pre-litigation demand)
        1. Sender's name and address
        2. Addressee (opposite party) name and address
        3. Facts in numbered paragraphs: what was bought/availed, when, amount paid
        4. The defect or deficiency, and when it arose
        5. Attempts already made to resolve it
        6. The legal basis (deficiency in service / defect in goods / unfair trade practice under
           the Consumer Protection Act, 2019 — cite only retrieved sections)
        7. Specific relief demanded, with a compliance window (commonly 15 days)
        8. Statement that a consumer complaint will follow on non-compliance
        9. Place, date, signature
    """),
    "consumer_complaint": textwrap.dedent("""\
        COMPLAINT UNDER SECTION 35, CONSUMER PROTECTION ACT, 2019
        1. Cause title: BEFORE THE <forum> AT <place>; Complainant vs Opposite Party
        2. Jurisdiction paragraph: pecuniary (value) and territorial (s.34(2) ground relied on)
        3. Limitation paragraph: date cause of action arose; within two years under s.69
        4. Facts in numbered paragraphs
        5. Deficiency / defect / unfair trade practice alleged
        6. Relief claimed (refund, replacement, compensation, costs)
        7. List of documents annexed
        8. Verification, place, date, signature
    """),
    "reply": textwrap.dedent("""\
        REPLY TO NOTICE
        1. Reference to the notice being replied to, with its date
        2. Paragraph-wise response: admitted / denied / not admitted for want of knowledge
        3. The sender's own version of facts
        4. Relief or position asserted
        5. Place, date, signature
    """),
}


def _opposing_counsel(ctx: TurnContext, user_argument: str):
    cf = ctx.case_file.known()
    return {
        "mode": "opposing_counsel",
        "user_argument": user_argument,
        "case_file": cf,
        "note": ("Now answer IN CHARACTER as counsel for the opposite party. Make the strongest honest "
                 "defences available on these facts — typically: the complainant is not a 'consumer' "
                 "(commercial purpose), delay/limitation, misuse or user damage, warranty terms and "
                 "exclusions, no deficiency because the service matched what was promised, wrong forum, "
                 "non-joinder of the manufacturer, and absence of documentary proof. Be brief and pointed. "
                 "Then step out of character under a heading 'How to answer this' and tell the user, in one "
                 "line each, what evidence or wording defeats each point. Do not cite any section or case "
                 "that was not retrieved this turn."),
    }


def _build_timeline(ctx: TurnContext):
    return analysis.build_timeline(ctx.case_file)


def _compute_claim_value(ctx: TurnContext, consideration=None, compensation=None,
                         interest_rate_pct=None, interest_from=None):
    res = analysis.compute_claim_value(
        consideration if consideration not in (None, "") else ctx.case_file.amount_paid,
        compensation, interest_rate_pct,
        interest_from if interest_from not in (None, "") else ctx.case_file.date_problem_arose)
    ctx.rule_results.append({"forum": {"forum": res.get("forum")}, "limitation": {}})
    return res


def _evidence_checklist(ctx: TurnContext):
    return analysis.evidence_checklist(ctx.case_file)


def _assess_strength(ctx: TurnContext):
    return analysis.assess_strength(ctx.case_file)


def _plan_next_steps(ctx: TurnContext):
    return analysis.plan_next_steps(ctx.case_file)


def _plan_next_steps_general(ctx: TurnContext):
    return analysis.plan_next_steps_general(ctx.case_file)


# Matches first- and third-person phrasing, because this is applied both to a
# user's own words and to the model's draft describing them ("her husband hit
# her"). A false positive only adds emergency numbers; a false negative
# withholds them, so this errs deliberately towards matching.
# Domains this consumer-law assistant does not cover.
_NON_CONSUMER_RE = re.compile(
    r"\b(landlord|tenan\w*|rent|evict\w*|lease|salary|wages|terminat\w*|fired|dismiss\w*|"
    r"employ\w*|police|fir\b|arrest|bail|criminal|divorce|custody|maintenance|matrimonial|"
    r"family law|property|inherit\w*|succession|will\b|tax|gst|income tax)\b", re.I)

_SAFETY_RE = re.compile(
    r"\b(violen\w*|assault\w*|beat\w*|abus\w*|threat\w*|harass\w*|stalk\w*|"
    r"rape|molest\w*|dowry|suicide|self[- ]harm|unsafe|domestic)\b"
    r"|\bhit\s+(me|her|him|them|us)\b"
    r"|\bafraid (for|of)\b", re.I)


def _escalate_to_human(ctx: TurnContext, reason: str, urgency: str = "normal"):
    """Hand-off pointers must fit the matter.

    Consumer pointers are useless, and arguably insulting, to someone who has
    just disclosed violence. Safety resources come first when the reason
    indicates harm, and the consumer helplines are offered only for consumer
    matters.
    """
    ctx.escalated = True
    safety = bool(_SAFETY_RE.search(reason or "")) or urgency == "urgent"
    # Offering the consumer helpline and e-Daakhil to someone facing eviction
    # is noise that makes the whole answer look automated. Only surface them
    # when the matter is actually a consumer one.
    non_consumer = bool(_NON_CONSUMER_RE.search(reason or ""))
    pointers = []
    if safety:
        pointers += [
            "Emergency: dial 112 (national emergency) or 100 (police) if anyone is in danger now",
            "Women's Helpline: 181 (24x7, nationwide)",
            "National Commission for Women complaint portal: ncwapps.nic.in",
        ]
    pointers.append(
        "District Legal Services Authority (DLSA) in the user's district — free legal aid for "
        "any matter, statutory body under the Legal Services Authorities Act, 1987; "
        "walk in or call the NALSA helpline 15100")
    if not safety and not non_consumer:
        pointers += [
            "National Consumer Helpline 1915 / consumerhelpline.gov.in — pre-litigation mediation",
            "e-Daakhil (edaakhil.nic.in) — official portal for filing consumer complaints online",
        ]
    if non_consumer and not safety:
        pointers.append(
            "This assistant also has a general mode that covers any legal problem "
            "(tenancy, employment, police, family, property). It gives practical next steps, "
            "though it cannot cite verified law. Switch to it from the sidebar.")
    return {
        "escalated": True, "reason": reason, "urgency": "urgent" if safety else urgency,
        "safety_concern": safety,
        "pointers": pointers,
        "note": ("Say plainly that this is outside what you can help with and why, in ONE "
                 "sentence. Then reproduce EVERY pointer above as a bullet, in full, including "
                 "phone numbers and web addresses - the person must leave this reply with "
                 "somewhere concrete to go. Never say you will 'connect' or 'transfer' them "
                 "or that someone will contact them: you cannot do any of that, and the "
                 "promise leaves them waiting for help that never comes. "
                 + ("Lead with the emergency numbers and be warm, not clinical. "
                    "Do not discuss consumer law at all." if safety else
                    "Do not improvise advice outside consumer law.")),
    }


# ------------------------------------------------------------------------ schemas
def _s(t, d, **kw):
    return {"type": t, "description": d, **kw}


def _opt(d, **kw):
    """Optional string parameter.

    Nullable because models routinely emit `"field": null` for facts they do
    not have. With a bare `"type": "string"` Groq's server-side tool-call
    validator rejects the whole call with HTTP 400 and the turn dies -- that
    killed 2 of 10 scenarios. Nulls are filtered out in CaseFile.update().
    """
    return {"type": ["string", "null"], "description": d, **kw}


SCHEMAS = [
    {
        "name": "search_statute",
        "description": "Search the Consumer Protection Act, 2019. Call before stating any legal "
                       "proposition. You may cite only sections it returns.",
        "parameters": {"type": "object", "properties": {
            "query": _s("string", "Issue in statutory language, e.g. 'reliefs the Commission can grant'."),
            "top_k": _s("integer", "Sections to return (default 3)."),
        }, "required": ["query"]},
    },
    {
        "name": "search_cases",
        "description": "Find consumer commission decisions on similar facts. Required before "
                       "describing how any case was decided.",
        "parameters": {"type": "object", "properties": {
            "facts_summary": _s("string", "User's facts in neutral legal English, 1-2 sentences."),
            "top_k": _s("integer", "Judgments to return (default 3)."),
        }, "required": ["facts_summary"]},
    },
    {
        "name": "update_case_file",
        "description": "Record facts the user gives. Returns which critical facts are still "
                       "missing. Omit fields you do not know; never guess them.",
        "parameters": {"type": "object", "properties": {
            "complainant_name": _opt("User's full name."),
            "complainant_city": _opt("User's city/district."),
            "opposite_party": _opt("Seller/company complained against."),
            "opposite_party_address": _opt("Their address."),
            "goods_or_service": _opt("What was bought or availed."),
            "amount_paid": _opt("Consideration paid, e.g. '18000', '4 lakh'."),
            "date_of_purchase": _opt("Purchase date, YYYY-MM-DD."),
            "date_problem_arose": _opt("Date the problem/refusal occurred (cause of action), YYYY-MM-DD."),
            "what_went_wrong": _opt("The defect or deficiency."),
            "steps_taken": _opt("Complaints already made."),
            "evidence_held": _opt("Documents held: invoice, warranty, chats."),
            "relief_wanted": _opt("Refund, replacement or compensation."),
        }, "required": []},
    },
    {
        "name": "check_forum_and_limitation",
        "description": "Compute the correct commission and whether the 2-year limit has expired. "
                       "Always use this; the printed Act has superseded figures.",
        "parameters": {"type": "object", "properties": {
            "amount_inr": _opt("Consideration paid. Omit to use the case file."),
            "cause_of_action_date": _opt("Date problem arose. Omit to use the case file."),
        }, "required": []},
    },
    {
        "name": "draft_document",
        "description": "Get a skeleton plus recorded facts for drafting. Refuses if facts are missing.",
        "parameters": {"type": "object", "properties": {
            "kind": _s("string", "legal_notice, consumer_complaint or reply.",
                       enum=["legal_notice", "consumer_complaint", "reply"]),
            "extra_instructions": _opt("Specific wording requested."),
        }, "required": ["kind"]},
    },
    {
        "name": "opposing_counsel",
        "description": "Role-play the other side so the user can rehearse. Use whenever they want "
                       "to practise, argue, or ask what the company will say.",
        "parameters": {"type": "object", "properties": {
            "user_argument": _s("string", "The argument to test."),
        }, "required": ["user_argument"]},
    },
    {
        "name": "escalate_to_human",
        "description": "Hand off to a real lawyer/helpline: outside consumer law, expired "
                       "limitation, nothing retrieved, or too serious for this tool.",
        "parameters": {"type": "object", "properties": {
            "reason": _s("string", "Why a human is needed, one sentence."),
            "urgency": _s("string", "normal or urgent.", enum=["normal", "urgent"]),
        }, "required": ["reason"]},
    },
]

ADVANCED_SCHEMAS = [
    {
        "name": "build_timeline",
        "description": "Build a dated chronology from the recorded facts and name any gaps or "
                       "contradictions. Use before drafting or assessing.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "compute_claim_value",
        "description": "Total the claim: consideration paid, compensation sought and indicative "
                       "interest. Forum depends on the consideration only.",
        "parameters": {"type": "object", "properties": {
            "consideration": _opt("Amount paid. Omit to use the case file."),
            "compensation": _opt("Extra compensation the user wants."),
            "interest_rate_pct": _opt("Interest rate, e.g. '9'."),
            "interest_from": _opt("Date interest runs from. Omit to use the case file."),
        }, "required": []},
    },
    {
        "name": "evidence_checklist",
        "description": "What proof the user appears to have and what has not been mentioned yet.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "assess_strength",
        "description": "What helps the case, what hurts it, what is still unknown. Never a "
                       "prediction of winning.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "plan_next_steps",
        "description": "Sequenced actions with real deadlines derived from the limitation date.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "plan_next_steps_general",
        "description": "Practical next steps for a non-consumer matter: written records, "
                       "evidence, replies, free legal aid. No deadlines or forums.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

# Tools the model sees while it is still gathering facts. Sending all twelve
# schemas on every call costs ~1.5k tokens before any content, which on an
# 8k-tokens/minute free tier is the difference between a 10s reply and a 60s
# one. Advanced tools appear once there are facts worth analysing.
CORE_TOOL_NAMES = {"update_case_file", "search_statute", "search_cases",
                   "check_forum_and_limitation", "escalate_to_human"}


def schemas_for(case_file, mode=None) -> list:
    """Tools visible to the model: filtered by mode, then by case-file progress.

    Mode comes first. The general agent must never even see search_statute or
    check_forum_and_limitation -- both are bound to the Consumer Protection Act
    and would produce confidently wrong answers for a tenancy or employment
    problem.
    """
    allowed = getattr(mode, "tool_names", None)
    pool = SCHEMAS + ADVANCED_SCHEMAS
    if allowed is not None:
        tools = [t for t in pool if t["name"] in allowed]
        core = [t for t in tools if t["name"] in CORE_TOOL_NAMES or t["name"] == "escalate_to_human"]
        return core if case_file.missing_critical() and core else tools
    if case_file.missing_critical():
        return [t for t in SCHEMAS if t["name"] in CORE_TOOL_NAMES]
    return pool


IMPLS = {
    "build_timeline": _build_timeline,
    "compute_claim_value": _compute_claim_value,
    "evidence_checklist": _evidence_checklist,
    "assess_strength": _assess_strength,
    "plan_next_steps": _plan_next_steps,
    "plan_next_steps_general": _plan_next_steps_general,
    "search_statute": _search_statute,
    "search_cases": _search_cases,
    "update_case_file": _update_case_file,
    "check_forum_and_limitation": _check_forum_and_limitation,
    "draft_document": _draft_document,
    "opposing_counsel": _opposing_counsel,
    "escalate_to_human": _escalate_to_human,
}


def dispatch(ctx: TurnContext, name: str, args: dict) -> str:
    fn = IMPLS.get(name)
    if fn is None:
        result = {"error": f"unknown tool {name!r}", "available": sorted(IMPLS)}
        ctx.tool_log.append({"tool": name, "args": args, "result_preview": str(result)[:400]})
        return json.dumps(result)
    try:
        result = fn(ctx, **(args or {}))
    except TypeError as e:
        result = {"error": "bad_arguments", "detail": str(e)}
    except Exception as e:  # noqa: BLE001
        result = {"error": type(e).__name__, "detail": str(e)}
    ctx.tool_log.append({"tool": name, "args": args, "result_preview": str(result)[:400]})
    return json.dumps(result, ensure_ascii=False, default=str)
