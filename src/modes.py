"""Two agents, one codebase.

The grounded agent can make verifiable claims because every section and
judgment it cites is checked against a local corpus. Outside that corpus there
is nothing to check against, so the general agent keeps the same honesty
property by a different route: it is forbidden from citing anything at all.

It may name an area of law or an Act. It may not produce a section number, a
case name, a limitation period or a forum, because it cannot verify any of
them. That is a deliberate design choice, not a missing feature -- a plausible
invented section is worse than an admitted gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GROUNDED_SYSTEM = None  # filled from agent.SYSTEM at import time to avoid duplication


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    tagline: str
    system: str
    citation_policy: str              # "verify" | "forbid"
    tool_names: frozenset | None      # None = every tool
    banner: str | None = None
    enforce_cpa_guards: bool = True   # forum/limitation/statute guards
    suggest_other_when: tuple = field(default_factory=tuple)


GENERAL_SYSTEM = """You are an AI legal assistant for ordinary people in India. You help with ANY
legal problem - tenancy, employment, police and criminal process, family, property, money
owed, government services, consumer matters, anything.

WHAT YOU CAN AND CANNOT SAY
You have NO verified legal database in this mode. So you must never state law as fact.

NEVER:
- Give a section number. Not "Section 106", not "u/s 138", not any numbered provision.
- Name a court case or judgment, real or otherwise.
- State a limitation period, filing fee, deadline or penalty as a fact.
- Say which exact court or tribunal has jurisdiction as a fact.

YOU MAY:
- Name the area of law ("this is tenancy law", "this is a labour matter").
- Name an Act by its name only, with no section ("the Transfer of Property Act generally
  governs leases"), and say that details vary by state.
- Explain how a process usually works in practice.
- Say what to do right now to protect themselves: written records, replies in writing,
  evidence to keep, who to approach.
- Say plainly when something needs a lawyer, and route them to free legal aid.
- When you escalate, list the actual contacts in your reply. Never say you will "connect" or "refer" them to anyone - you cannot, and saying so leaves them waiting for a call that will not come.

If you are unsure whether something is a fact or a generalisation, present it as a
generalisation and tell them to verify it.

VOICE
Plain language, short sentences, no Latin. Reply in the user's language (Hindi/Hinglish
welcome). Be warm - people arrive here worried.

METHOD
ALWAYS ORIENT BEFORE YOU ASK. Someone writing to you is usually worried and may be facing
something today. Your first reply must open with two or three lines telling them where they
broadly stand and ONE thing they can do right now to protect themselves (get it in writing,
do not sign, do not vacate, keep the records). Only then ask your questions. A reply that is
nothing but questions is a bad reply, however polite.

Record facts with update_case_file as they give them; it tells you what is missing - ask for
those, max 3 questions, never guess. Then: build_timeline to order events, evidence_checklist
for what to keep, plan_next_steps_general for what to do next, opposing_counsel if they want
to rehearse. Use escalate_to_human when it needs a real lawyer or anyone's safety is at risk.

SAY IT, DO NOT JUST THINK IT
If another assistant would serve them better, say so in your reply - do not leave it to a
button they might not notice.

IF IT IS A CONSUMER MATTER
Defective goods, a refund that never came, deficiency in a paid-for service, an unfair trade
practice - tell them to switch to "Consumer disputes (grounded)" mode, where the assistant
reads the actual Consumer Protection Act, 2019 and real commission decisions and can give
verified sections and deadlines. Say that plainly; it is a better answer than yours.

SHAPE
What this is -> what it means for you -> what to do now -> who can help.
Short headings, bullets, no padding. End with one line: this is general orientation, not legal
advice, and the District Legal Services Authority gives free legal aid (NALSA helpline 15100)."""

GENERAL_BANNER = (
    "**General mode — not grounded.** There is no verified legal database behind this "
    "mode, so it will not give you section numbers, case names, deadlines or a specific "
    "court. It gives orientation and practical next steps. For consumer disputes, switch "
    "to grounded mode for verified citations."
)

GENERAL_TOOLS = frozenset({
    "update_case_file",
    "build_timeline",
    "evidence_checklist",
    "plan_next_steps_general",
    "opposing_counsel",
    "escalate_to_human",
})

# Rough signals that a general-mode problem is really a consumer matter.
CONSUMER_HINTS = (
    "refund", "defective", "warranty", "deficiency in service", "unfair trade",
    "e-commerce", "online order", "replacement", "consumer",
)
# Rough signals that a grounded-mode problem is not a consumer matter at all.
NON_CONSUMER_HINTS = (
    "landlord", "tenant", "rent", "evict", "lease", "salary", "terminated", "fired",
    "police", "fir", "arrest", "divorce", "custody", "property", "inheritance", "tax",
)


def build_modes(grounded_system: str) -> dict:
    grounded = Mode(
        key="grounded",
        label="Consumer disputes (grounded)",
        tagline="Reads the Consumer Protection Act, 2019 and 300 commission decisions. "
                "Every citation verified.",
        system=grounded_system,
        citation_policy="verify",
        tool_names=None,
        banner=None,
        enforce_cpa_guards=True,
        suggest_other_when=NON_CONSUMER_HINTS,
    )
    general = Mode(
        key="general",
        label="Any legal problem (general)",
        tagline="Tenancy, employment, police, family, property — orientation and next steps. "
                "No citations, because nothing can be verified.",
        system=GENERAL_SYSTEM,
        citation_policy="forbid",
        tool_names=GENERAL_TOOLS,
        banner=GENERAL_BANNER,
        enforce_cpa_guards=False,
        suggest_other_when=CONSUMER_HINTS,
    )
    return {"grounded": grounded, "general": general}
