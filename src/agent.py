"""The agent loop.

A while-loop with a dispatch table, ~80 lines. Not LangGraph: there is one
loop, one conditional and seven tools, so a framework would add a dependency
and a mental model without changing behaviour.

Control flow is genuinely agentic -- the model chooses which tools to call and
when it has enough to answer. What is NOT left to the model: jurisdiction,
limitation, citation validity and escalation. Those are code (src/rules.py,
src/validate.py) and the validator audits the final message against them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import rules as R
from . import tools as T
from .llm import ToolCall, complete, describe  # noqa: F401
from .logging_setup import get as _get_logger
from .modes import build_modes
from .state import CaseFile, TurnContext
from .validate import is_degenerate, validate

log = _get_logger("agent")

MAX_TOOL_CALLS = 10
MODES = build_modes(None)  # populated below once SYSTEM is defined

# The user asking to rehearse is an instruction, not a hint. Observed: told
# "I want to practise arguing against them", gpt-oss-120b delivered a general
# advice note and never called opposing_counsel, even with the trigger phrases
# spelled out in the system prompt.
# The model reliably *recognises* an out-of-scope matter but does not reliably
# call the tool, so the user loses the helpline pointers. Detected from the
# model's own draft, which is precise: it has already decided it cannot help.
OUT_OF_SCOPE_RE = re.compile(
    r"outside (the )?(scope|my scope|what i)|not (a )?consumer (law|matter|dispute)|"
    r"beyond (my|the) (scope|remit)|i (can'?t|cannot) help with (this|that)|"
    r"\b(family|criminal|matrimonial|divorce|property|tax|employment) law\b|"
    r"specialis\w+ in (family|criminal)", re.I)

# "How strong is my case?" is a request for an assessment, not a prompt for
# more statute quoting. Observed: qwen produced a good-looking answer that
# never ran the assessment, so the user got prose instead of the honest
# helps/hurts/unknown breakdown.
ASSESS_RE = re.compile(
    r"\bhow (strong|good|solid)\b|\bmy chances\b|\bwill i win\b|\bdo i have a (good )?case\b|"
    r"\bassess\b|\bevaluate\b|\bworth (it|pursuing|fighting)\b|\bshould i (file|pursue|fight)\b",
    re.I)

PRACTISE_RE = re.compile(
    r"\b(practi[sc]e|rehears\w*|help me (argue|fight|respond)|how (do|should) i (respond|reply|argue)|"
    r"what will they (say|argue)|their side|counter ?argument|argue (against|with) them)\b", re.I)

SYSTEM = """You are an AI legal assistant for ordinary people in India. You cover ONE area:
consumer disputes under the Consumer Protection Act, 2019 - defective goods, deficiency in
service, refunds, unfair trade practices.

VOICE
Plain language, short sentences, no Latin. Explain any legal term on first use. Reply in the
user's language (Hindi/Hinglish welcome) but search in English legal terms.

CORRECTIONS
Users revise themselves ("actually it was 25,000", "sorry, March not May"). Treat the newest
statement as correct: call update_case_file again with the new value, say in one line what you
changed, and recompute anything that depended on it. Never argue with their correction.

METHOD
Record facts with update_case_file as the user gives them; it tells you what is missing - ask
for those, max 3 questions, never guess. Call search_statute before any legal proposition.
Call check_forum_and_limitation for venue or timing. Use search_cases to show how comparable
disputes went. Use draft_document for a notice/complaint/reply, opposing_counsel when they
want to rehearse or argue, escalate_to_human for anything outside consumer law, expired
limitation, or stakes too high for this tool.

ONCE THE FACTS ARE IN
Five more tools unlock. Use them - do not just quote law at the person:
  build_timeline       order the events and spot gaps or contradictions
  evidence_checklist   what proof they have vs what is missing
  assess_strength      what helps, what hurts, what is unknown
  compute_claim_value  consideration + compensation + interest
  plan_next_steps      sequenced actions with real deadlines
If they ask "how strong is my case", "what should I do" or "is it worth it",
run the relevant ones before answering.

TOOL DISCIPLINE
update_case_file alone is never enough. Any legal assessment needs search_statute,
and check_forum_and_limitation whenever an amount is known. Omit a field entirely
rather than sending it as null.

HARD RULES
- Cite nothing a tool did not return this turn. An admitted gap beats a confident invention.
- Never describe how a case was decided unless search_cases returned it this turn. No
  "Case A", no "a similar case", no invented outcomes. Use the real title or say you found none.
- Restate forum and limitation exactly as computed. Never recompute them yourself.
- Never predict they will win. Never state a filing fee.
- Outside consumer law: say so and escalate, and list the contacts the tool returns in full.
  Never say you will "connect" or "refer" them to anyone - you cannot. Do not improvise on
  criminal, family, property, employment or tax.
- End substantive advice with one line: this is information, not legal advice, and the District
  Legal Services Authority gives free help.

SHAPE
Where you stand -> what the law says -> what comparable cases did -> the concrete next step.
Short headings, bullets, no padding."""


MODES = build_modes(SYSTEM)


@dataclass
class Agent:
    case_file: CaseFile = field(default_factory=CaseFile)
    messages: list = field(default_factory=list)
    ctx: TurnContext = None
    mode: object = None
    _last_user: str = ""

    def __post_init__(self):
        self.ctx = TurnContext(case_file=self.case_file)
        if self.mode is None:
            self.mode = MODES["grounded"]
        if not self.messages:
            self.messages = [{"role": "system", "content": self.mode.system}]

    def _reconcile_amount(self):
        """Trust the user's words over the model's transcription of them."""
        recorded = self.case_file.amount_paid
        if not recorded:
            return
        stated = R.extract_amounts(self._last_user)
        if not stated:
            return
        val = R.parse_amount(recorded)
        if val is not None and any(abs(val - s) < 1 for s in stated):
            return  # matches something the user actually said
        if len(stated) == 1:
            corrected = f"{stated[0]:.0f}"
            log.warning("amount mismatch: model recorded %r but the user stated %s; "
                        "using the user's figure", recorded, corrected)
            self.case_file.amount_paid = corrected
            self.ctx.tool_log.append({
                "tool": "_reconcile_amount(code)",
                "args": {"model_said": recorded, "user_said": corrected},
                "result_preview": "corrected from the user's own wording",
            })

    def _suggests_other_mode(self) -> bool:
        """Is this problem better served by the other agent?"""
        blob = f"{self._last_user} {self.case_file.what_went_wrong} " \
               f"{self.case_file.goods_or_service}".lower()
        return any(h in blob for h in self.mode.suggest_other_when)

    def _limitation_expired(self) -> bool:
        return any((r.get("limitation") or {}).get("status") == "expired"
                   for r in self.ctx.rule_results)

    def _mandatory_calls(self, fired: set, draft: str = ""):
        """Grounding steps the model does not get to skip.

        Measured need: across 10 live scenarios, gpt-oss-120b recorded the facts
        and then answered WITHOUT computing jurisdiction or limitation in 3 of
        them, and gave substantive advice without retrieving a single section in
        another. Prompting did not fix it, and "usually calls the tool" is not a
        safety property. So when the model tries to finish without them, the
        harness runs them and makes it answer again with the results in hand.

        Each guard fires at most once per turn.
        """
        cf, forced = self.case_file, []

        # Said it cannot help, but never escalated -- so the user got no
        # helpline, no legal aid pointer, nothing actionable.
        if ("escalate" not in fired and draft and OUT_OF_SCOPE_RE.search(draft)
                and not self.ctx.escalated):
            fired.add("escalate")
            forced.append(ToolCall("escalate_to_human",
                                   {"reason": draft[:300], "urgency": "normal"}))
            return forced  # nothing else matters once we are out of scope

        # An assessment was asked for but never actually computed.
        if ("assess" not in fired and ASSESS_RE.search(self._last_user or "")
                and not cf.missing_critical()
                and not any(e["tool"] == "assess_strength" for e in self.ctx.tool_log)):
            fired.add("assess")
            forced.append(ToolCall("assess_strength", {}))

        # Rehearsal was explicitly requested but never role-played.
        if ("practise" not in fired and PRACTISE_RE.search(self._last_user or "")
                and not any(t["tool"] == "opposing_counsel" for t in self.ctx.tool_log)):
            fired.add("practise")
            forced.append(ToolCall("opposing_counsel", {"user_argument": self._last_user[:500]}))

        if not self.mode.enforce_cpa_guards:
            return forced  # the CPA guards below are bound to consumer law

        # Jurisdiction + limitation: mandatory whenever a claim value is known.
        if "forum" not in fired and cf.amount_paid and not self.ctx.rule_results:
            fired.add("forum")
            forced.append(ToolCall("check_forum_and_limitation", {}))

        # No substantive legal advice without at least one retrieved provision.
        if "statute" not in fired and not self.ctx.retrieved_sections and cf.what_went_wrong:
            fired.add("statute")
            q = " ".join(x for x in (cf.goods_or_service, cf.what_went_wrong) if x)[:180]
            forced.append(ToolCall("search_statute", {"query": q or "deficiency in service"}))

        return forced

    def ask(self, user_text: str, verbose=False) -> dict:
        self.ctx.reset_turn()
        self._last_user = user_text
        self.messages.append({"role": "user", "content": user_text})

        import time as _time
        t_start = _time.time()
        log.info("turn start | provider=%s | user=%r", describe(), user_text[:120])
        n_calls, final, fired = 0, None, set()  # `fired` guards fire once each
        while n_calls < MAX_TOOL_CALLS:
            reply = complete(self.messages, tools=T.schemas_for(self.case_file, self.mode))
            if not reply.tool_calls:
                forced = self._mandatory_calls(fired, reply.text or "")
                if forced:
                    if verbose:
                        print(f"    [enforced] {[f.name for f in forced]}")
                    # Run the required tools ourselves and hand the results back
                    # as a plain note. We deliberately do NOT fabricate an
                    # assistant tool-call turn: Gemini 3 attaches a
                    # `thought_signature` to every functionCall part it emits and
                    # rejects any functionCall we invent without one. A plain
                    # message is also portable across every provider.
                    lines = []
                    for tc in forced:
                        n_calls += 1
                        out = T.dispatch(self.ctx, tc.name, tc.args)
                        log.info("enforced tool %s -> %d chars", tc.name, len(out))
                        if tc.name == "update_case_file":
                            self._reconcile_amount()
                        lines.append(f"{tc.name} returned:\n{out}")
                    self.messages.append({"role": "user", "content":
                        "SYSTEM NOTE - required checks were run automatically because your "
                        "answer needed them:\n\n" + "\n\n".join(lines) +
                        "\n\nUse these results. Restate the forum and limitation exactly as "
                        "computed, cite only sections retrieved, and answer now."})
                    continue
                final = reply.text or ""
                break
            self.messages.append({"role": "assistant", "content": reply.text or "",
                                  "tool_calls": reply.tool_calls,
                                  "provider_turn": reply.provider_turn})
            for tc in reply.tool_calls:
                n_calls += 1
                t_tool = _time.time()
                out = T.dispatch(self.ctx, tc.name, tc.args)
                log.info("tool %s(%s) -> %d chars in %.2fs", tc.name,
                         json.dumps(tc.args, default=str)[:120], len(out), _time.time() - t_tool)
                if tc.name == "update_case_file":
                    self._reconcile_amount()
                if verbose:
                    print(f"    -> {tc.name}({json.dumps(tc.args, default=str)[:110]}) "
                          f"=> {out[:110]}")
                self.messages.append({"role": "tool", "name": tc.name,
                                      "id": tc.id, "content": out})
                # An expired limitation period changes everything and does not
                # depend on any other fact. Observed: with the clock dead since
                # 2023, the agent asked for the seller's name instead of saying
                # so. Surface it now, not after the rest of the intake.
                if (self.mode.enforce_cpa_guards and "expired_notice" not in fired
                        and self._limitation_expired()):
                    fired.add("expired_notice")
                    self.messages.append({"role": "user", "content":
                        "The limitation check reports that the two-year period under s.69 has "
                        "ALREADY EXPIRED. Tell the user this plainly in your next message, before "
                        "asking for any further facts, and explain condonation of delay under the "
                        "proviso to s.69(2). Do not bury it."})
        else:
            # Budget exhausted: force a text answer with no tools available.
            self.messages.append({"role": "user", "content":
                                  "You have used your tool budget for this turn. Answer now using only "
                                  "what the tools already returned. Do not cite anything new."})
            final = (complete(self.messages, tools=None).text or "").strip()

        if final is None:
            log.error("turn produced no final answer (tool budget exhausted?)")

        bad, why = is_degenerate(final or "")
        if bad:
            log.error("degenerate output rejected: %s", why)
            self.ctx.tool_log.append({"tool": "_degenerate_output(code)", "args": {"reason": why},
                                      "result_preview": "answer rejected and replaced"})
            final = (
                "I am sorry - I could not produce a reliable answer to that just now "
                "(the model returned malformed output, which I have discarded rather than "
                "show you).\n\n"
                "Please send your message again. If it keeps happening, switch `LLM_MODEL` "
                "in `.env` to another model.\n\n"
                "If this is urgent or involves anyone's safety, do not wait for me: "
                "dial **112** (emergency) or **181** (women's helpline), and your District "
                "Legal Services Authority gives free legal aid on any matter "
                "(NALSA helpline **15100**)."
            )
        checked = validate(final or "", self.ctx, policy=self.mode.citation_policy)
        log.info("turn done in %.1fs | tools=%s | flags=%d | answer=%d chars",
                 _time.time() - t_start, [t["tool"] for t in self.ctx.tool_log],
                 checked["report"]["n_issues"], len(checked["text"]))
        self.messages.append({"role": "assistant", "content": checked["text"]})
        return {
            "answer": checked["text"],
            "raw_answer": final,
            "validation": checked["report"],
            "tools_called": [t["tool"] for t in self.ctx.tool_log],
            "tool_log": list(self.ctx.tool_log),
            "case_file": self.case_file.as_dict(),
            "mode": self.mode.key,
            "suggest_other_mode": self._suggests_other_mode(),
            "provider": describe(),
        }
