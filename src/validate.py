"""Post-hoc verification of the assistant's final message.

The system prompt asks the model not to invent authorities. Prompts are not a
control. This module is the control: it re-reads the finished answer and checks
every legal assertion against what was actually retrieved or computed in the
same turn.

Four checks
  1. Section citations  -> must appear in ctx.retrieved_sections (returned by
                           search_statute this turn).
  2. Case citations     -> any "X vs Y" style case name must correspond to a
                           judgment title returned by search_cases this turn.
  3. Forum consistency  -> the commission named in prose must match what
                           check_forum_and_limitation computed.
  4. Stale pecuniary    -> catches the specific, likely failure of repeating
                           the bare Act's superseded "one crore" District limit.

Unverified items are marked inline and listed in a warning block. Nothing is
silently dropped: the user sees exactly what could not be stood behind.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "raw_outputs" / "validation.jsonl"

SECTION_RE = re.compile(r"\b(?:[Ss]ection|[Ss]ec\.?|[Ss]\.)\s*(\d+[A-Z]?)(?:\s*\(\s*\d+\s*\))?", re.U)
CASE_RE = re.compile(r"\b([A-Z][\w&.'\-/]*(?:\s+[\w&.'\-/]*[A-Za-z][\w&.'\-/]*){0,6}?)\s+(?:vs?\.?|versus|v/s)\s+([A-Z][\w&.'\-/]*(?:\s+[A-Z][\w&.'\-/]*){0,6})")

# Abbreviations whose trailing period must not be read as a sentence end.
_ABBR = r"(?:Ltd|Pvt|Co|Corp|Inc|No|Nos|Mr|Mrs|Ms|Smt|Shri|Dr|Govt|Hon|Anr|Ors|M/s|M/S|St|Rs|vs|v)"
_ABBR_RE = re.compile(rf"\b{_ABBR}\.", re.U)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
# Connectives that can be swept into a case name by the regex's left side.
_LEADERS = {"in", "see", "also", "and", "but", "the", "a", "an", "per", "cf",
            "here", "e", "g", "compare", "following", "unlike", "like", "as",
            "decided", "held", "case", "matter", "act", "section", "similarly"}


def _sentences(text):
    """Split on sentence ends without being fooled by 'Pvt. Ltd.'"""
    shielded = _ABBR_RE.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    for part in _SENT_SPLIT.split(shielded):
        yield part.replace("\x00", ".")


def _words(s):
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if w]


def _case_is_known(name, known_word_sets):
    """A cited case counts as verified if the distinctive words of the party
    names appear as WHOLE WORDS in a judgment title retrieved this turn."""
    left = name.split(" vs ")[0] if " vs " in name.lower() else name
    toks = [w for w in _words(left) if w not in _LEADERS and len(w) > 2]
    if not toks:
        return True  # nothing distinctive to check
    toks = toks[:3]
    return any(all(t in ws for t in toks) for ws in known_word_sets)
MARK = " ⚠[UNVERIFIED]"

# Sections of the CPA 2019 that the skeletons themselves reference; treated as
# known-good scaffolding rather than model-invented authority.
ALWAYS_OK = {"34", "35", "69"}

# Figures quoted as a jurisdictional ceiling, e.g. "does not exceed Rs 50,00,000".
_THRESHOLD_RE = re.compile(
    r"(?:does not exceed|not exceeding|exceeds?|up to|ceiling of|limit of|less than|more than)\s*"
    r"(?:\u20b9|Rs\.?|INR)?\s*(?P<amt>[\d,]+(?:\.\d+)?\s*(?:lakhs?|crores?|cr)?)",
    re.I)
_VALID_THRESHOLDS = {5_000_000.0, 20_000_000.0}


def _money(raw):
    import re as _re
    t = (raw or "").strip().lower()
    m = _re.match(r"([\d,]+(?:\.\d+)?)\s*(lakhs?|crores?|cr)?$", t)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (m.group(2) or "").rstrip("s")
    if unit == "lakh":
        val *= 100_000
    elif unit in ("crore", "cr"):
        val *= 10_000_000
    return val


_FILING_RE = re.compile(
    r"\b(file|filing|approach|lodge|submit|institute|present)\b|complaint (?:before|with|to)|"
    r"jurisdiction (?:lies|vests|rests)", re.I)

# Language that asserts what some other tribunal actually did. If no judgment
# was retrieved this turn, such a claim is unsourced -- and the first live run
# produced exactly this: two invented decisions labelled "Case A" and "Case B",
# which the "X vs Y" citation check could not see because no party names were
# given. This is the check that catches narrative fabrication.
_CASE_NARRATIVE_RE = re.compile(
    r"\bcase\s+[A-Z0-9]\b|\bthe commission (?:held|ordered|directed|awarded|dismissed|allowed|found)\b|"
    r"\b(?:in a |in another |a )?similar (?:case|matter|dispute)\b|\bcomparable (?:case|decision)\b|"
    r"\bearlier decision\b|\bprecedent\b|\bcourts have (?:held|found)\b|"
    r"\bcommissions have (?:held|found|ordered)\b|\bwas decided\b", re.I)

FORUMS = {
    "district": "District Consumer Disputes Redressal Commission",
    "state": "State Consumer Disputes Redressal Commission",
    "national": "National Consumer Disputes Redressal Commission",
}


def is_degenerate(text: str) -> tuple[bool, str]:
    """Detect a model stuck in a repetition loop.

    Observed live: qwen3.8-27b answered a domestic-violence disclosure with the
    word "really" repeated ~200 times. Shipping that to someone in distress is
    unacceptable, so it is caught and replaced rather than rendered.
    """
    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < 40:
        return False, ""
    top, n = Counter(words).most_common(1)[0]
    if n / len(words) > 0.25:
        return True, f"the word {top!r} is {n / len(words):.0%} of the answer ({n} times)"
    trigrams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    tri, tn = trigrams.most_common(1)[0]
    if tn > 12:
        return True, f"the phrase {' '.join(tri)!r} repeats {tn} times"
    return False, ""


def _validate_forbid(answer, marked, ctx):
    """General mode: flag any section number, case name or hard deadline."""
    issues = []
    seen = set()
    for m in SECTION_RE.finditer(answer or ""):
        val = f"Section {m.group(1)}"
        if val in seen:
            continue
        seen.add(val)
        issues.append({"type": "citation_not_permitted", "value": val,
                       "detail": "general mode has no verified corpus, so section numbers "
                                 "must not be given at all"})
        marked = re.sub(rf"(\b(?:[Ss]ection|[Ss]ec\.?|[Ss]\.)\s*{re.escape(m.group(1))}\b)"
                        rf"(?!{re.escape(MARK)})", r"\1" + MARK, marked)
    for sentence in _sentences(answer or ""):
        for m in CASE_RE.finditer(sentence):
            name = m.group(0).strip()
            if name in seen:
                continue
            seen.add(name)
            issues.append({"type": "citation_not_permitted", "value": name,
                           "detail": "case names cannot be verified in general mode"})
            marked = marked.replace(name, name + MARK, 1)
    for m in _HARD_DEADLINE_RE.finditer(answer or ""):
        val = m.group(0).strip()
        if val in seen:
            continue
        seen.add(val)
        issues.append({"type": "unverified_deadline", "value": val,
                       "detail": "limitation periods vary by statute and state and are not "
                                 "verified in general mode"})
        break

    if issues:
        bullets = "\n".join(f"- **{i['type']}** — `{i['value']}`: {i['detail']}" for i in issues)
        marked += ("\n\n---\n**⚠ This answer is not grounded in any verified source.** "
                   f"The automatic check flagged {len(issues)} item(s):\n{bullets}\n\n"
                   "Treat them as unverified. For consumer disputes, switch to grounded mode "
                   "where citations are checked against the Consumer Protection Act, 2019.")

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "general", "n_issues": len(issues), "issues": issues,
        "retrieved_sections": [], "retrieved_case_ids": [],
        "tools_called": [t["tool"] for t in ctx.tool_log], "escalated": ctx.escalated,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    return {"text": marked, "report": report}


_HARD_DEADLINE_RE = re.compile(
    r"\bwithin\s+(?:\d+|one|two|three|six|twelve)\s+(?:days?|weeks?|months?|years?)\b", re.I)


def _normalise_title(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def validate(answer: str, ctx, policy: str = "verify") -> dict:
    """policy="verify"  -> citations must match what was retrieved (grounded mode)
       policy="forbid"  -> ANY citation is an error (general mode, no corpus)

    Same machinery, opposite polarity. In general mode nothing can be checked,
    so the honesty property is kept by forbidding citation outright rather than
    by verifying it.
    """
    issues = []
    marked = answer or ""

    if policy == "forbid":
        return _validate_forbid(answer, marked, ctx)

    # --- 1. sections -------------------------------------------------------
    allowed_secs = set(ctx.retrieved_sections) | ALWAYS_OK
    bad_secs = []
    for m in SECTION_RE.finditer(marked):
        if m.group(1) not in allowed_secs:
            bad_secs.append(m.group(1))
    for s in sorted(set(bad_secs)):
        issues.append({"type": "unverified_section", "value": f"Section {s}",
                       "detail": "not returned by search_statute in this turn"})
        marked = re.sub(rf"(\b(?:[Ss]ection|[Ss]ec\.?|[Ss]\.)\s*{re.escape(s)}\b)(?!{re.escape(MARK)})",
                        r"\1" + MARK, marked)

    # --- 2. case names -----------------------------------------------------
    known = [set(_words(t)) for t in ctx.retrieved_case_titles.values()]
    flagged_names = set()
    for sentence in _sentences(answer or ""):
        for m in CASE_RE.finditer(sentence):
            name = m.group(0).strip()
            if name in flagged_names or _case_is_known(name, known):
                continue
            flagged_names.add(name)
            issues.append({"type": "unverified_case", "value": name,
                           "detail": "no matching judgment was retrieved in this turn"})
            if MARK not in marked[max(0, marked.find(name)):marked.find(name) + len(name) + 20]:
                marked = marked.replace(name, name + MARK, 1)

    # --- 3. forum consistency ---------------------------------------------
    computed = None
    for r in ctx.rule_results:
        f = (r.get("forum") or {}).get("forum")
        if f:
            computed = f
    if computed:
        expected = next(k for k, v in FORUMS.items() if v == computed)
        # Only a sentence that tells the user to FILE somewhere can be a forum
        # error. Naming another commission to explain a threshold ("exceeds the
        # Rs 2,00,00,000 ceiling for the State Commission") is correct writing,
        # and flagging it was a false positive on the first live run.
        for sentence in _sentences(answer or ""):
            low = sentence.lower()
            if not _FILING_RE.search(low):
                continue
            if FORUMS[expected].lower() in low or f"{expected} commission" in low:
                continue
            wrong = {k for k in FORUMS
                     if k != expected and (f"{k} commission" in low or FORUMS[k].lower() in low)}
            if wrong:
                issues.append({"type": "forum_mismatch", "value": ", ".join(sorted(wrong)),
                               "detail": f"code computed {computed}, but this sentence directs "
                                         f"filing at {sorted(wrong)}: {sentence.strip()[:160]!r}"})
                break

    # --- 4. stale pecuniary figure ----------------------------------------
    low = (answer or "").lower()
    if re.search(r"(one crore|1 ?crore|1,00,00,000)", low) and "district" in low:
        issues.append({"type": "stale_pecuniary_limit", "value": "one crore",
                       "detail": "District Commission limit was reduced to Rs 50,00,000 by the "
                                 "notification dated 30.12.2021; the bare Act text is superseded"})

    # --- 5. case narrative without retrieval -------------------------------
    if not ctx.retrieved_case_ids:
        for m in _CASE_NARRATIVE_RE.finditer(answer or ""):
            issues.append({
                "type": "uncited_case_narrative", "value": m.group(0).strip(),
                "detail": "the answer describes how a case was decided, but search_cases "
                          "returned nothing this turn, so this is unsourced",
            })
            break

    # --- 6. self-check: does the answer restate the amount correctly? -----
    # Guards the digit-drop class specifically (Rs 2,50,00,000 quoted back as
    # Rs 25,00,000). Only fires within one order of magnitude of the recorded
    # figure, so statutory thresholds and compensation figures do not trip it.
    recorded = getattr(getattr(ctx, "case_file", None), "amount_paid", "") or ""
    if recorded:
        from .rules import extract_amounts, parse_amount
        base = parse_amount(recorded)
        if base:
            for amt in extract_amounts(answer or ""):
                if abs(amt - base) < 1:
                    continue
                ratio = amt / base if base else 0
                if ratio in (0,):
                    continue
                if 0.09 < ratio < 0.12 or 8.5 < ratio < 11.5:
                    issues.append({
                        "type": "amount_misquoted", "value": f"{amt:,.0f}",
                        "detail": f"the case file records Rs {base:,.0f}; this differs by "
                                  f"about 10x, which is the classic lakh/crore transcription error",
                    })
                    break

    # --- 7. pecuniary threshold restated wrongly --------------------------
    # The model restated the District ceiling as "Rs 5,00,00,000" (5 crore)
    # instead of Rs 50,00,000. Check 4 only catches the specific stale "one
    # crore" figure and check 6 only compares against the claim amount, so a
    # mangled threshold slipped through both. Thresholds are the system's core
    # correctness claim, so they are checked explicitly.
    for m in _THRESHOLD_RE.finditer(answer or ""):
        raw = m.group("amt")
        val = _money(raw)
        if val is None or val in _VALID_THRESHOLDS:
            continue
        issues.append({
            "type": "wrong_pecuniary_threshold", "value": raw.strip(),
            "detail": "the statutory thresholds are Rs 50,00,000 (District) and "
                      "Rs 2,00,00,000 (State); this figure is neither",
        })
        break

    if issues:
        bullets = "\n".join(f"- **{i['type']}** — `{i['value']}`: {i['detail']}" for i in issues)
        marked += ("\n\n---\n**⚠ Automatic citation check flagged "
                   f"{len(issues)} item(s) in the answer above:**\n{bullets}\n\n"
                   "Marked items were not verified against the local statute index or the retrieved "
                   "judgments. Do not rely on them. Verify before using this in any filing.")

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_issues": len(issues),
        "issues": issues,
        "retrieved_sections": sorted(ctx.retrieved_sections),
        "retrieved_case_ids": sorted(ctx.retrieved_case_ids),
        "tools_called": [t["tool"] for t in ctx.tool_log],
        "escalated": ctx.escalated,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    return {"text": marked, "report": report}
