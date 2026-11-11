"""The case file: structured state the agent builds up across a conversation.

This is deliberately a plain dataclass and not something the LLM free-forms.
`missing_critical()` is what forces the agent to ask instead of inventing --
the tool result literally tells it which facts it still does not have.
"""
from __future__ import annotations

import difflib
import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

MATTERS = Path(__file__).resolve().parents[1] / "data" / "matters"


@dataclass
class CaseFile:
    complainant_name: str = ""
    complainant_city: str = ""
    opposite_party: str = ""
    opposite_party_address: str = ""
    goods_or_service: str = ""
    amount_paid: str = ""
    date_of_purchase: str = ""
    date_problem_arose: str = ""
    what_went_wrong: str = ""
    steps_taken: str = ""
    evidence_held: str = ""
    relief_wanted: str = ""

    # Every change, so a correction is visible rather than silently overwriting
    # an earlier fact. Users routinely revise ("actually it was 25,000").
    revisions: list = field(default_factory=list)
    matter_id: str = ""

    # Facts without which no responsible advice on forum/limitation is possible.
    CRITICAL = ("goods_or_service", "amount_paid", "date_problem_arose", "what_went_wrong", "opposite_party")

    # Common aliases the models actually emit, plus fuzzy matching. Observed:
    # gpt-oss-120b wrote `date_problem_arise`, the field was rejected, and the
    # agent then asked the user for a date they had already given. Losing a
    # fact the user stated is a worse failure than accepting a near-miss name.
    ALIASES = {
        "date_problem_arise": "date_problem_arose",
        "date_of_problem": "date_problem_arose",
        "problem_date": "date_problem_arose",
        "cause_of_action_date": "date_problem_arose",
        "incident_date": "date_problem_arose",
        "purchase_date": "date_of_purchase",
        "amount": "amount_paid",
        "amount_inr": "amount_paid",
        "price_paid": "amount_paid",
        "seller": "opposite_party",
        "company": "opposite_party",
        "vendor": "opposite_party",
        "product": "goods_or_service",
        "service": "goods_or_service",
        "item": "goods_or_service",
        "issue": "what_went_wrong",
        "problem": "what_went_wrong",
        "defect": "what_went_wrong",
        "evidence": "evidence_held",
        "documents": "evidence_held",
        "relief": "relief_wanted",
        "remedy_sought": "relief_wanted",
        "name": "complainant_name",
        "city": "complainant_city",
    }

    def _resolve(self, key, valid):
        if key in valid:
            return key, None
        if key in self.ALIASES:
            return self.ALIASES[key], key
        close = difflib.get_close_matches(key, valid, n=1, cutoff=0.82)
        return (close[0], key) if close else (None, key)

    def update(self, **kw):
        valid = {f.name for f in fields(self)}
        applied, ignored, remapped = {}, [], {}
        for k, v in kw.items():
            target, original = self._resolve(k, valid)
            if target is None:
                ignored.append(k)
                continue
            if v in (None, "", []):
                continue
            new = str(v).strip()
            old = getattr(self, target)
            if old and old != new:
                self.revisions.append({
                    "field": target, "from": old, "to": new,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            setattr(self, target, new)
            applied[target] = new
            if original:
                remapped[original] = target
        self.last_remapped = remapped
        return applied, ignored

    def missing_critical(self):
        return [k for k in self.CRITICAL if not getattr(self, k)]

    def known(self):
        skip = {"revisions", "matter_id"}
        return {k: v for k, v in asdict(self).items() if v and k not in skip}

    def as_dict(self):
        return asdict(self)

    def corrections(self):
        """Changes where the user revised a fact they had already given."""
        return list(self.revisions)

    # ---------------------------------------------------------- persistence
    def save(self, matter_id: str | None = None) -> Path:
        MATTERS.mkdir(parents=True, exist_ok=True)
        self.matter_id = matter_id or self.matter_id or uuid.uuid4().hex[:10]
        path = MATTERS / f"{self.matter_id}.json"
        payload = {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "case_file": self.as_dict()}
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, matter_id: str) -> "CaseFile":
        data = json.loads((MATTERS / f"{matter_id}.json").read_text(encoding="utf-8"))["case_file"]
        cf = cls()
        valid = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k in valid:
                setattr(cf, k, v)
        return cf

    @staticmethod
    def list_saved():
        if not MATTERS.exists():
            return []
        out = []
        for f in sorted(MATTERS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                cf = d["case_file"]
                out.append({"matter_id": f.stem, "saved_at": d.get("saved_at", ""),
                            "summary": " / ".join(x for x in (cf.get("opposite_party"),
                                                              cf.get("goods_or_service"),
                                                              cf.get("amount_paid")) if x) or "(empty)"})
            except Exception:  # noqa: BLE001
                continue
        return out


@dataclass
class TurnContext:
    """Everything the validator needs to audit one assistant turn."""
    case_file: CaseFile = field(default_factory=CaseFile)
    retrieved_sections: set = field(default_factory=set)   # e.g. {"35", "69"}
    retrieved_case_ids: set = field(default_factory=set)
    retrieved_case_titles: dict = field(default_factory=dict)
    rule_results: list = field(default_factory=list)
    escalated: bool = False
    tool_log: list = field(default_factory=list)

    def reset_turn(self):
        self.retrieved_sections = set()
        self.retrieved_case_ids = set()
        self.retrieved_case_titles = {}
        self.rule_results = []
        self.escalated = False
        self.tool_log = []
