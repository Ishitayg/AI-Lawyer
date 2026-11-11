"""Single LLM seam. Swap providers with one env var; nothing else changes.

    LLM_PROVIDER=gemini   (default, free tier)
    LLM_PROVIDER=ollama   (fully offline)
    LLM_PROVIDER=openrouter

Every provider implements the same contract:

    complete(messages, tools=None, temperature=0.2) -> Reply
    Reply.text        : str | None
    Reply.tool_calls  : list[ToolCall(name, args)]

Tool schemas are declared once, provider-neutral (src/tools.py), and
translated per provider here. That is the whole swap surface.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from .logging_setup import get as _get_logger

load_dotenv()
log = _get_logger("llm")


def _approx_tokens(payload) -> int:
    """Rough token estimate (~4 chars/token). Enough to explain a 429."""
    return len(json.dumps(payload, default=str)) // 4

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
_DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.1:8b",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4.1-mini",
}
MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(PROVIDER, "gemini-2.0-flash"))

# Every OpenAI-compatible provider differs only by base URL and key name.
OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
}


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    # OpenAI-compatible providers match each tool result to its call by id.
    # Deriving the id from the tool name breaks the moment one turn calls the
    # same tool twice, so every call gets a unique one.
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:16]}")


@dataclass
class Reply:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    # The provider's own representation of this assistant turn, replayed
    # verbatim on the next request. Gemini 3 thinking models attach a
    # `thought_signature` to each functionCall part and reject the follow-up
    # request if it is missing -- so the turn cannot be rebuilt from name+args,
    # it has to be handed back exactly as it came.
    provider_turn: Any = None


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- gemini
def _gemini_schema(node):
    """Translate the provider-neutral tool schema into what Gemini accepts.

    Tool schemas are declared once in src/tools.py and must suit every
    provider. Groq needs `"type": ["string", "null"]` or it rejects the whole
    tool call when a model sends a null; Gemini's SDK refuses union types and
    wants `nullable: true` instead. Reconciling that belongs in the adapter,
    not in the shared declaration -- this is precisely the seam the adapter
    exists for.
    """
    if isinstance(node, list):
        return [_gemini_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, list):
            concrete = [t for t in v if t != "null"] or ["string"]
            out["type"] = concrete[0]
            if "null" in v:
                out["nullable"] = True
        elif k in ("properties", "items"):
            out[k] = _gemini_schema(v)
        else:
            out[k] = _gemini_schema(v)
    return out


def _gemini(messages, tools, temperature, attempts=5):
    """Gemini, with retry on transient overload.

    503 UNAVAILABLE ("experiencing high demand") and 429 are both transient on
    this API and arrive often enough that a single attempt is unreliable --
    one 503 was enough to abort a whole benchmark run. Pinned model versions
    are steadier than the `-latest` aliases, which 503 most.
    """
    last = None
    for attempt in range(attempts):
        try:
            return _gemini_once(messages, tools, temperature)
        except Exception as e:  # noqa: BLE001
            text = str(e)
            transient = ("503" in text or "UNAVAILABLE" in text
                         or "429" in text or "RESOURCE_EXHAUSTED" in text
                         or "500" in text or "INTERNAL" in text)
            if not transient or attempt == attempts - 1:
                raise
            wait = min(2 ** attempt * 2, 30)
            log.warning("gemini transient error (%s); retrying in %.0fs (attempt %d/%d)",
                        text[:90], wait, attempt + 1, attempts)
            if _ON_WAIT:
                try:
                    _ON_WAIT(wait, attempt + 1, attempts)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(wait)
            last = e
    raise last  # unreachable


def _gemini_once(messages, tools, temperature):
    from google import genai
    from google.genai import types

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("Set GOOGLE_API_KEY in .env (free tier: aistudio.google.com/apikey)")
    # The SDK defaults to API version v1, which serves only a subset of models:
    # gemini-3-flash-preview and gemini-flash-latest both 404 there while
    # resolving fine on v1beta. Pin it so the chain's model names mean what
    # they say.
    client = genai.Client(api_key=key,
                          http_options=types.HttpOptions(api_version="v1beta"))

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=m["name"], response={"result": m["content"]})],
                )
            )
        elif m["role"] == "assistant" and m.get("tool_calls"):
            native = m.get("provider_turn")
            if native is not None:
                contents.append(native)  # preserves thought_signature
            else:
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_function_call(name=tc.name, args=tc.args)
                               for tc in m["tool_calls"]],
                    )
                )
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

    cfg: dict[str, Any] = {"temperature": temperature}
    if system:
        cfg["system_instruction"] = system
    if tools:
        cfg["tools"] = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters=_gemini_schema(t["parameters"]))
                for t in tools
            ])
        ]
    resp = client.models.generate_content(
        model=MODEL, contents=contents, config=types.GenerateContentConfig(**cfg)
    )
    calls, text, native = [], None, None
    for cand in resp.candidates or []:
        if cand.content is not None:
            native = cand.content
        for part in (cand.content.parts or []) if cand.content else []:
            if getattr(part, "function_call", None):
                calls.append(ToolCall(part.function_call.name, dict(part.function_call.args or {})))
            elif getattr(part, "text", None):
                text = (text or "") + part.text
    return Reply(text=text, tool_calls=calls, raw=resp,
                 provider_turn=native if calls else None)


# --------------------------------------------------------------------------- ollama
def _ollama(messages, tools, temperature):
    import requests

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    msgs = []
    for m in messages:
        if m["role"] == "tool":
            msgs.append({"role": "tool", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": tc.name, "arguments": tc.args}} for tc in m["tool_calls"]]})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    payload: dict[str, Any] = {"model": MODEL, "messages": msgs, "stream": False, "options": {"temperature": temperature}}
    if tools:
        payload["tools"] = [{"type": "function", "function": t} for t in tools]
    r = requests.post(f"{host}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    msg = r.json().get("message", {})
    calls = [ToolCall(c["function"]["name"],
                      c["function"]["arguments"] if isinstance(c["function"]["arguments"], dict)
                      else json.loads(c["function"]["arguments"]))
             for c in msg.get("tool_calls", [])]
    return Reply(text=msg.get("content") or None, tool_calls=calls, raw=msg)


# ------------------------------------------------------- openai-compatible (groq, openrouter)
def _openai_compatible(messages, tools, temperature, provider):
    import requests

    url, key_name = OPENAI_COMPATIBLE[provider]
    key = os.getenv(key_name)
    if not key:
        raise LLMError(f"Set {key_name} in .env")

    msgs = []
    for m in messages:
        if m["role"] == "tool":
            msgs.append({"role": "tool", "content": m["content"],
                         "tool_call_id": m.get("id") or m["name"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            msgs.append({"role": "assistant", "content": m.get("content") or None,
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.name,
                                                      "arguments": json.dumps(tc.args)}}
                                        for tc in m["tool_calls"]]})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})

    payload: dict[str, Any] = {"model": MODEL, "messages": msgs, "temperature": temperature}
    if tools:
        payload["tools"] = [{"type": "function", "function": t} for t in tools]
        payload["tool_choice"] = "auto"

    est = _approx_tokens(payload)
    log.debug("%s request: %d msgs, %d tools, ~%d tokens",
              provider, len(msgs), len(tools or []), est)
    bucket = _bucket(provider)
    bucket.reserve(est)
    t0 = time.time()
    r = _post_with_retry(url, key, payload, provider)
    usage = r.json().get("usage") or {}
    bucket.record_actual(est, usage.get("total_tokens") or 0)
    log.info("%s %s ok in %.1fs (prompt=%s completion=%s total=%s)", provider, MODEL,
             time.time() - t0, usage.get("prompt_tokens"), usage.get("completion_tokens"),
             usage.get("total_tokens"))
    msg = r.json()["choices"][0]["message"]
    calls = []
    for c in msg.get("tool_calls") or []:
        try:
            args = json.loads(c["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(c["function"]["name"], args, id=c.get("id") or f"call_{uuid.uuid4().hex[:16]}"))
    return Reply(text=msg.get("content"), tool_calls=calls, raw=msg)


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.I)

# Optional UI hook so a front end can say "waiting 21s for the rate limit"
# instead of showing an opaque spinner.
_ON_WAIT = None

# Never block a user for longer than this on a single call. Groq answered one
# overdraft with "try again in 650s"; an 11-minute silent stall is worse than
# an honest error telling the user to switch provider.
MAX_WAIT_SECONDS = float(os.getenv("LLM_MAX_WAIT_SECONDS", "45"))


class _TokenBucket:
    """Client-side pacing against a tokens-per-minute cap.

    Reacting to 429s is not enough. The free tier refills 8,000 tokens/minute,
    but one agent turn fires 4-8 calls of ~2,000 tokens within seconds. That
    overdraws the bucket, and the server repays the debt with escalating
    penalties -- measured: 111s, then 149s, then 650s. Staying under the line
    in the first place keeps every call fast.
    """

    def __init__(self, tpm: int):
        self.tpm = tpm
        self.events: list[tuple[float, int]] = []

    def _used(self, now):
        self.events = [(t, n) for t, n in self.events if t > now - 60]
        return sum(n for _, n in self.events)

    def reserve(self, est: int):
        if self.tpm <= 0:
            return
        ceiling = self.tpm * 0.85  # headroom: our estimate is approximate
        now = time.time()
        used = self._used(now)
        if used + est > ceiling and self.events:
            oldest = self.events[0][0]
            wait = max(0.0, 60 - (now - oldest)) + 0.5
            if wait > 0:
                log.info("pacing: %d tokens used in the last minute, this call ~%d; "
                         "waiting %.1fs to stay under %d TPM", used, est, wait, self.tpm)
                if _ON_WAIT:
                    try:
                        _ON_WAIT(wait, 0, 0)
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(wait)
                now = time.time()
                self._used(now)
        self.events.append((now, est))

    def record_actual(self, est: int, actual: int):
        """Replace the estimate with the server's real count."""
        if actual and self.events:
            t, _ = self.events[-1]
            self.events[-1] = (t, actual)


_BUCKETS: dict[str, _TokenBucket] = {}


def _bucket(provider: str) -> _TokenBucket:
    tpm = int(os.getenv("LLM_TPM", "8000" if provider == "groq" else "0"))
    if provider not in _BUCKETS:
        _BUCKETS[provider] = _TokenBucket(tpm)
    return _BUCKETS[provider]


def set_wait_callback(fn):
    global _ON_WAIT
    _ON_WAIT = fn


_TPD_RE = re.compile(r"\(TPD\): Limit (\d+), Used (\d+)")
_MALFORMED_TOOL_RE = re.compile(
    r"which was not in request\.tools|did not match schema|tool call validation failed", re.I)


def _quota_message(provider, body, wait):
    """Say which budget ran out, because the remedies are different.

    Per-minute (TPM): wait a moment and it refills.
    Per-day (TPD): this model is done for the day -- switch model or provider.
    Groq meters each model separately, so another model is usually still free.
    """
    tpd = _TPD_RE.search(body or "")
    if tpd:
        used, limit = int(tpd.group(2)), int(tpd.group(1))
        return (f"{provider}/{MODEL} has used its DAILY token budget "
                f"({used:,}/{limit:,}). Waiting will not help today. Groq meters each model "
                f"separately, so switch LLM_MODEL in .env to another model "
                f"(openai/gpt-oss-20b, qwen/qwen3.8-27b, openai/gpt-oss-120b), or set "
                f"LLM_PROVIDER=gemini with a free key from https://aistudio.google.com/apikey.")
    return (f"{provider}/{MODEL} is rate-limiting per minute: it asked for {wait:.0f}s "
            f"(cap {MAX_WAIT_SECONDS:.0f}s). The per-minute token budget is spent; "
            f"wait a moment and send again, or switch model/provider in .env.")


def _post_with_retry(url, key, payload, provider, attempts=6):
    """Free tiers are token-per-minute limited. A 429 is normal operation here,
    not an error: the response says exactly how long to wait, so wait."""
    import requests

    last = None
    for attempt in range(attempts):
        r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"},
                          json=payload, timeout=180)
        if r.status_code < 400:
            return r
        last = r
        if r.status_code == 429:
            m = _RETRY_AFTER_RE.search(r.text)
            wait = float(m.group(1)) + 0.75 if m else min(2 ** attempt, 30)
            try:
                wait = max(wait, float(r.headers.get("retry-after", 0)))
            except ValueError:
                pass
            if wait > MAX_WAIT_SECONDS:
                raise LLMError(_quota_message(provider, r.text, wait))
            log.warning("rate limited by %s; waiting %.1fs (attempt %d/%d)",
                        provider, wait, attempt + 1, attempts)
            if _ON_WAIT:
                try:
                    _ON_WAIT(wait, attempt + 1, attempts)
                except Exception:  # noqa: BLE001 - a UI callback must never break a retry
                    pass
            time.sleep(wait)
            continue
        # Weak models emit malformed tool calls in two ways, both of which
        # Groq rejects server-side with a 400 that kills the turn:
        #   1. an invented tool name    ("update Case_file")
        #   2. arguments that violate the declared schema
        # Measured on qwen3.8-27b: these two accounted for 3 of 17 scenarios
        # failing outright. A resample usually produces a well-formed call,
        # which is far cheaper than losing the conversation.
        if r.status_code == 400 and _MALFORMED_TOOL_RE.search(r.text or ""):
            if attempt < attempts - 1:
                log.warning("malformed tool call from model; resampling (%s)", r.text[:200])
                time.sleep(1)
                continue
            raise LLMError(
                f"{provider}/{MODEL} kept producing malformed tool calls "
                f"({attempts} attempts). This model handles function calling poorly; "
                f"switch LLM_MODEL in .env to openai/gpt-oss-120b or openai/gpt-oss-20b."
            )

        # "Request too large" is a 429 but has nothing to do with pacing --
        # waiting will never help, the context itself must shrink.
        if r.status_code == 429 and "Request too large" in (r.text or ""):
            raise LLMError(
                f"{provider}/{MODEL} rejected the request as too large for its per-request "
                f"limit. Shrink the context: lower LLM_KEEP_FULL_TOOL_RESULTS (currently "
                f"{KEEP_FULL_TOOL_RESULTS}) or LLM_OLD_TOOL_RESULT_CHARS "
                f"(currently {OLD_TOOL_RESULT_CHARS}), or use a model with a larger limit."
            )
        if 500 <= r.status_code < 600 and attempt < attempts - 1:
            time.sleep(2 ** attempt)
            continue
        break
    raise LLMError(f"{provider} HTTP {last.status_code}: {last.text[:400]}")


def _groq(messages, tools, temperature):
    return _openai_compatible(messages, tools, temperature, "groq")


def _openrouter(messages, tools, temperature):
    return _openai_compatible(messages, tools, temperature, "openrouter")


def _openai(messages, tools, temperature):
    return _openai_compatible(messages, tools, temperature, "openai")


# --------------------------------------------------------------------------- stub
# Deterministic scripted provider. No key, no network. Exists so the agent loop,
# tool dispatch and citation validator can be tested in CI and on a machine with
# no credentials -- the parts of this system that are supposed to be verifiable
# should not require a paid third party to verify.
_STUB_SCRIPT: list = []


def set_stub_script(steps):
    """steps: list of either [ToolCall, ...] or a final string."""
    global _STUB_SCRIPT
    _STUB_SCRIPT = list(steps)


def _stub(messages, tools, temperature):
    if not _STUB_SCRIPT:
        return Reply(text="[stub exhausted]")
    step = _STUB_SCRIPT.pop(0)
    if isinstance(step, str):
        return Reply(text=step)
    return Reply(text=None, tool_calls=list(step))


_PROVIDERS = {"gemini": _gemini, "ollama": _ollama, "openrouter": _openrouter,
              "groq": _groq, "openai": _openai, "stub": _stub}


# Tool results are the bulk of the context (statute sections, judgment
# passages). Once the model has acted on an older tool result, keeping it in
# full buys nothing and costs tokens -- which on a free tier is the binding
# constraint. Keep the most recent few intact; abbreviate the rest.
KEEP_FULL_TOOL_RESULTS = int(os.getenv("LLM_KEEP_FULL_TOOL_RESULTS", "3"))
OLD_TOOL_RESULT_CHARS = int(os.getenv("LLM_OLD_TOOL_RESULT_CHARS", "400"))


def _compact(messages):
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep = set(tool_idx[-KEEP_FULL_TOOL_RESULTS:])
    out = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and i not in keep and len(m.get("content", "")) > OLD_TOOL_RESULT_CHARS:
            m = {**m, "content": m["content"][:OLD_TOOL_RESULT_CHARS] + " ...[trimmed]"}
        out.append(m)
    return out


# --------------------------------------------------------------- model chain
# Free tiers meter per model per day. Hitting that limit is normal operation,
# not a failure, so it should not reach the user as an error: the chain simply
# moves to the next model. Smallest capable model first, both to conserve the
# larger budgets and because it is faster.
#
# Order is deliberate: gpt-oss-20b and -120b handle function calling reliably;
# qwen3.8-27b does not (it invented tool names and produced repetition loops in
# testing), so it sits last as a genuine last resort.
_DEFAULT_CHAIN = [
    ("groq", "openai/gpt-oss-20b"),
    ("groq", "openai/gpt-oss-120b"),
    ("gemini", "gemini-3.5-flash"),
    ("groq", "qwen/qwen3.8-27b"),
]

# Entries whose daily budget is spent. Remembered for the life of the process
# so we stop paying a failed request to rediscover it on every turn.
_EXHAUSTED: set = set()


def _chain():
    raw = os.getenv("LLM_CHAIN", "").strip()
    if raw:
        out = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            prov, _, mdl = item.partition(":")
            out.append((prov.strip(), mdl.strip()))
        return out
    # Dedupe on the MODEL name, not the (provider, model) pair: the same model
    # reached through a different provider entry is still the same daily budget,
    # and listing it twice just wastes a failed request rediscovering that.
    first = (PROVIDER, MODEL)
    out, seen = [], set()
    for entry in [first] + _DEFAULT_CHAIN:
        if entry[0] == "stub" or entry[1] in seen:
            continue
        seen.add(entry[1])
        out.append(entry)
    return out


def _has_credentials(provider: str) -> bool:
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if provider == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return provider == "stub" or provider == "ollama"


def _is_exhaustion(err: Exception) -> bool:
    t = str(err)
    return ("DAILY token budget" in t or "(TPD)" in t
            or "per day" in t.lower() or "RESOURCE_EXHAUSTED" in t
            or "quota" in t.lower())


def _is_unusable_model(err: Exception) -> bool:
    """A model this account/API version cannot serve.

    Measured the hard way: the chain fell through to a Gemini model that 404s
    on the SDK's default API version, and because a 404 is not 'exhaustion' it
    re-raised and killed 12 consecutive scenarios. An entry that cannot work is
    a reason to try the next one, not to abandon the run.
    """
    t = str(err)
    low = t.lower()
    return ("NOT_FOUND" in t or "is not found for API version" in t
            or ("404" in t and "model" in low)
            or "no longer available" in t
            or "does not exist" in low
            # Billing dead ends: a key that authenticates but cannot be used.
            # Measured across three Gemini keys: one project with 0 RPM quota,
            # one on free tier at 20 requests/day, and one paid project with
            # "prepayment credits are depleted" (HTTP 402). None of these get
            # better by waiting, so the chain should move on rather than abort.
            or "402" in t
            or "prepayment credits" in low
            or "billing" in low
            or "PERMISSION_DENIED" in t)


def complete(messages, tools=None, temperature=0.2) -> Reply:
    global PROVIDER, MODEL

    if PROVIDER == "stub":
        return _PROVIDERS["stub"](_compact(messages), tools, temperature)

    chain = [e for e in _chain() if _has_credentials(e[0]) and e not in _EXHAUSTED]
    if not chain:
        _EXHAUSTED.clear()  # a new day may have rolled; give them one more go
        chain = [e for e in _chain() if _has_credentials(e[0])]
    if not chain:
        raise LLMError("No usable provider. Set GROQ_API_KEY or GOOGLE_API_KEY in .env")

    errors = []
    for provider, model in chain:
        fn = _PROVIDERS.get(provider)
        if fn is None:
            continue
        prev = (PROVIDER, MODEL)
        PROVIDER, MODEL = provider, model
        t0 = time.time()
        try:
            reply = fn(_compact(messages), tools, temperature)
            log.debug("reply from %s:%s: %d tool_call(s)%s", provider, model,
                      len(reply.tool_calls),
                      f", {len(reply.text)} chars text" if reply.text else ", no text")
            return reply
        except Exception as e:  # noqa: BLE001
            PROVIDER, MODEL = prev
            if _is_exhaustion(e) or _is_unusable_model(e):
                _EXHAUSTED.add((provider, model))
                why = ("daily budget spent" if _is_exhaustion(e)
                       else "model unavailable to this account")
                errors.append(f"{provider}:{model} ({why})")
                log.warning("%s:%s unusable (%s); falling back to the next model",
                            provider, model, why)
                if _ON_WAIT:
                    try:
                        _ON_WAIT(0, 0, 0)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            log.error("provider %s:%s failed after %.1fs: %s: %s", provider, model,
                      time.time() - t0, type(e).__name__, str(e)[:300])
            raise

    raise LLMError(
        "Every configured model has spent its free-tier daily budget: "
        + "; ".join(errors)
        + ". Budgets roll over about 24h after first use. Add another provider key to .env "
          "(GOOGLE_API_KEY from https://aistudio.google.com/apikey) or set LLM_CHAIN to "
          "models you still have."
    )


def describe() -> str:
    return f"{PROVIDER}:{MODEL}"
