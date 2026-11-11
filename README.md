# AI Lawyer — Indian consumer disputes

A conversational agent for one narrow area of Indian law: **consumer disputes under the
Consumer Protection Act, 2019** — defective goods, deficiency in service, refunds that
never arrived, unfair trade practices.

It is a **tool-calling agent**, not a RAG pipeline with a chat skin. Retrieval is two of
its **twelve** tools; the rest reason over the facts themselves. All retrieval, embedding and indexing run **entirely on your machine**;
only text generation calls an API, and that is swappable with one environment variable
(including a fully offline Ollama path).

---

## Quick start

```bash
uv venv --python 3.12
uv pip install -r pyproject.toml

cp .env.example .env        # add a free Gemini key from aistudio.google.com/apikey

.venv/bin/python -I scripts/00_probe_sources.py   # are the sources alive?
.venv/bin/python -I scripts/01_statutes.py        # statute corpus  (~5s)
.venv/bin/python -I scripts/02_fetch_cases.py     # case corpus     (~8 min, polite crawl)
.venv/bin/python -I scripts/03_index.py           # local indexes   (~2 min)

```

### Two interfaces

```bash
.venv/bin/streamlit run app.py          # browser UI  → http://localhost:8501
.venv/bin/python -I chat.py             # terminal REPL (fastest for spot-checks)
```

Both show the **grounding**, not just the answer: which tools the agent chose, which
sections and judgments it actually retrieved, and anything the citation validator flagged.
In the browser that is the expander under each reply; in the terminal it is printed inline.

Terminal commands: `/file` (case file so far), `/tools` (tool trace), `/reset`, `/quit`.

Try these:

| Input | What to watch for |
|---|---|
| `I paid Rs 18,000 for a phone on 10 March 2025, screen died in May, shop refuses repair` | District Commission, within time, cites real sections |
| `they are not refunding me` | Asks questions instead of inventing facts |
| `I paid Rs 80,00,000 to a builder in 2024 for a flat, no possession yet` | Forum shifts to **State** — proves it is computed, not recited |
| `In Feb 2021 I paid Rs 30,000 for a laptop never delivered` | Says the 2-year limit has expired |
| `my husband hit me, what sections apply` | Refuses and escalates — out of domain |
| `the company says I installed it wrong, help me argue` | Role-plays the other side |

---

## Architecture

```
Streamlit chat
      │
      ▼
AGENT LOOP (function calling, max 8 tool calls/turn)   src/agent.py
      │
      │  RETRIEVAL (2)
      ├── search_statute ──────── hybrid RAG over CPA 2019     ┐ src/retrieve.py
      ├── search_cases ────────── hybrid RAG over 300 rulings  ┘
      │  REASONING OVER THE FACTS (5) — deterministic, src/analysis.py
      ├── build_timeline ──────── chronology + gaps + contradictions
      ├── compute_claim_value ─── consideration / compensation / interest
      ├── evidence_checklist ──── what proof is held vs missing
      ├── assess_strength ─────── helps / hurts / unknown (never a prediction)
      ├── plan_next_steps ─────── sequenced actions against real deadlines
      │  STATE, OUTPUT, SAFETY (5)
      ├── update_case_file ────── structured state, reports gaps  src/state.py
      ├── check_forum_and_limitation ── PURE PYTHON, cited        src/rules.py
      ├── draft_document ──────── notice / complaint / reply
      ├── opposing_counsel ────── role-plays the other side
      └── escalate_to_human ───── explicit, logged
      │
      ▼
CITATION VALIDATOR  → marks & lists anything unverifiable   src/validate.py
      │
      ▼
   answer
```

### Agent proposes, code disposes

| The LLM decides | Code decides |
|---|---|
| Which tool to call, and when | Which commission has jurisdiction |
| Whether it has enough facts to advise | Whether the 2-year limitation has run |
| How to phrase the answer | Whether a citation is real |
| What to draft | When to escalate to a human |

The system prompt tells the model not to invent authorities. **Prompts are not a control.**
`src/validate.py` re-reads the finished answer and checks it against what was actually
retrieved and computed in the same turn.

---

## How hallucination is actually prevented

Six mechanical checks on every final message, logged to `raw_outputs/validation.jsonl`:

1. **Section citations** — every `Section N` must have been returned by `search_statute`
   this turn, else it is marked `⚠[UNVERIFIED]` and listed in a warning block.
2. **Case citations** — any `X vs Y` must match a judgment title retrieved this turn.
3. **Forum consistency** — if the prose names a different commission than
   `check_forum_and_limitation` computed, it is flagged.
4. **Stale pecuniary limit** — see below.
5. **Unsourced case narrative** — language asserting what a tribunal *did* ("the Commission
   held", "Case A", "a similar case") is flagged whenever `search_cases` returned nothing that
   turn. Added after a live run produced two entirely invented decisions labelled "Case A" and
   "Case B", which check 2 could not see because they carried no party names.
6. **Misquoted amount** — an amount in the answer that differs from the recorded figure by
   about 10x, the classic lakh/crore transcription error.

Nothing is silently deleted. The user sees exactly which claims could not be stood behind.

### The bug this system is built around

Section 34 of the Consumer Protection Act, 2019 says the District Commission handles claims
up to **one crore rupees**. That figure was **superseded** by the Consumer Protection
(Jurisdiction of the District Commission, the State Commission and the National Commission)
Rules, 2021, notified **30.12.2021**, which cut it to **fifty lakh rupees**.

A pure RAG system retrieving s.34 would confidently send a user with a ₹70 lakh claim to the
wrong forum. That is why jurisdiction is computed in `src/rules.py` from the amending
instrument, why the rule result overrides retrieved text, and why check 4 exists.

Verified against *Gorantla Geosynthetics Pvt Ltd v. Akshaya Signature Homes* (NCDRC,
23.08.2023), which records the reduction; evidence kept in `data/raw/`.

---

## Data

| Corpus | Size | Source | Role |
|---|---|---|---|
| Consumer Protection Act, 2019 | 107 sections, 189 chunks | India Code → vendored (see NOTICE.md) | **Primary.** What the law says. |
| Consumer commission rulings | 300 judgments, 7,527 chunks | Indian Kanoon, `doctypes:ncdrc` | **Secondary.** How disputes were actually decided. |

Index total: **36 MB** — small enough to commit, which is the real constraint (the
deliverable is a link someone can open, not a 2 GB download).

Forum split: 239 State Commission, 61 National Commission.

### Why not the AWS Supreme Court corpus

Measured, not assumed: **1 consumer matter in 856 Supreme Court judgments for 2023.**
Consumer disputes are decided by consumer commissions. Indexing the SC corpus would have
built a retriever over documents that cannot answer this domain's questions.

---

## Retrieval

- **Two indexes, never mixed.** "What does the law say" and "how was this decided" are
  different questions; mixing lets a judgment's paraphrase of a section outrank the section.
- **Dense (`bge-small-en-v1.5`, 384-dim) + BM25, fused with Reciprocal Rank Fusion.** RRF
  needs no weights — any weighting here would be invented rather than measured. BM25 is what
  catches `Section 69`, `Rs 50,00,000` and case names.
- **numpy brute force, not FAISS.** At 7.7k vectors a full dot product is ~5 ms. FAISS would
  be a dependency earning nothing. Swap point is `_dense_scores`.
- **Statute sub-chunking with identity preserved.** Long sections are split, but every
  sub-chunk carries its section number and `search_statute` collapses them back to one hit
  per section, so a citation built from a fragment is still correct.
- **Query expansion.** The Act's headings are opaque: the provision listing refund /
  replacement / compensation is called *"Findings of District Commission"* (s.39) and never
  uses the word "relief", while ss.71–73 (*"Enforcement of orders"*, *"Appeal against order"*)
  absorb any query containing "order". Nine hand-written rules bridge lay wording to
  statutory vocabulary. **This is a known limitation** — it is not a general solution, and a
  query outside those patterns gets no help.

---

## Swapping the model

Set two lines in `.env`. Nothing else in the codebase changes.

```bash
LLM_PROVIDER=groq       LLM_MODEL=openai/gpt-oss-120b                     # default, free tier
LLM_PROVIDER=gemini     LLM_MODEL=gemini-2.0-flash                        # free tier
LLM_PROVIDER=ollama     LLM_MODEL=llama3.1:8b                             # fully offline, no key
LLM_PROVIDER=openrouter LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
LLM_PROVIDER=stub                                                          # scripted, for tests
```

`src/llm.py` is the entire swap surface — ~40 lines per provider, and Groq and OpenRouter
share one OpenAI-compatible implementation. Tool schemas are declared once,
provider-neutral, in `src/tools.py`.

## What it covers, and what it does not

**Covers:** goods and services bought for personal use; defects, deficiency in service,
refunds, unfair trade practices; which commission to approach; whether you are in time;
what the Act entitles you to; how comparable disputes were decided; a drafted legal notice,
complaint or reply; rehearsing the other side's arguments.

**Does not cover:** criminal, family, property, employment, tax, or anything else — it
refuses and escalates rather than improvising. State-specific rules. Live case status.
Anything after the corpus snapshot date.

**Deliberately not stated:** filing fees. They are set by the Consumer Protection (Consumer
Disputes Redressal Commissions) Rules, 2020, which this prototype has not verified against
source, so `check_forum_and_limitation` returns `fee: None` and the model is instructed never
to state one. A plausible wrong number is worse than an admitted gap.

**This is information, not legal advice.** Every substantive answer points to the District
Legal Services Authority (free legal aid), the National Consumer Helpline (1915) and
e-Daakhil (the official filing portal).

---

## What makes it an agent, not a retriever

**Progressive tool disclosure.** While facts are still being gathered the model sees 5 core
tools (~800 tokens of schema); once the case file is complete all 12 appear (~1,350). On an
8,000 tokens/minute free tier that is the difference between a 10-second reply and a 60-second
one.

**Corrections.** Users revise themselves ("actually it was ₹1,80,000"). The newest statement
wins, the change is recorded in `case_file.revisions`, and anything derived from it is
recomputed. Visible in both interfaces.

**A visible plan.** Facts → law → forum → assess → act, derived from state rather than asked
of the model, so it costs no tokens and cannot be hallucinated.

**Persistence.** Matters save to `data/matters/` and reload across sessions.

**Five code-enforced guards.** Each was added only after a live run proved the model skipped
something: mandatory jurisdiction computation, mandatory statute retrieval before advice,
immediate notice when limitation has expired, forced role-play when rehearsal is requested,
and amount reconciliation against the user's own words.

## Rate limits and the logs

Groq's free tier meters **per model**: 8,000 tokens/minute *and* 200,000 tokens/day. Three
suite runs exhausted the daily budget for `gpt-oss-120b`, which presented as the UI hanging.
Mitigations, all in `src/llm.py`:

* **Proactive pacing** — a rolling-window token bucket waits *before* overdrawing, because
  overdrawing earns escalating server-side penalties (measured: 111s, 149s, 650s).
* **45-second wait cap** — an 11-minute silent stall is worse than an honest error. The error
  names which budget ran out, since per-minute means "wait" and per-day means "switch model".
* **History compaction** — older tool results are abbreviated once acted on.

Everything is logged to `logs/ai-lawyer.log` (Streamlit swallows stdout from callbacks). The
UI reports each tool as it runs and counts down during rate-limit waits, so it never looks dead.

```bash
LOG_LEVEL=DEBUG .venv/bin/streamlit run app.py
tail -f logs/ai-lawyer.log
```

## Known limitations

1. **Query expansion is 9 hand-written rules**, tuned to the paths this prototype claims.
2. **300 judgments is a thin corpus.** Unusual fact patterns will retrieve weak analogues;
   the agent is instructed to say so, but retrieval confidence is not numerically scored.
3. **No reranker.** A cross-encoder would improve ordering at ~200 MB and added latency; at
   this corpus size it was not worth it. Deliberate omission, not an oversight.
4. **State Commission decisions dominate (239/300).** They are persuasive, not binding across
   states. The agent describes what commissions did; it does not assert precedent.
5. **No OCR.** Judgments served only as scanned PDFs are absent.
6. **India Code is unreachable** (HTTP 504 at build time), so the Act text is vendored. The
   fetch script tries the authoritative source first on every run.
7. **8 of 300 judgments were truncated** at 150k chars to keep the index committable.
8. **Interest in `compute_claim_value` is indicative arithmetic**, not an entitlement — the
   Commission decides whether to award it and at what rate.
9. **`evidence_checklist` is keyword matching** over what the user mentioned, so it reports
   "not mentioned" rather than "not held", and the agent is told to ask rather than assert.
10. **Self-critique is code, not an LLM pass.** The validator catches misquoted amounts and
    unsourced claims; it does not judge whether the advice is *good*.

See `NOTICE.md` for attribution and crawl conduct.
