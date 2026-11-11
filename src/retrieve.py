"""Hybrid retrieval over two separate local indexes: statutes and cases.

Design notes
------------
* Two indexes, never one mixed blob. "What does the law say?" and "how have
  commissions decided facts like mine?" are different questions and mixing them
  lets a judgment's paraphrase of a section outrank the section itself.
* Statute chunk == one whole section. Legal provisions are not split; a
  half-section is worse than useless because provisos invert meaning.
* Dense + BM25 fused with Reciprocal Rank Fusion. RRF needs no weights, which
  matters because any weighting scheme here would be invented rather than
  measured. BM25 is what catches "section 69", "Rs 50,00,000" and case names.
* numpy brute force, not FAISS: at this corpus size a full dot product is ~5ms.
  FAISS would be a dependency earning nothing. Swap point is `_dense_scores`.
* Everything runs locally. No embedding API, no network at query time.
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CASE_CHUNK_CHARS = 1400
CASE_CHUNK_OVERLAP = 200
# Long sections are sub-chunked. Measured reason: s.39 ("Findings of District
# Commission") is 3,198 chars, is the provision that actually lists refund /
# replacement / compensation, and never uses the word "relief". As one vector it
# was diluted below ss.71-73 for the query "what reliefs can the commission
# order". Every sub-chunk still carries its section header, so a retrieved
# fragment is never anonymous and the citation stays correct.
STATUTE_CHUNK_CHARS = 1100
STATUTE_CHUNK_OVERLAP = 150


@dataclass
class Hit:
    kind: str          # "statute" | "case"
    ref: str           # "Section 35" | doc_id
    title: str
    text: str
    url: str
    score: float
    meta: dict


# --------------------------------------------------------------------- model
@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts, query=False):
    # bge asks for an instruction prefix on queries only.
    if query:
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    v = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    return np.asarray(v, dtype=np.float32)


# -------------------------------------------------------------------- chunking
def chunk_statutes(sections):
    """One chunk per section, sub-chunked only when a section is long.

    A sub-chunk is never anonymous: it keeps the section number and heading, so
    a citation built from it is still correct, and search_statute collapses
    sub-chunks back to one hit per section before the agent sees them.
    """
    out = []
    for s in sections:
        head = f"Section {s['section_no']}. {s['heading']} — Consumer Protection Act, 2019"
        body = s["text"]
        pieces = ([body] if len(body) <= STATUTE_CHUNK_CHARS
                  else _split_chars(body, STATUTE_CHUNK_CHARS, STATUTE_CHUNK_OVERLAP))
        for i, piece in enumerate(pieces):
            out.append({
                "kind": "statute",
                "ref": f"Section {s['section_no']}",
                "title": head,
                "text": piece if i == 0 else f"{head}\n(continued) {piece}",
                "url": s["source_url"],
                "meta": {"section_no": s["section_no"], "heading": s["heading"],
                         "citation": s["citation"], "part": i, "parts": len(pieces),
                         "full_text": body if len(pieces) > 1 else None},
            })
    return out


def _split_chars(text, size, overlap):
    """Split on sentence-ish boundaries, falling back to hard cuts."""
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("; "), window.rfind(") "))
            if cut > size * 0.5:
                end = start + cut + 2
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if len(c) > 80]


def _split(text, size, overlap):
    paras, buf, out = re.split(r"\n{2,}", text), "", []
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf += ("\n\n" if buf else "") + p
        else:
            if buf:
                out.append(buf)
            buf = (buf[-overlap:] + "\n\n" + p) if overlap and buf else p
            while len(buf) > size:
                out.append(buf[:size])
                buf = buf[size - overlap:]
    if buf.strip():
        out.append(buf)
    return [c for c in out if len(c.strip()) > 200]


def chunk_cases(cases):
    out = []
    for c in cases:
        for i, piece in enumerate(_split(c["text"], CASE_CHUNK_CHARS, CASE_CHUNK_OVERLAP)):
            out.append({
                "kind": "case",
                "ref": c["doc_id"],
                "title": c["title"],
                "text": piece,
                "url": c["url"],
                "meta": {"doc_id": c["doc_id"], "date": c.get("date", ""), "court": c.get("court", ""),
                         "chunk": i, "sections_mentioned": c.get("sections_mentioned", [])},
            })
    return out


# ----------------------------------------------------------------------- index
def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def build(name, chunks):
    from rank_bm25 import BM25Okapi

    INDEX.mkdir(parents=True, exist_ok=True)
    texts = [f"{c['title']}\n{c['text']}" for c in chunks]
    vecs = embed(texts)
    np.save(INDEX / f"{name}.npy", vecs)
    (INDEX / f"{name}.chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    with open(INDEX / f"{name}.bm25.pkl", "wb") as f:
        pickle.dump(BM25Okapi([_tok(t) for t in texts]), f)
    return len(chunks), vecs.nbytes


@lru_cache(maxsize=4)
def _load(name):
    vecs = np.load(INDEX / f"{name}.npy")
    chunks = json.loads((INDEX / f"{name}.chunks.json").read_text(encoding="utf-8"))
    with open(INDEX / f"{name}.bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    return vecs, chunks, bm25


def _dense_scores(vecs, query):
    return vecs @ embed([query], query=True)[0]


def search(name, query, top_k=5, rrf_k=60, pool=40):
    """Hybrid dense+BM25 with Reciprocal Rank Fusion."""
    try:
        vecs, chunks, bm25 = _load(name)
    except FileNotFoundError:
        return []
    dense = _dense_scores(vecs, query)
    sparse = np.asarray(bm25.get_scores(_tok(query)), dtype=np.float32)

    d_rank = {i: r for r, i in enumerate(np.argsort(-dense)[:pool])}
    s_rank = {i: r for r, i in enumerate(np.argsort(-sparse)[:pool])}
    fused = {}
    for i in set(d_rank) | set(s_rank):
        fused[i] = (1.0 / (rrf_k + d_rank.get(i, pool))) + (1.0 / (rrf_k + s_rank.get(i, pool)))

    hits = []
    for i in sorted(fused, key=fused.get, reverse=True)[:top_k]:
        c = chunks[i]
        hits.append(Hit(c["kind"], c["ref"], c["title"], c["text"], c["url"], float(fused[i]), c["meta"]))
    return hits


# Hand-written bridge between how people describe a problem and how the Act
# titles its provisions. Needed because the Act's headings are opaque: the
# provision that lists refund / replacement / compensation is called "Findings
# of District Commission" (s.39) and never uses the word "relief", while
# ss.71-73 ("Enforcement of orders", "Appeal against order") soak up any query
# containing the word "order".
#
# LIMITATION, stated plainly: this is 9 hand-written rules covering the paths
# this prototype claims to handle. It is not a general solution to legal
# vocabulary mismatch, and a query outside these patterns gets no help. A
# production system would learn these mappings from query logs or build a
# proper legal thesaurus.
QUERY_EXPANSIONS = {
    r"relief|remedy|remedies|what can i get|compensat|refund|replace|return the price":
        "findings of District Commission return price replace goods compensation for loss or injury",
    r"time limit|deadline|too late|how long|limitation|expired":
        "limitation period two years from the date on which the cause of action has arisen",
    r"which (court|forum|commission)|where (do i|to) file|jurisdiction|pecuniary":
        "jurisdiction of District Commission value of goods or services paid as consideration",
    r"how (do i|to) (file|complain)|who can (file|complain)|procedure":
        "manner in which complaint shall be made procedure on admission of complaint",
    r"who is a consumer|am i a consumer|definition":
        "definitions consumer deficiency defect unfair trade practice service",
    r"appeal|challenge the order|not happy with the order":
        "appeal against order of District Commission State Commission",
    r"they (did not|didn't) (comply|pay)|not following the order|enforce":
        "enforcement of orders penalty for noncompliance of order",
    r"mediation|settle|settlement":
        "mediation reference to mediation settlement through mediation",
    r"product liability|manufacturer liable|defective product harm":
        "product liability action manufacturer product seller service provider",
}


def expand_query(query: str) -> str:
    """Append statutory vocabulary when a query uses lay wording."""
    extra = [v for pat, v in QUERY_EXPANSIONS.items() if re.search(pat, query, re.I)]
    return query + (" " + " ".join(extra) if extra else "")


def search_statute(query, top_k=5):
    """Collapse sub-chunks so the agent sees one hit per distinct section."""
    hits, seen, out = search("statutes", expand_query(query), top_k * 4), set(), []
    for h in hits:
        sec = h.meta["section_no"]
        if sec in seen:
            continue
        seen.add(sec)
        if h.meta.get("full_text"):
            h.text = h.meta["full_text"]  # hand the agent the whole provision
        out.append(h)
        if len(out) == top_k:
            break
    return out


def search_cases(query, top_k=4):
    # Collapse to one best chunk per judgment -- the agent wants distinct cases.
    hits, seen, out = search("cases", query, top_k * 4), set(), []
    for h in hits:
        if h.ref in seen:
            continue
        seen.add(h.ref)
        out.append(h)
        if len(out) == top_k:
            break
    return out


def index_stats():
    s = {}
    for name in ("statutes", "cases"):
        try:
            vecs, chunks, _ = _load(name)
            s[name] = {"chunks": len(chunks), "dim": int(vecs.shape[1]), "mb": round(vecs.nbytes / 1e6, 1)}
        except FileNotFoundError:
            s[name] = None
    return s
