#!/usr/bin/env python3
"""Phase 2: build the case-law corpus (the SECONDARY, illustrative source).

Why not the AWS Supreme Court bucket?
  Measured: 1 consumer matter in 856 SC judgments for 2023 (0.1%). Consumer
  disputes are decided by the NCDRC / State / District Commissions, not the
  Supreme Court. Indexing the SC corpus would have produced a retriever over
  documents that cannot answer this domain's questions.

Source: Indian Kanoon, filtered to doctypes:ncdrc.
Politeness: honours robots.txt (incl. its per-document denylist), 1 req/sec,
            descriptive User-Agent, on-disk cache so each doc is fetched once,
            hard cap on total documents. Attribution in NOTICE.md.

Output: data/cases/<docid>.json  +  data/cases/manifest.json
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "cases"
CACHE = ROOT / "data" / "raw" / "ik_cache"
BASE = "https://indiankanoon.org"
UA = {"User-Agent": "ai-lawyer-coursework-prototype/0.1 (student assignment; low-volume; respects robots.txt)"}
DELAY = 1.0

# Topics chosen to span the dispute types this assistant claims to cover.
QUERIES = [
    "refund deficiency in service",
    "defective goods replacement refund",
    "online shopping order not delivered refund",
    "mobile phone manufacturing defect warranty",
    "electronic appliance repair deficiency",
    "flight cancelled refund deficiency",
    "hotel booking refund deficiency in service",
    "builder possession delay refund consumer",
    "insurance claim repudiation deficiency",
    "bank unauthorised transaction deficiency service",
    "unfair trade practice misleading advertisement",
    "e-commerce platform liability consumer",
    "courier lost consignment compensation consumer",
    "service centre failed to repair compensation",
    "excess charge overcharging consumer complaint",
    "limitation two years consumer complaint condonation",
    "pecuniary jurisdiction district commission consumer",
    "mental agony compensation litigation cost consumer",
]


def load_robots_denylist(session):
    """Indian Kanoon lists individually de-indexed documents in robots.txt.
    Those are usually takedowns/right-to-be-forgotten. We must not fetch them."""
    try:
        r = session.get(f"{BASE}/robots.txt", headers=UA, timeout=30)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: could not read robots.txt ({type(e).__name__}); aborting to stay safe")
        sys.exit(1)
    denied = set(re.findall(r"^Disallow:\s*/docfragment/+(\d+)", r.text, re.M))
    denied |= set(re.findall(r"^Disallow:\s*/doc/+(\d+)", r.text, re.M))
    print(f"  robots.txt: {len(denied)} individually de-indexed documents will be skipped")
    return denied


# Only these forums belong in a consumer-law corpus. Enforced at search time
# AND again after parsing, because the first version of this script scraped
# every /doc/ link on the results page -- including citation and sidebar links
# -- and silently pulled in High Courts, tax tribunals and even bare Acts.
# 85 of the first 299 documents were off-domain before this filter existed.
CONSUMER_FORUM_RE = re.compile(r"consumer disputes redressal", re.I)


def search(session, query, pages, denied):
    """Parse the result list proper -- never every link on the page."""
    ids = []
    for page in range(pages):
        url = f"{BASE}/search/?formInput={quote(query + ' doctypes:ncdrc')}&pagenum={page}"
        try:
            r = session.get(url, headers=UA, timeout=30)
            if r.status_code != 200:
                break
        except Exception:  # noqa: BLE001
            break
        soup = BeautifulSoup(r.text, "lxml")
        results = soup.find_all(class_="result")
        kept = 0
        for res in results:
            src = res.find(class_="docsource")
            forum = src.get_text(" ", strip=True) if src else ""
            if not CONSUMER_FORUM_RE.search(forum):
                continue  # off-domain result, skip
            link = res.find("a", href=True)
            m = re.search(r"/(?:doc|docfragment)/(\d+)/", link["href"] if link else "")
            if not m or m.group(1) in denied:
                continue
            ids.append(m.group(1))
            kept += 1
        time.sleep(DELAY)
        if not results:
            break
    return list(dict.fromkeys(ids))


def fetch_doc(session, docid, retries=2):
    """Network failures skip a document; they never abort the run. The cache
    makes a re-run resume from where it stopped."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{docid}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="ignore"), True
    for attempt in range(retries + 1):
        try:
            r = session.get(f"{BASE}/doc/{docid}/", headers=UA, timeout=45)
            time.sleep(DELAY)
            if r.status_code != 200:
                return None, False
            tmp = cached.with_suffix(".part")
            tmp.write_text(r.text, encoding="utf-8")
            tmp.replace(cached)  # atomic: never leave a half-written cache file
            return r.text, False
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                print(f"    skip {docid}: {type(e).__name__}")
                return None, False
            time.sleep(2 * (attempt + 1))
    return None, False


DATE_RE = re.compile(r"on\s+(\d{1,2}\s+\w+,\s+\d{4})\s*$")
# Indian Kanoon's doctypes:ncdrc filter returns State and District commission
# matters alongside National ones, so the forum must be read off each page
# rather than assumed. Getting this wrong would mislabel every citation.
MAX_CHARS = 150_000  # one outlier was 2.3M chars; cap keeps the index committable


def parse_doc(docid, html):
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else f"Document {docid}"
    body = soup.find(class_="judgments") or soup.find(class_="akn-body") or soup.find("div", {"id": "main"})
    text = (body or soup).get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text))
    if len(text) < 800:
        return None
    truncated = len(text) > MAX_CHARS
    if truncated:
        text = text[:MAX_CHARS]
    src = soup.find(class_="docsource_main")
    court = src.get_text(" ", strip=True) if src else ""
    if not CONSUMER_FORUM_RE.search(court):
        return None  # belt and braces: never index a non-consumer forum
    dm = DATE_RE.search(title)
    sections = sorted(set(re.findall(r"[Ss]ection\s+(\d+[A-Z]?)", text)))
    return {
        "doc_id": docid,
        "title": title,
        "court": court,
        "truncated": truncated,
        "date": dm.group(1) if dm else "",
        "url": f"{BASE}/doc/{docid}/",
        "sections_mentioned": sections[:40],
        "text": text,
        "source": "indiankanoon.org",
        "sha1": hashlib.sha1(text.encode()).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=300, help="hard cap on documents fetched")
    ap.add_argument("--pages-per-query", type=int, default=3)
    args = ap.parse_args()

    CASES.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    print("Phase 2: case corpus (NCDRC via Indian Kanoon)")
    denied = load_robots_denylist(session)

    print(f"  searching {len(QUERIES)} topics x {args.pages_per_query} pages...")
    ids = []
    for q in QUERIES:
        got = search(session, q, args.pages_per_query, denied)
        ids.extend(got)
        print(f"    {len(got):>3} ids  {q}")
    ids = list(dict.fromkeys(ids))[: args.max_docs]
    print(f"  {len(ids)} unique documents after dedupe + cap")

    manifest, seen_hashes, cached_n = [], set(), 0
    for i, docid in enumerate(ids, 1):
        html, was_cached = fetch_doc(session, docid)
        cached_n += was_cached
        if not html:
            continue
        rec = parse_doc(docid, html)
        if not rec or rec["sha1"] in seen_hashes:
            continue
        seen_hashes.add(rec["sha1"])
        tmp = CASES / f"{docid}.json.part"
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CASES / f"{docid}.json")  # atomic write
        manifest.append({k: rec[k] for k in ("doc_id", "title", "date", "url", "court")})
        if i % 25 == 0:
            print(f"    {i}/{len(ids)} fetched ({cached_n} from cache)")

    (CASES / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    total = sum(len(json.loads((CASES / f"{m['doc_id']}.json").read_text(encoding="utf-8"))["text"])
                for m in manifest)
    print(f"\n  {len(manifest)} judgments saved, {total:,} chars ({cached_n} served from cache)")
    print(f"  -> {CASES.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
