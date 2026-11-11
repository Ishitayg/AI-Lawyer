#!/usr/bin/env python3
"""Phase 1: build the statute corpus (the PRIMARY grounding source).

Acquisition order:
  1. India Code (authoritative) -- currently returns HTTP 504, origin is down.
  2. Vendored copy in data/raw/ -- committed to the repo so a fresh clone
     always builds without network access.

Output: data/statutes/cpa2019.json  (one record per section)

Attribution: the vendored copy of the Act text was retrieved from Indian Kanoon
(indiankanoon.org/doc/48103131/), which publishes the Akoma Ntoso markup of
Act 35 of 2019. See NOTICE.md.
"""
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "statutes" / "cpa2019.json"
VENDORED = RAW / "cpa2019_indiankanoon.html"

INDIA_CODE = "https://www.indiacode.nic.in/handle/123456789/15256"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

ACT = {
    "act": "Consumer Protection Act, 2019",
    "act_no": "35 of 2019",
    "year": 2019,
    "published": "2019-08-09",
}

# Heading sits inline at the start of the section body: "34. Jurisdiction of ..."
HEADING_RE = re.compile(r"^\s*(\d+[A-Z]?)\s*\.\s*(.{3,160}?\.)\s", re.S)


def try_india_code(timeout=45):
    """Authoritative source. Returns HTML or None."""
    try:
        r = requests.get(INDIA_CODE, headers=UA, timeout=timeout)
        if r.status_code == 200 and len(r.content) > 50_000:
            return r.text
        print(f"  India Code returned HTTP {r.status_code} ({len(r.content)}B) -- falling back")
    except Exception as e:  # noqa: BLE001
        print(f"  India Code unreachable ({type(e).__name__}) -- falling back")
    return None


def normalise(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\s*\n\s*", "\n", s).strip()


def parse_sections(html: str):
    soup = BeautifulSoup(html, "lxml")
    body = soup.find(class_="akn-body") or soup
    out = []
    for node in body.select(".akn-section"):
        sid = node.get("id") or ""
        m = re.match(r"section_(\d+[A-Z]?)$", sid)
        text = normalise(node.get_text(" ", strip=True))
        if not text:
            continue
        hm = HEADING_RE.match(text)
        if m:
            section_no = m.group(1)
        elif hm:
            section_no = hm.group(1)
        else:
            continue
        heading = hm.group(2).rstrip(".").strip() if hm else ""
        out.append(
            {
                **ACT,
                "section_no": section_no,
                "heading": heading,
                "text": text,
                "citation": f"Section {section_no}, Consumer Protection Act, 2019",
                "source": "indiankanoon.org (Akoma Ntoso markup of Act 35 of 2019)",
                "source_url": "https://indiankanoon.org/doc/48103131/",
            }
        )
    # Deduplicate on section number, keeping the longest extraction.
    best = {}
    for rec in out:
        k = rec["section_no"]
        if k not in best or len(rec["text"]) > len(best[k]["text"]):
            best[k] = rec
    return sorted(best.values(), key=lambda r: (int(re.sub(r"\D", "", r["section_no"]) or 0), r["section_no"]))


def main():
    print("Phase 1: statute corpus")
    print("  trying India Code (authoritative)...")
    html = try_india_code()
    if html is None:
        if not VENDORED.exists():
            print(f"FATAL: no vendored copy at {VENDORED}", file=sys.stderr)
            return 1
        html = VENDORED.read_text(encoding="utf-8", errors="ignore")
        print(f"  using vendored copy: {VENDORED.relative_to(ROOT)}")

    secs = parse_sections(html)
    if len(secs) < 90:
        print(f"FATAL: parsed only {len(secs)} sections; expected ~107", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(secs, indent=1, ensure_ascii=False), encoding="utf-8")

    chars = sum(len(s["text"]) for s in secs)
    print(f"  parsed {len(secs)} sections, {chars:,} chars -> {OUT.relative_to(ROOT)}")
    print("\n  sanity check (sections that must exist for this domain):")
    by_no = {s["section_no"]: s for s in secs}
    for n, why in [("2", "definitions"), ("34", "District Commission jurisdiction"),
                   ("35", "manner of complaint"), ("38", "procedure"), ("39", "reliefs"),
                   ("69", "limitation period")]:
        s = by_no.get(n)
        print(f"    s.{n:<3} {'OK  ' if s else 'MISS'} {(s['heading'] if s else '') or why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
