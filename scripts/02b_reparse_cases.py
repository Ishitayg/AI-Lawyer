#!/usr/bin/env python3
"""Re-parse the cached Indian Kanoon HTML without re-fetching anything.

Used after a parser fix. The cache in data/raw/ik_cache/ is the source of
truth, so correcting metadata costs no further requests to the source site.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util

spec = importlib.util.spec_from_file_location("fetchmod", ROOT / "scripts" / "02_fetch_cases.py")
fetchmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetchmod)

CACHE = ROOT / "data" / "raw" / "ik_cache"
CASES = ROOT / "data" / "cases"


def main():
    files = sorted(CACHE.glob("*.html"))
    print(f"Re-parsing {len(files)} cached documents (no network)")
    manifest, seen, courts, trunc = [], set(), {}, 0
    for f in files:
        docid = f.stem
        rec = fetchmod.parse_doc(docid, f.read_text(encoding="utf-8", errors="ignore"))
        if not rec or rec["sha1"] in seen:
            continue
        seen.add(rec["sha1"])
        trunc += rec.get("truncated", False)
        courts[rec["court"]] = courts.get(rec["court"], 0) + 1
        (CASES / f"{docid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        manifest.append({k: rec[k] for k in ("doc_id", "title", "date", "url", "court")})
    (CASES / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    total = sum(len(json.loads((CASES / f"{m['doc_id']}.json").read_text(encoding='utf-8'))["text"]) for m in manifest)
    print(f"  {len(manifest)} judgments, {total:,} chars, {trunc} truncated")
    print("  forum distribution:")
    for c, n in sorted(courts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>4}  {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
