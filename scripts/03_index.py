#!/usr/bin/env python3
"""Phase 3: build the local vector + BM25 indexes.

Runs entirely offline after the embedding model is cached. Prints the on-disk
size because the index must stay small enough to commit to the repo -- the
deliverable is "a prototype link anyone can open", not a 2GB download.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    print("Phase 3: building local indexes")

    sec_path = ROOT / "data" / "statutes" / "cpa2019.json"
    if not sec_path.exists():
        print("FATAL: run scripts/01_statutes.py first", file=sys.stderr)
        return 1
    sections = json.loads(sec_path.read_text(encoding="utf-8"))
    s_chunks = retrieve.chunk_statutes(sections)
    n, nbytes = retrieve.build("statutes", s_chunks)
    print(f"  statutes: {n} chunks over {len(sections)} sections, {nbytes/1e6:.1f} MB vectors")

    case_dir = ROOT / "data" / "cases"
    files = sorted(p for p in case_dir.glob("*.json") if p.name != "manifest.json")
    if not files:
        print("  cases: none found -- run scripts/02_fetch_cases.py (statute-only mode still works)")
    else:
        cases = [json.loads(p.read_text(encoding="utf-8")) for p in files]
        c_chunks = retrieve.chunk_cases(cases)
        n, nbytes = retrieve.build("cases", c_chunks)
        print(f"  cases: {len(cases)} judgments -> {n} chunks, {nbytes/1e6:.1f} MB vectors")

    total = sum(p.stat().st_size for p in (ROOT / "data" / "index").glob("*"))
    print(f"\n  index dir total: {total/1e6:.1f} MB "
          f"({'OK to commit' if total < 100e6 else 'TOO BIG -- reduce corpus'})")
    print(f"  stats: {retrieve.index_stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
