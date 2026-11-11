#!/usr/bin/env python3
"""Phase 0: check every external source this project touches.

Run this before a demo. It tells you in ~30s which acquisition paths are
alive, so you never debug a dead government endpoint in front of an audience.
Stdlib only -- works before `uv pip install`.
"""
import concurrent.futures as cf
import sys
import time
import urllib.request

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

SOURCES = [
    ("India Code (root)",        "https://www.indiacode.nic.in/", "statute"),
    ("India Code (CPA 2019)",    "https://www.indiacode.nic.in/handle/123456789/15256", "statute"),
    ("SC judgments S3",          "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/?list-type=2&max-keys=1", "cases"),
    ("NCDRC",                    "https://ncdrc.nic.in/", "cases (optional)"),
    ("Dept of Consumer Affairs", "https://consumeraffairs.nic.in/", "statute (fallback)"),
    ("Gemini API",               "https://generativelanguage.googleapis.com/", "llm"),
    ("HuggingFace (models)",     "https://huggingface.co/BAAI/bge-small-en-v1.5", "embeddings"),
]


def probe(item, timeout=20):
    name, url, role = item
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return name, role, r.status, time.time() - t0, ""
    except urllib.error.HTTPError as e:
        # A 4xx still proves the host is alive and speaking HTTP.
        return name, role, e.code, time.time() - t0, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - we want every failure mode reported
        return name, role, 0, time.time() - t0, type(e).__name__


def main():
    print(f"{'SOURCE':<28} {'ROLE':<22} {'STATUS':<8} {'TIME':<8} NOTE")
    print("-" * 84)
    with cf.ThreadPoolExecutor(max_workers=len(SOURCES)) as ex:
        rows = list(ex.map(probe, SOURCES))
    for name, role, status, dt, err in rows:
        # Any HTTP response proves the host is alive; only connection
        # failures (status 0) mean the acquisition path is actually broken.
        mark = "OK" if status else "DEAD"
        print(f"{name:<28} {role:<22} {mark:<8} {dt:>5.1f}s   {err or status}")
    print("-" * 84)
    dead = [r[0] for r in rows if not r[2]]
    if dead:
        print(f"\n{len(dead)} source(s) unreachable: {', '.join(dead)}")
        print("Expected for India Code deep links. The statute corpus is vendored into")
        print("data/raw/ and committed, so the build never depends on them at runtime.")
    else:
        print("\nAll sources reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
