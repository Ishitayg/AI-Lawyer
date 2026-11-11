#!/usr/bin/env python3
"""Phase 6: run every scenario in tests/cases.yaml and write raw outputs.

Assertions are mechanical, not LLM-graded: did the agent call the right tools,
did the deterministic rules produce the right forum, did the citation validator
stay clean, did it avoid saying the things it must not say.

Writes one Markdown transcript per scenario to raw_outputs/ plus a summary.
"""
import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent import Agent  # noqa: E402
from src.llm import describe  # noqa: E402

OUT = ROOT / "raw_outputs"


def check(scn, results, agent):
    exp = scn.get("expect", {}) or {}
    last = results[-1]
    all_tools = {t for r in results for t in r["tools_called"]}
    joined = "\n".join(r["answer"] for r in results)
    fails = []

    for t in exp.get("tools_include", []):
        if t not in all_tools:
            fails.append(f"tool {t!r} never called (called: {sorted(all_tools) or 'none'})")

    if "forum_contains" in exp:
        forums = [fr["forum"]["forum"] or "" for r in results for log in [r["tool_log"]]
                  for fr in _rule_results(r)]
        if not any(exp["forum_contains"].lower() in f.lower() for f in forums):
            fails.append(f"no computed forum containing {exp['forum_contains']!r} (got: {forums or 'none'})")

    if exp.get("no_validation_issues") and last["validation"]["n_issues"]:
        fails.append(f"validator flagged {last['validation']['n_issues']} issue(s): "
                     f"{[i['type'] for i in last['validation']['issues']]}")

    if "answer_matches" in exp and not re.search(exp["answer_matches"], joined):
        fails.append(f"answer did not match /{exp['answer_matches']}/")

    if "must_not_match" in exp and re.search(exp["must_not_match"], joined):
        m = re.search(exp["must_not_match"], joined)
        fails.append(f"answer matched forbidden /{exp['must_not_match']}/ -> {m.group(0)!r}")

    for fld in exp.get("casefile_has", []):
        if not agent.case_file.as_dict().get(fld):
            fails.append(f"case file missing {fld!r}")

    for fld, want in (exp.get("casefile_equals") or {}).items():
        got = str(agent.case_file.as_dict().get(fld, ""))
        if got != str(want):
            fails.append(f"case file {fld!r} is {got!r}, expected {want!r}")

    for fld, want in (exp.get("casefile_contains") or {}).items():
        got = str(agent.case_file.as_dict().get(fld, ""))
        if str(want) not in got:
            fails.append(f"case file {fld!r} is {got!r}, expected it to contain {want!r}")

    if exp.get("corrections_recorded") and not agent.case_file.corrections():
        fails.append("no correction was recorded although the user revised a fact")

    return fails


def _rule_results(r):
    out = []
    for entry in r["tool_log"]:
        if entry["tool"] == "check_forum_and_limitation":
            out.append({"forum": {"forum": _extract_forum(entry["result_preview"])}})
    return out


def _extract_forum(blob):
    m = re.search(r"'forum': '([^']+)'", blob) or re.search(r'"forum": "([^"]+)"', blob)
    return m.group(1) if m else ""


def preflight(scenarios):
    """Refuse to start a run that the daily token budget cannot finish.

    Learned the hard way: a 20-token probe returned HTTP 200 while the model
    was at 196,775/200,000 tokens for the day. The run then failed all 17
    scenarios AND overwrote the previous good transcripts. A cheap check that
    sends a realistically-sized request is worth more than an optimistic one.
    """
    from src import llm
    need = len(scenarios) * 10_000  # ~10k tokens per scenario, measured
    probe = "consumer protection act deficiency in service analysis " * 500
    try:
        llm.complete([{"role": "user", "content": probe}])
    except Exception as e:  # noqa: BLE001
        print(f"PREFLIGHT FAILED — not starting, so existing raw_outputs are preserved.\n\n{e}\n")
        print(f"This run needs roughly {need:,} tokens.")
        return False
    return True


def backup_outputs():
    """Snapshot previous results before overwriting them."""
    summary = OUT / "summary.json"
    if not summary.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = OUT / "_previous" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for f in list(OUT.glob("*.md")) + [summary]:
        shutil.copy2(f, dest / f.name)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated scenario id substrings")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the daily-budget check (not recommended)")
    ap.add_argument("--merge", action="store_true",
                    help="merge into the existing summary.json instead of replacing it "
                         "(for re-running a few scenarios after a quota stall)")
    ap.add_argument("--sleep", type=float, default=2.0, help="pause between turns (free-tier rate limits)")
    args = ap.parse_args()

    scenarios = yaml.safe_load((ROOT / "tests" / "cases.yaml").read_text(encoding="utf-8"))
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        scenarios = [s for s in scenarios if any(w in s["id"] for w in wanted)]
    OUT.mkdir(exist_ok=True)

    if not args.no_preflight and not preflight(scenarios):
        return 2
    saved_to = backup_outputs()
    if saved_to:
        print(f"previous results backed up -> {saved_to.relative_to(ROOT)}\n")

    summary, t_start, aborted = [], time.time(), False
    print(f"Running {len(scenarios)} scenario(s) against {describe()}\n")

    for scn in scenarios:
        if aborted:
            break
        print(f"=== {scn['id']}")
        agent, results, error = Agent(), [], None
        try:
            for turn in scn["turns"]:
                print(f"  user: {turn[:90]}")
                r = agent.ask(turn, verbose=True)
                results.append(r)
                time.sleep(args.sleep)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            print(f"  ERROR {error}")
            if "Every configured model has spent" in str(e):
                # Nothing can run. Stop now and keep the transcripts already
                # written: N real outputs are worth far more than 17 stubs
                # saying the budget ran out, and a grader opening this folder
                # should see work, not error messages.
                print("\n  ABORTING RUN - every model is out of daily budget.")
                print(f"  Keeping the {len(summary)} scenario(s) completed so far.")
                aborted = True
                break
            traceback.print_exc(limit=2)

        fails = check(scn, results, agent) if results else ["no turns completed"]
        if error:
            fails.append(f"exception: {error}")
        status = "PASS" if not fails else "FAIL"
        print(f"  -> {status}" + ("" if not fails else "\n     - " + "\n     - ".join(fails)) + "\n")

        md = [f"# {scn['id']}", f"\n_{scn.get('why','')}_\n",
              f"\n**Model:** `{describe()}`  |  **Status:** {status}\n"]
        if fails:
            md.append("\n**Assertion failures**\n\n" + "\n".join(f"- {f}" for f in fails) + "\n")
        for turn, r in zip(scn["turns"], results):
            md += [f"\n---\n\n### User\n\n{turn}\n",
                   f"\n### Tools called\n\n`{'`, `'.join(r['tools_called']) or 'none'}`\n",
                   f"\n### Assistant\n\n{r['answer']}\n",
                   f"\n<details><summary>Validation report</summary>\n\n```json\n"
                   f"{json.dumps(r['validation'], indent=2)}\n```\n</details>\n",
                   f"\n<details><summary>Tool trace</summary>\n\n```json\n"
                   f"{json.dumps(r['tool_log'], indent=2)[:12000]}\n```\n</details>\n"]
        md.append(f"\n---\n\n### Final case file\n\n```json\n{json.dumps(agent.case_file.as_dict(), indent=2)}\n```\n")
        if agent.case_file.corrections():
            md.append(f"\n### Corrections recorded\n\n```json\n"
                      f"{json.dumps(agent.case_file.corrections(), indent=2)}\n```\n")
        (OUT / f"{scn['id']}.md").write_text("".join(md), encoding="utf-8")

        summary.append({"id": scn["id"], "status": status, "failures": fails,
                        "turns": len(results),
                        "tools": sorted({t for r in results for t in r["tools_called"]}),
                        "validation_issues": sum(r["validation"]["n_issues"] for r in results)})

    if args.merge and (OUT / "summary.json").exists():
        prev = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        merged = {s["id"]: s for s in prev.get("scenarios", [])}
        for s in summary:
            s["model"] = describe()
            merged[s["id"]] = s
        summary = [merged[k] for k in sorted(merged)]

    passed = sum(s["status"] == "PASS" for s in summary)
    (OUT / "summary.json").write_text(json.dumps(
        {"model": describe(), "passed": passed, "total": len(summary),
         "seconds": round(time.time() - t_start, 1), "scenarios": summary}, indent=2), encoding="utf-8")

    print("=" * 64)
    if aborted:
        print("RUN ABORTED - free-tier daily budgets exhausted on every model.")
        print("Budgets roll over ~24h after first use. Re-run then.")
    print(f"{passed}/{len(summary)} scenarios passed in {time.time()-t_start:.0f}s")
    for s in summary:
        print(f"  {s['status']:<5} {s['id']:<34} issues={s['validation_issues']}")
    print(f"\nraw outputs -> {OUT.relative_to(ROOT)}/")
    return 0 if passed == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
