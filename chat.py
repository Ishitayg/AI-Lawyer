#!/usr/bin/env python3
"""Terminal chat interface — the quickest way to poke at the agent.

    .venv/bin/python -I chat.py

Shows the grounding inline: which tools the agent chose, which sections and
judgments it actually retrieved, and anything the citation validator flagged.

Commands:  /file   show the case file    /plan    progress through the stages
           /tools  tool trace             /fixes   corrections the user made
           /save   persist this matter    /load ID restore one
           /mode   switch assistant      /reset   start over
           /quit   exit
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.agent import MODES, Agent  # noqa: E402
from src.analysis import progress  # noqa: E402
from src.state import CaseFile  # noqa: E402
from src.llm import describe  # noqa: E402
from src.retrieve import index_stats  # noqa: E402

B, D, G, R, Y, C = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m"
X = "\033[0m"


def main():
    print(f"\n{B}AI Lawyer — Indian consumer disputes{X}")
    print(f"{D}model: {describe()}   indexes: {index_stats()}{X}")
    print(f"{D}/mode to switch assistant · /file /plan /tools /save /reset /quit{X}\n")
    agent = Agent(mode=MODES["grounded"])
    last = None

    while True:
        try:
            msg = input(f"{B}you ▸ {X}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            return 0
        if msg == "/reset":
            agent = Agent()
            print(f"{D}(conversation reset){X}\n")
            continue
        if msg == "/file":
            print(json.dumps(agent.case_file.known(), indent=2) or "{}")
            print(f"{Y}missing: {agent.case_file.missing_critical()}{X}\n")
            continue
        if msg.startswith("/mode"):
            parts = msg.split()
            if len(parts) < 2 or parts[1] not in MODES:
                for k, m in MODES.items():
                    mark = "*" if agent.mode.key == k else " "
                    print(f" {mark} {k:<9} {m.label}\n     {m.tagline}")
                print("\n   usage: /mode grounded | /mode general\n")
            else:
                agent = Agent(case_file=agent.case_file, mode=MODES[parts[1]])
                print(f"{G}switched to {agent.mode.label} (case file kept){X}\n")
            continue
        if msg == "/plan":
            for st_ in progress(agent.case_file, agent.ctx):
                print(("  [x] " if st_["done"] else "  [ ] ") + st_["label"])
            print()
            continue
        if msg == "/fixes":
            fx = agent.case_file.corrections()
            print(json.dumps(fx, indent=2) if fx else "(no corrections)")
            print()
            continue
        if msg == "/save":
            p = agent.case_file.save()
            print(f"{G}saved as {agent.case_file.matter_id}{X}  ({p})\n")
            continue
        if msg.startswith("/load"):
            parts = msg.split()
            if len(parts) < 2:
                for m in CaseFile.list_saved():
                    print(f"  {m['matter_id']}  {m['saved_at']}  {m['summary']}")
                print()
            else:
                try:
                    agent = Agent(case_file=CaseFile.load(parts[1]))
                    print(f"{G}loaded {parts[1]}{X}\n")
                except FileNotFoundError:
                    print(f"{R}no such matter: {parts[1]}{X}\n")
            continue
        if msg == "/tools":
            print(json.dumps(last["tool_log"], indent=2) if last else "(nothing yet)")
            print()
            continue

        try:
            last = agent.ask(msg)
        except Exception as e:  # noqa: BLE001
            print(f"{R}error: {type(e).__name__}: {e}{X}\n")
            continue

        v = last["validation"]
        print(f"\n{C}tools:{X} {', '.join(last['tools_called']) or 'none'}")
        print(f"{C}retrieved:{X} sections {v['retrieved_sections'] or '-'} | "
              f"cases {v['retrieved_case_ids'] or '-'}")
        flag = (f"{G}citations verified{X}" if not v["n_issues"]
                else f"{R}{v['n_issues']} flagged: {[i['type'] for i in v['issues']]}{X}")
        print(f"{C}check:{X} {flag}\n")
        print(last["answer"])
        print()


if __name__ == "__main__":
    sys.exit(main())
