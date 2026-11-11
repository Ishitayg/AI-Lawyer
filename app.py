"""Streamlit chat UI — two independent assistants.

Each mode keeps its own agent, its own conversation and its own case file.
Switching modes does not merge anything: a tenancy conversation and a consumer
conversation are different matters, and bleeding one into the other would put
facts in front of a model that must not reason about them.

Beyond showing the answer, the UI does two jobs:

1. **Show the grounding.** Which tools ran, which sections and judgments were
   retrieved, and what the citation check flagged.
2. **Never look dead.** Free-tier rate limits can stall a turn for a minute.
   Every tool call and every wait is reported live, with elapsed time.

Everything is also written to logs/ai-lawyer.log, because Streamlit swallows
stdout from callbacks.
"""
import queue
import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import analysis, llm, retrieve  # noqa: E402
from src.agent import MODES, Agent  # noqa: E402
from src.logging_setup import LOG_FILE, get as get_logger  # noqa: E402
from src.state import CaseFile  # noqa: E402

log = get_logger("ui")

st.set_page_config(page_title="AI Lawyer — India", page_icon="⚖️", layout="wide")

# --------------------------------------------------------------- session state
if "agents" not in st.session_state:
    st.session_state.mode_key = "grounded"
    # One agent per mode, created up front and never swapped. Separate
    # conversations, separate case files, separate everything.
    st.session_state.agents = {k: Agent(mode=m) for k, m in MODES.items()}
    st.session_state.histories = {k: [] for k in MODES}
    log.info("new session")

MODE_KEY = st.session_state.mode_key
MODE = MODES[MODE_KEY]
agent = st.session_state.agents[MODE_KEY]
history = st.session_state.histories[MODE_KEY]


def other_key():
    return "general" if MODE_KEY == "grounded" else "grounded"


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Choose your assistant")
    keys = list(MODES)
    chosen = st.radio(
        "Which assistant?", keys,
        index=keys.index(MODE_KEY),
        format_func=lambda k: MODES[k].label,
        label_visibility="collapsed",
    )
    st.caption(MODES[chosen].tagline)
    if chosen != MODE_KEY:
        st.session_state.mode_key = chosen
        log.info("mode switched -> %s", chosen)
        st.rerun()

    n_other = len(st.session_state.histories[other_key()])
    if n_other:
        st.caption(f"↔ Your **{MODES[other_key()].label}** conversation "
                   f"({n_other} message{'s' if n_other != 1 else ''}) is kept separately.")

    st.divider()
    st.subheader("System")
    st.write(f"**Model:** `{llm.describe()}`")
    if llm._EXHAUSTED:
        st.caption("Fell back automatically — out of daily budget: "
                   + ", ".join(f"`{p}:{m}`" for p, m in sorted(llm._EXHAUSTED)))
    for name, s in retrieve.index_stats().items():
        st.write(f"**{name}:** " + (f"{s['chunks']} chunks, {s['mb']} MB" if s else "_not built_"))
    st.caption("Retrieval, embeddings and indexing run entirely on this machine.")
    if llm.PROVIDER == "groq":
        st.info("Groq's free tier allows 8,000 tokens/minute. A reply can take 30–90s while "
                "the client paces itself.", icon="⏳")

    st.divider()
    st.subheader(f"Progress — {MODE.label}")
    for stage in analysis.progress(agent.case_file, agent.ctx):
        st.write(("✅ " if stage["done"] else "⬜ ") + stage["label"])

    st.subheader("Case file")
    known = agent.case_file.known()
    if known:
        for k, v in known.items():
            st.write(f"**{k.replace('_',' ')}:** {v}")
    else:
        st.caption("_Nothing recorded yet._")
    missing = agent.case_file.missing_critical()
    if missing:
        st.warning("Still needed: " + ", ".join(m.replace("_", " ") for m in missing))

    # Opt-in only: never move facts between matters behind the user's back.
    other_known = st.session_state.agents[other_key()].case_file.known()
    if other_known and not known:
        if st.button(f"Copy details from {MODES[other_key()].label}", use_container_width=True):
            for k, v in other_known.items():
                agent.case_file.update(**{k: v})
            log.info("case file copied %s -> %s", other_key(), MODE_KEY)
            st.rerun()

    revisions = agent.case_file.corrections()
    if revisions:
        with st.expander(f"Corrections ({len(revisions)})"):
            for r in revisions:
                st.write(f"**{r['field'].replace('_',' ')}**: {r['from']} → {r['to']}")

    st.subheader("Matter")
    if st.button("Save matter", use_container_width=True):
        agent.case_file.save()
        st.success(f"Saved as `{agent.case_file.matter_id}`")
    saved = CaseFile.list_saved()
    if saved:
        pick = st.selectbox("Load a saved matter",
                            ["—"] + [f"{m['matter_id']} · {m['summary']}" for m in saved])
        if pick != "—" and st.button("Load", use_container_width=True):
            mid = pick.split(" · ")[0]
            st.session_state.agents[MODE_KEY] = Agent(case_file=CaseFile.load(mid), mode=MODE)
            st.session_state.histories[MODE_KEY] = []
            log.info("matter loaded <- %s into %s", mid, MODE_KEY)
            st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("Reset this", use_container_width=True,
                 help=f"Clears only the {MODE.label} conversation"):
        st.session_state.agents[MODE_KEY] = Agent(mode=MODE)
        st.session_state.histories[MODE_KEY] = []
        st.rerun()
    if c2.button("Reset all", use_container_width=True, help="Clears both conversations"):
        st.session_state.agents = {k: Agent(mode=m) for k, m in MODES.items()}
        st.session_state.histories = {k: [] for k in MODES}
        st.rerun()

    if st.button("Show log", use_container_width=True):
        st.session_state.show_log = not st.session_state.get("show_log", False)
    if st.session_state.get("show_log"):
        try:
            st.code("".join(LOG_FILE.read_text(encoding="utf-8").splitlines(keepends=True)[-40:]),
                    language="log")
        except FileNotFoundError:
            st.caption("_no log yet_")

    st.divider()
    st.caption("Sources: Consumer Protection Act, 2019; consumer commission decisions via "
               "Indian Kanoon. See NOTICE.md.")

# ------------------------------------------------------------------ main pane
st.title("⚖️ AI Lawyer — India")
st.caption(f"**{MODE.label}** · {MODE.tagline}")

if MODE.banner:
    st.warning(MODE.banner, icon="⚠️")
else:
    st.success("**Grounded mode.** Every section and judgment cited is checked against a local "
               "copy of the Consumer Protection Act, 2019 and 300 consumer commission decisions.",
               icon="✅")


def grounding_panel(meta):
    v = meta["validation"]
    label = MODES[meta.get("mode", "grounded")].label
    with st.expander(f"How this answer was grounded — {label} · "
                     f"{len(meta['tools_called'])} tool call(s), {v['n_issues']} flag(s)"):
        st.write("**Tools chosen:** " + (", ".join(f"`{t}`" for t in meta["tools_called"]) or "_none_"))
        st.write("**Sections retrieved:** " + (", ".join(v.get("retrieved_sections") or []) or "_none_"))
        st.write("**Judgments retrieved:** " + (", ".join(v.get("retrieved_case_ids") or []) or "_none_"))
        if v["issues"]:
            st.error("Citation check flagged:")
            st.json(v["issues"])
        elif meta.get("mode") == "general":
            st.info("No citations were attempted — correct for general mode, which has no corpus.")
        else:
            st.success("Every citation was verified against the local index.")
        st.write("**Tool trace**")
        st.json(meta["tool_log"])


for i, turn in enumerate(history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("meta"):
            grounding_panel(turn["meta"])
            if turn["meta"].get("suggest_other_mode") and i == len(history) - 1:
                st.info(f"This may suit **{MODES[other_key()].label}** better.", icon="↔️")
                if st.button(f"Switch to {MODES[other_key()].label}", key=f"sw{i}"):
                    st.session_state.mode_key = other_key()
                    st.rerun()


# ------------------------------------------------------------------- the turn
def run_turn(a, prompt, events: queue.Queue):
    """Runs on a worker thread so a rate-limit stall cannot freeze the UI."""
    llm.set_wait_callback(lambda secs, att, tot: events.put(("wait", secs, att, tot)))
    try:
        events.put(("done", a.ask(prompt)))
    except Exception as e:  # noqa: BLE001
        log.exception("turn failed")
        events.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        llm.set_wait_callback(None)


placeholder = ("Describe your problem, e.g. 'I paid ₹18,000 for a phone in March, the screen "
               "died in May, the shop refuses to repair it'") if MODE_KEY == "grounded" else (
               "Describe your problem, e.g. 'My rent agreement is valid but my landlord is "
               "threatening to evict me'")

if prompt := st.chat_input(placeholder):
    log.info("[%s] user: %r", MODE_KEY, prompt[:200])
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=True)
        events: queue.Queue = queue.Queue()
        worker = threading.Thread(target=run_turn, args=(agent, prompt, events), daemon=True)
        worker.start()

        result, error, t0, seen = None, None, time.time(), 0
        while True:
            try:
                ev = events.get(timeout=0.4)
            except queue.Empty:
                if len(agent.ctx.tool_log) > seen:
                    for entry in agent.ctx.tool_log[seen:]:
                        status.write(f"🔧 `{entry['tool']}`")
                    seen = len(agent.ctx.tool_log)
                status.update(label=f"Working… {time.time() - t0:.0f}s")
                if not worker.is_alive() and events.empty():
                    error = "the agent thread stopped without returning a result"
                    break
                continue
            if ev[0] == "wait":
                status.update(label=f"Rate limited — waiting {ev[1]:.0f}s")
                status.write(f"⏳ waiting {ev[1]:.0f}s for the token budget")
            elif ev[0] == "done":
                result = ev[1]
                break
            elif ev[0] == "error":
                error = ev[1]
                break

        elapsed = time.time() - t0
        if error:
            status.update(label=f"Failed after {elapsed:.0f}s", state="error")
            st.error(error)
            st.caption(f"Full trace in `{LOG_FILE}`")
            history.append({"role": "assistant",
                            "content": f"**Could not complete this turn.**\n\n{error}"})
        else:
            status.update(label=f"Done in {elapsed:.0f}s — "
                                f"{len(result['tools_called'])} tool call(s)", state="complete")
            st.markdown(result["answer"])
            history.append({"role": "assistant", "content": result["answer"], "meta": result})
        log.info("[%s] turn rendered in %.1fs (error=%s)", MODE_KEY, elapsed, bool(error))
    st.rerun()
