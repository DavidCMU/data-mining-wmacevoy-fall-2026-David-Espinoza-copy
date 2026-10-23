"""
RAG lab dashboard (Streamlit): tune the retrieval, then tune the conversation.

Phase 1, Retrieval (local, free): build an index per embedding model, search
it three ways side by side, and score a setup on questions.jsonl, the
questions with known answers. Every score is appended to data/compare.csv.

Phase 2, Conversation (Claude API): Zen Coder, a mentor bot that knows the
guide, talks with simulated novice developers, and a judge scores each
conversation. Pick system prompts and k (guide sections retrieved per turn),
start a run (zen/run.py, as its own process), and watch it fill in.

Chat: talk to Zen Coder yourself, and see what it retrieved each turn.

The pipeline in one line: vendor (the guide) -> chunks -> index (LanceDB) ->
ask / compare (phase 1) -> zen.coder (one turn) -> zen.run (judged runs) -> here.
"""
import json
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

import chunks
import compare
import index
import models
import store
from zen import coder, score
from zen import run as zrun

st.set_page_config(page_title="RAG lab", layout="wide")

# Reference categorical palette, fixed order (blue, orange, aqua, ...), as in ml/app.py.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#f4f8fd", "#104281"]         # one hue, light to dark
MODES = list(store.MODES)
MODE_COLORS = alt.Scale(domain=MODES, range=PALETTE[:3])

coder.load_env()
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


@st.cache_data
def guide_refs() -> set[str]:
    return {c.ref for c in chunks.chunk_guide(chunks.find_guide())}


def built_models() -> list[str]:
    names = set(store.connect().list_tables().tables)
    return [k for k in models.MODELS if store.table_name(k) in names]


def default_index(built: list[str]) -> int:
    """Where the default embedding model sits in a list of built ones (0 if absent)."""
    return built.index(models.DEFAULT) if models.DEFAULT in built else 0


def key_notice() -> None:
    if HAS_KEY:
        st.caption("Claude API key: found.")
    else:
        st.warning("No Claude API key. Put `ANTHROPIC_API_KEY=sk-ant-...` in `llm/.env` "
                   "(git-ignored) or export it, then restart the app. Phase 1 needs no key.")


with st.sidebar:
    page = st.radio("Phase", ["1 · Retrieval", "2 · Conversation", "Chat with Zen Coder"],
                    help="Phase 1 tunes what gets found. Phase 2 tunes what Zen Coder does "
                         "with it. Chat lets you try Zen Coder yourself.")


# =====================================================================================
# Phase 1: retrieval
# =====================================================================================
def retrieval_page() -> None:
    st.title("Phase 1 · Retrieval")
    st.caption("Retrieval is the R in RAG: given a question, find the guide sections that "
               "answer it. Everything on this page runs locally and costs nothing.")
    info = index.index_info()
    built = built_models()

    with st.sidebar:
        st.header("Index")
        st.caption("One LanceDB table per embedding model. Vectors from different models "
                   "can't be compared, so each model needs its own.")
        for k, m in models.MODELS.items():
            i = info.get(store.table_name(k))
            if k in built and i:
                st.caption(f"✓ **{k}** · {i['chunks']} chunks of ≤{i['max_chars']} chars · "
                           f"guide {i['guide']}")
            elif k in built:
                st.caption(f"✓ **{k}** · built before index.json existed")
            else:
                st.caption(f"– {k} · not built ({m.params}, {m.note})")
        with st.form("build"):
            key = st.selectbox("Model", list(models.MODELS), index=list(models.MODELS).index(
                models.DEFAULT), help="The first build of a model downloads it "
                                      "(90-200 MB) into ~/.cache/huggingface.")
            max_chars = st.select_slider(
                "Max chunk size (chars)", [500, 800, 1000, 1500, 2000, 2500, 3000],
                value=chunks.MAX_CHARS,
                help="Sections longer than this are split by paragraph. Smaller chunks are "
                     "more precise but lose context; larger ones may be truncated by the "
                     "model (MiniLM reads only 256 tokens). Rebuilding replaces the table, "
                     "and Phase 2 retrieves from whatever is built.")
            if st.form_submit_button("Build index", type="primary"):
                root = chunks.find_guide()
                with st.spinner(f"Embedding the guide with {key}…"):
                    cl = chunks.chunk_guide(root, max_chars)
                    t = index.index_model(store.connect(), key, cl, max_chars,
                                          index.guide_version(root))
                st.success(f"{key}: {len(cl)} chunks in {t['embed_s']:.1f}s, "
                           f"{t['truncated']} truncated")
                st.rerun()

        st.subheader("Experiments to try")
        with st.expander("Which model, which mode?"):
            st.markdown(
                "Build every model and score each in all three modes, with pinning on. "
                "October 2026, 1500-char chunks:\n\n"
                "| model | mode | hit@1 | hit@5 | MRR |\n|---|---|---|---|---|\n"
                "| e5-small | vector | 0.65 | 0.95 | **0.76** |\n"
                "| arctic-s | vector | 0.65 | 0.85 | 0.74 |\n"
                "| bge-small | vector | 0.57 | 0.93 | 0.72 |\n"
                "| minilm | vector | 0.60 | 0.88 | 0.70 |\n"
                "| (any) | fts | 0.50 | 0.75 | 0.62 |\n\n"
                "Hybrid didn't beat plain vector search once IDs were pinned. MiniLM "
                "truncates 115 of 246 chunks and still isn't last.")
        with st.expander("Chunk size"):
            st.markdown("Rebuild e5-small at **800** and **2500** characters and score it. "
                        "We got MRR 0.74 and 0.72, against 0.76 at 1500: barely a "
                        "difference. Does that hold for your questions?")
        with st.expander("Grow the test set"):
            st.markdown("Forty questions is small: one question is 2.5 points of hit@1. "
                        "Add questions in your own words on the **Test set** tab, "
                        "especially paraphrases, and see whether the ranking holds.")

    if not built:
        st.info("No index yet. Build one in the sidebar (e5-small is the default), or run "
                "`pixi run -e llm index`.")
        return

    tab_search, tab_score, tab_history, tab_questions = st.tabs(
        ["Search", "Score a setup", "History", "Test set"])

    with tab_search:
        st.caption("The same question, three ways. **vector** embeds it and finds the "
                   "nearest chunks by meaning; **fts** is keyword search (BM25); **hybrid** "
                   "merges the two rankings.")
        q = st.text_input("Question", "how do I handle a flaky bug I cannot reproduce")
        a, b, c = st.columns(3)
        key = a.selectbox("Model", built, default_index(built), key="search_model")
        k = b.slider("k", 1, 10, 5, key="search_k")
        pin = c.checkbox("Pin principles named by ID", True,
                         help="If the question names a principle (D10, SE1/O8), put its "
                              "section first, found by an exact filter.")
        if q:
            table = store.open_table(store.connect(), store.table_name(key))
            vector = models.embed_query(key, q)
            for col, mode in zip(st.columns(3), MODES):
                col.subheader(mode)
                for i, r in enumerate(store.search(table, mode, q, vector, k, pin), 1):
                    s = "by ID" if r["score"] == float("inf") else f"{r['score']:.3f}"
                    with col.expander(f"{i}. {r['heading'].split(' > ')[-1]} · {s}"):
                        st.caption(r["ref"])
                        st.markdown(r["text"].split("\n\n", 1)[-1])

    with tab_score:
        questions = compare.load_questions()
        st.caption(f"Rank every question in questions.jsonl ({len(questions)}) and find "
                   "where the section that answers it lands. **hit@1**: the answer is the "
                   "top result. **hit@5**: it is in the top five, which a k=5 prompt "
                   "would contain. **MRR**: the mean of 1/rank.")
        with st.form("score"):
            a, b, c = st.columns(3)
            key = a.selectbox("Model", built, default_index(built))
            modes = b.multiselect("Modes", MODES, MODES)
            pin = c.checkbox("Pin principles named by ID", True)
            go = st.form_submit_button("Score", type="primary")
        if go and modes:
            table = store.open_table(store.connect(), store.table_name(key))
            rows, ranks = [], {}
            with st.spinner("Searching…"):
                for mode in modes:
                    results = compare.evaluate(table, key, mode, questions, pin)
                    label = "(any)" if mode == "fts" else key
                    rows.append({"model": label, "mode": mode, **compare.summarize(results)})
                    ranks[mode] = [r["rank"] for r in results]
            df = pd.DataFrame(rows)
            i = info.get(store.table_name(key), {})
            compare.append_results(df, i.get("max_chars", chunks.MAX_CHARS),
                                   i.get("chunks", 0), pin, len(questions))
            st.session_state["scored"] = (df, ranks, questions)
        if "scored" in st.session_state:
            df, ranks, questions = st.session_state["scored"]
            st.dataframe(df, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%.2f")
                for c in df.columns if c not in ("model", "mode")})
            per = pd.DataFrame({"kind": [q["kind"] for q in questions],
                                "question": [q["q"] for q in questions],
                                **{f"rank ({m})": r for m, r in ranks.items()}})
            only_misses = st.checkbox("Only questions some mode missed at rank 1", True)
            if only_misses:
                rank_cols = [c for c in per.columns if c.startswith("rank")]
                per = per[(per[rank_cols] != 1).any(axis=1)]
            st.caption("Rank of the first section that answers it; empty = not in the top "
                       f"{compare.DEPTH}. Appended to data/compare.csv.")
            st.dataframe(per, hide_index=True, width="stretch")

    with tab_history:
        if not compare.RESULTS.exists():
            st.info("Nothing scored yet.")
        else:
            h = pd.read_csv(compare.RESULTS)
            h["setup"] = (h["model"] + " · " + h["mode"] + " · " + h["max_chars"].astype(str)
                          + h["pin"].astype(str).eq("True").map({True: " · pinned",
                                                                 False: ""}))
            latest = h.sort_values("when").groupby("setup").tail(1).sort_values(
                "MRR", ascending=False)
            st.caption("The most recent score for each setup (model · mode · chunk size).")
            st.altair_chart(
                alt.Chart(latest).mark_bar(cornerRadiusEnd=4, height=14).encode(
                    x=alt.X("MRR:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("setup:N", sort="-x", title=None),
                    color=alt.Color("mode:N", scale=MODE_COLORS,
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["setup", alt.Tooltip("MRR:Q", format=".3f"),
                             alt.Tooltip("hit@1:Q", format=".2f"),
                             alt.Tooltip("hit@5:Q", format=".2f"), "when", "questions"],
                ).properties(height=max(120, 24 * len(latest))),
                width="stretch")
            st.dataframe(h.drop(columns=["setup"]).iloc[::-1], hide_index=True,
                         width="stretch")

    with tab_questions:
        st.caption("The test set. **kind**: paraphrase (other words), keyword (the guide's "
                   "own term), or id (names a principle). **expect**: the sections that "
                   "answer it, separated by spaces, as guide/<file>.md#<anchor>. Find a "
                   "section's ref on the Search tab.")
        qs = pd.DataFrame([{"kind": q["kind"], "q": q["q"], "expect": " ".join(q["expect"])}
                           for q in compare.load_questions()])
        edited = st.data_editor(qs, num_rows="dynamic", width="stretch", column_config={
            "kind": st.column_config.SelectboxColumn(options=["paraphrase", "keyword", "id"],
                                                     required=True),
            "q": st.column_config.TextColumn("question", width="large", required=True),
            "expect": st.column_config.TextColumn(width="medium", required=True)})
        if st.button("Save questions.jsonl"):
            refs = guide_refs()
            rows = [{"kind": r.kind, "q": r.q.strip(), "expect": r.expect.split()}
                    for r in edited.dropna().itertuples()]
            bad = sorted({e for r in rows for e in r["expect"] if e not in refs})
            if bad:
                st.error("Not sections of the guide: " + ", ".join(bad))
            else:
                compare.QUESTIONS.write_text("".join(json.dumps(r) + "\n" for r in rows))
                st.success(f"Saved {len(rows)} questions.")


# =====================================================================================
# Phase 2: conversation
# =====================================================================================
PER_CONVERSATION = 0.15     # US$, a first guess for 3 turns with the default models


def cost_per_conversation() -> float:
    """Mean cost of past conversations, per Zen turn x 3; the first guess if none."""
    costs = []
    for p in zrun.list_runs():
        for f in p.glob("*.json"):
            if not f.name.endswith(".judge.json") and f.name not in (
                    "config.json", "status.json"):
                rec = zrun.read_json(f, {})
                if rec.get("cost"):
                    costs.append(rec["cost"] / max(1, len(rec.get("turns", []))) * 3)
    return sum(costs) / len(costs) if costs else PER_CONVERSATION


def run_label(p) -> str:
    s = zrun.read_json(p / "status.json", {})
    judged = len(list(p.glob("*.judge.json")))
    cost = f" · ${s['cost']:.2f}" if s.get("cost") else ""
    return f"{p.name} · {zrun.state(p)} · {judged} judged{cost}"


def show_turns(turns: list[dict]) -> None:
    for n, t in enumerate(turns, 1):
        with st.chat_message("user"):
            st.markdown(t["user"])
        with st.chat_message("assistant", avatar="🧘"):
            st.markdown(t["response"])
            if t.get("context") or t.get("query"):
                with st.expander(f"Turn {n}: what Zen Coder retrieved (the developer "
                                 "didn't see this)"):
                    if t.get("query"):
                        st.caption(f"query: {t['query']}")
                    for c in t.get("context", []):
                        st.caption(f"[{c['ref']}] {c['heading']}")
            st.caption(f"next query: {t.get('next_query', '')}")


def criteria_heatmap(table: pd.DataFrame, group: str) -> alt.Chart:
    long = table.reset_index().melt(id_vars=[group], value_vars=coder.CRITERIA,
                                    var_name="criterion", value_name="score")
    base = alt.Chart(long).encode(
        x=alt.X("criterion:N", sort=coder.CRITERIA, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{group}:N", title=None),
        tooltip=[f"{group}:N", "criterion:N", alt.Tooltip("score:Q", format=".2f")])
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("score:Q", scale=alt.Scale(domain=[1, 5], range=SEQUENTIAL),
                        legend=alt.Legend(title="score")))
    text = base.mark_text(fontSize=12).encode(
        text=alt.Text("score:Q", format=".1f"),
        color=alt.condition(alt.datum.score > 3.5, alt.value("white"), alt.value("#52514e")))
    return (cells + text).properties(height=max(100, 34 * len(table)))


def conversation_page() -> None:
    st.title("Phase 2 · Conversation")
    st.caption("Zen Coder talks with simulated novice developers who hold a few hidden "
               "facts; a judge model scores each conversation 1-5 on five criteria. Tune "
               "the system prompt and k, the number of guide sections retrieved per turn.")
    prompts, scenarios = coder.load_prompts(), coder.load_scenarios()
    built = built_models()

    with st.sidebar:
        key_notice()
        st.header("New run")
        with st.form("new_run"):
            name = st.text_input("Run name", time.strftime("%Y-%m-%d-%H%M"),
                                 help="A directory under data/zen/. Reusing a name resumes "
                                      "it: finished conversations are skipped.")
            sel_prompts = st.multiselect(
                "System prompts", list(prompts), [p for p in ("baseline", "door")
                                                  if p in prompts],
                help="Edit them, or add your own, on the Prompts & scenarios tab.")
            ks = st.multiselect("k (sections retrieved per turn)", [0, 1, 2, 3, 5, 8], [0, 3],
                                help="0 is the control: Zen Coder has only the condensed "
                                     "guide. Each turn retrieves for the query Zen asked "
                                     "for the turn before.")
            sel_scen = st.multiselect("Scenarios", list(scenarios), ["alarm", "plant"])
            repeats = st.slider("Repeats", 1, 5, 1,
                                help="Conversations vary run to run; repeats tell you how "
                                     "much. One per setting is an anecdote.")
            turns = st.slider("Zen turns per conversation", 2, 6, 3)
            with st.expander("Models"):
                names = list(coder.MODELS)
                zen_m = st.selectbox("Zen Coder", names, names.index(coder.ROLES["zen"]))
                nov_m = st.selectbox("Novice", names, names.index(coder.ROLES["novice"]))
                judge_m = st.selectbox("Judge", names, names.index(coder.ROLES["judge"]),
                                       help="Changing the judge changes the ruler: only "
                                            "compare runs judged by the same model.")
                r_model = st.selectbox("Retrieval model", built or [models.DEFAULT],
                                       default_index(built), help="Built in Phase 1.")
                r_mode = st.selectbox("Retrieval mode", MODES)
                workers = st.slider("Conversations at a time", 1, 8, 4)
            n = len(sel_prompts) * len(ks) * len(sel_scen) * repeats
            est = n * cost_per_conversation() * turns / 3
            st.caption(f"{n} conversations, about ${est:.2f} at the API's prices.")
            if st.form_submit_button("Start run", type="primary", disabled=not HAS_KEY):
                if not n:
                    st.error("Pick at least one prompt, k, and scenario.")
                elif any(k > 0 for k in ks) and r_model not in built:
                    st.error(f"Build the {r_model} index in Phase 1 first.")
                else:
                    zrun.start(name, {"prompts": sel_prompts, "ks": ks, "scenarios": sel_scen,
                                      "repeats": repeats, "turns": turns, "zen_model": zen_m,
                                      "novice_model": nov_m, "judge_model": judge_m,
                                      "retrieval_model": r_model, "retrieval_mode": r_mode,
                                      "workers": workers})
                    st.session_state["zen_run"] = name
                    time.sleep(1)       # let it write status.json
                    st.rerun()

        st.subheader("Experiments to try")
        with st.expander("Does retrieval help on top of the summary?"):
            st.markdown(
                "Our run 2026-10-22c (Sonnet, Opus judge, 2 scenarios, 2 prompts):\n\n"
                "| k | overall | grounded | tone | questions/turn | words/turn |\n"
                "|---|---|---|---|---|---|\n"
                "| 0 | **4.15** | 4.75 | 4.0 | 4.9 | 284 |\n"
                "| 3 | 3.70 | 4.5 | 3.0 | 6.0 | 342 |\n"
                "| 5 | 3.55 | 4.0 | 3.0 | 5.1 | 338 |\n\n"
                "With the summary already known, extra sections mostly added length and "
                "questions. Only 4 conversations per k: repeat it before believing it.")
        with st.expander("One door per reply"):
            st.markdown(
                "Told to lead through the door, Zen Coder stacked 4-10 questions per reply, "
                "and tone scores fell. The **door** prompt allows one question and 150 "
                "words. Run baseline vs door at k 0 and 3, with 2 repeats: does tone "
                "recover without losing *understands*?")
        with st.expander("The fact nobody finds"):
            st.markdown(
                "In every run so far, no conversation found out that the plant owner's "
                "Wi-Fi drops at night, so no conversation got to what the Pi should do "
                "offline. Write a prompt that asks *what could go wrong* before "
                "advising, and check `facts` on the Scores tab.")

        all_runs = zrun.list_runs()
        if all_runs:
            st.header("Runs")
            names = [p.name for p in all_runs]
            if st.session_state.get("zen_run") not in names:
                st.session_state["zen_run"] = names[0]
            st.radio("Show", names, key="zen_run", label_visibility="collapsed",
                     format_func=lambda nm: run_label(zrun.RUNS_DIR / nm))

    if not zrun.list_runs():
        st.info("No runs yet. Start one in the sidebar, or "
                "`pixi run -e llm zen-run --run first`.")
        prompts_tab(prompts, scenarios, None)
        return

    run_dir = zrun.RUNS_DIR / st.session_state["zen_run"]
    running = zrun.state(run_dir) == "running"
    tab_live, tab_scores, tab_conv, tab_compare, tab_edit = st.tabs(
        ["Progress", "Scores", "Conversations", "Compare runs", "Prompts & scenarios"])

    with tab_live:
        if running and st.button("Stop this run"):
            zrun.stop(run_dir)
            st.toast("Stopping: conversations finish the turn they are on…")
        st.fragment(run_every=3 if running else None)(show_progress)(run_dir)

    df = None
    try:
        df = score.load(run_dir.name)
    except SystemExit:
        pass

    with tab_scores:
        if df is None or df["overall"].isna().all():
            st.info("Scores appear as conversations are judged.")
        else:
            by = st.multiselect("Group by", ["prompt", "k", "scenario"], ["prompt", "k"])
            if by:
                cols = ["overall", *coder.CRITERIA, "facts", "guide_leaks",
                        "questions_per_turn", "words_per_turn", "relevant_recall", "cost"]
                cols = [c for c in cols if c in df and df[c].notna().any()]
                g = df.assign(group=df[by].astype(str).agg(" · ".join, axis=1))
                table = g.groupby("group")[cols].mean().round(2)
                table.insert(0, "n", g.groupby("group").size())
                table = table.sort_values("overall", ascending=False)
                st.caption("Mean of each group. overall is the mean of the five criteria. "
                           "With one or two conversations per group, a gap under about "
                           "0.4 is noise.")
                st.altair_chart(criteria_heatmap(table, "group"), width="stretch")
                st.dataframe(table, width="stretch")

    with tab_conv:
        if df is None:
            st.info("Conversations appear as they finish.")
        else:
            order = df.sort_values("overall", ascending=False)["id"].tolist()
            cid = st.selectbox("Conversation (best first)", order, format_func=lambda i: (
                f"{i} · {df.set_index('id').loc[i, 'overall']:.1f}"))
            rec = zrun.read_json(run_dir / f"{cid}.json", {})
            j = zrun.read_json(run_dir / f"{cid}.judge.json", {})
            if j:
                c = st.columns(6)
                for col, k in zip(c, coder.CRITERIA):
                    col.metric(k, j.get(k, "–"))
                m = rec.get("metrics", {})
                c[5].metric("leaks", m.get("guide_leaks", "–"),
                            help="principle IDs or mentions of the guide shown to the "
                                 "developer; should be 0")
                st.markdown(f"**Verdict.** {j.get('verdict', '')}")
                st.caption("Facts discovered: " + ("; ".join(j.get("facts_discovered", []))
                                                   or "none"))
                if j.get("evidence"):
                    with st.expander("The judge's evidence"):
                        st.markdown(j["evidence"])
            show_turns(rec.get("turns", []))

    with tab_compare:
        rows = []
        for p in zrun.list_runs():
            try:
                d = score.load(p.name)
            except SystemExit:
                continue
            cfg = zrun.read_json(p / "config.json", {})
            rows.append({"run": p.name, "n": len(d), "overall": d["overall"].mean(),
                         **{c: d[c].mean() for c in coder.CRITERIA},
                         "questions/turn": d["questions_per_turn"].mean(),
                         "words/turn": d["words_per_turn"].mean(),
                         "cost": d["cost"].sum() if "cost" in d and d["cost"].notna().any()
                         else None,
                         "judge": cfg.get("judge_model", "opus (workflow)")})
        if rows:
            cmp = pd.DataFrame(rows)
            st.caption("Every run's mean. Runs differ in prompts, k and scenarios, so "
                       "compare like with like; the Scores tab breaks one run down.")
            st.altair_chart(
                alt.Chart(cmp).mark_bar(cornerRadiusEnd=4, height=14, color=PALETTE[0]).encode(
                    x=alt.X("overall:Q", scale=alt.Scale(domain=[1, 5])),
                    y=alt.Y("run:N", sort=None, title=None),
                    tooltip=["run", "n", alt.Tooltip("overall:Q", format=".2f"), "judge"],
                ).properties(height=max(100, 26 * len(cmp))), width="stretch")
            st.dataframe(cmp.round(2), hide_index=True, width="stretch")

    with tab_edit:
        prompts_tab(prompts, scenarios, run_dir)


def show_progress(run_dir) -> None:
    status = zrun.read_json(run_dir / "status.json", {})
    st_now = zrun.state(run_dir)
    if st.session_state.get("_zen_seen") not in (None, st_now) and st_now != "running":
        st.session_state["_zen_seen"] = st_now
        st.rerun()                      # finished: redraw the whole page
    st.session_state["_zen_seen"] = st_now
    cfg = zrun.read_json(run_dir / "config.json", {})
    if not status:
        st.caption("Made by zen.workflow.js (Claude Code), so there is no live progress; "
                   "the other tabs show its results.")
        return
    total, done = status.get("total", 0), status.get("done", 0) + status.get("skipped", 0)
    c = st.columns(4)
    c[0].metric("State", st_now)
    c[1].metric("Conversations", f"{done} / {total}")
    c[2].metric("Failed", status.get("failed", 0))
    c[3].metric("Cost so far", f"${status.get('cost', 0):.2f}",
                help="Estimated from token counts at the prices in zen/coder.py.")
    if total:
        st.progress(done / total)
    st.caption(f"prompts {', '.join(cfg.get('prompts', []))} · k {cfg.get('ks')} · "
               f"scenarios {', '.join(cfg.get('scenarios', []))} · ×{cfg.get('repeats')} · "
               f"{cfg.get('turns')} turns · zen {cfg.get('zen_model')}, novice "
               f"{cfg.get('novice_model')}, judge {cfg.get('judge_model')} · retrieval "
               f"{cfg.get('retrieval_model') or models.DEFAULT} {cfg.get('retrieval_mode')}")
    for e in status.get("errors", []):
        st.error(e)
    if st_now == "lost":
        st.warning("This run stopped reporting without finishing (killed, or the machine "
                   "slept). Start it again with the same name to resume.")
    log = run_dir / "stdout.log"
    if st_now in ("failed", "lost") and log.exists():
        with st.expander("stdout.log"):
            st.code(log.read_text()[-4000:])
    events = zrun.read_progress(run_dir)
    if events:
        st.subheader("Latest")
        for e in events[-12:][::-1]:
            when = time.strftime("%H:%M:%S", time.localtime(e["t"]))
            if e["kind"] == "turn":
                st.caption(f"{when} · {e['id']} · Zen turn {e['n']} ({e['words']} words)")
            elif e["kind"] == "conversation":
                st.caption(f"{when} · **{e['id']} judged: {e['overall']:.1f}** "
                           f"(${e['cost']:.3f})")
            else:
                st.caption(f"{when} · {e['id']} · {e['kind']}: {e.get('error', '')}")


def prompts_tab(prompts: dict, scenarios: dict, run_dir) -> None:
    st.subheader("System prompts")
    st.caption("Zen Coder's instructions. Each is followed by the condensed guide "
               "(zen/guide-summary.md) and the rule never to cite it. Saved to "
               "zen/prompts.json, so commit the ones worth keeping.")
    pick = st.selectbox("Prompt", [*prompts, "(new prompt)"])
    text = st.text_area("Text", prompts.get(pick, prompts.get("baseline", "")), height=180)
    new_name = st.text_input("Save as", "" if pick == "(new prompt)" else pick,
                             help="A new name adds a prompt; the same name replaces it.")
    if st.button("Save prompt") and new_name.strip():
        prompts[new_name.strip()] = text.strip()
        coder.save_prompts(prompts)
        st.success(f"Saved {new_name.strip()}.")
    if run_dir is not None:
        used = zrun.read_json(run_dir / "config.json", {}).get("prompt_text", {})
        changed = [p for p, t in used.items() if prompts.get(p) != t]
        if changed:
            st.caption(f"Run {run_dir.name} used an earlier version of: {', '.join(changed)}.")
            for p in changed:
                with st.expander(f"{p}, as run"):
                    st.markdown(used[p])

    st.subheader("Scenarios")
    st.caption("Each simulated developer: an opening message, a persona, hidden facts they "
               "reveal only when asked, and the guide sections a strong answer would draw "
               "on (`relevant`, used for relevant_recall). Saved to zen/scenarios.json.")
    raw = st.text_area("scenarios.json", json.dumps(scenarios, indent=2), height=320)
    if st.button("Save scenarios"):
        try:
            new = json.loads(raw)
            need = {"opening", "persona", "facts", "relevant"}
            bad = [k for k, v in new.items() if not need <= set(v)]
            unknown = sorted({r for v in new.values() for r in v.get("relevant", [])
                              if r not in guide_refs()})
            if bad:
                st.error(f"Missing {sorted(need)} in: {', '.join(bad)}")
            elif unknown:
                st.error("Not sections of the guide: " + ", ".join(unknown))
            else:
                coder.SCENARIOS_FILE.write_text(json.dumps(new, indent=2) + "\n")
                st.success(f"Saved {len(new)} scenarios.")
        except json.JSONDecodeError as e:
            st.error(f"Not valid JSON: {e}")


# =====================================================================================
# Chat
# =====================================================================================
def chat_page() -> None:
    st.title("Chat with Zen Coder")
    prompts = coder.load_prompts()
    built = built_models()
    with st.sidebar:
        key_notice()
        prompt = st.selectbox("System prompt", list(prompts),
                              list(prompts).index("door") if "door" in prompts else 0)
        k = st.slider("k (sections retrieved per turn)", 0, 8, 3 if built else 0,
                      disabled=not built, help="Build an index in Phase 1 to turn this on.")
        names = list(coder.MODELS)
        zen_m = st.selectbox("Model", names, names.index(coder.ROLES["zen"]))
        r_model = st.selectbox("Retrieval model", built or [models.DEFAULT],
                               default_index(built))
        if st.button("New conversation"):
            st.session_state.pop("chat", None)
            st.session_state.pop("chat_judge", None)
    if not HAS_KEY:
        st.info("Chat needs a Claude API key (see the sidebar).")
        return
    if "claude" not in st.session_state:
        st.session_state["claude"] = coder.Claude()
    claude = st.session_state["claude"]
    turns = st.session_state.setdefault("chat", [])
    st.caption(f"Tell Zen Coder what you want to build. The expander under each reply "
               f"shows what it retrieved, which a real user wouldn't see. This session "
               f"has cost ${claude.cost:.3f}.")
    show_turns(turns)

    msg = st.chat_input("What do you want to build?")
    if msg:
        query = turns[-1]["next_query"] if turns else msg
        with st.chat_message("user"):
            st.markdown(msg)
        with st.spinner("Zen Coder is thinking…"):
            try:
                t = coder.zen_turn(claude, prompts[prompt], turns, msg, query, k, zen_m,
                                   {"model": r_model, "mode": "vector"})
            except Exception as e:   # show API errors and refusals; keep the conversation
                st.error(f"{type(e).__name__}: {e}")
                return
        turns.append(t)
        st.rerun()

    if turns:
        if st.button("Judge this conversation",
                     help="The same rubric as Phase 2, with you as the developer."):
            you = {"persona": "A real developer, typing for themselves.", "facts": [],
                   "relevant": []}
            with st.spinner("Judging…"):
                st.session_state["chat_judge"] = coder.judge(claude, you, turns)
        j = st.session_state.get("chat_judge")
        if j:
            c = st.columns(5)
            for col, key in zip(c, coder.CRITERIA):
                col.metric(key, j[key])
            st.markdown(f"**Verdict.** {j['verdict']}")


if page.startswith("1"):
    retrieval_page()
elif page.startswith("2"):
    conversation_page()
else:
    chat_page()
