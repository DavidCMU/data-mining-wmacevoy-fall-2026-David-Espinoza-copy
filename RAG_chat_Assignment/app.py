"""
RAG lab dashboard (Streamlit): build and tune retrieval over the Rust book,
then chat with it as a knowledge base.

Retrieval (local, free): build an index per embedding model, search it
three ways side by side, and score a setup on questions.jsonl, the
questions with known answers. Every score is appended to data/compare.csv.

Chat (Claude API): ask the book questions. Each reply is grounded in the
sections retrieved for that message and cites them — unlike llm/'s Zen
Coder, which deliberately hides its sources, this is a knowledge-base
assistant, so showing its sources is the point.

The pipeline in one line: vendor (the book) -> chunks -> index (LanceDB) ->
ask / compare (retrieval) -> chat.turn (one exchange) -> here.

Mirrors llm/app.py, minus Phase 2's simulated-conversation + judge system.
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

import chat
import chunks
import compare
import index
import models
import store

st.set_page_config(page_title="Rust book RAG", layout="wide")

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#f4f8fd", "#104281"]
MODES = list(store.MODES)
MODE_COLORS = alt.Scale(domain=MODES, range=PALETTE[:3])

chat.load_env()
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


@st.cache_data
def book_refs() -> set[str]:
    return {c.ref for c in chunks.chunk_book(chunks.find_book())}


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
        st.warning("No Claude API key. Put `ANTHROPIC_API_KEY=sk-ant-...` in `.env` "
                   "(git-ignored) or export it, then restart the app. Retrieval needs no key.")


with st.sidebar:
    page = st.radio("Page", ["Retrieval", "Chat"],
                    help="Retrieval tunes what gets found. Chat lets you ask the book "
                         "questions once something is indexed.")


# =====================================================================================
# Retrieval
# =====================================================================================
def retrieval_page() -> None:
    st.title("Retrieval")
    st.caption("Retrieval is the R in RAG: given a question, find the book sections that "
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
                           f"book {i['book']}")
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
                     "model (MiniLM reads only 256 tokens). Rebuilding replaces the table.")
            if st.form_submit_button("Build index", type="primary"):
                root = chunks.find_book()
                with st.spinner(f"Embedding the book with {key}…"):
                    cl = chunks.chunk_book(root, max_chars)
                    t = index.index_model(store.connect(), key, cl, max_chars,
                                          index.book_version(root))
                st.success(f"{key}: {len(cl)} chunks in {t['embed_s']:.1f}s, "
                           f"{t['truncated']} truncated")
                st.rerun()

        st.subheader("Experiments to try")
        with st.expander("Which model, which mode?"):
            st.markdown(
                "Build every model (`pixi run -e rag index --model all`) and run "
                "`pixi run -e rag compare`, or use the **Score a setup** tab for one model "
                "at a time. Compare hit@1/hit@5/MRR the same way llm/README.md does for the "
                "software-engineering guide, and record your own numbers for this book here.")
        with st.expander("Chunk size"):
            st.markdown("Rebuild e5-small at **800** and **2500** characters and score it. "
                        "llm/'s own experiment found barely a difference on its guide — "
                        "does that hold on a book this much larger?")
        with st.expander("Grow the test set"):
            st.markdown("45 questions is still small. Add your own, especially paraphrases, "
                        "on the **Test set** tab, and see whether the ranking holds.")

    if not built:
        st.info("No index yet. Build one in the sidebar (e5-small is the default), or run "
                "`pixi run -e rag index`.")
        return

    tab_search, tab_score, tab_history, tab_questions = st.tabs(
        ["Search", "Score a setup", "History", "Test set"])

    with tab_search:
        st.caption("The same question, three ways. **vector** embeds it and finds the "
                   "nearest chunks by meaning; **fts** is keyword search (BM25); **hybrid** "
                   "merges the two rankings.")
        q = st.text_input("Question", "how do I grow a list and read its elements back out")
        a, b = st.columns(2)
        key = a.selectbox("Model", built, default_index(built), key="search_model")
        k = b.slider("k", 1, 10, 5, key="search_k")
        if q:
            table = store.open_table(store.connect(), store.table_name(key))
            vector = models.embed_query(key, q)
            for col, mode in zip(st.columns(3), MODES):
                col.subheader(mode)
                for i, r in enumerate(store.search(table, mode, q, vector, k), 1):
                    with col.expander(f"{i}. {r['heading'].split(' > ')[-1]} · {r['score']:.3f}"):
                        st.caption(r["ref"])
                        st.markdown(r["text"].split("\n\n", 1)[-1])

    with tab_score:
        questions = compare.load_questions()
        st.caption(f"Rank every question in questions.jsonl ({len(questions)}) and find "
                   "where the section that answers it lands. **hit@1**: the answer is the "
                   "top result. **hit@5**: it is in the top five, which a k=5 prompt "
                   "would contain. **MRR**: the mean of 1/rank.")
        with st.form("score"):
            a, b = st.columns(2)
            key = a.selectbox("Model", built, default_index(built))
            modes = b.multiselect("Modes", MODES, MODES)
            go = st.form_submit_button("Score", type="primary")
        if go and modes:
            table = store.open_table(store.connect(), store.table_name(key))
            rows, ranks = [], {}
            with st.spinner("Searching…"):
                for mode in modes:
                    results = compare.evaluate(table, key, mode, questions)
                    label = "(any)" if mode == "fts" else key
                    rows.append({"model": label, "mode": mode, **compare.summarize(results)})
                    ranks[mode] = [r["rank"] for r in results]
            df = pd.DataFrame(rows)
            i = info.get(store.table_name(key), {})
            compare.append_results(df, i.get("max_chars", chunks.MAX_CHARS),
                                   i.get("chunks", 0), len(questions))
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
            h["setup"] = h["model"] + " · " + h["mode"] + " · " + h["max_chars"].astype(str)
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
        st.caption("The test set. **kind**: paraphrase (other words) or keyword (the "
                   "book's own term). **expect**: the sections that answer it, separated "
                   "by spaces, as <file>.md#<anchor>. Find a section's ref on the Search tab.")
        qs = pd.DataFrame([{"kind": q["kind"], "q": q["q"], "expect": " ".join(q["expect"])}
                           for q in compare.load_questions()])
        edited = st.data_editor(qs, num_rows="dynamic", width="stretch", column_config={
            "kind": st.column_config.SelectboxColumn(options=["paraphrase", "keyword"],
                                                     required=True),
            "q": st.column_config.TextColumn("question", width="large", required=True),
            "expect": st.column_config.TextColumn(width="medium", required=True)})
        if st.button("Save questions.jsonl"):
            refs = book_refs()
            rows = [{"kind": r.kind, "q": r.q.strip(), "expect": r.expect.split()}
                    for r in edited.dropna().itertuples()]
            bad = sorted({e for r in rows for e in r["expect"] if e not in refs})
            if bad:
                st.error("Not sections of the book: " + ", ".join(bad))
            else:
                compare.QUESTIONS.write_text("".join(json.dumps(r) + "\n" for r in rows))
                st.success(f"Saved {len(rows)} questions.")


# =====================================================================================
# Chat
# =====================================================================================
def show_turns(turns: list[dict]) -> None:
    for n, t in enumerate(turns, 1):
        with st.chat_message("user"):
            st.markdown(t["user"])
        with st.chat_message("assistant"):
            st.markdown(t["response"])
            if t.get("context"):
                with st.expander(f"Turn {n}: sections retrieved for this message"):
                    for c in t["context"]:
                        st.caption(f"[{c['ref']}] {c['heading']}")


def chat_page() -> None:
    st.title("Chat with the book")
    st.caption("Ask a question; each reply is grounded in the sections retrieved for it "
               "and cites them by [ref] — expand \"sections retrieved\" below a reply to "
               "see exactly what it was working from.")
    built = built_models()
    with st.sidebar:
        key_notice()
        k = st.slider("k (sections retrieved per message)", 0, 8, 5 if built else 0,
                      disabled=not built, help="Build an index on the Retrieval page to "
                                                "turn this on. 0 means the model answers "
                                                "from its own training, with nothing "
                                                "retrieved — a useful control.")
        names = list(chat.MODELS)
        model = st.selectbox("Model", names, names.index(chat.DEFAULT_MODEL))
        mode = st.selectbox("Search mode", MODES, disabled=not built)
        r_model = st.selectbox("Retrieval model", built or [models.DEFAULT],
                               default_index(built))
        if st.button("New conversation"):
            st.session_state.pop("chat", None)
    if not HAS_KEY:
        st.info("Chat needs a Claude API key (see the sidebar).")
        return
    if "claude" not in st.session_state:
        st.session_state["claude"] = chat.Claude()
    claude = st.session_state["claude"]
    turns = st.session_state.setdefault("chat", [])
    st.caption(f"This session has cost ${claude.cost:.3f}.")
    show_turns(turns)

    msg = st.chat_input("Ask the book something…")
    if msg:
        with st.chat_message("user"):
            st.markdown(msg)
        with st.spinner("Thinking…"):
            try:
                t = chat.turn(claude, turns, msg, k, model, {"model": r_model, "mode": mode})
            except Exception as e:   # show API errors and refusals; keep the conversation
                st.error(f"{type(e).__name__}: {e}")
                return
        turns.append(t)
        st.rerun()


if page == "Retrieval":
    retrieval_page()
else:
    chat_page()
