"""
Search the indexed guide and show what a RAG system would hand to an LLM.

    python ask.py "how do I handle a flaky bug I cannot reproduce?"
    python ask.py "SE1/O8" --mode fts
    python ask.py "should I commit my lock file" --model bge-small -k 3
    python ask.py "what is a K3 factory" --prompt     # print the full prompt

Retrieval is the "R" in RAG. --prompt prints the "AG" half's input: the
retrieved chunks, each labeled with its citation, wrapped in instructions to
answer only from them. Paste it into any chat model to see the whole loop.
"""
import argparse
import textwrap

import models
import store

PROMPT = """\
Answer the question using only the excerpts from the software-engineering guide
below. Cite the excerpts you use by their [ref]. If they do not answer it, say so.

{context}

Question: {question}"""


def retrieve(question: str, model: str = models.DEFAULT, mode: str = "vector",
             k: int = 5) -> list[dict]:
    table = store.open_table(store.connect(), store.table_name(model))
    vector = None if mode == "fts" else models.embed_query(model, question)
    return store.search(table, mode, question, vector, k)


def build_prompt(question: str, rows: list[dict]) -> str:
    context = "\n\n".join(f"[{r['ref']}]\n{r['text']}" for r in rows)
    return PROMPT.format(context=context, question=question)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("question")
    p.add_argument("--model", default=models.DEFAULT, choices=list(models.MODELS))
    p.add_argument("--mode", default="vector", choices=store.MODES)
    p.add_argument("-k", type=int, default=5, help="how many chunks to retrieve")
    p.add_argument("--prompt", action="store_true",
                   help="print the prompt an LLM would get instead of a summary")
    args = p.parse_args(argv)

    rows = retrieve(args.question, args.model, args.mode, args.k)
    if args.prompt:
        print(build_prompt(args.question, rows))
        return
    for i, r in enumerate(rows, 1):
        body = r["text"].split("\n\n", 1)[-1]
        score = "by ID" if r["score"] == float("inf") else f"{r['score']:.3f}"
        print(f"{i}. [{score}] {r['ref']}\n   {r['heading']}")
        print(textwrap.indent(textwrap.shorten(body, 240), "   "), end="\n\n")


if __name__ == "__main__":
    main()
