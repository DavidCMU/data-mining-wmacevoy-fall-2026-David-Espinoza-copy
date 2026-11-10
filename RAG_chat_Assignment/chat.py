"""
A knowledge-base chat bot over the indexed book: retrieve, then ask Claude
to answer *only* from what was retrieved, citing its sources.

This is deliberately much smaller than llm/zen/coder.py's Zen Coder: there's
no mentor persona hiding its sources, no simulated novice, no judge-scoring
pipeline. The chat here is a plain RAG assistant for a knowledge base — it
retrieves sections relevant to the question, answers from them, and says
which sections it used, which is exactly what you want from "ask the book a
question" rather than from a mentor that's supposed to make you think it
through yourself.

One exchange:

    retrieve k sections for the user's message (and, for a multi-turn chat,
    for the conversation so far, so a follow-up like "what about with Rc?"
    still retrieves something relevant)
    Claude reads the retrieved sections and the conversation, and answers
    using only what's there, citing each section it drew on by its [ref]

Every API call goes through Claude.call, which adds up tokens and cost, the
same bookkeeping llm/zen/coder.py does.
"""
import os
import re
from pathlib import Path

import ask
import models

ENV_FILE = Path(__file__).parent / ".env"          # optional ANTHROPIC_API_KEY=...; git-ignored

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
DEFAULT_MODEL = "sonnet"

# US$ per million tokens: (input, output, cache read). A cache write costs 1.25x input.
# From the Claude pricing page, October 2026. Prices change: check before trusting a total.
PRICES = {"claude-haiku-5-5": (0.10, 0.50, 0.01),
          "claude-sonnet-5-5": (2.00, 10.00, 0.20),
          "claude-opus-5-5": (4.00, 20.00, 0.20)}

SYSTEM = """\
You are a knowledge-base assistant for "The Rust Programming Language" book.
Answer only using the retrieved excerpts you are given below each question;
if they don't contain the answer, say so plainly rather than guessing or
using outside knowledge. Cite the excerpt(s) you used by their [ref] tag,
right after the sentence that relies on them."""


class ChatError(RuntimeError):
    """A call that returned no usable answer (a refusal, or cut off at max_tokens)."""


def load_env(path: Path = ENV_FILE) -> None:
    """Read ANTHROPIC_API_KEY from .env if it isn't already in the environment."""
    if os.environ.get("ANTHROPIC_API_KEY") or not path.exists():
        return
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ANTHROPIC_API_KEY" and value.strip():
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("'\"")


def cost(usage: dict[str, dict]) -> float:
    """US$ for {model: token counts}, at PRICES."""
    total = 0.0
    for model, u in usage.items():
        p_in, p_out, p_read = PRICES.get(model, (0, 0, 0))
        total += (u.get("input", 0) * p_in + u.get("cache_write", 0) * 1.25 * p_in
                  + u.get("cache_read", 0) * p_read + u.get("output", 0) * p_out) / 1e6
    return total


class Claude:
    """A Messages API client that keeps a tally of the tokens each model used."""

    def __init__(self, client=None):
        if client is None:
            import anthropic
            load_env()
            client = anthropic.Anthropic(max_retries=5)
        self.client = client
        self.usage: dict[str, dict] = {}

    @property
    def cost(self) -> float:
        return cost(self.usage)

    def call(self, model: str, system: str, user: str, max_tokens: int = 2000) -> str:
        model = MODELS.get(model, model)
        msg = self.client.messages.create(
            model=model, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}])
        u = self.usage.setdefault(model, {"calls": 0, "input": 0, "output": 0,
                                          "cache_read": 0, "cache_write": 0})
        u["calls"] += 1
        u["input"] += msg.usage.input_tokens
        u["output"] += msg.usage.output_tokens
        u["cache_read"] += msg.usage.cache_read_input_tokens or 0
        u["cache_write"] += msg.usage.cache_creation_input_tokens or 0
        if msg.stop_reason in ("refusal", "max_tokens"):
            raise ChatError(f"{model} stopped with {msg.stop_reason}")
        text = next((b.text for b in msg.content if b.type == "text"), "").strip()
        if not text:
            raise ChatError(f"{model} returned no text")
        return text


def retrieve(query: str, k: int, model: str | None = None, mode: str = "vector") -> list[dict]:
    """The top k book sections for a query, as {ref, heading, text}."""
    if k <= 0 or not query:
        return []
    rows = ask.retrieve(query, model or models.DEFAULT, mode, k)
    return [{"ref": r["ref"], "heading": r["heading"], "text": r["text"]} for r in rows]


def transcript(turns: list[dict]) -> str:
    out = []
    for t in turns:
        out.append(f"User: {t['user']}")
        if t.get("response"):
            out.append(f"Assistant: {t['response']}")
    return "\n".join(out)


def build_user_turn(history: list[dict], message: str, sections: list[dict]) -> str:
    context = ("\n\n".join(f"[{s['ref']}]\n{s['text']}" for s in sections)
               if sections else "(nothing retrieved for this message)")
    so_far = transcript(history)
    prefix = f"Conversation so far:\n{so_far}\n\n" if so_far else ""
    return f"{prefix}Retrieved excerpts:\n\n{context}\n\nUser: {message}"


def turn(claude: Claude, history: list[dict], message: str, k: int = 5,
        model: str = DEFAULT_MODEL, retrieval: dict | None = None) -> dict:
    """One exchange: retrieve for `message`, answer it. Returns the turn record."""
    sections = retrieve(message, k, **(retrieval or {}))
    response = claude.call(model, SYSTEM, build_user_turn(history, message, sections))
    return {"user": message, "context": sections, "response": response}


CITED = re.compile(r"\[([\w./-]+#?[\w-]*)\]")


def cited_refs(response: str, known_refs: set[str]) -> list[str]:
    """Which of the sections actually offered were cited in the reply, in order."""
    found = CITED.findall(response)
    return list(dict.fromkeys(ref for ref in found if ref in known_refs))
