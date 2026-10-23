"""
Zen Coder: a mentor bot for novice developers, and the pieces that test it.

One Zen turn:

    retrieve k guide sections for a query (k = 0: none, the control)
    Zen Coder reads its system prompt, the condensed guide, those sections, and the
    conversation so far, and replies with JSON: {"response": ..., "query": ...}
    `response` is what the developer sees; `query` is what Zen wants retrieved next turn

A test conversation alternates Zen turns with a simulated novice who holds a few
hidden facts (scenarios.json), and then a judge scores the transcript against a
rubric. run.py runs many conversations; app.py shows them and lets you chat.

Zen Coder knows the guide but never shows it: no principle IDs, no quotations,
no "the guide says". It leads the developer through the door rather than
pointing at the sign. metrics() counts any slip as a `guide_leak`.

Every API call goes through Claude.call, which adds up tokens and cost. Tests
pass in a fake client, so they run offline and free.
"""
import json
import os
import re
import threading
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE / "guide-summary.md"
PROMPTS_FILE = HERE / "prompts.json"
SCENARIOS_FILE = HERE / "scenarios.json"
ENV_FILE = HERE.parent / ".env"          # optional ANTHROPIC_API_KEY=...; git-ignored

MODELS = {"sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5", "opus": "claude-opus-5-5"}
ROLES = {"zen": "sonnet", "novice": "haiku", "judge": "opus"}      # the defaults

# US$ per million tokens: (input, output, cache read). A cache write costs 1.25x input.
# From the Claude pricing page, October 2026. Prices change: check before trusting a total.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01),
          "claude-opus-5-5": (4.00, 20.00, 0.20)}

CRITERIA = ["understands", "fits", "teaches", "grounded", "tone"]
CITED = re.compile(r"\b(?:SE1/)?([PUFRDTOWHA]\d{1,2})\b")
SHOWN = re.compile(r"\bthe guide\b|\bretriev\w*", re.IGNORECASE)   # naming its sources


class ZenError(RuntimeError):
    """A call that returned no usable answer (a refusal, or cut off at max_tokens)."""


# ---------- configuration on disk ----------

def load_env(path: Path = ENV_FILE) -> None:
    """Read ANTHROPIC_API_KEY from llm/.env if it isn't already in the environment."""
    if os.environ.get("ANTHROPIC_API_KEY") or not path.exists():
        return
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ANTHROPIC_API_KEY" and value.strip():
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("'\"")


def load_prompts() -> dict[str, str]:
    return json.loads(PROMPTS_FILE.read_text())


def save_prompts(prompts: dict[str, str]) -> None:
    PROMPTS_FILE.write_text(json.dumps(prompts, indent=2) + "\n")


def load_scenarios() -> dict[str, dict]:
    return json.loads(SCENARIOS_FILE.read_text())


def summary() -> str:
    return SUMMARY.read_text()


def principle_ids(text: str | None = None) -> set[str]:
    """Every principle ID the summary lists ("T3", ...): the IDs a reply must not show."""
    return set(re.findall(r"^- ([A-Z]\d+)\. ", text or summary(), re.MULTILINE))


# ---------- the API, with a running bill ----------

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

    def call(self, role: str, model: str, system: list[dict], user: str,
             schema: dict | None = None, effort: str | None = None,
             max_tokens: int = 16000):
        """One request. With a schema, the reply is parsed JSON; without, plain text."""
        model = MODELS.get(model, model)
        output_config = {}
        if effort:
            output_config["effort"] = effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        msg = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            **({"output_config": output_config} if output_config else {}))
        u = self.usage.setdefault(model, {"calls": 0, "input": 0, "output": 0,
                                          "cache_read": 0, "cache_write": 0})
        u["calls"] += 1
        u["input"] += msg.usage.input_tokens
        u["output"] += msg.usage.output_tokens
        u["cache_read"] += msg.usage.cache_read_input_tokens or 0
        u["cache_write"] += msg.usage.cache_creation_input_tokens or 0
        if msg.stop_reason in ("refusal", "max_tokens"):
            raise ZenError(f"{role} ({model}) stopped with {msg.stop_reason}")
        text = next((b.text for b in msg.content if b.type == "text"), "").strip()
        if not text:
            raise ZenError(f"{role} ({model}) returned no text")
        return json.loads(text) if schema else text


def cached(text: str) -> list[dict]:
    """A system prompt marked for prompt caching. The prompt and the guide summary
    are the same on every turn, so after the first call they bill at the cache-read rate."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


# ---------- retrieval ----------

_retrieving = threading.Lock()  # run.py's workers share one embedding model and database

def retrieve(query: str, k: int, model: str | None = None, mode: str = "vector") -> list[dict]:
    """The top k guide sections for a query, as {ref, heading, body}."""
    if k <= 0 or not query:
        return []
    # LanceDB and the embedding model load only when retrieval is on.
    import ask
    import models
    with _retrieving:
        rows = ask.retrieve(query, model or models.DEFAULT, mode, k)
    return [{"ref": r["ref"], "heading": r["heading"],
             "body": r["text"].split("\n\n", 1)[-1]} for r in rows]


# ---------- Zen Coder ----------

KNOWING = """\
What you know: a software-engineering guide, condensed, below. It is your private
understanding, not a reference for the developer. Never cite it: no principle IDs
(like SE1/T3), no quoting it, no "the guide says". Lead the developer through the door
rather than pointing at the sign: ask the question, or set the small step, that lets
them discover the principle for themselves, in plain words.

<guide>
{summary}
</guide>"""

ZEN_SCHEMA = {
    "type": "object",
    "properties": {
        "response": {"type": "string", "description": "exactly what the developer will see"},
        "query": {"type": "string",
                  "description": "a short search query into the guide for your next turn"},
    },
    "required": ["response", "query"],
    "additionalProperties": False,
}


def transcript(turns: list[dict], with_context: bool = False) -> str:
    """The conversation as text. with_context adds, in brackets, what Zen retrieved."""
    out = []
    for i, t in enumerate(turns, 1):
        s = f"Developer: {t['user']}\n"
        if with_context:
            refs = "; ".join(f'{c["ref"]} "{c["heading"]}"' for c in t.get("context", []))
            s += f"[retrieved for turn {i}: {refs}]\n" if refs else \
                f"[nothing retrieved for turn {i}]\n"
        if t.get("response"):
            s += f"Zen Coder: {t['response']}\n"
        out.append(s)
    return "\n".join(out)


def zen_system(prompt: str) -> list[dict]:
    return cached(f"{prompt}\n\n{KNOWING.format(summary=summary())}")


def zen_user(history: list[dict], user_msg: str, sections: list[dict]) -> str:
    background = ""
    if sections:
        body = "\n\n".join(f"[{s['ref']}] {s['heading']}\n\n{s['body']}" for s in sections)
        background = (
            "Private background for this turn: the sections of the full guide most "
            "relevant to the query you asked for. Like the summary, use what helps this "
            "developer, ignore what doesn't, and never mention, quote, or cite them.\n\n"
            f"<sections>\n{body}\n</sections>\n\n")
    so_far = transcript(history) or "(this is the first message)"
    return (f"{background}Conversation so far:\n{so_far}\nDeveloper: {user_msg}\n\n"
            "Return `response`, your reply exactly as the developer will see it, and "
            "`query`, a short search query into the guide that would help on your next turn.")


def zen_turn(claude: Claude, prompt: str, history: list[dict], user_msg: str, query: str,
             k: int, model: str = ROLES["zen"], retrieval: dict | None = None) -> dict:
    """One Zen turn: retrieve for `query`, reply to `user_msg`. Returns the turn record."""
    sections = retrieve(query, k, **(retrieval or {}))
    z = claude.call("zen", model, zen_system(prompt), zen_user(history, user_msg, sections),
                    schema=ZEN_SCHEMA, max_tokens=16000)
    return {"user": user_msg, "query": query if k else None,
            "context": [{"ref": s["ref"], "heading": s["heading"]} for s in sections],
            "response": z["response"], "next_query": z["query"]}


# ---------- the simulated novice ----------

def novice_system(scenario: dict) -> list[dict]:
    facts = "\n".join(f"- {f}" for f in scenario["facts"])
    return cached(f"""\
You are role-playing a novice developer chatting with a mentor called Zen Coder.

Who you are: {scenario['persona']}
Facts about your situation. Reveal one only when Zen Coder asks about it or it
naturally comes up; never volunteer them all at once:
{facts}

Write only your next chat message: 1-3 casual sentences, the way a novice would write.
Stay in character, and never mention that this is a role-play.""")


def novice_turn(claude: Claude, scenario: dict, turns: list[dict],
                model: str = ROLES["novice"]) -> str:
    return claude.call("novice", model, novice_system(scenario),
                       f"Conversation so far:\n{transcript(turns)}\nYour next message:",
                       effort="low", max_tokens=4000)


# ---------- a whole conversation ----------

def converse(claude: Claude, prompt: str, k: int, scenario: dict, turns: int = 3,
             models: dict | None = None, retrieval: dict | None = None,
             on_turn=None, should_stop=None) -> list[dict]:
    """Zen turn, novice turn, ... ending on a Zen turn. on_turn(n, turn) after each Zen
    turn; should_stop() is checked between turns."""
    models = {**ROLES, **(models or {})}
    out: list[dict] = []
    user_msg = query = scenario["opening"]
    for n in range(1, turns + 1):
        t = zen_turn(claude, prompt, out, user_msg, query, k, models["zen"], retrieval)
        out.append(t)
        if on_turn:
            on_turn(n, t)
        if n == turns or (should_stop and should_stop()):
            break
        user_msg = novice_turn(claude, scenario, out, models["novice"])
        query = t["next_query"]
    return out


def metrics(scenario: dict, turns: list[dict], ids: set[str] | None = None) -> dict:
    """Free, objective numbers about a conversation (the judge's scores are the other half)."""
    ids = principle_ids() if ids is None else ids
    retrieved = {c["ref"] for t in turns for c in t.get("context", [])}
    leaks = questions = words = 0
    for t in turns:
        r = t["response"]
        # SE1/X9 always counts; a bare ID only if it is real ("A0" is more likely a pin name).
        leaks += sum(1 for m in CITED.finditer(r) if m[0].startswith("SE1/") or m[1] in ids)
        leaks += len(SHOWN.findall(r))
        questions += r.count("?")
        words += len(r.split())
    relevant = scenario.get("relevant", [])
    return {"relevant_recall": sum(r in retrieved for r in relevant) / len(relevant)
            if relevant else None,
            "guide_leaks": leaks,
            "questions_per_turn": questions / len(turns),
            "words_per_turn": words / len(turns)}


# ---------- the judge ----------

SCORE = {"type": "integer", "description": "1 (poor) to 5 (excellent)"}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string",
                     "description": "the evidence for each criterion, written before scoring"},
        **{c: SCORE for c in CRITERIA},
        "facts_discovered": {"type": "array", "items": {"type": "string"},
                             "description": "hidden facts the developer revealed"},
        "first_recommendation_turn": {
            "type": "integer", "description": "first Zen turn with a concrete next step; 0 if none"},
        "verdict": {"type": "string", "description": "one or two sentences"},
    },
    "required": ["evidence", *CRITERIA, "facts_discovered", "first_recommendation_turn",
                 "verdict"],
    "additionalProperties": False,
}

RUBRIC = """\
Score each criterion from 1 (poor) to 5 (excellent). Write out the evidence first.
- understands: asks the questions that matter for this project, no more than one or two
  at a time, and doesn't interrogate.
- fits: recommends the simplest workable path for this developer and their actual constraints.
- teaches: leads the developer through the door rather than pointing at the sign. A
  question they can answer, or a small step they take, lets them discover why it matters,
  so they could reuse the idea on the next project. Lecturing a principle scores lower.
- grounded: the advice is consistent with the guide (condensed in your instructions),
  anything drawn from retrieved sections is used correctly, and irrelevant sections are
  ignored rather than forced in. Zen Coder must never show its sources: any principle ID,
  quotation, or mention of a guide or retrieval in a reply is a serious fault.
- tone: calm, concise, encouraging, not preachy.
Also list the hidden facts the developer actually revealed, and the first Zen Coder turn
(1-based) with a concrete next step."""


def judge(claude: Claude, scenario: dict, turns: list[dict],
          model: str = ROLES["judge"]) -> dict:
    facts = "\n".join(f"- {f}" for f in scenario["facts"])
    system = cached(
        "You judge conversations between Zen Coder, a mentor bot for novice developers, "
        "and a developer. Its goal: learn what the developer wants to build, then recommend "
        "good ways to build it and explain why, so they learn to build good systems.\n\n"
        f"Zen Coder knows this guide, condensed:\n<guide>\n{summary()}\n</guide>\n\n{RUBRIC}")
    user = (f"The developer: {scenario['persona']}\n"
            f"Hidden facts the developer would reveal only if asked:\n{facts}\n"
            f"Guide sections a strong answer would likely draw on: "
            f"{', '.join(scenario.get('relevant', [])) or '(none listed)'}\n\n"
            "Transcript (bracketed lines show what Zen Coder retrieved before each reply; "
            f"the developer did not see them):\n{transcript(turns, with_context=True)}")
    return claude.call("judge", model, system, user, schema=JUDGE_SCHEMA, max_tokens=32000)
