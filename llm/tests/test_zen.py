"""
Offline tests for Zen Coder: a fake API client stands in for Claude, so these
check the turn protocol, the metrics, the bill and the run files, not whether
any model is any good. That is what a run and its judge are for.
"""
import json
from types import SimpleNamespace

import pytest

import chunks
from zen import coder, score
from zen import run as zrun


class FakeMessages:
    """Answers like each role would: Zen with {response, query}, the judge with scores,
    the novice with plain text. Records every request."""

    def __init__(self, stop_reason="end_turn"):
        self.requests, self.stop_reason = [], stop_reason

    def create(self, **req):
        self.requests.append(req)
        schema = req.get("output_config", {}).get("format", {}).get("schema", {})
        props = schema.get("properties", {})
        n = len(self.requests)
        if "response" in props:
            text = json.dumps({"response": f"What would done look like? (reply {n})",
                               "query": f"query {n}"})
        elif "verdict" in props:
            text = json.dumps({"evidence": "fine", **{c: 4 for c in coder.CRITERIA},
                               "facts_discovered": ["MacBook"],
                               "first_recommendation_turn": 2, "verdict": "Solid."})
        else:
            text = f"novice message {n}"
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=3000, cache_creation_input_tokens=0))


REAL_CLAUDE = coder.Claude


def fake(stop_reason="end_turn") -> coder.Claude:
    return REAL_CLAUDE(SimpleNamespace(messages=FakeMessages(stop_reason)))


SCENARIO = {"opening": "I want an alarm clock.", "persona": "A student.",
            "facts": ["Uses a MacBook."], "relevant": ["guide/08-work.md#w6"]}


def test_a_conversation_alternates_and_chains_queries():
    claude = fake()
    turns = coder.converse(claude, "Be Zen.", 0, SCENARIO, turns=3)
    assert len(turns) == 3
    assert turns[0]["user"] == SCENARIO["opening"]
    assert turns[1]["user"].startswith("novice message")
    assert all(t["context"] == [] and t["query"] is None for t in turns)    # k = 0
    reqs = claude.client.messages.requests
    assert len(reqs) == 5                        # three Zen turns, two novice turns
    # The guide summary rides in the cached system prompt, not in each message.
    system = reqs[0]["system"][0]
    assert "<guide>" in system["text"] and system["cache_control"] == {"type": "ephemeral"}


def test_retrieval_uses_the_previous_turns_query(monkeypatch):
    asked = []
    monkeypatch.setattr(coder, "retrieve", lambda q, k, **kw: asked.append(q) or [
        {"ref": "guide/08-work.md#w6", "heading": "W6", "body": "thin slice"}])
    turns = coder.converse(fake(), "Be Zen.", 3, SCENARIO, turns=2)
    assert asked == [SCENARIO["opening"], turns[0]["next_query"]]
    assert coder.metrics(SCENARIO, turns)["relevant_recall"] == 1.0


def test_metrics_count_shown_sources_but_not_pin_names():
    turns = [{"response": "See SE1/T3, and T3 again. Wire it to A0.", "context": []},
             {"response": "As the guide says: retrieval found it. Ready?", "context": []}]
    m = coder.metrics(SCENARIO, turns, ids={"T3"})
    assert m["guide_leaks"] == 4                 # SE1/T3, T3, "the guide", "retrieval"
    assert m["questions_per_turn"] == 0.5


def test_a_refusal_is_an_error_not_an_empty_reply():
    with pytest.raises(coder.ZenError, match="refusal"):
        coder.zen_turn(fake("refusal"), "Be Zen.", [], "hi", "hi", 0)


def test_the_bill_counts_every_token_type():
    claude = fake()
    coder.zen_turn(claude, "Be Zen.", [], "hi", "hi", 0, model="sonnet")
    p_in, p_out, p_read = coder.PRICES[coder.MODELS["sonnet"]]
    assert claude.cost == pytest.approx((1000 * p_in + 200 * p_out + 3000 * p_read) / 1e6)


def test_a_run_writes_its_files_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(zrun, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(score, "RUNS", tmp_path)
    monkeypatch.setattr(coder, "Claude", lambda: fake())
    argv = ["--run", "t", "--prompts", "baseline", "--ks", "0", "--scenarios", "alarm",
            "--repeats", "2", "--turns", "2", "--workers", "2"]
    zrun.main(argv)
    run_dir = tmp_path / "t"
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["state"], status["done"], status["failed"]) == ("done", 2, 0)
    assert zrun.state(run_dir) == "done"
    df = score.load("t")
    assert len(df) == 2 and (df["overall"] == 4).all() and (df["cost"] > 0).all()

    zrun.main(argv)                              # same name: nothing left to do
    status = json.loads((run_dir / "status.json").read_text())
    assert (status["done"], status["skipped"]) == (0, 2)


def test_prompts_and_scenarios_are_usable():
    prompts, scenarios = coder.load_prompts(), coder.load_scenarios()
    assert {"baseline", "door"} <= set(prompts)
    refs = {c.ref for c in chunks.chunk_guide(chunks.find_guide())}
    for name, s in scenarios.items():
        assert s["opening"] and s["persona"] and s["facts"], name
        assert set(s["relevant"]) <= refs, name


def test_a_conversation_that_fails_is_counted_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(zrun, "RUNS_DIR", tmp_path)

    def no_key():
        raise RuntimeError("no API key")
    monkeypatch.setattr(coder, "Claude", no_key)
    zrun.main(["--run", "t", "--prompts", "baseline", "--ks", "0", "--scenarios", "alarm"])
    status = json.loads((tmp_path / "t" / "status.json").read_text())
    assert (status["state"], status["failed"]) == ("failed", 1)
    assert "no API key" in status["errors"][0]
