import asyncio
import json

import pytest

from carl.data.__main__ import boundary_pairs, choose_buckets, prepare
from carl.eval.__main__ import summarize
from carl.eval.scoring import extract, reward, score
from carl.inference.batch import run_batch
from carl.io import examples
from carl.core.tool import ToolResult


def test_nested_box_and_paper_vs_strict():
    assert extract(r"The answer is \boxed{\text{New York}}.", "hotpotqa") == r"\text{New York}"
    assert extract(r"\boxed{unfinished", "gsm8k") is None
    assert score("Born in New York City", ["New York"], "hotpotqa")
    assert not score("Born in New York City", "New York", "hotpotqa", paper=False)
    assert not score("yesterday", "yes", "hotpotqa")
    assert score("1,000.00", "1000", "gsm8k")
    assert not score("nan", "0", "finqa")


@pytest.mark.parametrize("text", [None, "", " ", "\t\r\n"])
@pytest.mark.parametrize("dataset", ["gsm8k", "hotpotqa", "2wiki", "finqa", "musique"])
def test_empty_answers_are_incorrect(text, dataset):
    assert extract(text, dataset) is None
    assert not score(extract(text, dataset), "4", dataset)
    assert reward(text, "4", dataset) == 0.0


def test_macro_average_is_not_population_weighted():
    rows = [{"dataset": d, "paper_em": d == "gsm8k", "strict_em": False,
             "tool_calls": 0, "generated_tokens": 1, "termination": "eos"}
            for d in ["gsm8k", "hotpotqa", "hotpotqa", "2wiki", "finqa", "musique"]]
    metrics = summarize(rows)
    assert metrics["avg5"] == pytest.approx(0.2)
    assert metrics["overall"]["em"] == pytest.approx(1/6)


def test_finqa_conversion_preserves_context_and_source_split(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps([{"id": "a", "pre_text": ["In millions"], "table": [["2001", 10]],
                               "post_text": ["after table"], "qa": {"question": "total?", "exe_ans": 10}}]))
    row = list(prepare(path, "finqa", "test"))[0]
    assert "In millions" in row["question"] and "| 2001 | 10 |" in row["question"]
    assert row["split"] == "test" and row["gold"] == "10"
    normalized = tmp_path / "normalized.jsonl"
    normalized.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Training"):
        examples(normalized, training=True)


def test_four_bucket_selection_fails_instead_of_inventing_data():
    with pytest.raises(ValueError, match="Bucket"):
        choose_buckets([], 4, 0.5, 1)
    rows = [{"tier": t, "prompt_mode": m, "cluster": 0} for t in ("tier1", "tier2")
            for m in ("no_tool", "forced_tool")]
    assert len(choose_buckets(rows, 4, 0.5, 1)) == 4


def test_async_batch_keeps_trajectories_independent_and_persists():
    class Backend:
        def render_prompt(self, question):
            return question

        async def __call__(self, prompt, **kwargs):
            await asyncio.sleep(0)
            if "[TOOL OUTPUT]" in prompt:
                text, finish = "<context>four</context>", "stop"
            elif "four" in prompt:
                text, finish = r"\boxed{4}", "eos"
            else:
                text, finish = "```python\nprint(2+2)\n```", "stop"
            return {"text": text, "finish_reason": finish, "ids": [1]}

    async def execute(code):
        return ToolResult("test", "success", "raw")

    records = []
    async def persist(row, trajectory):
        records.append((row["sample_id"], trajectory))

    asyncio.run(run_batch([{"sample_id": str(i), "question": str(i)} for i in range(4)],
                          Backend(), execute, persist, concurrency=2))
    assert len(records) == 4
    assert all(t.termination_reason == "eos" and "[TOOL OUTPUT]" not in t.full_context for _, t in records)
