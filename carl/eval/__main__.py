"""Evaluate JSONL questions with resumable local records and recomputable metrics."""
import argparse
import asyncio
from collections import defaultdict, Counter
from dataclasses import asdict
import json
from pathlib import Path

from carl.eval.scoring import extract, score
from carl.io import digest, examples, write_json, snapshot_identity


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["dataset"]].append(row)
    result = {}
    for dataset, rows in groups.items():
        n = len(rows)
        result[dataset] = {
            "n": n, "em": sum(r["paper_em"] for r in rows)/n,
            "strict_em": sum(r["strict_em"] for r in rows)/n,
            "tool_calls": sum(r["tool_calls"] for r in rows)/n,
            "generated_tokens": sum(r["generated_tokens"] for r in rows)/n,
            "termination": dict(Counter(r["termination"] for r in rows))}
    result["overall"] = {"n": len(records), "em": sum(r["paper_em"] for r in records)/len(records)}
    for name, datasets in (("avg3", ("hotpotqa", "2wiki", "musique")),
                           ("avg5", ("gsm8k", "hotpotqa", "2wiki", "finqa", "musique"))):
        result[name] = sum(result[d]["em"] for d in datasets)/len(datasets) if all(d in result for d in datasets) else None
    return result


async def evaluate(args):
    from carl.core.rollout import rollout
    from carl.inference.batch import run_batch
    from carl.inference.backend import HFGenerator
    from carl.learning import load_model
    from carl.tools import make_executor
    rows = examples(args.data)
    identity = {k: v for k, v in vars(args).items() if k != "resume"}
    identity["data_sha256"] = digest(args.data)
    identity["model_files"] = snapshot_identity(args.model)
    identity["index_segments_sha256"] = digest(Path(args.index) / "segments_2") if args.index else None
    output = Path(args.out)
    if args.resume:
        saved = json.loads((output / "run.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Resume model/data/configuration changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "records").mkdir()
        write_json(output / "run.json", {"identity": identity, "status": "running"})
    known = {}
    lookup = {str(r["sample_id"]): r for r in rows}
    for path in (output / "records").glob("*.json"):
        import hashlib
        record = json.loads(path.read_text())
        key = str(record["sample_id"])
        if key not in lookup or path.stem != hashlib.sha256(key.encode()).hexdigest():
            raise ValueError("Unexpected resume record")
        payload = record["payload"]
        if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != record["sha256"]:
            raise ValueError("Corrupt resume record")
        example = lookup[key]
        trajectory = payload["trajectory"]
        final = trajectory["segments"][-1]["generated_text"] if trajectory["termination_reason"] == "eos" else None
        prediction = extract(final, example["dataset"])
        if (str(payload["sample_id"]) != key or payload["dataset"] != example["dataset"]
                or payload["termination"] != trajectory["termination_reason"]
                or payload["prediction"] != prediction
                or payload["paper_em"] != score(prediction, example["gold"], example["dataset"])
                or payload["strict_em"] != score(prediction, example["gold"], example["dataset"], paper=False)):
            raise ValueError("Saved prediction disagrees with its terminal trajectory or scorer")
        known[key] = payload
    execute = make_executor(args)

    async def persist(example, trajectory):
        import hashlib
        prediction = extract(trajectory.final_answer_text, example["dataset"])
        payload = {"sample_id": example["sample_id"], "dataset": example["dataset"], "prediction": prediction,
                   "paper_em": score(prediction, example["gold"], example["dataset"]),
                   "strict_em": score(prediction, example["gold"], example["dataset"], paper=False),
                   "termination": trajectory.termination_reason, "tool_calls": trajectory.total_tool_calls,
                   "generated_tokens": sum(len(s.generated_ids) for s in trajectory.segments),
                   "trajectory": asdict(trajectory)}
        key = str(example["sample_id"])
        record = {"sample_id": example["sample_id"], "payload": payload,
                  "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
        await asyncio.to_thread(write_json, output / "records" / (hashlib.sha256(key.encode()).hexdigest() + ".json"), record)
        known[key] = payload

    todo = [r for r in rows if str(r["sample_id"]) not in known]
    if not todo:
        pass
    elif args.backend == "vllm":
        from carl.inference.vllm import VLLMGenerator
        backend = VLLMGenerator(args.model, args.revision)
        try:
            await run_batch(todo, backend, execute.execute, persist, concurrency=args.concurrency)
        finally:
            backend.close()
    else:
        import torch
        device = torch.device(args.device)
        model, tokenizer = load_model(args.model, args.revision, device)
        generate = HFGenerator(model, tokenizer)
        for example in todo:
            trajectory = await asyncio.to_thread(
                rollout, example["question"], generate, execute,
                initial_context=generate.render_prompt(example["question"]))
            await persist(example, trajectory)
    records = [known[str(r["sample_id"])] for r in rows]
    metrics = summarize(records)
    write_json(output / "metrics.json", metrics)
    write_json(output / "run.json", {"identity": identity, "status": "complete", "n": len(records)})
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--index")
    parser.add_argument("--container-image")
    parser.add_argument("--allow-trusted-execution", action="store_true")
    args = parser.parse_args()
    if not Path(args.model).is_dir() and not args.revision:
        parser.error("Remote models require an explicit revision")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
