"""Prepare public examples, competence tiers, tool rollouts and critic boundaries."""
import argparse
from collections import defaultdict, Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

from carl.io import examples, read_jsonl


def prepare(source, dataset, split):
    """Convert the original public JSON schemas without oracle context injection."""
    data = read_jsonl(source) if str(source).endswith(".jsonl") else json.loads(Path(source).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data", data)
    if not isinstance(data, list):
        raise ValueError("Source must be a JSON array, JSONL, or a data array")
    for i, item in enumerate(data):
        source_id = str(item.get("_id", item.get("id", i)))
        if dataset == "finqa":
            parts = item.get("pre_text", []) + [
                "\n".join("| " + " | ".join(map(str, row)) + " |" for row in item["table"])
            ] + item.get("post_text", []) + [item["qa"]["question"]]
            question = "\n\n".join(parts)
            gold = str(item["qa"]["exe_ans"])
        else:
            question = item["question"]
            answer = item["answer"]
            gold = answer.rsplit("####", 1)[-1].strip() if dataset == "gsm8k" else answer
        yield {"sample_id": f"{dataset}:{split}:{source_id}", "dataset": dataset,
               "split": split, "question": question, "gold": gold}


def choose_buckets(records, count, tier2_fraction, seed):
    buckets = defaultdict(list)
    for row in records:
        buckets[(row["tier"], row["prompt_mode"])].append(row)
    targets = {
        ("tier2", "no_tool"): round(count * tier2_fraction / 2),
        ("tier2", "forced_tool"): round(count * tier2_fraction / 2),
    }
    remaining = count - sum(targets.values())
    targets.update({("tier1", "no_tool"): remaining // 2, ("tier1", "forced_tool"): remaining - remaining // 2})
    rng = random.Random(seed)
    selected = []
    for key, n in targets.items():
        if len(buckets[key]) < n:
            raise ValueError(f"Bucket {key} has {len(buckets[key])} trajectories; needs {n}")
        clusters = defaultdict(list)
        for row in buckets[key]:
            clusters[row.get("cluster", 0)].append(row)
        for pool in clusters.values():
            rng.shuffle(pool)
        for i in range(n):
            available = [c for c in sorted(clusters) if clusters[c]]
            cluster = available[i % len(available)]
            selected.append(clusters[cluster].pop())
    return selected


def boundary_pairs(records):
    for row in records:
        for i, segment in enumerate(row["trajectory"]["segments"]):
            yield {
                "dataset": row["example"]["dataset"], "q_idx": row["example"]["sample_id"],
                "split": row["example"]["split"],
                "rollout_idx": row["rollout_idx"], "prompt_mode": row["prompt_mode"],
                "tier": row["tier"], "bucket": f"{row['tier']}/{row['prompt_mode']}",
                "boundary_kind": "pre_segment", "boundary_index": i,
                "context": segment["start_context"], "prompt_ids": segment["prompt_ids"],
                "V_target": row["trajectory"]["reward"],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "tiers", "collect", "pairs", "cluster"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "hotpotqa", "2wiki", "finqa", "musique"))
    parser.add_argument("--split", choices=("train", "dev", "validation", "test"))
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index")
    parser.add_argument("--container-image")
    parser.add_argument("--allow-trusted-execution", action="store_true")
    parser.add_argument("--count", type=int, default=32000)
    parser.add_argument("--tier2-fraction", type=float, default=0.35)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--sign-labels", help="Independent boundary-sign validation labels; never used as training targets")
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        parser.error("Use a new output file; inputs are never overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        if not args.dataset or not args.split:
            parser.error("prepare requires --dataset and --split")
        result = prepare(args.data, args.dataset, args.split)
    elif args.action == "pairs":
        if args.count < 4 or not 0 < args.tier2_fraction < 1:
            parser.error("Warm-up count must be at least4 and the tier fraction in (0,1)")
        records = choose_buckets(read_jsonl(args.data), args.count, args.tier2_fraction, args.seed)
        pairs = list(boundary_pairs(records))
        if args.sign_labels:
            labels = {}
            for label in read_jsonl(args.sign_labels):
                key = tuple(str(label[k]) for k in ("dataset", "q_idx", "prompt_mode", "rollout_idx", "boundary_index"))
                if key in labels or label["expected_value_change"] not in (-1, 1) or not label.get("value_change_source"):
                    raise ValueError("Invalid/duplicate independent sign label")
                labels[key] = label
            matched = set()
            for row in pairs:
                key = tuple(str(row[k]) for k in ("dataset", "q_idx", "prompt_mode", "rollout_idx", "boundary_index"))
                if key in labels:
                    row.update({k: labels[key][k] for k in ("expected_value_change", "value_change_source")})
                    matched.add(key)
            if matched != set(labels):
                raise ValueError("Sign labels refer to unavailable boundary states")
        result = pairs
    elif args.action == "cluster":
        if not args.embedding_revision:
            parser.error("Clustering requires an explicit --embedding-revision")
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        rows = examples(args.data, training=True)
        encoder = SentenceTransformer(args.embedding_model, revision=args.embedding_revision, device=args.device)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["dataset"]].append(row)
        result = []
        for dataset, group in grouped.items():
            n = 30 if dataset == "gsm8k" else 50
            features = encoder.encode([r["question"] for r in group])
            labels = KMeans(n_clusters=min(n, len(group)), random_state=args.seed, n_init=10).fit_predict(features)
            result.extend(dict(row, cluster=int(label)) for row, label in zip(group, labels))
    else:
        import torch
        from carl.learning import load_model
        from carl.inference.backend import HFGenerator
        from carl.core.prompt import render_initial_context
        from carl.core.rollout import rollout, CODE_STOPS, CARL_SYSTEM_PROMPT
        from carl.eval.scoring import reward
        from carl.tools import make_executor
        if not args.model:
            parser.error("Collection requires --model and a pinned revision or local snapshot")
        torch.manual_seed(args.seed)
        model, tokenizer = load_model(args.model, args.revision, torch.device(args.device))
        generate = HFGenerator(model, tokenizer, temperature=1)
        rows = examples(args.data, training=True)
        execute = make_executor(args) if args.action == "collect" else None

        def collect():
            for row in rows:
                if args.action == "tiers":
                    prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": "Answer directly without tools or code. Put your answer inside \\boxed{}."},
                        {"role": "user", "content": row["question"]}],
                        tokenize=False, add_generation_prompt=True)
                    outputs = [generate(prompt, CODE_STOPS, 2048) for _ in range(5)]
                    successes = [int(o["finish_reason"] == "eos" and reward(o["text"], row["gold"], row["dataset"])) for o in outputs]
                    yield {**row, "tier": "tier2" if any(successes) else "tier1",
                           "no_tool_successes": successes, "no_tool_outputs": outputs}
                else:
                    if row.get("tier") not in ("tier1", "tier2") or len(row.get("no_tool_outputs", [])) != 5:
                        raise ValueError("collect requires output from the tiers command")
                    example = {k: row[k] for k in ("sample_id", "dataset", "split", "question", "gold")}
                    o = row["no_tool_outputs"][0]
                    # Preserve the actual no-tool generation, not a resampled answer.
                    from carl.core.segment import Segment, Trajectory
                    segment = Segment(start_context=tokenizer.decode(o["prompt_ids"], skip_special_tokens=False,
                        clean_up_tokenization_spaces=False), generated_text=o["text"], generated_ids=o["ids"],
                        prompt_ids=o["prompt_ids"], generation_finish_reason=o["finish_reason"],
                        termination="eos" if o["finish_reason"] == "eos" else "truncated")
                    tr = Trajectory(segments=[segment], full_context=segment.start_context+o["text"],
                                    termination_reason="eos" if o["finish_reason"] == "eos" else "token_limit")
                    tr.reward = reward(tr.final_answer_text, row["gold"], row["dataset"])
                    yield {"example": example, "tier": row["tier"], "cluster": row.get("cluster", 0),
                           "rollout_idx": 0, "prompt_mode": "no_tool", "trajectory": asdict(tr)}
                    initial = tokenizer.apply_chat_template([
                        {"role": "system", "content": CARL_SYSTEM_PROMPT + "\nYou must invoke a tool before answering."},
                        {"role": "user", "content": row["question"]}],
                        tokenize=False, add_generation_prompt=True)
                    tr = rollout(row["question"], generate, execute, initial_context=initial)
                    tr.reward = reward(tr.final_answer_text, row["gold"], row["dataset"])
                    if tr.total_tool_calls == 0:
                        raise ValueError("Forced-tool collection produced no tool call; do not silently relabel it")
                    yield {"example": example, "tier": row["tier"], "cluster": row.get("cluster", 0),
                           "rollout_idx": 1, "prompt_mode": "forced_tool", "trajectory": asdict(tr)}
        result = collect()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        n = 0
        for row in result:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            n += 1
    temporary.rename(destination)
    print(json.dumps({"records": n, "output": str(destination)}))


if __name__ == "__main__":
    main()
