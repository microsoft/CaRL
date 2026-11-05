"""Full-parameter critic warm-up, SFT and policy optimization."""
import argparse
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import random

import torch

from carl.distributed import aggregate_training_metrics, average_gradients, barrier, initialize, rank_world
from carl.io import digest, examples, read_jsonl, write_json, snapshot_identity
from carl.learning import autocast, load_model, sft_loss, target_tokens


def restore_rng(state, rank):
    rng = state["rng_states"][rank]
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"].cpu())


def split_questions(rows, fraction, seed):
    keys = sorted({(x["dataset"], str(x["q_idx"])) for x in rows})
    if not 0 < fraction < 1 or len(keys) < 2:
        raise ValueError("Validation needs at least two questions and a fraction in (0,1)")
    random.Random(seed).shuffle(keys)
    held = set(keys[:max(1, min(len(keys)-1, int(len(keys)*fraction)))])
    return ([x for x in rows if (x["dataset"], str(x["q_idx"])) not in held],
            [x for x in rows if (x["dataset"], str(x["q_idx"])) in held])


def critic_step(value, rows, tokenizer, optimizer, device):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    value.eval()
    for row in rows:
        ids = row.get("prompt_ids") or tokenizer.encode(row["context"], add_special_tokens=False)
        if not ids or len(ids) > value.backbone.config.max_position_embeddings:
            raise ValueError("Invalid or overlong critic boundary")
        tokens = torch.tensor([ids], device=device)
        with autocast(device):
            estimate = value(tokens, torch.ones_like(tokens))
            loss = (estimate - float(row["V_target"])).square().mean() / len(rows)
        loss.backward()
        total += float(loss.detach())
    average_gradients(value.parameters())
    torch.nn.utils.clip_grad_norm_(value.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    return total


def critic(args, model, tokenizer, device, output, start, state):
    from carl.critic.head import ValueModel
    from carl.critic.validation import validation_metrics, passes_calibration_gate
    data = read_jsonl(args.data)
    if any(x.get("V_target") not in (0, 1) for x in data):
        raise ValueError("Warm-up targets must be actual binary terminal outcomes")
    if any(x.get("split") != "train" or x.get("dataset") not in ("gsm8k", "hotpotqa", "2wiki") for x in data):
        raise ValueError("Critic warm-up accepts only the three training datasets and train splits")
    train_rows, validation = split_questions(data, 0.05, args.seed)
    value = ValueModel(model.base_model)
    optimizer = torch.optim.AdamW([
        {"params": value.backbone.parameters(), "lr": 5e-7, "weight_decay": 0.01},
        {"params": value.head.parameters(), "lr": 5e-6, "weight_decay": 0.0}])
    rng = random.Random(args.seed + rank_world()[0])
    if state:
        value.head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rank_rngs"][rank_world()[0]])
        restore_rng(state, rank_world()[0])
    else:
        torch.manual_seed(args.seed + rank_world()[0])
    buckets = {}
    for row in train_rows:
        buckets.setdefault(row["bucket"], []).append(row)
    if len(buckets) != 4:
        raise ValueError("Critic warm-up requires all four tier/prompt buckets")
    passed = False
    for step in range(start, args.steps):
        for group, lr in zip(optimizer.param_groups, (5e-7, 5e-6)):
            group["lr"] = lr * min(1.0, (step + 1) / 100)
        batch = [rng.choice(buckets[rng.choice(sorted(buckets))]) for _ in range(args.batch_size // rank_world()[1])]
        loss = critic_step(value, batch, tokenizer, optimizer, device)
        metrics = {"loss": loss}
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            scores = []
            with torch.no_grad():
                for row in validation:
                    ids = row.get("prompt_ids") or tokenizer.encode(row["context"], add_special_tokens=False)
                    tokens = torch.tensor([ids], device=device)
                    scores.append(float(value(tokens, torch.ones_like(tokens))))
            metrics = validation_metrics(validation, scores)
            passed = step + 1 >= args.gate_after and passes_calibration_gate(metrics, 0.70, 0.45, 0.60)
            metrics["passed_gate"] = passed
            save(output, step + 1, model, tokenizer, {
                "head": value.head.state_dict(), "optimizer": optimizer.state_dict(), "metrics": metrics,
                "cfg": {"model_name": args.model, "auc_gate": 0.70, "ev_gate": 0.45, "sign_gate": 0.60},
            }, rng)
        log(output, step + 1, metrics)
        if passed:
            break
    if not passed:
        raise RuntimeError("Critic finished without passing the calibration gate; checkpoints are diagnostic only")


def supervised(args, model, tokenizer, device, output, start, state):
    records = read_jsonl(args.data)
    eligible = []
    for record in records:
        example = record["example"]
        if example["split"] != "train" or example["dataset"] not in ("gsm8k", "hotpotqa", "2wiki"):
            raise ValueError("SFT accepts only the three training datasets")
        trajectory = record["trajectory"]
        from carl.eval.scoring import reward
        if (trajectory["termination_reason"] == "eos" and
                reward(trajectory["segments"][-1]["generated_text"], example["gold"], example["dataset"]) == 1):
            eligible.append(trajectory)
    if not eligible:
        raise ValueError("No complete correct self-generated trajectories for SFT")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    reference = None
    if args.reference_kl:
        reference, _ = load_model(args.model, args.revision, device)
        reference.requires_grad_(False).eval()
    rng = random.Random(args.seed + rank_world()[0])
    if state:
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rank_rngs"][rank_world()[0]])
        restore_rng(state, rank_world()[0])
    else:
        torch.manual_seed(args.seed + rank_world()[0])
    for step in range(start, args.steps):
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        batch = [rng.choice(eligible) for _ in range(args.batch_size // rank_world()[1])]
        total = 0.0
        for trajectory in batch:
            for segment in trajectory["segments"]:
                prompt, action = target_tokens(segment, tokenizer)
                with autocast(device):
                    loss = sft_loss(model, reference, prompt, action, args.reference_kl)
                loss = loss / (len(batch) * len(trajectory["segments"]))
                loss.backward()
                total += float(loss.detach())
        average_gradients(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        log(output, step + 1, {"loss": total})
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            save(output, step + 1, model, tokenizer, {"optimizer": optimizer.state_dict()}, rng)


def optimize(args, model, tokenizer, device, output, start, state):
    from carl.inference.backend import HFGenerator
    from carl.ppo.shared import SharedPPOTrainer
    from carl.ppo.trainer import PPOConfig
    from carl.tools import make_executor
    raw = json.loads(Path(args.config).read_text()) if args.config else {}
    if set(raw) - {f.name for f in fields(PPOConfig)}:
        raise ValueError("Unknown PPO configuration keys")
    cfg = PPOConfig(**{**raw, "actor_model": args.model, "num_steps": args.steps})
    world = rank_world()[1]
    if cfg.rollout_batch_prompts % world or cfg.ppo_mini_batch % world:
        raise ValueError("Global rollout/minibatch sizes must be divisible by process count")
    cfg.rollout_batch_prompts //= world
    cfg.ppo_mini_batch //= world
    reference, _ = load_model(args.model, args.revision, device)
    trainer = SharedPPOTrainer(model, reference, tokenizer, cfg)
    if state:
        if state.get("ppo_update_rule") != "combined_adam_v1":
            raise ValueError("PPO resume requires a combined-optimizer checkpoint; dual-optimizer states cannot be converted")
        trainer.critic.head.load_state_dict(state["head"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.policy_version = start
        trainer._rng.setstate(state["rank_rngs"][rank_world()[0]])
        restore_rng(state, rank_world()[0])
    elif args.critic_checkpoint:
        warm = torch.load(Path(args.critic_checkpoint) / "state.pt", map_location=device, weights_only=False)
        warm_policy, _ = load_model(str(Path(args.critic_checkpoint) / "model"), None, device)
        warm["backbone"] = warm_policy.base_model.state_dict()
        trainer.load_warmup(warm)
        del warm_policy
    elif cfg.method == "carl":
        raise ValueError("CARL requires a gate-passing --critic-checkpoint")
    if not state:
        torch.manual_seed(args.seed + rank_world()[0])
    rows = examples(args.data, training=True)
    generate = HFGenerator(model, tokenizer, temperature=1)
    execute = make_executor(args)
    for step in range(start, args.steps):
        factor = min(1.0, (step + 1) / cfg.warmup_steps)
        for group, lr in zip(trainer.optimizer.param_groups, (cfg.actor_lr, cfg.critic_value_head_lr)):
            group["lr"] = lr * factor
        selected = trainer._rng.choices(rows, k=cfg.rollout_batch_prompts)
        generate.policy_version = trainer.policy_version
        with autocast(device):
            metrics = trainer.update(trainer.collect(selected, generate, execute))
        log(output, step + 1, metrics)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.steps:
            save(output, step + 1, model, tokenizer, {
                "head": trainer.critic.head.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                "ppo_update_rule": "combined_adam_v1",
            }, trainer._rng)


def log(output, step, metrics):
    metrics = aggregate_training_metrics(metrics)
    if rank_world()[0] == 0:
        record = {"step": step, **metrics}
        print(json.dumps(record, allow_nan=False), flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def save(output, step, model, tokenizer, state, rng):
    import torch.distributed as dist
    local = {"python": rng.getstate(), "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
    states = [None] * rank_world()[1]
    if dist.is_initialized():
        dist.all_gather_object(states, local)
    else:
        states = [local]
    if rank_world()[0] == 0:
        path = output / f"step-{step:06d}"
        staging = output / f".step-{step:06d}.tmp"
        staging.mkdir()
        model.save_pretrained(staging / "model")
        tokenizer.save_pretrained(staging / "model")
        torch.save({**state, "step": step, "rank_rngs": [s["python"] for s in states], "rng_states": states,
                    "world_size": rank_world()[1]}, staging / "state.pt")
        write_json(staging / "complete.json", {"step": step, "status": "complete"})
        staging.rename(path)
    barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("critic", "sft", "ppo"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", help="Completed checkpoint directory in this output run")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--gate-after", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--reference-kl", type=float, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config")
    parser.add_argument("--critic-checkpoint")
    parser.add_argument("--index")
    parser.add_argument("--container-image")
    parser.add_argument("--allow-trusted-execution", action="store_true")
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (2400 if args.method == "critic" else 500)
    args.batch_size = args.batch_size if args.batch_size is not None else (256 if args.method == "critic" else 128)
    if (min(args.steps, args.batch_size, args.checkpoint_every, args.gate_after) < 1 or
            not math.isfinite(args.lr) or args.lr <= 0 or
            not math.isfinite(args.reference_kl) or args.reference_kl < 0):
        parser.error("Training sizes/rates must be positive; reference KL finite and nonnegative")
    device = initialize()
    rank, world = rank_world()
    if args.batch_size % world:
        parser.error("Global batch size must be divisible by world size")
    identity = {k: v for k, v in vars(args).items() if k != "resume"}
    identity.update(data_sha256=digest(args.data), config_sha256=digest(args.config) if args.config else None, world_size=world)
    identity["model_files"] = snapshot_identity(args.model)
    identity["index_segments_sha256"] = digest(Path(args.index) / "segments_2") if args.index else None
    output = Path(args.out)
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("Resume checkpoint must belong to this output run")
    if rank == 0:
        if args.resume:
            if json.loads((output / "run.json").read_text()) != identity:
                raise ValueError("Resume model/data/config/world size differs")
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "run.json", identity)
    barrier()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(str(Path(args.resume) / "model") if args.resume else args.model,
                                  None if args.resume else args.revision, device)
    state = torch.load(Path(args.resume) / "state.pt", map_location=device, weights_only=False) if args.resume else None
    start = state["step"] if state else 0
    if state:
        if state["world_size"] != world or not (Path(args.resume) / "complete.json").is_file():
            raise ValueError("Resume requires a complete checkpoint from the same world size")
        restore_rng(state, rank)
    model.gradient_checkpointing_enable()
    {"critic": critic, "sft": supervised, "ppo": optimize}[args.method](args, model, tokenizer, device, output, start, state)


if __name__ == "__main__":
    main()
