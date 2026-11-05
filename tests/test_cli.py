import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def local_model(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "four": 4, "4": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}assistant: "
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=6, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=8192,
        bos_token_id=2, eos_token_id=3, pad_token_id=0))
    path = tmp_path / "model"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def command(*args):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    result = subprocess.run([sys.executable, "-m", *map(str, args)], cwd=root, env=env,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_sft_cli_checkpoint_and_resume(local_model, tmp_path):
    data = tmp_path / "trajectories.jsonl"
    example = {"sample_id": "test", "dataset": "gsm8k", "split": "train", "question": "two plus two?", "gold": "4"}
    data.write_text(json.dumps({"example": example, "trajectory": {"termination_reason": "eos", "segments": [
        {"segment_type": "synthesize", "generated_text": "\\boxed{4}", "prompt_ids": [2, 4], "generated_ids": [5, 3]}
    ]}}) + "\n")
    out = tmp_path / "sft"
    args = ["carl.train", "sft", "--model", local_model, "--data", data, "--out", out,
            "--steps", "1", "--batch-size", "1", "--checkpoint-every", "1"]
    command(*args)
    assert (out / "step-000001" / "complete.json").is_file()
    command(*args, "--resume", out / "step-000001")
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 1


def test_evaluation_cli_and_resume_do_not_change_records(local_model, tmp_path):
    data = tmp_path / "questions.jsonl"
    data.write_text(json.dumps({"sample_id": "one", "dataset": "gsm8k", "split": "test",
                               "question": "Q", "gold": "4"}) + "\n")
    out = tmp_path / "eval"
    args = ["carl.eval", "--model", local_model, "--data", data, "--out", out,
            "--device", "cpu", "--allow-trusted-execution"]
    command(*args)
    saved = {p.name: p.read_bytes() for p in (out / "records").glob("*.json")}
    command(*args, "--resume")
    assert {p.name: p.read_bytes() for p in (out / "records").glob("*.json")} == saved
    assert json.loads((out / "metrics.json").read_text())["overall"]["n"] == 1


@pytest.mark.parametrize("method", ["trajectory_ppo", "grpo"])
def test_policy_training_cli_collects_updates_and_saves(local_model, tmp_path, method):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"sample_id": "one", "dataset": "gsm8k", "split": "train",
                               "question": "Q", "gold": "4"}) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"method": method, "rollout_batch_prompts": 1, "n_samples_per_prompt": 2,
                                  "ppo_mini_batch": 1, "ppo_epochs": 1,
                                  "invoke_max_tokens": 8, "synthesize_max_tokens": 8}))
    out = tmp_path / method
    args = ["carl.train", "ppo", "--model", local_model, "--data", data, "--out", out,
            "--steps", "1", "--batch-size", "1", "--config", config,
            "--checkpoint-every", "1", "--allow-trusted-execution"]
    command(*args)
    assert (out / "step-000001" / "complete.json").is_file()
    metrics = json.loads((out / "metrics.jsonl").read_text().splitlines()[0])
    assert metrics["policy_version"] == 1 and metrics["n_trajectories"] == 2
    import torch
    state = torch.load(out / "step-000001" / "state.pt", weights_only=False)
    assert state["ppo_update_rule"] == "combined_adam_v1"
    assert "optimizer" in state and "actor_optimizer" not in state and "critic_optimizer" not in state
    command(*args, "--resume", out / "step-000001")
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 1
