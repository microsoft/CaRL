# CARL

Competence-aware reinforcement learning for LLM tool use.

The same controller runs three kinds of segment during training and inference:

1. **Invoke:** generate reasoning and a Python block; execute the code.
2. **Assimilate:** extract useful output into a `<context>` block.
3. **Commit:** answer the question and end with the model's natural EOS.

Raw tool output is visible during assimilation, then replaced by the extraction
before decoding continues. The critic assigns segment credit from changes in
predicted terminal success.

## Setup and quick start

Use Python 3.11+ and a PyTorch build suitable for your CPU/CUDA/ROCm environment.
Run commands from the repository root.

```shell
pip install -e ".[model,test]"
pytest
python examples/rollout.py
```

The example uses scripted model responses and executes fixed arithmetic on the
host. It demonstrates Invoke -> Assimilate -> Commit without downloading weights.

## Tool execution

For generated code, build the restricted Linux worker image with Docker running:

```shell
docker build -t carl-python docker
```

Pass `--container-image carl-python` to collection, PPO, inference and evaluation.
Each call starts a fresh worker with bounded runtime/output and no retained
variables. Containers run unprivileged, read-only and without network access.
Startup failure does not fall back to host execution.

Alternatively, explicitly use `--allow-trusted-execution` for trusted host code.
**Host execution is not a security sandbox:** generated code retains your
filesystem/network permissions. Use a disposable environment without secrets;
Windows also lacks the host worker's POSIX resource/process-group guarantees.
Container restrictions are not a production security certification.

For `search(query)`, install `pip install -e ".[retrieval]"`, provision a Wikipedia
Pyserini Lucene index and Java, then add `--index INDEX` to tool-using commands.
The parent handles search requests; workers do not need network access. BM25 uses
top-k 3, k1=0.9 and b=0.4. The paper's index snapshot is `20210120.d1b9e6`.

## Inference and evaluation

`MODEL` below is a local model/tokenizer directory with a chat template. For a
remote model, supply its identifier and `--revision COMMIT`. Transformers
inference/evaluation also accepts `--device cpu`.

```shell
python -m carl.inference --model MODEL --question "What is 347 times 28?" --container-image carl-python
python -m carl.eval --model MODEL --data data/eval.jsonl --out outputs/eval --container-image carl-python
```

Add `--index INDEX` for retrieval questions. Evaluation saves per-example
trajectories and metrics: paper/strict answer matching, tool calls, tokens and
terminations. Failures remain in accuracy denominators. Add `--resume` to the same
evaluation command to retain completed examples. For asynchronous evaluation,
install `.[vllm]` and add `--backend vllm --concurrency 8`; the adapter targets
vLLM 0.7/0.8's V0 engine.

## Training

### Data and critic warm-up

Obtain the original [GSM8K](https://github.com/openai/grade-school-math),
[HotpotQA](https://hotpotqa.github.io/),
[2WikiMQA](https://github.com/Alab-NII/2wikimultihop),
[FinQA](https://github.com/czyssrs/FinQA) and
[MuSiQue](https://github.com/StonyBrookNLP/musique) releases separately.
Normalized JSONL rows have this format; `gold` may also be a list of aliases:

```json
{"sample_id":"example:1","dataset":"gsm8k","split":"train","question":"What is two plus two?","gold":"4"}
```

Convert each dataset with `prepare`, preserving its source split. Combine only
GSM8K, HotpotQA and 2Wiki training splits into `data/train.jsonl`; keep evaluation
questions separate. [sample_data/](sample_data/) contains small synthetic
format illustrations.

```shell
python -m carl.data prepare --dataset gsm8k --split train --data raw/train.jsonl --out data/gsm8k_train.jsonl
pip install -e ".[data]"
python -m carl.data cluster --data data/train.jsonl --out data/clustered.jsonl --embedding-revision COMMIT
python -m carl.data tiers --model MODEL --data data/clustered.jsonl --out data/tiers.jsonl
python -m carl.data collect --model MODEL --data data/tiers.jsonl --out data/rollouts.jsonl --index INDEX --container-image carl-python
python -m carl.data pairs --data data/rollouts.jsonl --out data/pairs.jsonl --sign-labels data/signs.jsonl
python -m carl.train critic --model MODEL --data data/pairs.jsonl --out outputs/critic --steps 2400
```

Collection saves actual no-tool and forced-tool trajectories. Warm-up samples
four tier/prompt buckets and holds out whole questions for validation.
`signs.jsonl` must contain independently checked transition directions, not
directions inferred from final reward. Each label identifies a boundary and its
change to the next boundary:

```json
{"dataset":"hotpotqa","q_idx":"source-id","prompt_mode":"forced_tool","rollout_idx":1,"boundary_index":1,"expected_value_change":1,"value_change_source":"checked retrieval evidence"}
```

Warm-up must pass competence AUC >= 0.70, transition-sign accuracy >= 0.60 and
explained variance >= 0.45, eligible from step 1800. Missing evidence cannot pass.
Use the actual passing checkpoint below; step 1800 is only an example.

### CaRL, trajectory PPO and GRPO

```shell
python -m carl.train ppo --model MODEL --data data/train.jsonl --out outputs/carl --config configs/carl.json --critic-checkpoint outputs/critic/step-001800 --index INDEX --container-image carl-python
python -m carl.train ppo --model MODEL --data data/train.jsonl --out outputs/ppo --config configs/trajectory_ppo.json --index INDEX --container-image carl-python
python -m carl.train ppo --model MODEL --data data/train.jsonl --out outputs/grpo --config configs/grpo.json --index INDEX --container-image carl-python
```

CaRL uses adjacent boundary-value differences and a final reward residual.
Trajectory PPO shares an initial-value-baselined return across segments; GRPO
uses within-prompt relative returns. PPO uses a frozen reference and one combined
actor/weighted-critic AdamW update on the shared backbone.

### Rejection-sampled SFT

```shell
python -m carl.train sft --model MODEL --data data/rollouts.jsonl --out outputs/sft --steps 500
```

SFT accepts only complete correct training trajectories and supervises generated
action tokens, not prompts or tool observations. Optional `--reference-kl 0.001`
adds full-vocabulary KL(reference || policy); the default is zero.

Training saves every 25 updates by default (`--checkpoint-every`), including
optimizer and per-rank RNG states. Resume with the original command plus
`--resume outputs/RUN/step-NNNNNN`. Only complete, compatible, trusted checkpoints
are accepted. Evaluate saved checkpoints separately by passing their `model`
subdirectory to the evaluation command above.

Training uses native PyTorch. For multiple GPUs, replace `python` with
`torchrun --standalone --nproc-per-node 8`. Each device must fit the full policy,
reference and optimizer. Global batches must divide the process count. Gradients
and logged training means are averaged across ranks; trajectory counts are summed.

## Code flow

- [core/rollout.py](carl/core/rollout.py): segment state machine and context replacement.
- [inference/](carl/inference/): generation and native boundary stopping; vLLM can reuse unchanged prefixes.
- [sandbox/](carl/sandbox/) and [retrieval/](carl/retrieval/): Python execution and parent-side search.
- [ppo/shared.py](carl/ppo/shared.py): collection, frozen old scores/advantages and combined updates.
- [train.py](carl/train.py) and [eval/](carl/eval/): training/checkpoints and answer scoring.

## Citation

Abhijit Kumar, Zoey Wu, and Mohit Suley. **Knowing When to Ask: Segment-Level
Credit Assignment for LLM Tool Use.** 2026.
[arXiv:2605.27788](https://arxiv.org/abs/2605.27788).
[DOI](https://doi.org/10.48550/arXiv.2605.27788).
Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

## License

Code uses the [MIT license](LICENSE); models, datasets and retrieval corpora
retain their own licenses.
