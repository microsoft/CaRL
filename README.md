# CaRL

Competence-aware reinforcement learning for LLM tool use.

CaRL separates a tool-using answer into **invoke**, **assimilate** and
**commit** segments. Python produces an observation; the model extracts the
useful information into a `<context>` block. The runtime replaces raw output
with that extraction before the next decision.

## Current scope

This change introduces the reviewed protocol and credit-assignment primitives:

- The shared rollout controller and trajectory records.
- Structured tool results and Python/context boundary detection.
- Checkpoint-specific initial prompt rendering.
- Segment advantages: `V(next) - V(current)`, with terminal `R - V(last)`.

The controller accepts generation and execution callbacks. This foundation does
not include a model backend, Python executor, retriever, optimizer or benchmark
runner, and does not claim to reproduce the paper's results.

## Layout

```text
carl/
  core/          # rollout, records, boundaries and initial prompt
  ppo/           # segment-level advantage calculation
tests/           # focused checks for the included modules
```

Additional training and inference modules will be introduced separately after
review. Empty modules and placeholder entry points are deliberately omitted.

## Development

Python 3.11 or later:

```shell
pip install -e ".[test]"
python -m pytest -q
```

These tests run without a model download, retrieval index or GPU. They cover
state transitions, observation replacement, comment preservation, recorded
diagnostics and the telescoping segment-credit identity.

## License

The included source retains its [MIT license](LICENSE).