"""Run one question through the CARL controller."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from carl.core.rollout import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="Required for a remote model")
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index", help="Optional prebuilt Wikipedia Lucene index")
    parser.add_argument("--container-image", help="Restricted Linux worker image")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--allow-trusted-execution", action="store_true",
                        help="Allow generated Python with your filesystem permissions; NOT a security sandbox")
    args = parser.parse_args()
    if not args.allow_trusted_execution and not args.container_image:
        parser.error("Choose --container-image or explicit --allow-trusted-execution")
    if not Path(args.model).is_dir() and not args.revision:
        parser.error("A remote model requires an explicit --revision")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from carl.inference.backend import HFGenerator
    from carl.tools import make_executor
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16).to(args.device)
    generate = HFGenerator(model, tokenizer)
    tool = make_executor(args)
    trajectory = rollout(args.question, generate, tool, invoke_max_tokens=args.max_new_tokens,
                         initial_context=generate.render_prompt(args.question))
    print(json.dumps(asdict(trajectory), ensure_ascii=True))
    if trajectory.termination_reason != "eos":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
