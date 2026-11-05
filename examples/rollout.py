"""Exercise a tool round without downloading model weights."""
from dataclasses import asdict
import json

from carl.core.rollout import rollout
from carl.sandbox.trusted import TrustedExecutor


def main():
    responses = iter([
        ("I will multiply the two supplied numbers.\n```python\nprint(347 * 28)\n```", "stop"),
        ("<context>The calculated product of 347 and 28 is 9716.</context>", "stop"),
        (r"The calculated product is \boxed{9716}.", "eos"),
    ])

    def generate(**request):
        text, finish = next(responses)
        return {"text": text, "finish_reason": finish}

    trajectory = rollout("What is 347 times 28?", generate,
                         TrustedExecutor(allow_trusted_execution=True))
    print(json.dumps(asdict(trajectory), indent=2))


if __name__ == "__main__":
    main()
