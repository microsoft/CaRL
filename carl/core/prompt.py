"""Initial chat rendering shared by inference and on-policy collection."""
from typing import Protocol

from carl.core.rollout import CARL_SYSTEM_PROMPT


class ChatTokenizer(Protocol):
    chat_template: str | dict | None

    def apply_chat_template(self, messages: list[dict[str, str]], *,
                            tokenize: bool, add_generation_prompt: bool) -> str: ...


def render_initial_context(tokenizer: ChatTokenizer, question: str) -> str:
    if not tokenizer.chat_template:
        raise ValueError("A checkpoint-specific chat template is required")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty")
    return tokenizer.apply_chat_template([
        {"role": "system", "content": CARL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ], tokenize=False, add_generation_prompt=True)
