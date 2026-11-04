import pytest

from carl.core.prompt import render_initial_context
from carl.core.rollout import CARL_SYSTEM_PROMPT


class Tokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        assert messages == [
            {"role": "system", "content": CARL_SYSTEM_PROMPT},
            {"role": "user", "content": "Question"},
        ]
        return "Rendered question with assistant prefix"


def test_initial_context_uses_checkpoint_chat_template():
    assert render_initial_context(Tokenizer(), "Question") == "Rendered question with assistant prefix"


def test_missing_template_or_empty_question_is_rejected():
    tokenizer = Tokenizer()
    with pytest.raises(ValueError, match="Question"):
        render_initial_context(tokenizer, " ")
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        render_initial_context(tokenizer, "Question")
