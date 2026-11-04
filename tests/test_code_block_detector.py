from dataclasses import asdict

import pytest

from carl.core.code_block_detector import detect_code_block


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_comments_remain_in_executable_body_without_separate_storage(newline):
    body = newline.join([
        "# Explain the inputs",
        "current_age = 8",
        "# Calculate elapsed years, not the future age",
        "print(3 * current_age - current_age)",
    ])
    block = f"```python{newline}{body}{newline}```"
    text = "Reasoning before the call." + newline + block + newline
    result = detect_code_block(text)
    assert result is not None
    assert result.code == result.executable == body
    assert text[result.start:result.end] == block
    assert "comments" not in asdict(result)


@pytest.mark.parametrize("body", ["", "   ", "# comment only", "\n# comment only\n\n# another comment"])
def test_empty_and_comment_only_blocks_have_no_executable_code(body):
    result = detect_code_block(f"```python\n{body}\n```")
    assert result is not None
    assert result.code == body.strip()
    assert result.executable == ""


def test_incomplete_block_is_not_closed_by_the_detector():
    assert detect_code_block("```python\n# A comment\nprint(4)") is None
