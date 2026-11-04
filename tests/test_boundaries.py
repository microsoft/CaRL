import pytest

from carl.core.boundaries import python_boundary
from carl.core.context_block_detector import detect_context_block


@pytest.mark.parametrize("label", ["python", "py", "Python"])
def test_python_boundary_accepts_complete_labeled_block(label):
    text = f"Reasoning.\n```{label}\n# Keep this explanation\nprint(4)\n```"
    detected = python_boundary(text)
    assert detected is not None
    assert detected.executable == "# Keep this explanation\nprint(4)"


@pytest.mark.parametrize("text", [
    "```\nprint(4)\n```", "```javascript\nprint(4)\n```",
    "```python\nprint(4)", "No tool block",
])
def test_python_boundary_rejects_unlabeled_unsupported_and_incomplete_blocks(text):
    assert python_boundary(text) is None


def test_context_detection_preserves_offsets_and_accepts_empty_extraction():
    text = "prefix<context>\nUseful fact\n</context>suffix"
    result = detect_context_block(text)
    assert result.content == "Useful fact"
    assert text[result.start:result.end] == "<context>\nUseful fact\n</context>"
    assert detect_context_block("<context></context>").content == ""
    assert detect_context_block("<context>unfinished") is None
