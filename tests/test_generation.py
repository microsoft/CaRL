import pytest

from carl.inference.backend import HFGenerator, SegmentBoundary


class Tokens:
    def __init__(self, value):
        self.value = value
        self.shape = (1, len(value))

    def __getitem__(self, key):
        return Tokens(self.value[key[1]])

    def tolist(self):
        return self.value


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


@pytest.mark.parametrize("prefix,body,assimilate", [
    ("Prompt contains ```python\nprint(1)\n```", "", False),
    ("", "```python\nprint(1)", False),
    ("", "<context>unfinished", True),
])
def test_stop_ignores_prompt_and_partial_blocks(prefix, body, assimilate):
    check = SegmentBoundary(Tokenizer(), len(prefix), assimilate)
    assert not check(Tokens(list(map(ord, prefix + body))), None)


@pytest.mark.parametrize("body,assimilate", [
    ("```python\nprint(1)\n```", False), ("<context>fact</context>", True),
])
def test_stop_recognizes_native_structural_boundary(body, assimilate):
    check = SegmentBoundary(Tokenizer(), 0, assimilate)
    assert check(Tokens(list(map(ord, body))), None)


def test_model_setup_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="Temperature"):
        HFGenerator(None, None, temperature=float("nan"))
