"""Structural boundaries on generated text only, never on a rendered prompt."""
import re

from carl.core.code_block_detector import CodeBlockDetection, detect_code_block

_PYTHON_OPEN = re.compile(r"^```(?:python|py|Python)[ \t]*\r?\n", re.MULTILINE)


def python_boundary(generated_text: str) -> CodeBlockDetection | None:
    detected = detect_code_block(generated_text)
    if detected is not None and _PYTHON_OPEN.match(generated_text, detected.start):
        return detected
    return None
