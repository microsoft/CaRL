"""Detect <context>...</context> assimilation blocks."""
import re
from dataclasses import dataclass


@dataclass
class ContextBlockDetection:
    content: str
    start: int
    end: int


_CTX = re.compile(r"<context>(.*?)</context>", re.DOTALL)


def detect_context_block(text: str) -> ContextBlockDetection | None:
    m = _CTX.search(text)
    if m is None:
        return None
    return ContextBlockDetection(content=m.group(1).strip(), start=m.start(), end=m.end())
