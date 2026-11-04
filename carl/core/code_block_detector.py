"""Extract a complete fenced code block without inventing delimiters."""
import re
from dataclasses import dataclass


@dataclass
class CodeBlockDetection:
    code: str
    executable: str
    start: int
    end: int


_CODE_BLOCK_PATTERN = re.compile(
    r"(^```(?:python|py|Python)?[ \t]*\r?\n)(.*?)(^```[ \t]*(?=\r?$))",
    re.DOTALL | re.MULTILINE,
)


def detect_code_block(text: str) -> CodeBlockDetection | None:
    m = _CODE_BLOCK_PATTERN.search(text)
    if m is None:
        return None
    body = m.group(2)
    has_code = any(line.strip() and not line.lstrip().startswith("#") for line in body.splitlines())
    return CodeBlockDetection(
        code=body.strip(),
        executable=body.strip() if has_code else "",
        start=m.start(),
        end=m.end(),
    )
