"""Terminal-answer normalization; the paper metric also accepts multiword containment."""
from decimal import Decimal, InvalidOperation
import re
import string


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join(text.translate(str.maketrans("", "", string.punctuation)).split())


def number(text):
    try:
        value = Decimal(str(text).strip().replace(",", "").replace("$", "").replace("%", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def extract(text, dataset):
    if not text or not text.strip():
        return None
    answers = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answers:
        return answers[-1].strip()
    start = text.rfind("\\boxed{")
    if start >= 0:
        depth = 1
        for i in range(start + 7, len(text)):
            depth += int(text[i] == "{") - int(text[i] == "}")
            if not depth:
                return text[start + 7:i].strip()
        return None
    if dataset in ("gsm8k", "finqa"):
        matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return matches[-1] if matches else None
    return text.strip().splitlines()[-1].strip()


def score(prediction, gold, dataset, paper=True):
    if prediction is None:
        return False
    if isinstance(gold, (list, tuple)):
        return any(score(prediction, g, dataset, paper) for g in gold)
    if dataset in ("gsm8k", "finqa"):
        p, g = number(prediction), number(gold)
        if p is not None and g is not None:
            return p == g
    p, g = normalize(prediction), normalize(gold)
    return bool(g) and (p == g or paper and len(g.split()) > 1 and g in p)


def reward(text, gold, dataset):
    return float(score(extract(text, dataset), gold, dataset))
