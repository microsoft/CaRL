"""JSON persistence, input validation and file identity checks."""
import hashlib
import json
import os
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def snapshot_identity(path):
    directory = Path(path)
    if not directory.is_dir():
        return None
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".model", ".jinja"))
    if not files:
        raise ValueError("Local model directory contains no model/tokenizer files")
    return {p.name: digest(p) for p in files}


def examples(path, training=False):
    rows = read_jsonl(path)
    ids = set()
    for row in rows:
        if not {"sample_id", "dataset", "question", "gold", "split"} <= row.keys():
            raise ValueError("Examples require sample_id, dataset, question, gold and split")
        if (not isinstance(row["sample_id"], str) or not row["sample_id"]
                or not isinstance(row["question"], str) or not row["question"].strip()
                or row["sample_id"] in ids):
            raise ValueError("Questions must be nonempty and sample IDs unique")
        if row["dataset"] not in ("gsm8k", "hotpotqa", "2wiki", "finqa", "musique"):
            raise ValueError("Unsupported dataset name")
        if training and (row["split"] != "train" or row["dataset"] not in ("gsm8k", "hotpotqa", "2wiki")):
            raise ValueError("Training accepts only GSM8K, HotpotQA and 2Wiki training splits")
        ids.add(row["sample_id"])
    if not rows:
        raise ValueError("Input examples are empty")
    return rows
