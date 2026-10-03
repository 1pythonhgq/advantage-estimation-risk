"""Data alignment and validation statistics for the base-policy diagnostic critic."""
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question_key(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def question_text(row):
    for key in ("query", "problem", "question"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key].strip()
    raise ValueError("Missing question text")


def gold_answer(row):
    if isinstance(row.get("_final_answer"), str):
        return row["_final_answer"].strip()
    return row["answer"].rsplit("####", 1)[-1].strip()


def question_split(rows, validation_fraction, seed):
    if not 0 < validation_fraction < 1 or len(rows) < 2:
        raise ValueError("Need >=2 questions and validation fraction in (0,1)")
    keys = [question_key(r["question"]) for r in rows]
    ids = [r["source_index"] for r in rows]
    if len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate critic question text or source_index")
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    count = max(1, min(len(ids)-1, round(len(ids)*validation_fraction)))
    validation = sorted(ids[:count])
    train = sorted(ids[count:])
    return train, validation


def value_span(prompt_length, response_length):
    if prompt_length < 1 or response_length < 1:
        raise ValueError("Empty prompt or response")
    # At position P-1, the causal hidden state contains s0 (only the prompt).
    # End is exclusive; do not supervise the terminal state's final position.
    return prompt_length-1, prompt_length+response_length-1


def validate_examples(rows, allowed_questions):
    seen = set()
    for row in rows:
        key = (row["source_index"], row["rollout_index"])
        if key in seen or key[0] not in allowed_questions:
            raise ValueError("Repeated rollout or question in the wrong split")
        seen.add(key)
        prompt, response = row["prompt_length"], row["response_length"]
        if any(type(x) is not int or x < 0 for x in row["input_ids"]):
            raise ValueError("Invalid token IDs")
        if len(row["input_ids"]) != prompt+response or not math.isfinite(row["reward"]):
            raise ValueError("Invalid token count or reward")
        value_span(prompt, response)
    if not rows or {r["source_index"] for r in rows} != set(allowed_questions):
        raise ValueError("Missing questions in prepared data")


class RegressionStatistics:
    """Stable merge of batch population moments, without storing all predictions."""
    def __init__(self):
        self.n = 0
        self.mean_y = self.mean_e = self.m2_y = self.m2_e = self.sse = 0.0

    def add(self, n, mean_y, m2_y, mean_e, m2_e, sse):
        if n < 1 or not all(math.isfinite(x) for x in (mean_y, m2_y, mean_e, m2_e, sse)):
            raise ValueError("Invalid validation moments")
        total = self.n+n
        for suffix, mean, m2 in (("y", mean_y, m2_y), ("e", mean_e, m2_e)):
            delta = mean-getattr(self, "mean_"+suffix)
            setattr(self, "m2_"+suffix, getattr(self, "m2_"+suffix)+m2+delta*delta*self.n*n/total)
            setattr(self, "mean_"+suffix, getattr(self, "mean_"+suffix)+delta*n/total)
        self.n = total
        self.sse += sse

    def result(self):
        if not self.n:
            raise ValueError("No supervised validation positions")
        return dict(prefix_count=self.n, mse=self.sse/self.n, target_mean=self.mean_y,
                    target_variance=self.m2_y/self.n, residual_variance=self.m2_e/self.n,
                    explained_variance=None if self.m2_y <= 1e-12*self.n else 1-self.m2_e/self.m2_y)
