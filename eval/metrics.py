"""Metrics for comparing pipeline conditions.

Mirrors Anthropic's own reporting (accuracy delta vs. cost delta) and adds a
third column: percentage of flagged nodes resolved without any LLM call — the
number that makes the classical-tool tier's case in the article.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


# Pricing per million tokens (2026-06)
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00},
    "claude-opus-4-8":           {"input": 5.00, "output": 25.00},
}


@dataclass
class ConditionResult:
    condition: str
    samples: int
    correct: int
    cloud_calls: int
    classical_calls: int
    total_input_tokens: int
    total_output_tokens: int
    latency_ms: float
    cost_usd: float = 0.0  # set explicitly; accounts for multi-model conditions
    per_dataset: dict[str, tuple[int, int]] = field(default_factory=dict)  # dataset → (correct, total)

    @property
    def accuracy(self) -> float:
        return self.correct / self.samples if self.samples else 0.0

    @property
    def cloud_avoided_pct(self) -> float:
        total = self.cloud_calls + self.classical_calls
        return self.classical_calls / total if total else 0.0

    @property
    def estimated_cost_usd(self) -> float:
        return self.cost_usd

    def as_row(self) -> dict:
        return {
            "Condition": self.condition,
            "Accuracy": f"{self.accuracy:.1%}",
            "Cloud calls": self.cloud_calls,
            "Classical calls": self.classical_calls,
            "Cloud avoided": f"{self.cloud_avoided_pct:.1%}",
            "Est. cost": f"${self.estimated_cost_usd:.4f}",
            "Latency (ms)": f"{self.latency_ms:.0f}",
        }


@dataclass
class EvalScore:
    correct: bool
    prediction: str
    score: float


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9./+-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _number(text: str) -> str:
    return re.sub(r"[^0-9.\-]", "", text)


def _last_label(text: str, labels: dict[str, list[str]]) -> str:
    haystack = text.lower()
    matches: list[tuple[int, str]] = []
    for canonical, variants in labels.items():
        for variant in variants:
            pattern = r"(?<![a-z0-9])" + re.escape(variant.lower()) + r"(?![a-z0-9])"
            matches.extend((m.start(), canonical) for m in re.finditer(pattern, haystack))
    if not matches:
        return ""
    return max(matches, key=lambda item: item[0])[1]


def _arc_answer(text: str) -> str:
    tail = text[-500:]
    patterns = [
        r"(?i)(?:final\s+answer|answer|correct\s+(?:answer|option)|option)\s*(?:is|:)?\s*[\(\"']?([A-E])\b",
        r"(?i)\b([A-E])\s*(?:is\s+)?(?:the\s+)?(?:correct|answer)\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, tail))
        if matches:
            return matches[-1].group(1).upper()

    standalone = list(re.finditer(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", tail))
    return standalone[-1].group(1).upper() if standalone else ""


def _squad_normalize(text: str) -> list[str]:
    """Standard SQuAD normalization: lowercase, strip articles + punctuation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return [t for t in text.split() if t]


def _squad_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _squad_normalize(prediction)
    truth_tokens = _squad_normalize(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _squad_answer(text: str) -> str:
    """Extract the answer phrase from Gemma's output.

    Uses the LAST "Answer:" line so Sonnet's synthesis (appended after Gemma's
    raw chain) takes precedence over any wrong "Answer:" Gemma wrote mid-chain.
    """
    matches = list(re.finditer(r"(?i)\bAnswer:\s*(.+)", text))
    if matches:
        return matches[-1].group(1).strip().split("\n")[0].strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


def _gsm8k_answer(text: str) -> str:
    matches = list(re.finditer(r"(?i)answer\s*:\s*(-?[\d,]+(?:\.\d+)?)", text))
    if matches:
        return _number(matches[-1].group(1))

    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _number(numbers[-1]) if numbers else ""


def _extraction_score(expected: str, output: str) -> EvalScore:
    expected_tokens = set(_clean(expected).split())
    output_tokens = set(_clean(output).split())
    if not expected_tokens:
        return EvalScore(correct=not output_tokens, prediction="", score=1.0 if not output_tokens else 0.0)

    overlap = expected_tokens & output_tokens
    precision = len(overlap) / len(output_tokens) if output_tokens else 0.0
    recall = len(overlap) / len(expected_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Extraction output is often verbose. For this article/demo eval, require
    # near-complete recall while tolerating some extra explanatory text.
    return EvalScore(
        correct=recall >= 0.85,
        prediction=" ".join(sorted(overlap)),
        score=f1,
    )


def evaluate_output(sample: object, output: str) -> EvalScore:
    """Extract and score the answer for one eval sample.

    This avoids inflated scores from substring matching against verbose chain-of-
    thought output, especially for multiple-choice letters and label lists.
    """
    dataset = getattr(sample, "dataset")
    expected = str(getattr(sample, "ground_truth"))

    if dataset == "ag_news":
        labels = {
            "World": ["world"],
            "Sports": ["sports", "sport"],
            "Business": ["business"],
            "Science/Technology": ["science/technology", "science and technology", "technology", "science"],
        }
        pred = _last_label(output, labels)
        return EvalScore(correct=pred == expected, prediction=pred, score=1.0 if pred == expected else 0.0)

    if dataset == "sst2":
        labels = {"positive": ["positive"], "negative": ["negative"]}
        pred = _last_label(output, labels)
        return EvalScore(correct=pred == expected, prediction=pred, score=1.0 if pred == expected else 0.0)

    if dataset == "stsb":
        labels = {
            "very similar": ["very similar"],
            "dissimilar": ["dissimilar", "not similar"],
            "similar": ["similar"],
        }
        pred = _last_label(output, labels)
        return EvalScore(correct=pred == expected, prediction=pred, score=1.0 if pred == expected else 0.0)

    if dataset in ("arc_challenge", "mmlu"):
        pred = _arc_answer(output)
        return EvalScore(correct=pred == expected, prediction=pred, score=1.0 if pred == expected else 0.0)

    if dataset == "gsm8k":
        pred = _gsm8k_answer(output)
        truth = _number(expected)
        return EvalScore(correct=pred == truth, prediction=pred, score=1.0 if pred == truth else 0.0)

    if dataset == "wikiann":
        return _extraction_score(expected, output)

    if dataset in ("squad", "hotpotqa"):
        pred = _squad_answer(output)
        f1 = _squad_f1(pred, expected)
        return EvalScore(correct=f1 >= 0.5, prediction=pred, score=f1)

    pred = _clean(output)
    truth = _clean(expected)
    return EvalScore(correct=pred == truth, prediction=pred, score=1.0 if pred == truth else 0.0)


def max_tokens_for(sample: object) -> int:
    dataset = getattr(sample, "dataset")
    if dataset in {"ag_news", "sst2", "stsb", "arc_challenge"}:
        return 64
    if dataset == "wikiann":
        return 96
    if dataset == "gsm8k":
        return 512
    if dataset == "mmlu":
        return 512
    if dataset in ("squad", "hotpotqa"):
        return 512
    return 128


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, PRICING["claude-sonnet-4-6"])
    return input_tokens * p["input"] / 1_000_000 + output_tokens * p["output"] / 1_000_000


def compare_datasets(results: list[ConditionResult]) -> None:
    """Print per-dataset accuracy breakdown for each condition."""
    if not results:
        return

    all_datasets = sorted(
        {ds for r in results for ds in r.per_dataset}
    )
    if not all_datasets:
        return

    conditions = [r.condition.split("(")[0].strip() for r in results]
    col0_w = max(len(ds) for ds in all_datasets) + 2
    col_w = max(max(len(c) for c in conditions) + 2, 10)

    header_parts = [f"{'Dataset':<{col0_w}}"] + [f"{c:<{col_w}}" for c in conditions]
    print(" | ".join(header_parts))
    print("-+-".join(["-" * col0_w] + ["-" * col_w for _ in conditions]))

    for ds in all_datasets:
        row = [f"{ds:<{col0_w}}"]
        for r in results:
            if ds in r.per_dataset:
                c, t = r.per_dataset[ds]
                row.append(f"{c}/{t} ({c/t:.0%})" if t else f"0/0 (n/a)")
            else:
                row.append(f"{'—':<{col_w}}")
        print(" | ".join(f"{cell:<{col_w}}" if i > 0 else cell for i, cell in enumerate(row)))


def compare(results: list[ConditionResult]) -> None:
    """Print a fixed-width comparison table to stdout."""
    if not results:
        return

    headers = list(results[0].as_row().keys())
    rows = [list(r.as_row().values()) for r in results]

    col_w = [
        max(len(h), max(len(str(row[i])) for row in rows)) + 2
        for i, h in enumerate(headers)
    ]
    sep = "-+-".join("-" * w for w in col_w)
    fmt = " | ".join(f"{{:<{w}}}" for w in col_w)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
