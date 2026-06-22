import pytest
from eval.metrics import (
    evaluate_output, token_cost, EvalScore,
    _gsm8k_answer, _arc_answer, _extraction_score,
)
from eval.datasets import EvalSample


def _sample(dataset, ground_truth, node_type="classification"):
    return EvalSample(prompt="", ground_truth=ground_truth,
                      node_type=node_type, dataset=dataset)


# ── GSM8K ────────────────────────────────────────────────────────────────────

def test_gsm8k_finds_answer_line():
    assert _gsm8k_answer("Step 1... Step 2...\nAnswer: 18") == "18"


def test_gsm8k_answer_with_comma():
    assert _gsm8k_answer("Answer: 1,000") == "1000"


def test_gsm8k_fallback_to_last_number():
    assert _gsm8k_answer("she earns 18 dollars") == "18"


def test_gsm8k_evaluate_correct():
    s = _sample("gsm8k", "18", "reasoning")
    score = evaluate_output(s, "Step by step...\nAnswer: 18")
    assert score.correct
    assert score.score == 1.0


def test_gsm8k_evaluate_wrong():
    s = _sample("gsm8k", "18", "reasoning")
    score = evaluate_output(s, "Answer: 10")
    assert not score.correct


# ── ARC-Challenge ─────────────────────────────────────────────────────────────

def test_arc_extracts_letter():
    assert _arc_answer("The answer is B") == "B"


def test_arc_extracts_last_letter():
    assert _arc_answer("Could be A or B. The correct answer is C.") == "C"


def test_arc_standalone_letter():
    assert _arc_answer("B") == "B"


def test_arc_evaluate_correct():
    s = _sample("arc_challenge", "C", "reasoning")
    score = evaluate_output(s, "C")
    assert score.correct


def test_arc_evaluate_wrong():
    s = _sample("arc_challenge", "C", "reasoning")
    score = evaluate_output(s, "A")
    assert not score.correct


# ── SST-2 ─────────────────────────────────────────────────────────────────────

def test_sst2_positive():
    s = _sample("sst2", "positive")
    score = evaluate_output(s, "positive")
    assert score.correct


def test_sst2_picks_last_label():
    s = _sample("sst2", "negative")
    score = evaluate_output(s, "starts positive ends negative")
    assert score.correct


# ── WikiANN extraction ────────────────────────────────────────────────────────

def test_extraction_exact_match():
    score = _extraction_score("Kanye West", "Kanye West")
    assert score.correct


def test_extraction_partial_recall():
    score = _extraction_score("Kanye West Jamie Foxx Gold Digger", "Kanye West Jamie Foxx Gold Digger")
    assert score.correct


def test_extraction_missing_entity_fails():
    score = _extraction_score("Kanye West Jamie Foxx Gold Digger", "Kanye West")
    assert not score.correct  # recall < 0.85


def test_extraction_empty_expected():
    score = _extraction_score("", "")
    assert score.correct


# ── Pricing ───────────────────────────────────────────────────────────────────

def test_token_cost_sonnet():
    cost = token_cost("claude-sonnet-4-6", 1_000_000, 0)
    assert cost == pytest.approx(3.0)


def test_token_cost_haiku():
    cost = token_cost("claude-haiku-4-5-20251001", 0, 1_000_000)
    assert cost == pytest.approx(5.0)


def test_token_cost_unknown_model_defaults_to_sonnet():
    cost = token_cost("unknown-model", 1_000_000, 0)
    assert cost == pytest.approx(3.0)
