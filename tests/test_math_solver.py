import pytest
from frugal.tools.math_solver import solve


def test_correct_subtraction_confirmed():
    r = solve("she has 16 - 3 = 13 eggs left.")
    assert r["confidence"] == pytest.approx(0.92)
    assert "13" in r["label"]
    assert r["label"] == "she has 16 - 3 = 13 eggs left."


def test_wrong_subtraction_corrected():
    r = solve("she has 16 - 3 = 10 eggs left.")
    assert r["confidence"] == pytest.approx(0.92)
    assert "= 13" in r["label"]


def test_wrong_multiplication_corrected():
    r = solve("she makes 9 * $2 = $20.")
    assert r["confidence"] == pytest.approx(0.92)
    assert "= 18" in r["label"]


def test_correct_multiplication_unchanged():
    r = solve("she makes 9 * $2 = $18.")
    assert r["confidence"] == pytest.approx(0.92)
    assert r["label"] == "she makes 9 * $2 = $18."


def test_percentage_wrong_corrected():
    r = solve("40% of 200 = 90 GB")
    assert r["confidence"] == pytest.approx(0.92)
    assert "= 80" in r["label"]


def test_percentage_correct_unchanged():
    r = solve("40% of 200 = 80 GB")
    assert r["confidence"] == pytest.approx(0.92)
    assert r["label"] == "40% of 200 = 80 GB"


def test_no_equation_returns_zero_confidence():
    r = solve("She starts with 16 eggs and eats 3, so")
    assert r["confidence"] == 0.0
    assert r["label"] == "She starts with 16 eggs and eats 3, so"


def test_no_equation_header():
    r = solve("Here is how to solve this step by step:")
    assert r["confidence"] == 0.0


def test_addition_corrected():
    r = solve("total cost is 40 + 60 = 90 dollars")
    assert r["confidence"] == pytest.approx(0.92)
    assert "= 100" in r["label"]


def test_empty_string():
    r = solve("")
    assert r["confidence"] == 0.0
