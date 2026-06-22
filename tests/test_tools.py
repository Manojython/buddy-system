import pytest
from frugal.tools.classifier import classify
from frugal.tools.registry import ToolRegistry


# ── Classifier ────────────────────────────────────────────────────────────────

def test_classifier_sports():
    r = classify("The team won the championship game last season")
    assert r["label"] == "Sports"
    assert r["confidence"] > 0


def test_classifier_business():
    r = classify("The company reported record revenue and profit this quarter")
    assert r["label"] == "Business"


def test_classifier_technology():
    r = classify("The new AI model uses neural networks and algorithms")
    assert r["label"] == "Science/Technology"


def test_classifier_returns_confidence():
    r = classify("some text")
    assert "confidence" in r
    assert 0.0 <= r["confidence"] <= 1.0


def test_classifier_returns_all_scores():
    r = classify("text")
    assert "all_scores" in r
    assert set(r["all_scores"].keys()) == {"Sports", "Business", "Science/Technology", "World"}


# ── ToolRegistry ──────────────────────────────────────────────────────────────

def test_registry_run_unknown_type_returns_none():
    reg = ToolRegistry()
    assert reg.run("unknown_type", "some text") is None


def test_registry_clears_bar_true():
    reg = ToolRegistry()
    reg.register("test", lambda t: {"label": t, "confidence": 0.9}, confidence_bar=0.8)
    assert reg.clears_bar("test", 0.9)


def test_registry_clears_bar_false():
    reg = ToolRegistry()
    reg.register("test", lambda t: {"label": t, "confidence": 0.5}, confidence_bar=0.8)
    assert not reg.clears_bar("test", 0.5)


def test_registry_run_returns_result():
    reg = ToolRegistry()
    reg.register("sentiment", lambda t: {"label": "positive", "confidence": 0.9})
    result = reg.run("sentiment", "great movie")
    assert result is not None
    assert result["confidence"] == pytest.approx(0.9)
    assert result["result"]["label"] == "positive"


def test_registry_registered_types():
    reg = ToolRegistry()
    reg.register("classification", lambda t: {"label": "World", "confidence": 0.8})
    reg.register("sentiment", lambda t: {"label": "positive", "confidence": 0.9})
    assert set(reg.registered_types()) == {"classification", "sentiment"}


def test_registry_tool_exception_returns_none():
    reg = ToolRegistry()
    reg.register("bad_tool", lambda t: 1 / 0)  # always raises
    assert reg.run("bad_tool", "text") is None
