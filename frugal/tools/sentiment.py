"""Coarse sentiment scoring via VADER lexicon.

VADER is designed for short social-media text and works without any model download.
It is adequate for the coarse positive/negative/neutral split that most sentiment
nodes require; fine-grained cases fall through to the cloud buddy.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def score(text: str) -> dict:
    """
    Score sentiment of text.

    Returns {"label": "positive"|"negative"|"neutral", "compound": float, "confidence": float}.
    """
    polarity = _analyzer().polarity_scores(text)
    compound = polarity["compound"]  # in [-1, 1]

    if compound >= 0.05:
        label = "positive"
        confidence = min(0.95, 0.55 + compound * 0.4)
    elif compound <= -0.05:
        label = "negative"
        confidence = min(0.95, 0.55 + abs(compound) * 0.4)
    else:
        label = "neutral"
        # VADER is less reliable in the neutral band — keep confidence low
        confidence = 0.45

    return {"label": label, "compound": compound, "confidence": confidence}
