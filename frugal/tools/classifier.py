"""Text classification via keyword-scored heuristic + optional spaCy textcat.

The heuristic is used when no trained spaCy textcat model is available.
It is intentionally lightweight — the point is to be fast and cheap, not to
beat a fine-tuned LLM. Nodes that score low confidence here fall through to
the cloud buddy automatically.
"""
from __future__ import annotations

from functools import lru_cache

_KEYWORD_MAP: dict[str, list[str]] = {
    "Sports": [
        "game", "player", "team", "score", "match", "win", "loss",
        "championship", "league", "season", "tournament", "athlete",
    ],
    "Business": [
        "company", "market", "stock", "revenue", "profit", "ceo",
        "merger", "acquisition", "earnings", "economic", "startup", "investor",
    ],
    "Science/Technology": [
        "research", "study", "technology", "software", "computer", "ai",
        "space", "scientific", "data", "algorithm", "model", "neural",
    ],
    "World": [
        "country", "government", "president", "war", "election",
        "international", "foreign", "military", "nation", "political", "treaty",
    ],
}


@lru_cache(maxsize=1)
def _spacy_nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def classify(text: str, categories: list[str] | None = None) -> dict:
    """
    Classify text into one of the given categories (default: AG-News style).

    Returns {"label": str, "confidence": float, "all_scores": dict}.
    """
    cats = categories or list(_KEYWORD_MAP.keys())
    text_lower = text.lower()

    scores: dict[str, float] = {}
    for cat in cats:
        keywords = _KEYWORD_MAP.get(cat, [])
        if keywords:
            scores[cat] = sum(kw in text_lower for kw in keywords) / len(keywords)
        else:
            scores[cat] = 0.0

    best = max(scores, key=lambda k: scores[k])
    raw_score = scores[best]
    # Scale: each 10% keyword hit rate → ~0.5 confidence; cap at 0.92
    confidence = min(0.92, raw_score * 5.0) if raw_score > 0 else 0.15

    return {"label": best, "confidence": confidence, "all_scores": scores}
