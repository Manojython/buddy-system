"""Semantic similarity via sentence-transformers (all-MiniLM-L6-v2).

Input convention: two texts separated by ' ||| '.
This separator is injected by the pipeline when it detects a similarity node.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def score(text: str) -> dict:
    """
    Compute cosine similarity between two sentences.

    Returns {"score": float [0,1], "label": str, "confidence": float}.
    """
    if " ||| " not in text:
        return {"score": 0.5, "label": "unknown", "confidence": 0.1}

    a, b = text.split(" ||| ", 1)
    emb_a, emb_b = _model().encode([a.strip(), b.strip()])

    cos = float(
        np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9)
    )
    # Normalize cosine from [-1, 1] to [0, 1]
    normalized = (cos + 1.0) / 2.0

    if normalized > 0.80:
        label = "very similar"
    elif normalized > 0.55:
        label = "similar"
    elif normalized > 0.35:
        label = "somewhat similar"
    else:
        label = "dissimilar"

    # Confidence: higher when the score is far from the ambiguous midpoint (0.5)
    confidence = min(0.92, abs(normalized - 0.5) * 2.0)

    return {"score": normalized, "label": label, "confidence": confidence}
