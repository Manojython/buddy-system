"""Named entity recognition via spaCy en_core_web_sm."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        ) from e


def extract(text: str) -> dict:
    """
    Extract named entities from text.

    Returns {"entities": [...], "label": str summary, "confidence": float}.

    spaCy en_core_web_sm doesn't expose per-entity probabilities; confidence
    is approximated from entity density (more entities found = more confident
    the model engaged meaningfully with the text).
    """
    doc = _nlp()(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    if entities:
        label = "; ".join(f"{e['text']} ({e['label']})" for e in entities)
        # Density proxy: saturates at 0.88 for 4+ entities
        confidence = min(0.88, 0.45 + len(entities) * 0.1)
    else:
        label = "(no entities found)"
        confidence = 0.2

    return {"entities": entities, "label": label, "confidence": confidence}
