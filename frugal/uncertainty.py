"""Extract what Gemma is uncertain about from per-token entropy logs.

Instead of sending an entire clause to Sonnet, we find the specific token with
the highest entropy in the window, then use spaCy to widen it to the smallest
meaningful linguistic unit (named entity → noun chunk → word) that Sonnet can
fact-check against a document chunk.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UncertainSpan:
    text: str
    entropy: float
    clause: str


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_uncertain_span(clause: str, token_entropies: list[float]) -> UncertainSpan | None:
    """Return the highest-entropy *named entity or noun chunk* in clause.

    The naive approach — find max-entropy token, return the word at that index —
    fires constantly on function words (and, with, in, of) because those positions
    have high structural entropy (many possible continuations) but zero factual
    uncertainty. We need NER-gated extraction:

    1. Run spaCy NER + noun chunks on the clause.
    2. For each candidate span, map its character offsets to approximate token
       indices and compute average entropy across those tokens.
    3. Return the candidate with the highest average entropy — this targets
       factual spans ("Byzantine Empire", "Seljuk Turks") over function words.
    4. Only fall back to the raw max-entropy word if no candidates found.
    """
    if not token_entropies or not clause.strip():
        return None

    n_tokens = len(token_entropies)
    n_chars = len(clause)

    def _char_to_token_idx(char_idx: int) -> int:
        """Linear interpolation from char offset to approximate token index."""
        return min(int(char_idx / max(n_chars, 1) * n_tokens), n_tokens - 1)

    def _span_entropy(start_char: int, end_char: int) -> float:
        t0 = _char_to_token_idx(start_char)
        t1 = _char_to_token_idx(end_char)
        if t0 > t1:
            t0, t1 = t1, t0
        window = token_entropies[t0:t1 + 1]
        return max(window) if window else 0.0

    try:
        nlp = _get_nlp()
        doc = nlp(clause)

        best_text = None
        best_entropy = 0.0

        # Named entities are the primary targets — specific, factual, checkable
        for ent in doc.ents:
            e = _span_entropy(ent.start_char, ent.end_char)
            if e > best_entropy:
                best_entropy = e
                best_text = ent.text

        # Noun chunks as fallback — broader but still semantic
        if best_text is None:
            for chunk in doc.noun_chunks:
                # Skip determiners-only chunks ("the", "a")
                content = [t for t in chunk if t.pos_ not in ("DET", "PRON")]
                if not content:
                    continue
                e = _span_entropy(chunk.start_char, chunk.end_char)
                if e > best_entropy:
                    best_entropy = e
                    best_text = chunk.text

        if best_text:
            return UncertainSpan(best_text, best_entropy, clause)

    except Exception:
        pass

    # Last resort: return the max-entropy word, but strip function words
    _STOP = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
             "of", "for", "with", "by", "is", "are", "was", "were", "be",
             "it", "its", "this", "that", "also", "as", "from", "not"}
    words = clause.split()
    if not words:
        return None
    pairs = sorted(
        ((i, e) for i, (e, w) in enumerate(zip(token_entropies, words))
         if w.strip(".,;:!?\"'()").lower() not in _STOP),
        key=lambda x: x[1], reverse=True,
    )
    if not pairs:
        return None
    idx, entropy = pairs[0]
    word = words[min(idx, len(words) - 1)].strip(".,;:!?\"'()")
    return UncertainSpan(word, entropy, clause) if word else None
