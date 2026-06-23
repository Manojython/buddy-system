"""Document chunking and semantic retrieval for targeted Sonnet queries.

When Gemma is uncertain about a specific span, we need the smallest passage
excerpt that answers the uncertainty — not the full document. sentence-transformers
cosine similarity finds the most relevant chunk.
"""
from __future__ import annotations

import re

import numpy as np

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _sentence_chunks(document: str, window: int = 4, step: int = 2) -> list[str]:
    """Sliding window of sentences — overlap helps avoid missing spans at edges."""
    sentences = re.split(r'(?<=[.!?])\s+', document.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= window:
        return [document]
    chunks = []
    for i in range(0, len(sentences) - window + 1, step):
        chunks.append(" ".join(sentences[i:i + window]))
    if not chunks:
        chunks = [document]
    return chunks


def retrieve_relevant_chunk(document: str, query: str) -> str:
    """Return the document chunk most semantically similar to the query.

    For short documents (SQuAD passages ~300 words) this will often return the
    full document. For longer docs it returns the most relevant window.
    """
    chunks = _sentence_chunks(document)
    if len(chunks) == 1:
        return document

    encoder = _get_encoder()
    chunk_vecs = encoder.encode(chunks, show_progress_bar=False)
    query_vec = encoder.encode([query], show_progress_bar=False)[0]

    norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
    sims = np.dot(chunk_vecs, query_vec) / norms

    return chunks[int(np.argmax(sims))]
