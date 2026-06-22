"""Classical arithmetic verifier — catches wrong computations in generated clauses.

Looks for explicit equations like "16 - 3 = 10" or "40% of 200 = 90" in the
clause text and verifies them. If wrong, corrects the number in-place. Returns
low confidence when no equation is found so the router falls through to Sonnet.

No SymPy needed for basic arithmetic — reserved for symbolic/algebraic clauses.
"""
from __future__ import annotations

import re


def _num(s: str) -> float:
    return float(re.sub(r"[$,%]", "", s).replace(",", ""))


_EQ_RE = re.compile(
    r"(\$?-?[\d,]+(?:\.\d+)?)\s*([+\-×÷*])\s*(\$?-?[\d,]+(?:\.\d+)?)"
    r"\s*=\s*(\$?-?[\d,]+(?:\.\d+)?)"
)
_PCT_RE = re.compile(
    r"(-?[\d]+(?:\.\d+)?)\s*%\s+of\s+(-?[\d]+(?:\.\d+)?)\s*=\s*(-?[\d]+(?:\.\d+)?)"
)

_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "×": lambda a, b: a * b,
    "÷": lambda a, b: a / b if b else None,
}


def solve(text: str) -> dict:
    """Verify arithmetic in a reasoning clause. Returns corrected text + confidence."""
    result = text
    found = False

    for m in _PCT_RE.finditer(text):
        found = True
        pct, base, claimed = float(m.group(1)), float(m.group(2)), float(m.group(3))
        actual = pct / 100 * base
        if abs(actual - claimed) > 0.01:
            result = result.replace(m.group(0), f"{m.group(1)}% of {m.group(2)} = {actual:g}")

    for m in _EQ_RE.finditer(text):
        found = True
        try:
            a, b, claimed = _num(m.group(1)), _num(m.group(3)), _num(m.group(4))
        except ValueError:
            continue
        fn = _OPS.get(m.group(2))
        if fn is None:
            continue
        actual = fn(a, b)
        if actual is not None and abs(actual - claimed) > 0.01:
            result = result.replace(
                m.group(0),
                f"{m.group(1)} {m.group(2)} {m.group(3)} = {actual:g}",
            )

    return {"label": result, "confidence": 0.92 if found else 0.0}
