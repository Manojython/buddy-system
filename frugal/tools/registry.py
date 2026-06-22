"""Central registry mapping node types to classical ML/NLP tool callables."""
from __future__ import annotations

from typing import Callable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[[str], dict]] = {}
        self._bars: dict[str, float] = {}

    def register(
        self, node_type: str, fn: Callable[[str], dict], confidence_bar: float = 0.8
    ) -> None:
        """Register a classical tool for a node type.

        fn must accept a text string and return a dict with at least:
            {"label": str, "confidence": float, ...}
        """
        self._tools[node_type] = fn
        self._bars[node_type] = confidence_bar

    def run(self, node_type: str, text: str) -> dict | None:
        """Run the registered tool for node_type.

        Returns {"result": <tool output dict>, "confidence": float, "bar": float}
        or None if no tool is registered or the tool raises.
        """
        if node_type not in self._tools:
            return None
        try:
            result = self._tools[node_type](text)
            return {
                "result": result,
                "confidence": float(result.get("confidence", 0.0)),
                "bar": self._bars[node_type],
            }
        except Exception:
            return None

    def clears_bar(self, node_type: str, confidence: float) -> bool:
        bar = self._bars.get(node_type, 1.0)
        return confidence >= bar

    def registered_types(self) -> list[str]:
        return list(self._tools.keys())


_default_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _default_registry
