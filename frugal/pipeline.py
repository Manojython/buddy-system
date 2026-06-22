"""
Main orchestration pipeline.

Flow per turn:
  1. Local model generates token by token, yielding (token_text, entropy).
  2. EntropyMonitor (Rust) watches entropy; fires interrupt at a clause boundary
     when entropy crossed the threshold inside the current window.
  3. On interrupt: the current node is finalized and sent to the Router (Rust).
  4. Router checks the node type against the classical-tool registry.
     - Classical tool available and confident → use its result directly.
     - No tool, or tool is unsure → ask the CloudBuddy.
  5. The patch (from either source) is spliced into the ReasoningGraph (Rust)
     and appended to the output. Generation continues with this context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from frugal.model import LocalModel
from frugal.buddy import CloudBuddy
from frugal.tools.registry import ToolRegistry

try:
    from frugal._bridge import EntropyMonitor, ReasoningGraph, Router

    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False

# Sentence / clause boundary: end-of-sentence punctuation, or a comma before
# a coordinating conjunction (simplified but adequate for demo).
_BOUNDARY_RE = re.compile(
    r"[.!?]\s*$|;\s*$|,\s+(?:and|but|so|because|however|therefore|which|while)\b",
    re.IGNORECASE,
)


@dataclass
class Escalation:
    node_id: str
    node_type: str
    original_text: str
    routed_to: str  # "classical" or "cloud"
    tool_name: str | None
    result_text: str
    confidence: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class PipelineResult:
    output: str
    graph_nodes: list[dict]
    cloud_calls: int
    classical_calls: int
    total_input_tokens: int
    total_output_tokens: int
    escalations: list[Escalation] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        local_model: LocalModel,
        buddy: CloudBuddy,
        registry: ToolRegistry,
        entropy_threshold: float = 2.5,
        min_tokens_per_boundary: int = 10,
        router_confidence_bar: float = 0.8,
    ):
        self.local_model = local_model
        self.buddy = buddy
        self.registry = registry

        if _BRIDGE_AVAILABLE:
            self._monitor = EntropyMonitor(entropy_threshold, min_tokens_per_boundary)
            self._graph = ReasoningGraph()
            self._router = Router(router_confidence_bar)
            for tool_type in registry.registered_types():
                self._router.register(tool_type, tool_type)
        else:
            self._monitor = None
            self._graph = None
            self._router = None

    def run(
        self,
        prompt: str,
        node_type_hint: str = "claim",
        max_tokens: int = 512,
        temp: float = 0.0,
    ) -> PipelineResult:
        formatted = self.local_model.format_prompt(prompt)

        output_parts: list[str] = []
        current_tokens: list[str] = []
        current_entropies: list[float] = []
        parent_id: str | None = None
        escalations: list[Escalation] = []
        cloud_calls = 0
        classical_calls = 0
        total_input = 0
        total_output = 0

        for token_text, entropy in self.local_model.generate(
            formatted,
            max_tokens=max_tokens,
            temp=temp,
        ):
            current_tokens.append(token_text)
            current_entropies.append(entropy)

            accumulated = "".join(current_tokens)
            is_boundary = bool(_BOUNDARY_RE.search(accumulated))

            should_interrupt = False
            if self._monitor and is_boundary:
                should_interrupt = self._monitor.update(entropy, is_boundary)
            elif self._monitor:
                self._monitor.update(entropy, False)

            if is_boundary and current_tokens:
                node_text = accumulated.strip()
                avg_entropy = sum(current_entropies) / len(current_entropies)
                # Convert average entropy to a confidence proxy in [0, 1]
                avg_confidence = 1.0 / (1.0 + avg_entropy)

                if self._graph:
                    node_id = self._graph.add_node(
                        node_type_hint, node_text, parent_id, avg_confidence
                    )
                else:
                    node_id = f"n{len(output_parts)}"

                if should_interrupt:
                    node = {
                        "id": node_id,
                        "type": node_type_hint,
                        "text": node_text,
                        "confidence": avg_confidence,
                    }
                    esc = self._escalate(node)
                    escalations.append(esc)

                    if esc.routed_to == "classical":
                        classical_calls += 1
                    else:
                        cloud_calls += 1
                        total_input += esc.input_tokens
                        total_output += esc.output_tokens

                    output_parts.append(esc.result_text)
                else:
                    output_parts.append(node_text)

                parent_id = node_id
                current_tokens = []
                current_entropies = []

        # Flush any trailing tokens that didn't end on a boundary
        if current_tokens:
            output_parts.append("".join(current_tokens))

        # For reasoning tasks with cloud escalations: Gemma's KV cache can't see
        # Sonnet's clause corrections, so its final Answer: line may be wrong.
        # One synthesis call reads the corrected chain and extracts the right answer.
        if node_type_hint == "reasoning" and cloud_calls > 0:
            corrected_chain = " ".join(output_parts)
            synthesis = self.buddy.final_answer(corrected_chain)
            output_parts.append("\n" + synthesis["text"])
            cloud_calls += 1
            total_input += synthesis["input_tokens"]
            total_output += synthesis["output_tokens"]

        graph_nodes = self._graph.all_nodes() if self._graph else []

        return PipelineResult(
            output=" ".join(output_parts),
            graph_nodes=graph_nodes,
            cloud_calls=cloud_calls,
            classical_calls=classical_calls,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            escalations=escalations,
        )

    def _escalate(self, node: dict) -> Escalation:
        node_type = node["type"]

        # 1. Try classical tool first
        decision = self._router.decide(node_type) if self._router else "cloud"
        if decision.startswith("classical:"):
            tool_name = decision.split(":", 1)[1]
            classical_result = self.registry.run(node_type, node["text"])
            if classical_result and self.registry.clears_bar(
                node_type, classical_result["confidence"]
            ):
                label = classical_result["result"].get("label", node["text"])
                return Escalation(
                    node_id=node["id"],
                    node_type=node_type,
                    original_text=node["text"],
                    routed_to="classical",
                    tool_name=tool_name,
                    result_text=label,
                    confidence=classical_result["confidence"],
                )

        # 2. Fall through to cloud buddy
        ancestors = self._graph.ancestor_chain(node["id"]) if self._graph else []
        result = self.buddy.ask(node, ancestors)

        if self._graph:
            self._graph.patch_node(node["id"], result["patch"], 0.95)

        return Escalation(
            node_id=node["id"],
            node_type=node_type,
            original_text=node["text"],
            routed_to="cloud",
            tool_name=None,
            result_text=result["patch"],
            confidence=0.95,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
        )
