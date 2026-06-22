"""
Full tiered pipeline demo:
  local MLX model → entropy monitor (Rust) → reasoning graph (Rust)
  → router (Rust) → classical tool OR cloud buddy → patched output

Usage:
    python examples/full_pipeline.py "Analyze the sentiment and classify the topic of: ..."
"""
from __future__ import annotations

import argparse
import sys

from frugal.model import LocalModel
from frugal.buddy import CloudBuddy, BUDDY_MODELS
from frugal.pipeline import Pipeline
from frugal.tools.registry import ToolRegistry
from frugal.tools import classifier, ner, similarity, sentiment


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("classification", classifier.classify, confidence_bar=0.80)
    reg.register("extraction", ner.extract, confidence_bar=0.75)
    reg.register("similarity", similarity.score, confidence_bar=0.70)
    reg.register("sentiment", sentiment.score, confidence_bar=0.85)
    return reg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=(
        "Analyze the sentiment of this review and classify the product category: "
        "'The battery life on this laptop is exceptional, lasting 14 hours on a charge. "
        "However, the keyboard feels mushy compared to the competition.'"
    ))
    parser.add_argument("--buddy", default="sonnet", choices=list(BUDDY_MODELS),
                        help="Cloud buddy model tier (default: sonnet)")
    parser.add_argument("--local-model", default="mlx-community/Qwen3.5-9B-OptiQ-4bit")
    args = parser.parse_args()

    prompt = args.prompt
    print(f"Prompt: {prompt}")
    print(f"Buddy: {args.buddy} ({BUDDY_MODELS[args.buddy]})\n")

    pipeline = Pipeline(
        local_model=LocalModel(args.local_model),
        buddy=CloudBuddy(model=BUDDY_MODELS[args.buddy]),
        registry=build_registry(),
        entropy_threshold=2.5,
        min_tokens_per_boundary=10,
        router_confidence_bar=0.8,
    )

    print("Generating...\n")
    result = pipeline.run(prompt)

    print(f"\nOutput:\n{result.output}\n")
    print("─" * 60)
    print(f"Cloud LLM calls  : {result.cloud_calls}")
    print(f"Classical calls  : {result.classical_calls}")
    print(f"Cloud tokens     : {result.total_input_tokens} in / {result.total_output_tokens} out")
    print(f"Graph nodes      : {len(result.graph_nodes)}")

    if result.escalations:
        print("\nEscalation log:")
        for e in result.escalations:
            tool = e.tool_name or "cloud-buddy"
            print(f"  [{e.node_id}] {e.node_type} → {tool}")
            print(f"    original : {e.original_text[:80]}…")
            print(f"    result   : {e.result_text[:80]}…")
            print(f"    conf     : {e.confidence:.2f}")


if __name__ == "__main__":
    main()
