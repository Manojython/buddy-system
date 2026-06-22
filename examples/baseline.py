"""
Advisor System baseline: local model generates a full response, then the cloud
reviews it unconditionally — regardless of confidence. One cloud call per query,
every time. This is the Anthropic Advisor pattern in a local-first setup, and is
the cost/accuracy baseline the Buddy System improves on.

Usage:
    python examples/baseline.py "What caused the 2008 financial crisis?"
    python examples/baseline.py --buddy haiku "..."
"""
from __future__ import annotations

from frugal.model import LocalModel
from frugal.buddy import CloudBuddy, BUDDY_MODELS


def run(
    prompt: str,
    model_path: str = "lmstudio-community/Qwen3-4B-Thinking-2507-MLX-6bit",
    buddy_model: str = "sonnet",
) -> dict:
    local = LocalModel(model_path)
    buddy = CloudBuddy(model=BUDDY_MODELS.get(buddy_model, buddy_model))

    # Step 1: full local generation
    formatted = local.format_prompt(prompt)
    tokens: list[str] = []
    for token_text, _ in local.generate(formatted):
        tokens.append(token_text)
        print(token_text, end="", flush=True)
    print()
    local_response = "".join(tokens)

    # Step 2: unconditional cloud review of the full response
    review = buddy.ask(
        node={"id": "full_response", "type": "claim", "text": local_response, "confidence": 0.5},
        ancestor_chain=[],
    )

    return {
        "local_response": local_response,
        "verdict": review["verdict"],
        "patch": review["patch"],
        "input_tokens": review["input_tokens"],
        "output_tokens": review["output_tokens"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?",
                        default="Explain the main causes of the 2008 financial crisis in three sentences.")
    parser.add_argument("--buddy", default="sonnet", choices=list(BUDDY_MODELS))
    parser.add_argument("--local-model",
                        default="lmstudio-community/Qwen3-4B-Thinking-2507-MLX-6bit")
    args = parser.parse_args()

    print(f"\nPrompt: {args.prompt}\n")
    print("Local model output:\n")
    result = run(args.prompt, model_path=args.local_model, buddy_model=args.buddy)
    print(f"\nCloud buddy ({result['verdict']}):\n{result['patch']}")
    print(f"\nTokens: {result['input_tokens']} in / {result['output_tokens']} out")
