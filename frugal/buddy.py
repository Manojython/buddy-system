"""Cloud buddy: sends a flagged reasoning node to a cloud LLM for verification."""
from __future__ import annotations

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Named aliases for article demos — pass to CloudBuddy(model=...) or use a
# full model ID directly. Cost order: haiku < sonnet < opus.
BUDDY_MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

# System prompt for the buddy's node-verification role — cached across calls.
_SYSTEM_PROMPT = (
    "You are the cloud buddy in a tiered AI inference system. "
    "A local language model generates text token by token. When its per-token entropy "
    "spikes at a clause boundary — meaning the model is genuinely uncertain — the "
    "flagged reasoning node is escalated to you.\n\n"
    "Your job: verify the flagged step against the prior reasoning context and either "
    "confirm it or correct it.\n\n"
    "Rules:\n"
    "- Respond with EXACTLY one of the two formats below. No preamble, no explanation.\n"
    "- CONFIRMED: <the original text, verbatim>\n"
    "- CORRECTED: <your corrected version — one concise sentence>\n"
    "- Correct only factual errors or logical contradictions. Do not rephrase for style."
)


_WORKER_SYSTEM = "Answer the following question concisely and accurately."

_SYNTHESIS_SYSTEM = (
    "You are the final-answer extractor in a tiered AI system. "
    "You receive step-by-step reasoning where uncertain steps have already been corrected. "
    "Output ONLY the final answer on the last line as:\nAnswer: <value>\n"
    "No explanation, no preamble."
)


class CloudBuddy:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 512,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> dict:
        """Direct generation — used by the Haiku worker in the Anthropic Advisor condition."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": _WORKER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage
        return {
            "text": response.content[0].text.strip(),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_created": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read": getattr(usage, "cache_read_input_tokens", 0),
        }

    def final_answer(self, corrected_chain: str) -> dict:
        """One synthesis call after reasoning generation: extract the final answer."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=128,
            system=[{"type": "text", "text": _SYNTHESIS_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": corrected_chain}],
        )
        usage = response.usage
        return {
            "text": response.content[0].text.strip(),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }

    def ask(self, node: dict, ancestor_chain: list[dict]) -> dict:
        """
        Send a flagged reasoning node and its ancestor chain to the cloud buddy.

        The system prompt is marked for prompt caching — on the first call the
        API writes the cache; every subsequent call in the same session reads it,
        paying only for the dynamic user content.

        Returns:
            verdict: "confirmed" | "corrected"
            patch: the corrected or confirmed text
            input_tokens / output_tokens: billed tokens (cache reads are cheaper)
            cache_created / cache_read: token counts for cost transparency
        """
        ancestor_text = "\n".join(
            f"  [{a['type']}] {a['text']}"
            for a in ancestor_chain
            if a["id"] != node["id"]
        ) or "(no prior context)"

        user_content = (
            f"Prior reasoning context:\n{ancestor_text}\n\n"
            f"Flagged step — type: {node['type']}, confidence: {node['confidence']:.3f}\n"
            f"{node['text']}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("CONFIRMED:"):
            verdict = "confirmed"
            patch = raw[len("CONFIRMED:"):].strip()
        elif raw.startswith("CORRECTED:"):
            verdict = "corrected"
            patch = raw[len("CORRECTED:"):].strip()
        else:
            verdict = "corrected"
            patch = raw

        usage = response.usage
        return {
            "verdict": verdict,
            "patch": patch,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_created": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read": getattr(usage, "cache_read_input_tokens", 0),
            "model": self.model,
        }
