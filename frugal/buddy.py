"""Cloud buddy: sends a flagged reasoning node to a cloud LLM for verification."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()


def _with_retry(fn, max_retries: int = 5, base_delay: float = 5.0):
    """Retry fn on 529 Overloaded with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise

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

_TARGETED_SYSTEM = (
    "You are a fact-checker in a tiered AI inference system. "
    "A local 4B model is answering a question from a document. "
    "When the model is uncertain about a specific fact mid-generation, it sends you: "
    "the relevant document passage, the original question, and the specific span it is uncertain about.\n\n"
    "Your job: verify whether that span is correct based only on the passage.\n\n"
    "Reply with EXACTLY one of:\n"
    "CORRECT: <the span, verbatim>\n"
    "WRONG: <the correct value from the passage, as briefly as possible>\n\n"
    "No preamble. No explanation. One line only."
)

_SYNTHESIS_SYSTEM = (
    "You are the final-answer extractor in a tiered AI system. "
    "You receive step-by-step reasoning where uncertain steps have already been corrected. "
    "Output ONLY the final answer on the last line as:\nAnswer: <value>\n"
    "No explanation, no preamble."
)

_SYNTHESIS_DOC_SYSTEM = (
    "You are the final-answer extractor in a tiered AI inference system. "
    "A local model attempted to answer a question from a passage, but may have reached a wrong conclusion. "
    "Your job: read the source passage and the question, then output the correct answer.\n\n"
    "Rules:\n"
    "1. The source passage is the GROUND TRUTH. If the reasoning chain contradicts the passage, IGNORE the chain.\n"
    "2. Answer only from what the passage explicitly states. No outside knowledge.\n"
    "3. Use the passage's exact wording for names, dates, and quantities.\n"
    "4. Output ONLY the final answer on the last line as: Answer: <value>\n"
    "5. Keep the answer concise — a short phrase matching the passage's phrasing.\n"
    "6. No preamble. No explanation. Just: Answer: <value>"
)


class CloudBuddy:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 256,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> dict:
        """Direct generation — used by the Haiku worker in the Anthropic Advisor condition."""
        response = _with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": _WORKER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        ))
        usage = response.usage
        return {
            "text": response.content[0].text.strip(),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_created": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read": getattr(usage, "cache_read_input_tokens", 0),
        }

    def final_answer(
        self,
        corrected_chain: str,
        document: str = "",
        question: str = "",
    ) -> dict:
        """One synthesis call after reasoning generation: extract the final answer.

        For document QA: uses a specialized system prompt that grounds the answer
        in the source passage and explicitly knows the question. max_tokens raised
        to 256 so multi-clause answers aren't cut off.
        """
        if document:
            content = (
                f"Source passage:\n{document}\n\n"
                f"Question: {question}\n\n"
                f"Corrected reasoning chain:\n{corrected_chain}"
            )
            system_text = _SYNTHESIS_DOC_SYSTEM
            max_tok = 256
        else:
            content = corrected_chain
            system_text = _SYNTHESIS_SYSTEM
            max_tok = 256
        response = _with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=max_tok,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
        ))
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

        response = _with_retry(lambda: self.client.messages.create(
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
        ))

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

    def ask_targeted(
        self,
        uncertain_span: str,
        doc_chunk: str,
        original_question: str,
    ) -> dict:
        """Targeted fact-check: is this specific span correct given the document?

        Sends only the uncertain span and the most relevant passage chunk —
        not the entire reasoning chain. Sonnet returns CORRECT or WRONG + correction.
        """
        user_content = (
            f"Document passage:\n{doc_chunk}\n\n"
            f"Question: {original_question}\n\n"
            f"The model wrote this fact (it is uncertain about it):\n{uncertain_span}\n\n"
            "Is this fact correct based on the passage above?"
        )
        response = _with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=128,
            system=[{"type": "text", "text": _TARGETED_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_content}],
        ))
        usage = response.usage
        raw = response.content[0].text.strip()
        if raw.startswith("CORRECT:"):
            verdict, correction = "confirmed", raw[len("CORRECT:"):].strip()
        elif raw.startswith("WRONG:"):
            verdict, correction = "corrected", raw[len("WRONG:"):].strip()
        else:
            verdict, correction = "corrected", raw
        return {
            "verdict": verdict,
            "patch": correction or uncertain_span,
            "original": uncertain_span,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }

    def ask_targeted_batch(
        self,
        queries: list[tuple],  # list of (UncertainSpan, doc_chunk)
        original_question: str,
        max_workers: int = 8,
    ) -> list[dict]:
        """Fire all targeted queries concurrently via a thread pool.

        Gemma's generation is already done by the time this is called — this is
        the async phase: multiple Sonnet calls in parallel instead of serial.
        I/O-bound API calls benefit from threading even under the GIL.
        """
        def _ask(args):
            span, chunk = args
            return self.ask_targeted(span.text, chunk, original_question)

        with ThreadPoolExecutor(max_workers=min(len(queries), max_workers, 4)) as pool:
            futures = [pool.submit(_ask, q) for q in queries]
            return [f.result() for f in futures]
