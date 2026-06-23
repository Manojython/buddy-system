"""
Eval comparing three conditions on the same four datasets.

  1. Local only           — free, local 4B model, no cloud
  2. Anthropic Advisor    — Haiku generates + Opus reviews every response (their pattern)
  3. Buddy System         — local 4B generates, Sonnet only at entropy spikes via classical tools first

Usage:
    python -m eval.run --mode buddy --n 20          # free run, no cloud
    python -m eval.run --mode advisor --n 20        # Haiku + Opus baseline
    python -m eval.run --mode all --n 20            # full comparison table
"""
from __future__ import annotations

import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()  # loads HF_HOME + ANTHROPIC_API_KEY before any HF imports
import time
from datetime import datetime
from pathlib import Path

from frugal.model import LocalModel, load_local_model
from frugal.buddy import CloudBuddy, BUDDY_MODELS
from frugal.pipeline import Pipeline
from frugal.tools.registry import ToolRegistry
from frugal.tools import classifier, ner, similarity, sentiment, math_solver

from eval.datasets import load_all, EvalSample
from eval.metrics import ConditionResult, evaluate_output, max_tokens_for, token_cost, compare, compare_datasets


_log_file: "TextIO | None" = None


def _log(msg: str) -> None:
    """Timestamp-prefix, print to stdout, and flush-write to the run log file."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("classification", classifier.classify, confidence_bar=0.80)
    reg.register("extraction", ner.extract, confidence_bar=0.75)
    reg.register("similarity", similarity.score, confidence_bar=0.70)
    reg.register("sentiment", sentiment.score, confidence_bar=0.85)
    reg.register("reasoning", math_solver.solve, confidence_bar=0.90)
    return reg


def _run_local(local: LocalModel, samples: list[EvalSample], verbose: bool = False) -> ConditionResult:
    """Local 4B model only — zero cloud cost, baseline accuracy floor."""
    correct = 0
    per_dataset: dict[str, list[int]] = {}
    t0 = time.perf_counter()

    for i, s in enumerate(samples):
        t_sample = time.perf_counter()
        tokens: list[str] = []
        for text, _ in local.generate(
            local.format_prompt(s.prompt),
            max_tokens=max_tokens_for(s),
            temp=0.0,
        ):
            tokens.append(text)
        output = "".join(tokens)
        score = evaluate_output(s, output)
        if score.correct:
            correct += 1
        ds = s.dataset
        if ds not in per_dataset:
            per_dataset[ds] = [0, 0]
        per_dataset[ds][1] += 1
        if score.correct:
            per_dataset[ds][0] += 1
        sample_ms = (time.perf_counter() - t_sample) * 1000
        if verbose:
            _log(f"  ── local [{i+1}/{len(samples)}] ──────────────────────────")
            _log(f"     task      : {s.node_type}")
            _log(f"     dataset   : {s.dataset}")
            _log(f"     prompt    : {s.prompt[:120].replace(chr(10), ' ')}")
            _log(f"     ground_truth: {s.ground_truth!r}")
            _log(f"     prediction: {score.prediction!r}  score={score.score:.2f}")
            _log(f"     output    : {output[:300].replace(chr(10), ' ')}")
            _log(f"     hit       : {'✓ CORRECT' if score.correct else '✗ WRONG'}")
            _log(f"     latency   : {sample_ms:.0f}ms")

    elapsed_ms = (time.perf_counter() - t0) * 1000 / max(len(samples), 1)
    return ConditionResult(
        condition="Local only (free)",
        samples=len(samples), correct=correct,
        cloud_calls=0, classical_calls=0,
        total_input_tokens=0, total_output_tokens=0,
        latency_ms=elapsed_ms, cost_usd=0.0,
        per_dataset={ds: (v[0], v[1]) for ds, v in per_dataset.items()},
    )


def _run_anthropic_advisor(
    haiku: CloudBuddy, opus: CloudBuddy, samples: list[EvalSample], verbose: bool = False,
) -> ConditionResult:
    """Anthropic Advisor pattern: Haiku generates → Opus reviews every response.

    Both models are paid cloud. Haiku handles the bulk generation; Opus reviews
    unconditionally on every query regardless of confidence. This is what people
    are running today when they follow Anthropic's Advisor Strategy.
    """
    correct = 0
    cost = 0.0
    total_in = total_out = 0
    per_dataset: dict[str, list[int]] = {}
    t0 = time.perf_counter()

    for i, s in enumerate(samples):
        t_sample = time.perf_counter()

        # Step 1: Haiku generates
        haiku_result = haiku.generate(s.prompt)
        haiku_cost = token_cost(haiku.model, haiku_result["input_tokens"], haiku_result["output_tokens"])
        cost += haiku_cost

        # Step 2: Opus reviews unconditionally
        opus_result = opus.ask(
            node={"id": "full", "type": s.node_type,
                  "text": haiku_result["text"], "confidence": 0.5},
            ancestor_chain=[],
        )
        opus_cost = token_cost(opus.model, opus_result["input_tokens"], opus_result["output_tokens"])
        cost += opus_cost

        score = evaluate_output(s, opus_result["patch"])
        if score.correct:
            correct += 1
        ds = s.dataset
        if ds not in per_dataset:
            per_dataset[ds] = [0, 0]
        per_dataset[ds][1] += 1
        if score.correct:
            per_dataset[ds][0] += 1
        total_in += haiku_result["input_tokens"] + opus_result["input_tokens"]
        total_out += haiku_result["output_tokens"] + opus_result["output_tokens"]
        sample_ms = (time.perf_counter() - t_sample) * 1000

        if verbose:
            _log(f"  ── advisor [{i+1}/{len(samples)}] ────────────────────────")
            _log(f"     task           : {s.node_type}")
            _log(f"     prompt         : {s.prompt[:120].replace(chr(10), ' ')}")
            _log(f"     ground_truth   : {s.ground_truth!r}")
            _log(f"     prediction     : {score.prediction!r}  score={score.score:.2f}")
            _log(f"     haiku_output   : {haiku_result['text'][:300].replace(chr(10), ' ')}")
            _log(f"     haiku_tokens   : in={haiku_result['input_tokens']}  out={haiku_result['output_tokens']}  cost=${haiku_cost:.5f}")
            _log(f"     haiku_cache    : created={haiku_result.get('cache_created',0)}  read={haiku_result.get('cache_read',0)}")
            _log(f"     opus_verdict   : {opus_result['verdict']}")
            _log(f"     opus_patch     : {opus_result['patch'][:300].replace(chr(10), ' ')}")
            _log(f"     opus_tokens    : in={opus_result['input_tokens']}  out={opus_result['output_tokens']}  cost=${opus_cost:.5f}")
            _log(f"     opus_cache     : created={opus_result.get('cache_created',0)}  read={opus_result.get('cache_read',0)}")
            _log(f"     hit            : {'✓ CORRECT' if score.correct else '✗ WRONG'}")
            _log(f"     total_cost     : ${haiku_cost+opus_cost:.5f}")
            _log(f"     latency        : {sample_ms:.0f}ms")

    elapsed_ms = (time.perf_counter() - t0) * 1000 / max(len(samples), 1)
    return ConditionResult(
        condition="Anthropic Advisor (Haiku+Opus)",
        samples=len(samples), correct=correct,
        cloud_calls=len(samples) * 2,  # 1 Haiku + 1 Opus per sample
        classical_calls=0,
        total_input_tokens=total_in, total_output_tokens=total_out,
        latency_ms=elapsed_ms, cost_usd=cost,
        per_dataset={ds: (v[0], v[1]) for ds, v in per_dataset.items()},
    )


def _run_buddy(
    local: LocalModel, buddy: CloudBuddy,
    samples: list[EvalSample], threshold: float,
    verbose: bool = False,
) -> ConditionResult:
    """Buddy System: local 4B generates free → Sonnet only at entropy spikes → classical tools first."""
    pipeline = Pipeline(
        local_model=local, buddy=buddy,
        registry=_build_registry(),
        entropy_threshold=threshold,
        min_tokens_per_boundary=3,
        router_confidence_bar=0.8,
    )

    correct = cloud = classical_c = in_tok = out_tok = 0
    cost = 0.0
    per_dataset: dict[str, list[int]] = {}
    t0 = time.perf_counter()

    for i, s in enumerate(samples):
        t_sample = time.perf_counter()
        result = pipeline.run(
            s.prompt,
            node_type_hint=s.node_type,
            max_tokens=max_tokens_for(s),
            document=s.document,
        )
        score = evaluate_output(s, result.output)
        if score.correct:
            correct += 1
        ds = s.dataset
        if ds not in per_dataset:
            per_dataset[ds] = [0, 0]
        per_dataset[ds][1] += 1
        if score.correct:
            per_dataset[ds][0] += 1
        cloud += result.cloud_calls
        classical_c += result.classical_calls
        in_tok += result.total_input_tokens
        out_tok += result.total_output_tokens
        sample_cost = token_cost(buddy.model, result.total_input_tokens, result.total_output_tokens)
        cost += sample_cost
        sample_ms = (time.perf_counter() - t_sample) * 1000

        if verbose:
            _log(f"  ── buddy [{i+1}/{len(samples)}] ──────────────────────────")
            _log(f"     task           : {s.node_type}")
            _log(f"     prompt         : {s.prompt[:120].replace(chr(10), ' ')}")
            _log(f"     ground_truth   : {s.ground_truth!r}")
            _log(f"     prediction     : {score.prediction!r}  score={score.score:.2f}")
            _log(f"     output         : {result.output[:300].replace(chr(10), ' ')}")
            _log(f"     hit            : {'✓ CORRECT' if score.correct else '✗ WRONG'}")
            _log(f"     cloud_calls    : {result.cloud_calls}")
            _log(f"     classical_calls: {result.classical_calls}")
            for esc in result.escalations:
                tool = esc.tool_name or "sonnet"
                _log(f"       escalation → {esc.routed_to}:{tool}  conf={esc.confidence:.2f}  '{esc.original_text[:60]}'")
            _log(f"     tokens         : in={result.total_input_tokens}  out={result.total_output_tokens}")
            _log(f"     cost           : ${sample_cost:.5f}")
            _log(f"     latency        : {sample_ms:.0f}ms")

    elapsed_ms = (time.perf_counter() - t0) * 1000 / max(len(samples), 1)
    return ConditionResult(
        condition="Buddy System (local+Sonnet)",
        samples=len(samples), correct=correct,
        cloud_calls=cloud, classical_calls=classical_c,
        total_input_tokens=in_tok, total_output_tokens=out_tok,
        latency_ms=elapsed_ms, cost_usd=cost,
        per_dataset={ds: (v[0], v[1]) for ds, v in per_dataset.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20,
                        help="Samples per dataset (×4 datasets = total)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Entropy threshold for buddy interrupts")
    parser.add_argument("--local-model", default="lmstudio-community/Qwen3-4B-Thinking-2507-MLX-6bit",
                        help="Local MLX model for Local-only and Buddy conditions")
    parser.add_argument("--buddy-model", default="sonnet", choices=list(BUDDY_MODELS),
                        help="Cloud model for Buddy System (default: sonnet)")
    parser.add_argument(
        "--mode", default="buddy", choices=["local", "advisor", "buddy", "all"],
        help="local | advisor (Haiku+Opus) | buddy (local+Sonnet) | all",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample logs: output, verdict, tokens, cost, latency")
    parser.add_argument("--dataset", default=None,
                        help="Only run samples from this dataset (e.g. mmlu, gsm8k). Default: all.")
    args = parser.parse_args()

    # Set up log file: logs/<mode>_n<n>_<timestamp>.log
    global _log_file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_tag = f"_{args.dataset}" if args.dataset else ""
    log_path = log_dir / f"{args.mode}_n{args.n}{ds_tag}_{ts}.log"
    _log_file = open(log_path, "w", buffering=1)  # line-buffered
    _log(f"Run started: {datetime.now().isoformat()}  mode={args.mode}  n={args.n}  "
         f"local_model={args.local_model}  buddy_model={args.buddy_model}  verbose={args.verbose}")
    _log(f"Log file: {log_path.resolve()}\n")

    total = args.n * 7
    _log(f"Loading datasets ({args.n} × 7 = {total} samples)...")
    samples = load_all(args.n)
    if args.dataset:
        samples = [s for s in samples if s.dataset == args.dataset]
        _log(f"Filtered to dataset={args.dataset!r}: {len(samples)} samples.")
    _log(f"Loaded {len(samples)} samples.\n")

    results: list[ConditionResult] = []

    if args.mode in ("local", "buddy", "all"):
        _log(f"Loading local model: {args.local_model}")
        local = load_local_model(args.local_model)
        _log("Local model ready.\n")

    if args.mode in ("buddy", "all"):
        sonnet = CloudBuddy(model=BUDDY_MODELS[args.buddy_model])

    if args.mode in ("advisor", "all"):
        haiku = CloudBuddy(model=BUDDY_MODELS["haiku"])
        opus  = CloudBuddy(model=BUDDY_MODELS["opus"])

    _log("─" * 50)

    if args.mode in ("local", "all"):
        _log("1. Local only …")
        r = _run_local(local, samples, verbose=args.verbose)
        results.append(r)
        _log(f"   accuracy={r.accuracy:.1%}  cost=$0.00\n")

    if args.mode in ("buddy", "all"):
        _log(f"2. Buddy System (local + {args.buddy_model}) …")
        r = _run_buddy(local, sonnet, samples, args.threshold, verbose=args.verbose)
        results.append(r)
        _log(f"   accuracy={r.accuracy:.1%}  cloud={r.cloud_calls}  classical={r.classical_calls}  cost=${r.cost_usd:.4f}\n")

    if args.mode in ("advisor", "all"):
        _log("3. Anthropic Advisor (Haiku + Opus) …")
        r = _run_anthropic_advisor(haiku, opus, samples, verbose=args.verbose)
        results.append(r)
        _log(f"   accuracy={r.accuracy:.1%}  cloud_calls={r.cloud_calls}  cost=${r.cost_usd:.4f}\n")

    _log("─" * 50)
    # compare() writes to stdout — capture and mirror to log
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    compare(results)
    print()
    compare_datasets(results)
    sys.stdout = old_stdout
    table_str = buf.getvalue()
    _log(table_str)

    buddy_r  = next((r for r in results if "Buddy"   in r.condition), None)
    advisor_r = next((r for r in results if "Advisor" in r.condition), None)
    if buddy_r and advisor_r and advisor_r.cost_usd > 0:
        savings_pct    = (1 - buddy_r.cost_usd / advisor_r.cost_usd) * 100
        accuracy_delta = (buddy_r.accuracy - advisor_r.accuracy) * 100
        _log(f"Buddy vs Advisor:  {savings_pct:.0f}% cheaper,  {accuracy_delta:+.1f}pp accuracy")

    _log(f"\nRun finished: {datetime.now().isoformat()}")
    _log_file.close()


if __name__ == "__main__":
    main()
