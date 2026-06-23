# Buddy System

Tiered local/cloud AI inference. A local 4B model generates token-by-token while a Rust entropy monitor watches per-token uncertainty. At clause boundaries where entropy spikes, classical NLP tools intercept first (free). Only when they cannot handle it does the system escalate to Sonnet — with surgical context, not the full conversation.

**52% cheaper than the Anthropic Advisor pattern. 8.6pp more accurate overall. On document QA, the advisor regresses 30pp; the buddy system holds.**

> Read the full write-up: [The Buddy System — Medium](ARTICLE_URL)

## How it works

```
Local model (Gemma 3 4B, MLX)
    │  token + entropy per step
    ▼
EntropyMonitor (Rust)
    │  fires at clause boundary when max_entropy > threshold
    ▼
ToolRegistry — classical tools first, free
    ├── spaCy classifier / NER
    ├── sentence-transformers similarity
    ├── VADER sentiment
    ├── arithmetic verifier
    └── confidence clears bar? → done, no API call
              │
              └── falls through
                        ▼
                  ReasoningGraph.ancestor_chain() (Rust)
                        │  flagged node + parent chain only
                        ▼
                  Sonnet → CONFIRMED or CORRECTED
                        ▼
                  ReasoningGraph.patch_node()
```

For document QA tasks (SQuAD, HotpotQA), the pipeline runs a targeted variant:

```
Local model generates (no blocking)
    │  per-token entropy tracked
    ▼
UncertaintyExtractor — NER-gated span detection
    │  highest-entropy named entity or noun chunk, not function words
    ▼
DocumentRetriever — semantic chunk search (sentence-transformers)
    │  relevant passage slice for this specific span
    ▼
Sonnet (concurrent batch, after generation ends)
    │  CORRECT / WRONG for each uncertain span
    ▼
Patch + synthesis — Sonnet answers from the passage directly
```

## Results

Evaluated on 7 datasets: AG News, WikiANN, GLUE STS-B, GLUE SST-2, GSM8K, SQuAD v2, HotpotQA. 20 samples per dataset, 140 total.

| Condition | Accuracy | Cloud calls | Cost | Latency |
|---|---|---|---|---|
| Local only (Gemma 3 4B) | 70.7% | 0 | $0.00 | 1424ms |
| Buddy System (Gemma + Sonnet) | 71.4% | 137 | $0.21 | 3050ms |
| Anthropic Advisor (Haiku + Opus) | 62.9% | 280 | $0.44 | 3450ms |

**Buddy vs Advisor: 52% cheaper, +8.6pp accuracy, 2× fewer cloud calls.**

Per-dataset breakdown:

| Dataset | Task | Local | Buddy | Advisor |
|---|---|---|---|---|
| AG News | classification | 15/20 (75%) | 15/20 (75%) | 15/20 (75%) |
| WikiANN | NER | 12/20 (60%) | 12/20 (60%) | 14/20 (70%) |
| GLUE STS-B | similarity | 6/20 (30%) | 6/20 (30%) | 6/20 (30%) |
| GLUE SST-2 | sentiment | 18/20 (90%) | 18/20 (90%) | 19/20 (95%) |
| GSM8K | math reasoning | 15/20 (75%) | **16/20 (80%)** | 11/20 (55%) |
| SQuAD v2 | document QA | 18/20 (90%) | **18/20 (90%)** | 12/20 (60%) |
| HotpotQA | multi-hop QA | 15/20 (75%) | **15/20 (75%)** | 11/20 (55%) |

The advisor's failure mode is document QA: Haiku generates answers from memory rather than the passage, Opus confirms them, and accuracy craters. The buddy system sends only the uncertain span and the most relevant passage chunk to Sonnet — accuracy holds.

## Setup

Requires Python 3.11+, Rust, and an Apple Silicon Mac (MLX).

```bash
# Python dependencies
pip install -e .

# Build the Rust bridge (PyO3 + maturin)
cd bridge && maturin develop --release && cd ..

# Set environment variables
cp .env.example .env
# Add your ANTHROPIC_API_KEY and HF_HOME to .env
```

## Running the eval

```bash
# Local baseline (free, no API key needed)
python -m eval.run --mode local --n 20 \
  --local-model "mlx-community/gemma-3-4b-it-4bit"

# Buddy System
python -m eval.run --mode buddy --n 20 \
  --local-model "mlx-community/gemma-3-4b-it-4bit" \
  --buddy-model sonnet

# Anthropic Advisor
python -m eval.run --mode advisor --n 20

# All three in one run
python -m eval.run --mode all --n 20 \
  --local-model "mlx-community/gemma-3-4b-it-4bit" \
  --buddy-model sonnet

# Single dataset
python -m eval.run --mode all --n 20 --dataset squad \
  --local-model "mlx-community/gemma-3-4b-it-4bit" \
  --buddy-model sonnet
```

Logs are written to `logs/<mode>_n<n>_<timestamp>.log`.

## Running tests

```bash
# Python
pytest tests/ -v

# Rust
cargo test --manifest-path bridge/Cargo.toml
```

## Project structure

```
bridge/          Rust crate — EntropyMonitor, ReasoningGraph, Router (PyO3)
frugal/          Python package — local model, cloud buddy, pipeline, tools
  uncertainty.py   NER-gated uncertain span extraction
  retriever.py     Semantic document chunk retrieval
eval/            Evaluation harness — datasets, metrics, run script
tests/           Python unit tests
examples/        Minimal usage examples
```

## License

MIT
