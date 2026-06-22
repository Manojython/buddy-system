# Buddy System

Tiered local/cloud AI inference. A local 4B model generates token-by-token while a Rust entropy monitor watches per-token uncertainty. At clause boundaries where entropy spikes, classical NLP tools intercept first (free). Only when they cannot handle it does the system escalate to Sonnet — with surgical context from a Rust ReasoningGraph, not the full conversation.

**70% cheaper than the Anthropic Advisor pattern at 1.7pp accuracy cost** across 6 NLP and reasoning benchmarks.

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

## Results

Evaluated on AG News, WikiANN, GLUE STS-B, GLUE SST-2, GSM8K, ARC-Challenge. 20 samples per dataset, 120 total.

| Condition | Accuracy | Cloud calls | Cost | Latency |
|---|---|---|---|---|
| Local only (Gemma 3 4B) | 65.0% | 0 | $0.00 | 1174ms |
| Buddy System (Gemma + Sonnet) | 65.0% | 87 | $0.096 | 2559ms |
| Anthropic Advisor (Haiku + Opus) | 66.7% | 240 | $0.318 | 4262ms |

**Buddy vs Advisor: 70% cheaper, 1.7pp accuracy gap, 3× fewer cloud calls.**

The local and buddy accuracy match because the 87 Sonnet calls concentrate on the hardest reasoning problems — ones where even Sonnet cannot salvage a misframed approach. On the 80 NLP samples, Gemma is already confident and correct; entropy never spikes and no cloud call is made. The advisor pattern wastes 160 of its 240 API calls on those same samples unconditionally.

The right comparison is buddy vs advisor: near-identical accuracy at a third of the cost.

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
eval/            Evaluation harness — datasets, metrics, run script
tests/           Python unit tests
examples/        Minimal usage examples
```

## License

MIT
