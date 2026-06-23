"""Load and prepare benchmark datasets for the tiered-router eval.

Classical-tool arm:
  - AG News (classification)
  - CoNLL-2003 (NER / extraction)
  - GLUE STS-B (similarity)
  - GLUE SST-2 (sentiment)

Each pulls a fixed slice from HuggingFace `datasets` — no custom labeling needed,
all have standard accuracy metrics for comparison in the article.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalSample:
    prompt: str
    ground_truth: str
    node_type: str  # maps directly to router registry keys
    dataset: str
    document: str = ""  # source document for document_qa tasks


def load_ag_news(n: int = 200) -> list[EvalSample]:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split=f"test[:{n}]")
    labels = ["World", "Sports", "Business", "Science/Technology"]
    return [
        EvalSample(
            prompt=(
                "Classify the following news article into exactly one of these categories: "
                "World, Sports, Business, Science/Technology.\n"
                "Return only the category name.\n\n"
                f"{row['text']}"
            ),
            ground_truth=labels[row["label"]],
            node_type="classification",
            dataset="ag_news",
        )
        for row in ds
    ]


def load_wikiann_ner(n: int = 200) -> list[EvalSample]:
    from datasets import load_dataset

    ds = load_dataset("unimelb-nlp/wikiann", "en", split=f"test[:{n}]")
    return [
        EvalSample(
            prompt=(
                "Extract all named entities (people, locations, organizations, and other proper names such as works, events, or titles) from the following text.\n"
                "Return only the entity names separated by semicolons. If there are no entities, return NONE.\n\n"
                f"{' '.join(row['tokens'])}"
            ),
            # spans are like ["LOC: India", "PER: Gandhi"] — extract just entity names
            ground_truth=" ".join(s.split(": ", 1)[1] for s in row["spans"]) if row["spans"] else "",
            node_type="extraction",
            dataset="wikiann",
        )
        for row in ds
        if row["spans"]  # skip sentences with no entities
    ][:n]


def load_stsb(n: int = 200) -> list[EvalSample]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "stsb", split=f"validation[:{n}]")
    return [
        EvalSample(
            prompt=(
                "Are these two sentences very similar, similar, or dissimilar?\n"
                "Return only one of: very similar, similar, dissimilar.\n\n"
                f"Sentence 1: {row['sentence1']}\n"
                f"Sentence 2: {row['sentence2']}"
            ),
            ground_truth="very similar" if row["label"] >= 4.0
            else "similar" if row["label"] >= 2.5
            else "dissimilar",
            node_type="similarity",
            dataset="stsb",
        )
        for row in ds
    ]


def load_sst2(n: int = 200) -> list[EvalSample]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "sst2", split=f"validation[:{n}]")
    return [
        EvalSample(
            prompt=(
                "Is the sentiment of the following text positive or negative?\n"
                "Return only: positive or negative.\n\n"
                f"{row['sentence']}"
            ),
            ground_truth="positive" if row["label"] == 1 else "negative",
            node_type="sentiment",
            dataset="sst2",
        )
        for row in ds
    ]


def load_gsm8k(n: int = 20) -> list[EvalSample]:
    """Grade-school math word problems — multi-step arithmetic reasoning."""
    from datasets import load_dataset
    import re

    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    samples = []
    for row in ds:
        # Answer is after "####" e.g. "#### 42"
        match = re.search(r"####\s*(.+)", row["answer"])
        if not match:
            continue
        ground_truth = match.group(1).strip().replace(",", "")
        samples.append(EvalSample(
            prompt=(
                "Solve the following math problem step by step. "
                "Show your reasoning, then on the last line write: Answer: <number>\n\n"
                f"{row['question']}"
            ),
            ground_truth=ground_truth,
            node_type="reasoning",
            dataset="gsm8k",
        ))
    return samples[:n]



def load_squad(n: int = 20) -> list[EvalSample]:
    """SQuAD v2 reading comprehension — questions grounded in a passage.

    Perfect for the targeted buddy flow: Gemma reads a passage and answers a
    question. When entropy spikes on a name/date/fact, the uncertain span is
    sent to Sonnet with the relevant passage chunk — not the full reasoning chain.
    Sonnet fact-checks the specific span against the source text.
    """
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split=f"validation[:{n * 3}]")
    samples = []
    for row in ds:
        if not row["answers"]["text"]:
            continue  # skip unanswerable questions
        ground_truth = row["answers"]["text"][0]
        context = row["context"]
        question = row["question"]
        samples.append(EvalSample(
            prompt=(
                f"Read the following passage carefully, then think through the question "
                f"step by step before answering. On the last line write: Answer: <answer>\n\n"
                f"Passage: {context}\n\n"
                f"Question: {question}"
            ),
            ground_truth=ground_truth,
            node_type="document_qa",
            dataset="squad",
            document=context,
        ))
        if len(samples) >= n:
            break
    return samples[:n]


def load_hotpotqa(n: int = 20) -> list[EvalSample]:
    """HotpotQA — multi-hop Wikipedia QA requiring 2-step reasoning.

    Distractor config gives 10 paragraphs (2 supporting + 8 distractors), making
    the retriever's job meaningful: find the right chunk for the uncertain span.
    Bridge questions require connecting two facts; comparison questions are yes/no.
    Both types trigger real entropy spikes at the pivot entity or comparison value.
    """
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{n * 3}]")
    samples = []
    for row in ds:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]

        paragraphs = [
            f"[{title}]\n" + " ".join(sents)
            for title, sents in zip(titles, sentences)
        ]
        document = "\n\n".join(paragraphs)

        samples.append(EvalSample(
            prompt=(
                "Read the following passages carefully, then think through the "
                "multi-hop question step by step before answering. "
                "On the last line write: Answer: <answer>\n\n"
                f"Passages:\n{document}\n\n"
                f"Question: {row['question']}"
            ),
            ground_truth=row["answer"],
            node_type="document_qa",
            dataset="hotpotqa",
            document=document,
        ))
        if len(samples) >= n:
            break
    return samples[:n]


def load_all(n_per_dataset: int = 20) -> list[EvalSample]:
    loaders = [load_ag_news, load_wikiann_ner, load_stsb, load_sst2,
               load_gsm8k, load_squad, load_hotpotqa]
    samples: list[EvalSample] = []
    for fn in loaders:
        try:
            samples.extend(fn(n_per_dataset))
        except Exception as e:
            print(f"Warning: {fn.__name__} failed — {e}")
    return samples
