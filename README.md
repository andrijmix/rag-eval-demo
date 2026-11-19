# rag-eval-demo

A small, dependency-free harness for **measuring a RAG pipeline automatically** against a reference Q&A table, instead of eyeballing answers.

It shows the evaluation loop I use for LLM answer quality: a reference table written by domain experts, automatic scoring, and a quality gate in CI.

## What it measures

| Metric | Meaning |
|---|---|
| `hit@k` | Is the expected document in the top-k retrieved? |
| `MRR` | Mean reciprocal rank of the expected document |
| `answer F1` | Token-level F1 between the produced answer and the reference answer |

## Run

```bash
python -m rag_eval.evaluate            # prints per-question results and totals
python -m rag_eval.evaluate --min-hit 0.8   # exits 1 if hit@3 drops below 0.8 (CI gate)
pip install pytest && pytest -q
```

Example output on the bundled data (14 questions over 6 synthetic documents):

```
n=14  hit@3=1.00  MRR=0.96  answer F1=0.91
```

The one weak case is a paraphrase ("Can I get my money back after six weeks?"): BM25 ranks the right document 2nd and the extractive answer scores F1 0.21. That is the point of the harness: it makes such weak spots visible and measurable.

## Structure

```
rag_eval/retriever.py   BM25 retriever (baseline, no dependencies)
rag_eval/evaluate.py    metrics, extractive baseline answerer, CLI
data/docs/*.md          synthetic knowledge base (fictional company)
data/qa.csv             reference questions, expected document, expected answer
tests/                  unit tests + regression gates on the bundled data
.github/workflows/      CI: tests and quality gate
```

## Extending it

- Replace `extractive_answer` in `evaluate.py` with an LLM call (any provider) to score a real generator with the same metrics.
- Swap `BM25` for an embedding / vector retriever and compare `hit@k` and `MRR` side by side.
- Grow `data/qa.csv` with real questions; the gate then protects against regressions when prompts, models or chunking change.

The data is fully synthetic. No real company or customer data is used.
