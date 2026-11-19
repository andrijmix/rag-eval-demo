"""Evaluate a retrieval + answering pipeline against a reference Q&A table.

Metrics:
  hit@k      - is the expected document among the top-k retrieved?
  MRR        - mean reciprocal rank of the expected document
  answer F1  - token-level F1 between the produced answer and the reference answer
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from .retriever import BM25, tokenize


def load_docs(folder: str) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.md"))}


def load_qa(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def token_f1(pred: str, gold: str) -> float:
    p, g = Counter(tokenize(pred)), Counter(tokenize(gold))
    overlap = sum((p & g).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def extractive_answer(doc_text: str, question: str) -> str:
    """Baseline 'generator': the sentence with the most word overlap with the question.

    Swap this for an LLM call to evaluate a real generator with the same metrics.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", doc_text.replace("\n", " ")) if s.strip()]
    q = set(tokenize(question))
    return max(sentences, key=lambda s: len(q & set(tokenize(s))))


def evaluate(
    docs: dict[str, str],
    qa: list[dict[str, str]],
    k: int = 3,
    answer_fn: Callable[[str, str], str] = extractive_answer,
) -> dict:
    index = BM25(docs)
    hits, rr, f1s, rows = 0, 0.0, 0.0, []
    for item in qa:
        ranked = [d for d, _ in index.search(item["question"], k=len(docs))]
        rank = ranked.index(item["expected_doc"]) + 1 if item["expected_doc"] in ranked else None
        hit = rank is not None and rank <= k
        answer = answer_fn(docs[ranked[0]], item["question"])
        f1 = token_f1(answer, item["expected_answer"])
        hits += hit
        rr += 1 / rank if rank else 0.0
        f1s += f1
        rows.append({"question": item["question"], "rank": rank, "hit": hit, "f1": round(f1, 2)})
    n = len(qa)
    return {"n": n, "hit_at_k": hits / n, "mrr": rr / n, "answer_f1": f1s / n, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--qa", default="data/qa.csv")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--min-hit", type=float, default=None, help="exit 1 if hit@k is below this (CI gate)")
    args = ap.parse_args(argv)

    result = evaluate(load_docs(args.docs), load_qa(args.qa), k=args.k)
    for r in result["rows"]:
        mark = "ok " if r["hit"] else "MISS"
        print(f"{mark} rank={r['rank']} f1={r['f1']:.2f}  {r['question']}")
    print(f"\nn={result['n']}  hit@{args.k}={result['hit_at_k']:.2f}  "
          f"MRR={result['mrr']:.2f}  answer F1={result['answer_f1']:.2f}")
    if args.min_hit is not None and result["hit_at_k"] < args.min_hit:
        print(f"FAIL: hit@{args.k} below {args.min_hit}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
