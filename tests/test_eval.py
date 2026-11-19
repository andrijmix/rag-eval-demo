from rag_eval.evaluate import evaluate, extractive_answer, load_docs, load_qa, token_f1
from rag_eval.retriever import BM25


def test_retriever_finds_obvious_document():
    docs = {"a": "cats and dogs", "b": "refund policy for customers"}
    assert BM25(docs).search("refund", k=1)[0][0] == "b"


def test_token_f1_bounds():
    assert token_f1("same words", "same words") == 1.0
    assert token_f1("one", "two") == 0.0
    assert 0 < token_f1("a b c", "a b d") < 1


def test_extractive_answer_picks_best_sentence():
    text = "Intro line. The refund window is 30 days. Other info."
    assert extractive_answer(text, "how long is the refund window") == "The refund window is 30 days."


def test_shipped_dataset_meets_quality_gate():
    result = evaluate(load_docs("data/docs"), load_qa("data/qa.csv"), k=3)
    assert result["n"] == 14
    assert result["hit_at_k"] >= 0.8
    # regression gates, set just below the current baseline (MRR 0.96, F1 0.91)
    assert result["mrr"] >= 0.9
    assert result["answer_f1"] >= 0.85
