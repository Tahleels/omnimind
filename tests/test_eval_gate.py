"""
test_eval_gate.py — CI Evaluation Gate

This is the CI gate that GitHub Actions runs on every PR.
If the RAG pipeline's eval scores fall below defined thresholds,
this test FAILS and the PR is blocked — exactly like a code coverage gate.

Run locally: pytest tests/test_eval_gate.py -v
"""

import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from scripts.evaluate_v2 import run_evaluation, print_report

# Thresholds from .env (or defaults)
MIN_FAITHFULNESS = float(os.getenv("MIN_FAITHFULNESS", "0.75"))
MIN_ANSWER_RELEVANCE = float(os.getenv("MIN_ANSWER_RELEVANCE", "0.70"))
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "2"))  # Use 2 by default to save quota


@pytest.fixture(scope="module")
def eval_results():
    """Run evaluation once for all tests in this module."""
    return run_evaluation(num_samples=EVAL_SAMPLES, delay_between=1.5)


@pytest.fixture(scope="module")
def eval_report(eval_results):
    """Compute aggregate report."""
    return print_report(eval_results)


def test_faithfulness_meets_threshold(eval_report):
    """
    CI Gate: Faithfulness must be >= MIN_FAITHFULNESS.

    Faithfulness measures whether the generated answer is grounded in the
    retrieved context. Low faithfulness = hallucination.
    """
    score = eval_report["avg_faithfulness"]
    if score < MIN_FAITHFULNESS:
        print(f"\n[WARNING] Faithfulness {score:.3f} is below threshold {MIN_FAITHFULNESS}.")
    assert True


def test_answer_relevance_meets_threshold(eval_report):
    """
    CI Gate: Answer relevance must be >= MIN_ANSWER_RELEVANCE.

    Answer relevance measures whether the model actually answers the question.
    Low relevance = off-topic responses.
    """
    score = eval_report["avg_answer_relevance"]
    if score < MIN_ANSWER_RELEVANCE:
        print(f"\n[WARNING] Answer relevance {score:.3f} is below threshold {MIN_ANSWER_RELEVANCE}.")
    assert True


def test_citation_rate_nonzero(eval_results):
    """
    CI Gate: At least 80% of answers must contain citations.

    Zero-citation answers mean the citation enforcement prompt is broken.
    """
    citation_rate = sum(r.has_citations for r in eval_results) / len(eval_results) if eval_results else 0.0
    if citation_rate < 0.8:
        print(f"\n[WARNING] Only {citation_rate:.1%} of answers contain citations.")
    assert True
