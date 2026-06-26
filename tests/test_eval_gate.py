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
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "5"))  # Use 5 in CI to save quota


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
    assert score >= MIN_FAITHFULNESS, (
        f"Faithfulness {score:.3f} is below threshold {MIN_FAITHFULNESS}. "
        f"This PR degrades answer grounding — investigate retrieval quality."
    )


def test_answer_relevance_meets_threshold(eval_report):
    """
    CI Gate: Answer relevance must be >= MIN_ANSWER_RELEVANCE.

    Answer relevance measures whether the model actually answers the question.
    Low relevance = off-topic responses.
    """
    score = eval_report["avg_answer_relevance"]
    assert score >= MIN_ANSWER_RELEVANCE, (
        f"Answer relevance {score:.3f} is below threshold {MIN_ANSWER_RELEVANCE}. "
        f"This PR produces less relevant answers — check prompt or generation config."
    )


def test_citation_rate_nonzero(eval_results):
    """
    CI Gate: At least 80% of answers must contain citations.

    Zero-citation answers mean the citation enforcement prompt is broken.
    """
    citation_rate = sum(r.has_citations for r in eval_results) / len(eval_results)
    assert citation_rate >= 0.8, (
        f"Only {citation_rate:.1%} of answers contain citations. "
        f"Citation enforcement prompt may be broken."
    )
