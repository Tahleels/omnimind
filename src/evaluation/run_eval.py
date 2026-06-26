"""
run_eval.py — LLM-as-a-Judge Evaluation Runner

Evaluates the RAG pipeline on the curated test dataset using
LLM-as-a-judge scoring for:
  - Faithfulness: Is the answer grounded in the context?
  - Answer Relevance: Does the answer address the question?
  - Citation Coverage: Did the model cite at least one source?

The judge LLM is Gemini 2.5 Flash (same as generation, no extra cost).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.evaluation.test_dataset import EVAL_DATASET
from src.generation.chain import RAGPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt for LLM-as-a-judge scoring
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are an objective evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the following Q&A pair on these two dimensions. Return ONLY a JSON object.

Question: {question}
Reference Answer: {reference}
Generated Answer: {generated}

Score each dimension from 0.0 to 1.0:
1. faithfulness: Is the generated answer factually consistent with the reference? (1.0 = fully consistent, 0.0 = contradicts)
2. answer_relevance: Does the generated answer address the question? (1.0 = directly answers, 0.0 = off-topic)

Return ONLY this JSON format (no other text):
{{"faithfulness": 0.0, "answer_relevance": 0.0}}"""


class EvalResult(NamedTuple):
    question: str
    generated_answer: str
    faithfulness: float
    answer_relevance: float
    has_citations: bool
    num_sources: int


def run_evaluation(
    num_samples: int = 10,
    delay_between: float = 2.0,
) -> list[EvalResult]:
    """
    Run the RAG pipeline on a subset of the test dataset and score each result.

    Args:
        num_samples: Number of test cases to evaluate (default 10 to save API quota).
        delay_between: Seconds to wait between API calls (rate limiting).

    Returns:
        List of EvalResult objects.
    """
    logger.info("Initializing RAG Pipeline for evaluation...")
    pipeline = RAGPipeline()

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "openrouter":
        judge_llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "google/gemini-2.5-flash:free"),
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AskMyDocs RAG Evaluation",
            }
        )
    else:
        judge_llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
        )

    dataset = EVAL_DATASET[:num_samples]
    results = []

    logger.info(f"Evaluating {len(dataset)} test cases...\n")

    for i, sample in enumerate(dataset, 1):
        question = sample["question"]
        reference = sample["ground_truth"]

        logger.info(f"[{i}/{len(dataset)}] Q: {question[:60]}...")

        # Run RAG pipeline
        try:
            rag_result = pipeline.ask(question)
            generated = rag_result["answer"]
            num_sources = len(rag_result["sources"])
            has_citations = bool(rag_result["sources"])
        except Exception as e:
            logger.error(f"  Pipeline error: {e}")
            results.append(EvalResult(
                question=question,
                generated_answer=f"ERROR: {e}",
                faithfulness=0.0,
                answer_relevance=0.0,
                has_citations=False,
                num_sources=0,
            ))
            continue

        # LLM-as-a-judge scoring
        try:
            judge_prompt = JUDGE_PROMPT.format(
                question=question,
                reference=reference,
                generated=generated,
            )
            judge_response = judge_llm.invoke(judge_prompt)
            judge_text = judge_response.content.strip()

            # Parse JSON (handle markdown code blocks)
            if "```" in judge_text:
                judge_text = judge_text.split("```")[1]
                if judge_text.startswith("json"):
                    judge_text = judge_text[4:]

            scores = json.loads(judge_text)
            faithfulness = float(scores.get("faithfulness", 0.0))
            relevance = float(scores.get("answer_relevance", 0.0))

        except Exception as e:
            logger.warning(f"  Judge scoring failed: {e}. Defaulting to 0.")
            faithfulness = 0.0
            relevance = 0.0

        result = EvalResult(
            question=question,
            generated_answer=generated,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            has_citations=has_citations,
            num_sources=num_sources,
        )
        results.append(result)

        logger.info(
            f"  ✓ Faithfulness: {faithfulness:.2f} | "
            f"Relevance: {relevance:.2f} | "
            f"Sources: {num_sources}"
        )

        if i < len(dataset):
            time.sleep(delay_between)

    return results


def print_report(results: list[EvalResult]) -> dict:
    """Print a formatted evaluation report and return aggregate scores."""
    if not results:
        print("No results to report.")
        return {}

    avg_faith = sum(r.faithfulness for r in results) / len(results)
    avg_rel = sum(r.answer_relevance for r in results) / len(results)
    citation_rate = sum(r.has_citations for r in results) / len(results)

    print("\n" + "=" * 60)
    print("  RAG PIPELINE EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples evaluated   : {len(results)}")
    print(f"  Avg Faithfulness    : {avg_faith:.3f}  (threshold: 0.75)")
    print(f"  Avg Answer Relevance: {avg_rel:.3f}  (threshold: 0.70)")
    print(f"  Citation Rate       : {citation_rate:.1%}")
    print("=" * 60)

    faith_ok = avg_faith >= float(os.getenv("MIN_FAITHFULNESS", "0.75"))
    rel_ok = avg_rel >= float(os.getenv("MIN_ANSWER_RELEVANCE", "0.70"))

    print(f"\n  Faithfulness gate  : {'✅ PASS' if faith_ok else '❌ FAIL'}")
    print(f"  Relevance gate     : {'✅ PASS' if rel_ok else '❌ FAIL'}")
    print(f"  Overall            : {'✅ ALL PASS' if faith_ok and rel_ok else '❌ SOME FAILED'}")
    print("=" * 60)

    return {
        "avg_faithfulness": avg_faith,
        "avg_answer_relevance": avg_rel,
        "citation_rate": citation_rate,
        "faithfulness_pass": faith_ok,
        "relevance_pass": rel_ok,
        "all_pass": faith_ok and rel_ok,
    }
