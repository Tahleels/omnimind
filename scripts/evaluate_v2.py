"""
evaluate.py — LLM-as-a-Judge Evaluation Runner for v2 Pipeline

Evaluates RAGPipelineV2 on the curated test dataset.  Scores each
response for faithfulness, answer relevance, and citation coverage
using the same LLM-as-judge approach as v1, but now routes through
the full v2 graph (memory, router, workspace-scoped retrieval).

Usage:
    python scripts/evaluate_v2.py [--samples N] [--workspace WORKSPACE_ID]

Options:
    --samples N          Number of test cases to evaluate (default: 10).
    --workspace ID       Workspace to query against (default: "default").
    --delay SECONDS      Sleep between API calls to avoid rate limits (default: 2).
    --output FILE        Write JSON report to this file (optional).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.v2.evaluation.test_dataset import EVAL_DATASET
from src.v2.generation.chain import RAGPipelineV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """\
You are an objective evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the following Q&A pair on two dimensions. Return ONLY a JSON object.

Question: {question}
Reference Answer: {reference}
Generated Answer: {generated}

Score each dimension from 0.0 to 1.0:
1. faithfulness: Is the generated answer factually consistent with the reference? (1.0 = fully consistent)
2. answer_relevance: Does the generated answer address the question? (1.0 = directly answers)

Return ONLY this JSON (no other text):
{{"faithfulness": 0.0, "answer_relevance": 0.0}}\
"""


class EvalResult(NamedTuple):
    question: str
    generated_answer: str
    faithfulness: float
    answer_relevance: float
    has_citations: bool
    num_sources: int
    route: str


def _make_judge(provider: str, model: str):
    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            default_headers={"HTTP-Referer": "http://localhost:8000"},
        )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )


def run_evaluation(
    num_samples: int = 10,
    workspace_id: str = "default",
    delay_between: float = 2.0,
) -> list[EvalResult]:
    logger.info("Initialising RAGPipelineV2 for evaluation...")
    pipeline = RAGPipelineV2()

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    judge = _make_judge(provider, model)

    dataset = EVAL_DATASET[:num_samples]
    results: list[EvalResult] = []

    logger.info(f"Evaluating {len(dataset)} test cases (workspace={workspace_id})...\n")

    for i, sample in enumerate(dataset, 1):
        question = sample["question"]
        reference = sample["ground_truth"]
        logger.info(f"[{i}/{len(dataset)}] Q: {question[:65]}...")

        # Run v2 pipeline (creates a fresh session per question for isolation)
        try:
            rag = pipeline.ask(
                question=question,
                workspace_id=workspace_id,
                search_mode="combined",
            )
            generated = rag["answer"]
            num_sources = len(rag["sources"])
            has_citations = bool(rag["sources"])
            route = rag.get("route", "rag")
        except Exception as exc:
            logger.error(f"  Pipeline error: {exc}")
            results.append(EvalResult(question, f"ERROR: {exc}", 0.0, 0.0, False, 0, "error"))
            continue

        # LLM-as-judge scoring
        try:
            judge_prompt = JUDGE_PROMPT.format(
                question=question, reference=reference, generated=generated
            )
            resp = judge.invoke(judge_prompt)
            text = resp.content.strip()
            # Handle markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            scores = json.loads(text)
            faithfulness = float(scores.get("faithfulness", 0.0))
            relevance = float(scores.get("answer_relevance", 0.0))
        except Exception as exc:
            logger.warning(f"  Judge failed: {exc} — defaulting to 0.0")
            faithfulness = relevance = 0.0

        results.append(EvalResult(question, generated, faithfulness, relevance, has_citations, num_sources, route))
        logger.info(
            f"  ✓ Faith: {faithfulness:.2f} | Rel: {relevance:.2f} | "
            f"Sources: {num_sources} | Route: {route}"
        )

        if i < len(dataset):
            time.sleep(delay_between)

    return results


def print_report(results: list[EvalResult], output_path: str | None = None) -> dict:
    if not results:
        print("No results.")
        return {}

    avg_faith = sum(r.faithfulness for r in results) / len(results)
    avg_rel = sum(r.answer_relevance for r in results) / len(results)
    citation_rate = sum(r.has_citations for r in results) / len(results)
    route_counts: dict[str, int] = {}
    for r in results:
        route_counts[r.route] = route_counts.get(r.route, 0) + 1

    print("\n" + "=" * 62)
    print("  RAG PIPELINE v2 — EVALUATION REPORT")
    print("=" * 62)
    print(f"  Samples evaluated   : {len(results)}")
    print(f"  Avg Faithfulness    : {avg_faith:.3f}  (threshold: 0.75)")
    print(f"  Avg Answer Relevance: {avg_rel:.3f}  (threshold: 0.70)")
    print(f"  Citation Rate       : {citation_rate:.1%}")
    print(f"  Route distribution  : {route_counts}")
    print("=" * 62)

    faith_ok = avg_faith >= float(os.getenv("MIN_FAITHFULNESS", "0.75"))
    rel_ok = avg_rel >= float(os.getenv("MIN_ANSWER_RELEVANCE", "0.70"))

    print(f"\n  Faithfulness gate  : {'✅ PASS' if faith_ok else '❌ FAIL'}")
    print(f"  Relevance gate     : {'✅ PASS' if rel_ok else '❌ FAIL'}")
    print(f"  Overall            : {'✅ ALL PASS' if faith_ok and rel_ok else '❌ SOME FAILED'}")
    print("=" * 62)

    report = {
        "avg_faithfulness": avg_faith,
        "avg_answer_relevance": avg_rel,
        "citation_rate": citation_rate,
        "faithfulness_pass": faith_ok,
        "relevance_pass": rel_ok,
        "all_pass": faith_ok and rel_ok,
        "route_counts": route_counts,
        "results": [r._asdict() for r in results],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2))
        logger.info(f"\nReport saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAGPipelineV2.")
    parser.add_argument("--samples", type=int, default=10, help="Number of test cases (default: 10)")
    parser.add_argument("--workspace", default="default", help="Workspace ID to query (default: default)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between API calls (default: 2)")
    parser.add_argument("--output", default=None, help="Optional JSON output file path")
    args = parser.parse_args()

    results = run_evaluation(
        num_samples=args.samples,
        workspace_id=args.workspace,
        delay_between=args.delay,
    )
    report = print_report(results, output_path=args.output)
    sys.exit(0 if report.get("all_pass") else 1)


if __name__ == "__main__":
    main()
