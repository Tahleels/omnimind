"""
evaluate.py — CLI Evaluation Runner
Usage: python scripts/evaluate.py [--samples N]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.evaluation.run_eval import run_evaluation, print_report

def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation and print quality report.")
    parser.add_argument("--samples", type=int, default=10, help="Number of test samples to evaluate (default: 10).")
    args = parser.parse_args()

    results = run_evaluation(num_samples=args.samples)
    report = print_report(results)

    if not report.get("all_pass", False):
        print("\n⚠ Quality gate FAILED — scores below threshold.")
        sys.exit(1)
    else:
        print("\n✅ All quality gates PASSED.")

if __name__ == "__main__":
    main()
