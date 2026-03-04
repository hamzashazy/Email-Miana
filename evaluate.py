"""
Evaluate the LLM classifier against the labeled dataset.

Usage:
    python evaluate.py                          # Evaluate all 513 emails
    python evaluate.py --sample 50              # Evaluate a random sample of 50
    python evaluate.py --sample 50 --seed 42    # Reproducible sample
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from classifier.config import CATEGORIES, MAX_CONCURRENT_REQUESTS
from classifier.data_loader import load_emails
from classifier.llm_client import classify_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "email-classification-report.xlsx"


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute precision, recall, F1 per class and overall accuracy."""
    categories = sorted(set(y_true + y_pred))

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    per_class = {}
    for cat in categories:
        p = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0.0
        r = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp[cat] + fn[cat]
        per_class[cat] = {"precision": p, "recall": r, "f1": f1, "support": support}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    return {"accuracy": accuracy, "correct": correct, "total": len(y_true), "per_class": per_class}


def build_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    label_idx = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            matrix[label_idx[t]][label_idx[p]] += 1
    return matrix


def print_report(metrics: dict, confusion: list[list[int]], labels: list[str], misclassified: list[dict]):
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT — Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    print("=" * 70)

    print(f"\n{'Category':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 58)
    for cat in labels:
        if cat in metrics["per_class"]:
            m = metrics["per_class"][cat]
            print(f"{cat:<18} {m['precision']:>10.1%} {m['recall']:>10.1%} {m['f1']:>10.1%} {m['support']:>10}")

    print(f"\n{'Confusion Matrix':^58}")
    print("-" * 58)
    true_pred = "True \\ Pred"
    header = f"{true_pred:<18}" + "".join(f"{l[:8]:>10}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:<18}" + "".join(f"{confusion[i][j]:>10}" for j in range(len(labels)))
        print(row)

    if misclassified:
        print(f"\n{'Misclassified Examples (up to 20)':^58}")
        print("-" * 58)
        for item in misclassified[:20]:
            print(f"  ID: {item['id']}")
            print(f"  True: {item['true']} -> Predicted: {item['pred']} (confidence: {item['confidence']:.0%})")
            print(f"  Subject: {item['subject'][:80]}")
            print(f"  Reasoning: {item['reasoning'][:100]}")
            print()


async def main():
    parser = argparse.ArgumentParser(description="Evaluate email classifier")
    parser.add_argument("--sample", type=int, default=0, help="Evaluate a random sample (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--output", type=str, default="", help="Save detailed results to JSON")
    args = parser.parse_args()

    emails = load_emails(DATA_PATH)
    logger.info("Loaded %d emails from dataset", len(emails))

    if args.sample > 0:
        random.seed(args.seed)
        emails = random.sample(emails, min(args.sample, len(emails)))
        logger.info("Sampled %d emails for evaluation", len(emails))

    y_true = [e["classification"] for e in emails]

    logger.info("Classifying %d emails with concurrency=%d ...", len(emails), args.concurrency)
    start = time.time()
    results = await classify_batch(emails, max_concurrent=args.concurrency)
    elapsed = time.time() - start
    logger.info("Classification complete in %.1fs (%.2fs/email)", elapsed, elapsed / len(emails))

    y_pred = [r["classification"] for r in results]

    labels = [c for c in CATEGORIES if c in set(y_true + y_pred)]
    metrics = compute_metrics(y_true, y_pred)
    confusion = build_confusion_matrix(y_true, y_pred, labels)

    misclassified = []
    for email, result in zip(emails, results):
        if email["classification"] != result["classification"]:
            misclassified.append({
                "id": email["id"],
                "true": email["classification"],
                "pred": result["classification"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "subject": email.get("subject", ""),
                "snippet": (email.get("snippet", "") or "")[:200],
            })

    print_report(metrics, confusion, labels, misclassified)

    if args.output:
        output_data = {
            "metrics": metrics,
            "confusion_labels": labels,
            "confusion_matrix": confusion,
            "misclassified": misclassified,
            "all_results": [
                {
                    "id": r["id"],
                    "true_label": e["classification"],
                    "pred_label": r["classification"],
                    "confidence": r["confidence"],
                    "reasoning": r["reasoning"],
                }
                for e, r in zip(emails, results)
            ],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        logger.info("Detailed results saved to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
