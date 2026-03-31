"""
03_baseline_retrieval.py
~~~~~~~~~~~~~~~~~~~~~~~~
Step 3 — Baseline BM25 retrieval and evaluation.

Run standalone:
    python project/03_baseline_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyterrier as pt
from project.config import BM25_MODEL, DATASET_NAME, INDEX_DIR
from project.utils import evaluate_run, save_metrics, save_results


def main() -> dict:
    dataset   = pt.get_dataset(DATASET_NAME)
    index_ref = pt.IndexRef.of(INDEX_DIR)
    retriever = pt.BatchRetrieve(index_ref, wmodel=BM25_MODEL)

    queries_df = dataset.get_topics()
    print("=== BASELINE BM25 ===")
    print(f"Retrieving for {len(queries_df)} queries...")
    results = retriever.transform(queries_df)
    save_results(results, "baseline")

    print("\n=== EVALUATION ===")
    metrics = evaluate_run(results, dataset.get_qrels())
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    save_metrics(metrics, "baseline")

    print("\n✅ Baseline done.")
    return metrics


if __name__ == "__main__":
    main()
