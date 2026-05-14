"""
04_classical_qe.py
~~~~~~~~~~~~~~~~~~
Step 4 — Classical query expansion via Pseudo-Relevance Feedback (Bo1).

Run standalone:
    python project/04_classical_qe.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyterrier as pt
from project.config import BM25_MODEL, DATASET_NAME, INDEX_DIR, PRF_FB_DOCS, PRF_FB_TERMS
from project.utils import evaluate_run, save_metrics, save_results


def main() -> dict:
    dataset   = pt.get_dataset(DATASET_NAME)
    index_ref = pt.IndexRef.of(INDEX_DIR)

    bm25 = pt.BatchRetrieve(index_ref, wmodel=BM25_MODEL)
    qe_pipeline = (
        bm25
        >> pt.rewrite.QueryExpansion(index_ref, fb_docs=PRF_FB_DOCS, fb_terms=PRF_FB_TERMS)
        >> pt.BatchRetrieve(index_ref, wmodel=BM25_MODEL)
    )

    queries_df = dataset.get_topics()
    print("=== CLASSICAL QE (PRF / Bo1) ===")
    print(f"  fb_docs={PRF_FB_DOCS}  fb_terms={PRF_FB_TERMS}")
    print(f"Retrieving for {len(queries_df)} queries...")
    results = qe_pipeline.transform(queries_df)
    save_results(results, "classical_qe")

    print("\n=== EVALUATION ===")
    metrics = evaluate_run(results, dataset.get_qrels())
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    save_metrics(metrics, "classical_qe")

    print("\n[OK] Classical QE done.")
    return metrics


if __name__ == "__main__":
    main()
