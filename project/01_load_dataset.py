"""
01_load_dataset.py
~~~~~~~~~~~~~~~~~~
Step 1 — Load and explore the dataset.

Run standalone:
    python project/01_load_dataset.py
"""

import os
import sys

# Allow running as a standalone script from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyterrier as pt
from project.config import DATASET_NAME


def main() -> None:
    print(f"Loading '{DATASET_NAME}' dataset...")
    dataset = pt.get_dataset(DATASET_NAME)

    queries_df = dataset.get_topics()
    qrels_df   = dataset.get_qrels()

    print("\n=== QUERIES ===")
    print(f"Total: {len(queries_df)}")
    print(queries_df.head(3).to_string(index=False))

    print("\n=== QRELS ===")
    print(f"Total: {len(qrels_df)}")
    print(f"Avg relevant docs / query: {len(qrels_df) / len(queries_df):.2f}")
    print(qrels_df.head(3).to_string(index=False))

    print("\n=== CORPUS ===")
    sample: list = []
    total = 0
    for doc in dataset.get_corpus_iter():
        if len(sample) < 3:
            sample.append(doc)
        total += 1
    print(f"Total documents: {total}")
    for i, doc in enumerate(sample):
        print(f"  Doc {i}: {doc}")

    print("\n✅ Dataset loaded.")


if __name__ == "__main__":
    main()
