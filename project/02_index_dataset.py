"""
02_index_dataset.py
~~~~~~~~~~~~~~~~~~~
Step 2 — Build a PyTerrier BM25 index over the corpus.

Run standalone:
    python project/02_index_dataset.py [--force]

Flags:
    --force   Delete and recreate the index even if it already exists.
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyterrier as pt
from project.config import DATASET_NAME, INDEX_DIR


def main(force_reindex: bool = False) -> None:
    if os.path.exists(INDEX_DIR) and not force_reindex:
        print(f"Index already exists at: {INDEX_DIR}")
        print("  Pass force_reindex=True (or --force) to rebuild.")
        return

    print(f"Loading '{DATASET_NAME}' dataset...")
    dataset = pt.get_dataset(DATASET_NAME)

    if os.path.exists(INDEX_DIR):
        print(f"Removing existing index at: {INDEX_DIR}")
        shutil.rmtree(INDEX_DIR)

    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Creating index at: {INDEX_DIR}")

    indexer   = pt.IterDictIndexer(INDEX_DIR)
    index_ref = indexer.index(dataset.get_corpus_iter())

    stats = pt.IndexFactory.of(index_ref).getCollectionStatistics()
    print("\n=== INDEX STATISTICS ===")
    print(f"  Documents : {stats.getNumberOfDocuments()}")
    print(f"  Tokens    : {stats.getNumberOfTokens()}")
    print(f"  Vocab     : {stats.getNumberOfUniqueTerms()}")
    print(f"  Avg len   : {stats.getAverageDocumentLength():.2f}")

    print("\n[OK] Index created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild index even if it exists")
    args = parser.parse_args()
    main(force_reindex=args.force)
