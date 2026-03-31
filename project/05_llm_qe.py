"""
05_llm_qe.py
~~~~~~~~~~~~
Step 5 — LLM-based query expansion using Llama 3.2 (via Ollama).

Prerequisites:
    ollama serve &
    ollama pull llama3.2:3b

Run standalone:
    python project/05_llm_qe.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyterrier as pt
from project.config import BM25_MODEL, DATASET_NAME, INDEX_DIR, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL
from project.llama_service import LlamaService
from project.utils import evaluate_run, sanitize_terrier_query, save_metrics, save_results


def main() -> dict:
    svc = LlamaService(base_url=OLLAMA_URL, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT)
    if not svc.is_available():
        print(f"⚠️  Neither Ollama nor llama-cpp-python is available.")
        print(f"   Option 1: Start Ollama:  ollama serve &")
        print(f"   Option 2: Install CPU backend:")
        print(f"     CMAKE_ARGS=\"-DGGML_METAL=OFF\" pip install llama-cpp-python")
        print(f"   Then re-pull the model:  ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

    print(f"=== LLM-BASED QE (model: {OLLAMA_MODEL}) ===")
    dataset    = pt.get_dataset(DATASET_NAME)
    queries_df = dataset.get_topics()
    total      = len(queries_df)
    print(f"Expanding {total} queries...")

    expanded_rows = []
    for i, (_, row) in enumerate(queries_df.iterrows()):
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  Progress: {i + 1}/{total}")
        expanded_rows.append({
            "qid":            row["qid"],
            "query":          sanitize_terrier_query(svc.expand_query(row["query"])),
            "original_query": row["query"],
        })

    expanded_df = pd.DataFrame(expanded_rows)
    save_results(expanded_df, "llm_expanded_queries")

    print("Sample expansions:")
    for _, r in expanded_df.head(3).iterrows():
        print(f"  [{r['qid']}] {r['original_query']!r}  →  {r['query']!r}")

    print("\n=== RETRIEVING WITH EXPANDED QUERIES ===")
    index_ref = pt.IndexRef.of(INDEX_DIR)
    retriever = pt.BatchRetrieve(index_ref, wmodel=BM25_MODEL)
    results   = retriever.transform(expanded_df[["qid", "query"]])
    save_results(results, "llm_qe")

    print("\n=== EVALUATION ===")
    metrics = evaluate_run(results, dataset.get_qrels())
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    save_metrics(metrics, "llm_qe")

    print("\n✅ LLM QE done.")
    return metrics


if __name__ == "__main__":
    main()
