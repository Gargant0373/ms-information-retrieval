"""
utils.py
~~~~~~~~
Shared evaluation helpers and I/O utilities for the IR pipeline.
"""

import os

import pandas as pd
import pytrec_eval

from project.config import DISPLAY_METRICS, EVAL_METRICS, OUTPUT_DIR


def ensure_output_dir() -> str:
    """Create OUTPUT_DIR if it doesn't exist and return its path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def sanitize_terrier_query(query: str) -> str:
    """
    Strip characters that Terrier's query parser cannot handle
    (apostrophes, quotes, parentheses, operators, etc.).
    """
    import re
    # Remove single quotes (apostrophes), double quotes, and Terrier-special chars
    query = re.sub(r"[\"'`#^()\[\]{}|!+\-]", " ", query)
    # Collapse runs of whitespace
    query = re.sub(r"\s+", " ", query).strip()
    return query


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_run_dict(results_df: pd.DataFrame) -> dict:
    run: dict = {}
    for _, row in results_df.iterrows():
        qid = str(row["qid"])
        run.setdefault(qid, {})[str(row["docno"])] = float(row["score"])
    return run


def _build_qrels_dict(qrels_df: pd.DataFrame) -> dict:
    qrels: dict = {}
    for _, row in qrels_df.iterrows():
        qid = str(row["qid"])
        qrels.setdefault(qid, {})[str(row["docno"])] = int(row["label"])
    return qrels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_run(results_df: pd.DataFrame, qrels_df: pd.DataFrame) -> dict:
    """
    Evaluate retrieval results against qrels using pytrec_eval.

    Returns a dict mapping metric name → score averaged across all queries.
    """
    run   = _build_run_dict(results_df)
    qrels = _build_qrels_dict(qrels_df)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, EVAL_METRICS)
    per_query = evaluator.evaluate(run)

    return {
        metric: sum(per_query[qid][metric] for qid in per_query) / len(per_query)
        for metric in DISPLAY_METRICS
    }


def save_results(results_df: pd.DataFrame, name: str) -> str:
    """Save a retrieval results DataFrame to OUTPUT_DIR; returns the file path."""
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, f"results_{name}.csv")
    results_df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return path


def save_metrics(metrics: dict, name: str) -> str:
    """Save an averaged metrics dict as a single-row CSV; returns the file path."""
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, f"metrics_{name}.csv")
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f"  Saved → {path}")
    return path
