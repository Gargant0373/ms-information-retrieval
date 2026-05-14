"""
pipeline.py
~~~~~~~~~~~
End-to-end IR pipeline runner.

Usage (from repo root):
    python project/pipeline.py [--force-index] [--skip-llm]

Flags:
    --force-index   Rebuild the BM25 index even if it already exists.
    --skip-llm      Skip LLM-based QE (Ollama not required).
                    A zero-score stub is written so the comparison step still runs.

Steps:
    1  Load & inspect the dataset
    2  Build the BM25 index (skipped if already exists unless --force-index)
    3  Baseline BM25 retrieval + evaluation
    4  Classical QE (PRF / Bo1) + evaluation
    5  LLM-based QE (Llama via Ollama) + evaluation
    6  Compare all methods, produce tables and a chart
"""

import argparse
import importlib.util
import os
import sys

# Ensure the repo root is on sys.path so all `project.*` imports resolve
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from project.config import DISPLAY_METRICS, OUTPUT_DIR
from project.utils import ensure_output_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_step(filename: str):
    """
    Load a project script by file path and return its module.

    Using importlib allows loading files whose names start with a digit,
    which Python's standard import machinery would reject.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def _write_zero_metrics(name: str) -> None:
    """Write a zero-score metrics stub so the comparison step can still run."""
    ensure_output_dir()
    stub_path = os.path.join(OUTPUT_DIR, f"metrics_{name}.csv")
    if not os.path.exists(stub_path):
        pd.DataFrame([{m: 0.0 for m in DISPLAY_METRICS}]).to_csv(stub_path, index=False)
        print(f"  Stub metrics written → {stub_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(force_index: bool = False, skip_llm: bool = False) -> None:
    _section("STEP 1 — Dataset exploration")
    _load_step("01_load_dataset.py").main()

    _section("STEP 2 — Indexing")
    _load_step("02_index_dataset.py").main(force_reindex=force_index)

    _section("STEP 3 — Baseline BM25")
    _load_step("03_baseline_retrieval.py").main()

    _section("STEP 4 — Classical QE (PRF / Bo1)")
    _load_step("04_classical_qe.py").main()

    _section("STEP 5 — LLM-based QE")
    if skip_llm:
        print("  Skipped (--skip-llm flag set).")
        _write_zero_metrics("llm_qe")
    else:
        try:
            _load_step("05_llm_qe.py").main()
        except SystemExit as exc:
            print(f"  [WARNING] LLM QE step exited early (code {exc.code}). Writing zero-score stub.")
            _write_zero_metrics("llm_qe")
        except Exception as exc:
            print(f"  [WARNING] LLM QE step failed: {exc}")
            _write_zero_metrics("llm_qe")

    _section("STEP 6 — Compare all methods")
    _load_step("06_compare_all_methods.py").main()

    _section("PIPELINE COMPLETE")
    print(f"  All outputs saved to: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full IR project pipeline.")
    parser.add_argument(
        "--force-index", action="store_true",
        help="Rebuild the BM25 index even if it already exists.",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM-based QE step (Ollama not required).",
    )
    args = parser.parse_args()
    run_pipeline(force_index=args.force_index, skip_llm=args.skip_llm)


if __name__ == "__main__":
    main()
