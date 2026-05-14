"""
06_compare_all_methods.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Step 6 — Compare all retrieval approaches with a summary table and visualizations.

Run standalone:
    python project/06_compare_all_methods.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # headless-safe; won't block when no display is present

import matplotlib.pyplot as plt
import pandas as pd
from project.config import OUTPUT_DIR
from project.utils import ensure_output_dir


def _load_metrics(name: str) -> pd.Series:
    path = os.path.join(OUTPUT_DIR, f"metrics_{name}.csv")
    return pd.read_csv(path).iloc[0]


def main() -> None:
    ensure_output_dir()

    baseline  = _load_metrics("baseline")
    classical = _load_metrics("classical_qe")
    llm       = _load_metrics("llm_qe")

    comparison = pd.DataFrame({
        "Baseline BM25":      baseline,
        "Classical QE (PRF)": classical,
        "LLM QE":             llm,
    })

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(comparison.to_string())

    improvements = pd.DataFrame({
        "Classical QE (%)": ((classical - baseline) / baseline * 100).round(2),
        "LLM QE (%)":       ((llm       - baseline) / baseline * 100).round(2),
    })

    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 70)
    print(improvements.to_string())

    # Save CSVs
    comparison.to_csv(os.path.join(OUTPUT_DIR, "comparison_all_methods.csv"))
    improvements.to_csv(os.path.join(OUTPUT_DIR, "improvements_over_baseline.csv"))

    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    metrics = list(comparison.index)
    x = range(len(metrics))
    width = 0.25

    for offset, col in enumerate(comparison.columns):
        ax1.bar([i + (offset - 1) * width for i in x], comparison[col], width, label=col, alpha=0.8)
    ax1.set_ylabel("Score")
    ax1.set_title("Retrieval Performance Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    improvements.plot(kind="bar", ax=ax2, alpha=0.8)
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("Improvement Over Baseline BM25")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "comparison_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {plot_path}")

    print("\n=== BEST METHOD PER METRIC ===")
    for metric in comparison.index:
        best  = comparison.loc[metric].idxmax()
        score = comparison.loc[metric, best]
        print(f"  {metric:15} → {best} ({score:.4f})")

    print("\n[OK] Comparison complete.")


if __name__ == "__main__":
    main()
