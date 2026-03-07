"""
Phase 7: Generate all figures for the report.
1. Token length distribution of title vs plot (reproduces Figure 2)
2. Total tokens/latency/cost vs selectivity for both orderings (reproduces Figure 3)
3. Ordering accuracy comparison: local vs sample vs oracle
4. Sample budget tradeoff curve (novel)
5. Table 2 reproduction: per-operator stats
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.tokenizer import count_tokens

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def figure1_token_distribution():
    """Reproduce Figure 2: Token length distribution of title vs plot."""
    df = pd.read_csv("data/movies_full.csv")

    df["title_tokens"] = df["title"].apply(count_tokens)
    df["plot_tokens"] = df["plot"].apply(count_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["title_tokens"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Input Length (Title)")
    axes[0].set_ylabel("Occurrences")
    axes[0].set_title("(a) Title Token Distribution")

    axes[1].hist(df["plot_tokens"], bins=50, color="coral", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Input Length (Plot)")
    axes[1].set_ylabel("Occurrences")
    axes[1].set_title("(b) Plot Token Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_token_distribution.png"), dpi=150)
    plt.close()
    print("  Saved fig2_token_distribution.png")


def figure2_selectivity_comparison():
    """Reproduce Figure 3: Tokens/latency/cost vs selectivity."""
    comparison_path = os.path.join(RESULTS_DIR, "comparison", "full_results.json")
    if not os.path.exists(comparison_path):
        print("  Skipping fig3: no comparison results found")
        return

    with open(comparison_path) as f:
        results = json.load(f)

    # Group by first operator selectivity
    data = []
    for r in results:
        for ordering in ["lang_first", "genre_first"]:
            if ordering in r.get("real_results", {}):
                stats = r["real_results"][ordering]
                data.append({
                    "config": r["config"],
                    "lang_sel": r["lang_selectivity"],
                    "genre_sel": r["genre_selectivity"],
                    "ordering": "title->plot" if ordering == "lang_first" else "plot->title",
                    "total_tokens": stats["total_tokens"],
                    "total_latency": stats["total_latency"],
                    "total_cost": stats["total_cost"],
                    "tuples_to_llm": sum(stats.get("tuples_per_stage", [])),
                })

    if not data:
        print("  No data for fig3")
        return

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ordering, color, marker in [("title->plot", "steelblue", "o"), ("plot->title", "coral", "s")]:
        sub = df[df["ordering"] == ordering].sort_values("lang_sel")
        axes[0].plot(sub["lang_sel"], sub["tuples_to_llm"], marker=marker,
                     color=color, label=ordering, linewidth=2)
        axes[1].plot(sub["lang_sel"], sub["total_tokens"], marker=marker,
                     color=color, label=ordering, linewidth=2)
        axes[2].plot(sub["lang_sel"], sub["total_latency"], marker=marker,
                     color=color, label=ordering, linewidth=2)

    axes[0].set_xlabel("Selectivity of First Semantic Operator")
    axes[0].set_ylabel("Total Tuples to LLM")
    axes[0].legend()

    axes[1].set_xlabel("Selectivity of First Semantic Operator")
    axes[1].set_ylabel("Total Tokens Processed")
    axes[1].legend()

    axes[2].set_xlabel("Selectivity of First Semantic Operator")
    axes[2].set_ylabel("Total Time (s)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_selectivity_comparison.png"), dpi=150)
    plt.close()
    print("  Saved fig3_selectivity_comparison.png")


def figure3_ordering_accuracy():
    """Ordering accuracy: local vs sample vs oracle across 25 configs."""
    summary_path = os.path.join(RESULTS_DIR, "comparison", "summary.csv")
    if not os.path.exists(summary_path):
        print("  Skipping ordering accuracy: no summary found")
        return

    df = pd.read_csv(summary_path)

    methods = ["local_correct", "sample_correct", "oracle_correct"]
    labels = ["Local Only", "Small-Sample", "Oracle"]
    accuracies = [df[m].mean() * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=["steelblue", "coral", "green"],
                  edgecolor="black", alpha=0.8)
    ax.set_ylabel("Ordering Accuracy (%)")
    ax.set_title("Ordering Accuracy vs Oracle")
    ax.set_ylim(0, 105)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.0f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_ordering_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved fig4_ordering_accuracy.png")


def figure4_ablation_curve():
    """Sample budget tradeoff curve (novel contribution)."""
    ablation_path = os.path.join(RESULTS_DIR, "ablation", "ablation_summary.csv")
    if not os.path.exists(ablation_path):
        print("  Skipping ablation curve: no ablation results found")
        return

    df = pd.read_csv(ablation_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["sample_size"], df["accuracy"] * 100, "o-",
            color="coral", linewidth=2, markersize=8)
    ax.set_xlabel("Sample Budget (LLM calls per operator)")
    ax.set_ylabel("Ordering Accuracy (%)")
    ax.set_title("Cost/Quality Tradeoff: Sample Budget vs Accuracy")
    ax.set_ylim(0, 105)
    ax.set_xticks(df["sample_size"])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_ablation_curve.png"), dpi=150)
    plt.close()
    print("  Saved fig5_ablation_curve.png")


def generate_all():
    print("Generating figures...")
    figure1_token_distribution()
    figure2_selectivity_comparison()
    figure3_ordering_accuracy()
    figure4_ablation_curve()
    print("Done!")


if __name__ == "__main__":
    generate_all()
