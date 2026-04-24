"""
Phase 7: Generate all figures for the report.
1. Token length distribution of title vs plot (reproduces Figure 2)
2. Total tokens/latency/cost vs selectivity for both orderings (reproduces Figure 3)
3. Ordering accuracy comparison: local vs sample vs oracle
4. Sample budget tradeoff curve (novel)
5. Table 2 reproduction: per-operator stats
6. Eddy vs baselines accuracy bar chart
7. Eddy convergence curve
8. Routing policy comparison
9. Non-stationary adaptivity
10. Cost savings: eddy vs fixed-sample
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


def figure5_eddy_accuracy():
    """Eddy vs baselines accuracy bar chart."""
    summary_path = os.path.join(RESULTS_DIR, "eddy_comparison", "summary.csv")
    if not os.path.exists(summary_path):
        print("  Skipping fig6: no eddy comparison results found")
        return

    df = pd.read_csv(summary_path)

    methods = ["local_correct", "sample_correct", "oracle_correct", "eddy_correct"]
    labels = ["Local Only", "Fixed-Sample", "Oracle", "Eddy (Thompson)"]
    colors = ["steelblue", "coral", "green", "darkorchid"]
    accuracies = [df[m].mean() * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Ordering Accuracy (%)")
    ax.set_title("Ordering Accuracy: Eddy vs Baselines (25 configs)")
    ax.set_ylim(0, 110)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.0f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig6_eddy_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved fig6_eddy_accuracy.png")


def figure6_convergence():
    """Eddy convergence curve: cumulative accuracy over tuple index."""
    details_path = os.path.join(RESULTS_DIR, "eddy_convergence",
                                "convergence_details.json")
    if not os.path.exists(details_path):
        print("  Skipping fig7: no convergence results found")
        return

    with open(details_path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        per_tuple = r.get("per_tuple", [])
        if not per_tuple:
            continue
        idxs = [p["tuple_idx"] for p in per_tuple]
        accs = [p["cumulative_accuracy"] * 100 for p in per_tuple]
        ax.plot(idxs, accs, alpha=0.3, linewidth=1, color="steelblue")

    # Average across all configs
    max_len = max(len(r.get("per_tuple", [])) for r in results)
    if max_len > 0:
        avg_acc = []
        for i in range(max_len):
            vals = []
            for r in results:
                pt = r.get("per_tuple", [])
                if i < len(pt):
                    vals.append(pt[i]["cumulative_accuracy"] * 100)
            if vals:
                avg_acc.append(np.mean(vals))
        ax.plot(range(len(avg_acc)), avg_acc, color="darkred",
                linewidth=3, label="Average across configs")

    ax.set_xlabel("Tuple Index")
    ax.set_ylabel("Cumulative Routing Accuracy (%)")
    ax.set_title("Eddy Convergence: Routing Accuracy Over Time")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig7_convergence.png"), dpi=150)
    plt.close()
    print("  Saved fig7_convergence.png")


def figure7_policy_comparison():
    """Routing policy comparison: accuracy and cost."""
    summary_path = os.path.join(RESULTS_DIR, "eddy_policies", "policy_summary.csv")
    if not os.path.exists(summary_path):
        print("  Skipping fig8: no policy comparison results found")
        return

    df = pd.read_csv(summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["darkorchid", "teal", "coral", "steelblue"]
    x = range(len(df))

    # Accuracy
    bars1 = axes[0].bar(x, df["accuracy"] * 100, color=colors[:len(df)],
                        edgecolor="black", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["policy"], rotation=15, ha="right")
    axes[0].set_ylabel("Ordering Accuracy (%)")
    axes[0].set_title("Accuracy by Routing Policy")
    axes[0].set_ylim(0, 110)
    for bar, acc in zip(bars1, df["accuracy"] * 100):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{acc:.0f}%", ha="center", va="bottom", fontweight="bold")

    # Average tokens
    bars2 = axes[1].bar(x, df["avg_tokens"], color=colors[:len(df)],
                        edgecolor="black", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["policy"], rotation=15, ha="right")
    axes[1].set_ylabel("Avg Total Tokens")
    axes[1].set_title("Token Cost by Routing Policy")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig8_policy_comparison.png"), dpi=150)
    plt.close()
    print("  Saved fig8_policy_comparison.png")


def figure8_nonstationary():
    """Non-stationary adaptivity: eddy routing decisions over time."""
    ns_path = os.path.join(RESULTS_DIR, "eddy_nonstationary",
                           "nonstationary_results.json")
    if not os.path.exists(ns_path):
        print("  Skipping fig9: no non-stationary results found")
        return

    with open(ns_path) as f:
        results = json.load(f)

    for r in results:
        config = r["config"]
        per_tuple = r.get("per_tuple_routing", [])
        if not per_tuple:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Top: routing decisions over time
        idxs = [p["tuple_idx"] for p in per_tuple]
        is_lang_first = [int(p["is_lang_first"]) for p in per_tuple]

        # Smooth with rolling average
        window = max(1, len(idxs) // 20)
        rolling = pd.Series(is_lang_first).rolling(window, min_periods=1).mean()
        axes[0].fill_between(idxs, rolling, alpha=0.4, color="steelblue",
                             label="lang_first fraction")
        axes[0].fill_between(idxs, rolling, 1, alpha=0.4, color="coral",
                             label="genre_first fraction")
        axes[0].set_ylabel("Routing Fraction")
        axes[0].set_title(f"Eddy Routing Decisions Over Time ({config})")
        axes[0].legend(loc="upper right")
        axes[0].set_ylim(0, 1)

        # Bottom: per-segment selectivity
        segments = r.get("segment_stats", [])
        if segments:
            seg_mids = [(s["start"] + s["end"]) / 2 for s in segments]
            lang_sels = [s["lang_sel"] for s in segments]
            genre_sels = [s["genre_sel"] for s in segments]
            axes[1].plot(seg_mids, lang_sels, "o-", color="steelblue",
                         label="lang selectivity", linewidth=2)
            axes[1].plot(seg_mids, genre_sels, "s-", color="coral",
                         label="genre selectivity", linewidth=2)
            axes[1].set_ylabel("True Selectivity")
            axes[1].set_xlabel("Tuple Index")
            axes[1].legend()
            axes[1].set_ylim(0, 1)

        # Add shift line for shift configs
        n_rows = r["n_rows"]
        if "shift50" in config:
            for ax in axes:
                ax.axvline(x=n_rows // 2, color="red", linestyle="--",
                           alpha=0.7, label="shift point")
        elif "shift75" in config:
            for ax in axes:
                ax.axvline(x=int(n_rows * 0.75), color="red",
                           linestyle="--", alpha=0.7, label="shift point")

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"fig9_nonstationary_{config}.png"),
                    dpi=150)
        plt.close()
        print(f"  Saved fig9_nonstationary_{config}.png")


def figure9_cost_savings():
    """Cost savings: eddy vs fixed-sample wasted calls."""
    eddy_path = os.path.join(RESULTS_DIR, "eddy_comparison", "full_results.json")
    comp_path = os.path.join(RESULTS_DIR, "comparison", "full_results.json")

    if not os.path.exists(eddy_path):
        print("  Skipping fig10: no eddy comparison results found")
        return

    with open(eddy_path) as f:
        eddy_results = json.load(f)

    # Collect eddy tokens vs best-fixed tokens vs sample overhead
    data = []
    for r in eddy_results:
        real = r.get("real_results", {})
        if not real:
            continue
        best_fixed = min(
            real.get("lang_first", {}).get("total_tokens", float("inf")),
            real.get("genre_first", {}).get("total_tokens", float("inf")),
        )
        data.append({
            "config": r["config"],
            "best_fixed": best_fixed,
            "eddy_tokens": r["eddy_tokens"],
            "sample_overhead": r.get("sample_overhead_cost", 0) * 1e6,
        })

    if not data:
        print("  No data for fig10")
        return

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["best_fixed"], width,
           label="Best Fixed Ordering", color="steelblue", alpha=0.8)
    ax.bar([i + width / 2 for i in x], df["eddy_tokens"], width,
           label="Eddy (Thompson)", color="darkorchid", alpha=0.8)

    ax.set_xlabel("Config")
    ax.set_ylabel("Total Tokens")
    ax.set_title("Eddy vs Best Fixed Ordering: Total Token Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=45, ha="right", fontsize=7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig10_cost_savings.png"), dpi=150)
    plt.close()
    print("  Saved fig10_cost_savings.png")


def generate_all():
    """Generate all figures (original + eddy extensions)."""
    print("Generating figures...")
    figure1_token_distribution()
    figure2_selectivity_comparison()
    figure3_ordering_accuracy()
    figure4_ablation_curve()
    figure5_eddy_accuracy()
    figure6_convergence()
    figure7_policy_comparison()
    figure8_nonstationary()
    figure9_cost_savings()
    print("Done!")


if __name__ == "__main__":
    generate_all()
