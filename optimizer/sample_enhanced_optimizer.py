"""
Phase 5 - Step 5.3: Sample-enhanced optimizer.
Runs small-sample estimator for each operator, then uses selectivity-aware cost model.
"""

from itertools import permutations
import pandas as pd
from cost_model.local_estimator import estimate_operator_cost
from sampler.small_sample_estimator import SmallSampleEstimator


def find_best_ordering_sampled(operators: list, relation: pd.DataFrame,
                               sample_size: int = 20,
                               ground_truth_cols: dict = None,
                               objective: str = "tokens",
                               accuracy_threshold: float = 0.5) -> dict:
    """
    1. Run small-sample estimator for each operator
    2. Use selectivity estimates to compute plan costs
    3. Pick ordering with minimum cost

    ground_truth_cols: {operator_name: column_name} for accuracy estimation
    """
    tuples = relation.to_dict("records")
    n_total = len(relation)

    # Step 1: Sample each operator
    estimator = SmallSampleEstimator(sample_size=sample_size)
    sample_stats = {}
    total_sample_cost = 0

    for op in operators:
        gt_col = ground_truth_cols.get(op.name) if ground_truth_cols else None
        stats = estimator.estimate(op, relation, ground_truth_col=gt_col)
        sample_stats[op.name] = stats
        total_sample_cost += stats["total_sample_cost"]

    # Step 2: Enumerate orderings with selectivity-aware cost
    best_ordering = None
    best_cost = float("inf")
    all_plans = []

    for perm in permutations(range(len(operators))):
        ordered_ops = [operators[i] for i in perm]
        total_cost = 0
        n_remaining = n_total
        plan_detail = []

        for op in ordered_ops:
            est = estimate_operator_cost(op, tuples, n_remaining)
            if objective == "tokens":
                cost = est["total_compute_cost"]
            else:
                cost = est["total_monetary_cost"]
            total_cost += cost

            # Get selectivity from samples
            op_sample = sample_stats[op.name]
            selectivity = op_sample["estimated_selectivity"]
            accuracy = op_sample.get("estimated_accuracy")

            # Adjust selectivity based on accuracy
            if accuracy is not None and accuracy < accuracy_threshold:
                penalty = 1.0 - accuracy
                effective_sel = selectivity + penalty * (1.0 - selectivity)
            else:
                effective_sel = selectivity

            plan_detail.append({
                "operator": op.name,
                "n_remaining": n_remaining,
                "cost": cost,
                "selectivity": selectivity,
                "accuracy": accuracy,
                "effective_selectivity": effective_sel,
            })

            n_remaining = int(n_remaining * effective_sel)

        plan = {
            "ordering": " -> ".join(op.name for op in ordered_ops),
            "total_cost": total_cost,
            "detail": plan_detail,
        }
        all_plans.append(plan)

        if total_cost < best_cost:
            best_cost = total_cost
            best_ordering = plan

    return {
        "best": best_ordering,
        "all_plans": all_plans,
        "sample_stats": sample_stats,
        "total_sample_cost": total_sample_cost,
        "method": "sample_enhanced",
    }


def find_best_ordering_oracle(operators: list, relation: pd.DataFrame,
                              objective: str = "tokens") -> dict:
    """
    Oracle optimizer: uses ground truth selectivity.
    """
    tuples = relation.to_dict("records")
    n_total = len(relation)

    # Compute true selectivity from ground truth
    true_stats = {}
    for op in operators:
        if op.name == "lang_filter":
            passing = sum(1 for r in tuples
                          if str(r.get("language", "")).lower() == op.filter_value.lower())
        elif op.name == "genre_filter":
            passing = sum(1 for r in tuples
                          if str(r.get("genre", "")).lower() == op.filter_value.lower())
        else:
            passing = n_total
        true_stats[op.name] = {"true_selectivity": passing / n_total if n_total > 0 else 1.0}

    best_ordering = None
    best_cost = float("inf")
    all_plans = []

    for perm in permutations(range(len(operators))):
        ordered_ops = [operators[i] for i in perm]
        total_cost = 0
        n_remaining = n_total
        plan_detail = []

        for op in ordered_ops:
            est = estimate_operator_cost(op, tuples, n_remaining)
            if objective == "tokens":
                cost = est["total_compute_cost"]
            else:
                cost = est["total_monetary_cost"]
            total_cost += cost

            sel = true_stats[op.name]["true_selectivity"]
            plan_detail.append({
                "operator": op.name,
                "n_remaining": n_remaining,
                "cost": cost,
                "true_selectivity": sel,
            })
            n_remaining = int(n_remaining * sel)

        plan = {
            "ordering": " -> ".join(op.name for op in ordered_ops),
            "total_cost": total_cost,
            "detail": plan_detail,
        }
        all_plans.append(plan)

        if total_cost < best_cost:
            best_cost = total_cost
            best_ordering = plan

    return {
        "best": best_ordering,
        "all_plans": all_plans,
        "true_stats": true_stats,
        "method": "oracle",
    }
