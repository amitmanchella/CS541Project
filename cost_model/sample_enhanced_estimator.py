"""
Phase 5 - Step 5.2: Sample-enhanced cost model.
Extends local cost model with selectivity/accuracy from small-sample estimation.
Key formula: cost(A->B) = cost(A)*N + cost(B)*N*selectivity(A)
"""

from cost_model.local_estimator import estimate_operator_cost


def estimate_plan_cost_local(operators: list, tuples: list[dict],
                             n_total: int = None) -> dict:
    """
    Local-only plan cost: assumes no selectivity info.
    Just sums up cost(op) * N_remaining where N_remaining = N for all ops
    (since we don't know selectivity without sampling).
    """
    if n_total is None:
        n_total = len(tuples)

    total_tokens = 0
    total_monetary = 0
    n_remaining = n_total

    per_op = []
    for op in operators:
        est = estimate_operator_cost(op, tuples, n_remaining)
        total_tokens += est["total_compute_cost"]
        total_monetary += est["total_monetary_cost"]
        per_op.append({**est, "n_remaining": n_remaining})
        # Local model doesn't know selectivity, so assume all pass
        # (conservative: no filtering benefit assumed)

    return {
        "ordering": " -> ".join(op.name for op in operators),
        "total_estimated_tokens": total_tokens,
        "total_estimated_monetary": total_monetary,
        "per_operator": per_op,
    }


def estimate_plan_cost_with_samples(operators: list, tuples: list[dict],
                                    sample_stats: dict,
                                    n_total: int = None,
                                    accuracy_threshold: float = 0.5) -> dict:
    """
    Sample-enhanced plan cost.
    Uses selectivity and accuracy from small-sample estimator.

    cost(A->B) = cost(A)*N + cost(B)*N*effective_selectivity(A)
    where effective_selectivity accounts for accuracy.
    """
    if n_total is None:
        n_total = len(tuples)

    total_tokens = 0
    total_monetary = 0
    n_remaining = n_total

    per_op = []
    for op in operators:
        est = estimate_operator_cost(op, tuples, n_remaining)
        total_tokens += est["total_compute_cost"]
        total_monetary += est["total_monetary_cost"]

        # Get sample-based selectivity for this operator
        op_stats = sample_stats.get(op.name, {})
        selectivity = op_stats.get("estimated_selectivity", 1.0)
        accuracy = op_stats.get("estimated_accuracy", 1.0)

        # Adjust selectivity based on accuracy:
        # Low-accuracy operator may incorrectly filter, making apparent
        # selectivity unreliable. Penalize by pushing selectivity toward 1.0
        if accuracy < accuracy_threshold:
            penalty = 1.0 - accuracy
            effective_selectivity = selectivity + penalty * (1.0 - selectivity)
        else:
            effective_selectivity = selectivity

        per_op.append({
            **est,
            "n_remaining": n_remaining,
            "selectivity": selectivity,
            "accuracy": accuracy,
            "effective_selectivity": effective_selectivity,
        })

        # Update remaining tuples for next operator
        n_remaining = int(n_remaining * effective_selectivity)

    return {
        "ordering": " -> ".join(op.name for op in operators),
        "total_estimated_tokens": total_tokens,
        "total_estimated_monetary": total_monetary,
        "per_operator": per_op,
    }


def estimate_plan_cost_oracle(operators: list, tuples: list[dict],
                              ground_truth: dict,
                              n_total: int = None) -> dict:
    """
    Oracle plan cost: uses true selectivity from ground truth labels.
    """
    if n_total is None:
        n_total = len(tuples)

    total_tokens = 0
    total_monetary = 0
    n_remaining = n_total

    per_op = []
    for op in operators:
        est = estimate_operator_cost(op, tuples, n_remaining)
        total_tokens += est["total_compute_cost"]
        total_monetary += est["total_monetary_cost"]

        true_selectivity = ground_truth.get(op.name, {}).get("true_selectivity", 1.0)

        per_op.append({
            **est,
            "n_remaining": n_remaining,
            "true_selectivity": true_selectivity,
        })

        n_remaining = int(n_remaining * true_selectivity)

    return {
        "ordering": " -> ".join(op.name for op in operators),
        "total_estimated_tokens": total_tokens,
        "total_estimated_monetary": total_monetary,
        "per_operator": per_op,
    }
