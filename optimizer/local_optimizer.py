"""
Phase 4 - Step 4.2: Local optimizer.
Uses only token-based cost estimation (no LLM calls) to pick best operator ordering.
"""

from itertools import permutations
from cost_model.local_estimator import estimate_operator_cost


def find_best_ordering_local(operators: list, tuples: list[dict],
                             n_total: int = None,
                             objective: str = "tokens") -> dict:
    """
    Enumerate all orderings and pick the one with minimum estimated cost.
    For 2 operators there are only 2 orderings.

    objective: "tokens" or "monetary"
    """
    if n_total is None:
        n_total = len(tuples)

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
            plan_detail.append({
                "operator": op.name,
                "n_remaining": n_remaining,
                "cost": cost,
                "mean_input_tokens": est["mean_input_tokens"],
            })
            # Local optimizer doesn't know selectivity, assumes all tuples pass
            # But the key insight: it orders by per-tuple cost (shorter input first)

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
        "method": "local_only",
    }
