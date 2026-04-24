"""
Routing policies for the Eddy adaptive query router.

Each policy decides, for a given tuple, which operator ordering to use.
Policies differ in how they balance exploration (trying both orderings to
gather statistics) vs. exploitation (using the currently-best ordering).

Four policies implemented:
  1. Thompson Sampling  - Bayesian; samples selectivity from Beta posteriors
  2. UCB               - Optimistic; explores arms with high uncertainty
  3. Epsilon-Greedy    - Simple; random exploration with decaying probability
  4. Lottery           - Original Eddies mechanism; weighted random tickets
"""

import numpy as np
from itertools import permutations


class RoutingPolicy:
    """Base class for eddy routing policies."""

    def __init__(self, name: str):
        self.name = name

    def choose(self, operators: list, op_stats: dict,
               local_costs: dict, tuple_idx: int) -> list:
        """
        Choose an operator ordering for the current tuple.

        Args:
            operators: list of SemanticOperator instances
            op_stats: per-operator running statistics dict
            local_costs: per-operator local cost estimates (from tokenizer)
            tuple_idx: index of current tuple (0-based)

        Returns:
            list of operators in the chosen execution order
        """
        raise NotImplementedError


def _expected_cost(ordering, op_stats, local_costs):
    """Compute expected cost of an ordering using current selectivity estimates."""
    cost = 0.0
    n_remaining = 1.0  # fraction of tuples still alive
    for op in ordering:
        per_tuple = local_costs[op.name]["mean_compute_cost"]
        cost += per_tuple * n_remaining
        # estimate selectivity from running stats
        s = op_stats[op.name]
        total = s["passes"] + s["fails"]
        sel = s["passes"] / total if total > 0 else 0.5
        n_remaining *= sel
    return cost


# --------------------------------------------------------------------------- #
# 1. Thompson Sampling
# --------------------------------------------------------------------------- #

class ThompsonSamplingPolicy(RoutingPolicy):
    """
    Bayesian approach: maintain Beta(alpha, beta) posterior over each
    operator's selectivity.  For every tuple, *sample* from each
    posterior, compute expected plan cost under those samples, and
    pick the cheapest ordering.

    Naturally explores less as the posteriors tighten.
    """

    def __init__(self):
        super().__init__("thompson_sampling")

    def choose(self, operators, op_stats, local_costs, tuple_idx):
        # Sample selectivity for each operator from Beta posterior
        sampled_sel = {}
        for op in operators:
            s = op_stats[op.name]
            alpha = s["passes"] + 1   # +1 uninformative prior
            beta = s["fails"] + 1
            sampled_sel[op.name] = np.random.beta(alpha, beta)

        best_ordering = None
        best_cost = float("inf")

        for perm in permutations(operators):
            cost = 0.0
            n_remaining = 1.0
            for op in perm:
                per_tuple = local_costs[op.name]["mean_compute_cost"]
                cost += per_tuple * n_remaining
                n_remaining *= sampled_sel[op.name]
            if cost < best_cost:
                best_cost = cost
                best_ordering = list(perm)

        return best_ordering


# --------------------------------------------------------------------------- #
# 2. Upper Confidence Bound (UCB)
# --------------------------------------------------------------------------- #

class UCBPolicy(RoutingPolicy):
    """
    Treat each ordering as a bandit arm.  Track mean observed cost
    per ordering.  Select the arm whose lower confidence bound is
    smallest (we want to *minimise* cost, so we subtract the
    exploration bonus).

    c controls exploration aggressiveness.
    """

    def __init__(self, c: float = 1.0):
        super().__init__("ucb")
        self.c = c
        # {ordering_key: {"total_cost": float, "count": int}}
        self.ordering_stats = {}

    def _ordering_key(self, ordering):
        return tuple(op.name for op in ordering)

    def record(self, ordering, cost):
        """Called by the router after executing a tuple to record observed cost."""
        key = self._ordering_key(ordering)
        if key not in self.ordering_stats:
            self.ordering_stats[key] = {"total_cost": 0.0, "count": 0}
        self.ordering_stats[key]["total_cost"] += cost
        self.ordering_stats[key]["count"] += 1

    def choose(self, operators, op_stats, local_costs, tuple_idx):
        all_orderings = list(permutations(operators))

        # Ensure every ordering is tried at least once
        for perm in all_orderings:
            key = self._ordering_key(perm)
            if key not in self.ordering_stats or self.ordering_stats[key]["count"] == 0:
                self.ordering_stats.setdefault(key, {"total_cost": 0.0, "count": 0})
                return list(perm)

        total_count = sum(s["count"] for s in self.ordering_stats.values())

        best_ordering = None
        best_score = float("inf")

        for perm in all_orderings:
            key = self._ordering_key(perm)
            s = self.ordering_stats[key]
            mean_cost = s["total_cost"] / s["count"]
            exploration = self.c * np.sqrt(np.log(total_count) / s["count"])
            score = mean_cost - exploration  # lower is better
            if score < best_score:
                best_score = score
                best_ordering = list(perm)

        return best_ordering


# --------------------------------------------------------------------------- #
# 3. Epsilon-Greedy
# --------------------------------------------------------------------------- #

class EpsilonGreedyPolicy(RoutingPolicy):
    """
    With probability epsilon, choose a random ordering (explore).
    Otherwise, choose the ordering with lowest expected cost using
    current selectivity estimates (exploit).

    Epsilon decays as: epsilon = epsilon_0 / (1 + tuple_idx / decay_rate)
    """

    def __init__(self, epsilon_0: float = 0.3, decay_rate: float = 50.0):
        super().__init__("epsilon_greedy")
        self.epsilon_0 = epsilon_0
        self.decay_rate = decay_rate

    def choose(self, operators, op_stats, local_costs, tuple_idx):
        epsilon = self.epsilon_0 / (1.0 + tuple_idx / self.decay_rate)

        if np.random.random() < epsilon:
            # Explore: random ordering
            perm = list(operators)
            np.random.shuffle(perm)
            return perm
        else:
            # Exploit: best ordering by current estimates
            best_ordering = None
            best_cost = float("inf")
            for perm in permutations(operators):
                cost = _expected_cost(perm, op_stats, local_costs)
                if cost < best_cost:
                    best_cost = cost
                    best_ordering = list(perm)
            return best_ordering


# --------------------------------------------------------------------------- #
# 4. Lottery Scheduling (original Eddies mechanism)
# --------------------------------------------------------------------------- #

class LotteryPolicy(RoutingPolicy):
    """
    Each ordering holds 'tickets' proportional to how good it is.
    A weighted random draw determines the ordering for each tuple.
    Tickets are re-computed every `update_interval` tuples based on
    running selectivity/cost estimates.
    """

    def __init__(self, base_tickets: int = 100, update_interval: int = 10):
        super().__init__("lottery")
        self.base_tickets = base_tickets
        self.update_interval = update_interval
        self.tickets = {}       # ordering_key -> ticket count
        self._orderings = None  # cached list of permutations

    def choose(self, operators, op_stats, local_costs, tuple_idx):
        # Lazily build ordering list and initial tickets
        if self._orderings is None:
            self._orderings = list(permutations(operators))
            for perm in self._orderings:
                key = tuple(op.name for op in perm)
                self.tickets[key] = self.base_tickets

        # Periodically re-weight tickets based on current cost estimates
        if tuple_idx > 0 and tuple_idx % self.update_interval == 0:
            costs = {}
            for perm in self._orderings:
                key = tuple(op.name for op in perm)
                costs[key] = _expected_cost(perm, op_stats, local_costs)

            max_cost = max(costs.values()) if costs else 1.0
            for key in self.tickets:
                # Invert: lower cost -> more tickets
                inv = max_cost - costs.get(key, 0) + 1.0
                self.tickets[key] = max(1, int(self.base_tickets * inv / max_cost))

        # Weighted random draw
        keys = list(self.tickets.keys())
        weights = np.array([self.tickets[k] for k in keys], dtype=float)
        probs = weights / weights.sum()
        chosen_idx = np.random.choice(len(keys), p=probs)
        chosen_key = keys[chosen_idx]

        # Map key back to operator objects
        name_to_op = {op.name: op for op in operators}
        return [name_to_op[n] for n in chosen_key]
