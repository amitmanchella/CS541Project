
The Original Paper

The foundational paper is:

Ron Avnur and Joseph M. Hellerstein. "Eddies: Continuously Adaptive Query Processing." SIGMOD 2000.

  • DOI link: https://doi.org/10.1145/342009.335420
  • Free PDF (Berkeley): https://dsf.berkeley.edu/papers/sigmod00-eddy.pdf

It's one of the most cited papers in adaptive query processing (~1,800+ citations). Any DB conference
reviewer will instantly recognize it.

────────────────────────────────────────────────────────────────────────────────────────────────────────────

How Traditional Eddies Work

In a normal database, once the optimizer picks a query plan (e.g., "filter by age, then filter by city, then
join"), it's locked in for the entire execution. If the optimizer's statistics were wrong, tough luck --
you're stuck with a bad plan.

Eddies flips this on its head. Instead of committing to a fixed operator ordering, there's a router (the
"eddy") that sits between the data source and the operators. For each tuple, the eddy decides which operator
to send it to first. After the operator processes the tuple (pass or fail), the result comes back to the
eddy, which routes it to the next operator or discards it. The eddy tracks running statistics and adjusts
its routing decisions over time.

The original paper uses lottery scheduling: each operator gets "tickets" proportional to how desirable it is
to route tuples there first. The ticket counts are updated as the eddy observes real selectivities and
costs.

────────────────────────────────────────────────────────────────────────────────────────────────────────────

Why This Is a Perfect Fit for Semantic Operators

Here's where it gets interesting. The Eddies idea was designed for traditional operators (comparisons that
cost microseconds). Semantic operators are radically different -- each one is an expensive LLM call. This
creates new challenges and opportunities that the original Eddies paper never had to address:

1. Zero wasted LLM calls

Look at your current sample-enhanced approach:

sampler/small_sample_estimator.py:
    # Samples 20 tuples, runs them through the LLM
    # Collects selectivity statistics
    # These 20 LLM calls are ONLY used for statistics
    # They don't contribute to the final query result

That's 40 LLM calls (20 per operator) spent purely on planning. With the eddy approach, every single LLM 
call contributes to the actual query result. The first 20-40 tuples might be routed suboptimally as the eddy
explores, but their results still count. This is a concrete, measurable advantage.

2. Exploration is expensive, so the routing policy matters more

In traditional Eddies, routing one tuple to the "wrong" operator first costs microseconds -- who cares. In
your setting, routing a tuple to genre_filter first (when lang_filter first would have been cheaper) wastes
~130 tokens worth of LLM cost. So the exploration-exploitation trade-off is much sharper. This motivates
more sophisticated routing policies than simple lottery scheduling:

  • Thompson Sampling: Maintain a Beta distribution over each operator's selectivity. For each tuple, sample
  from each distribution, compute expected cost under those samples, and route optimistically. This
  naturally explores less as uncertainty decreases.
  • UCB (Upper Confidence Bound): Route to minimize an optimistic cost estimate. Explores less aggressively
  than lottery scheduling.
  • Epsilon-greedy with decay: With probability epsilon, explore (try the non-preferred ordering); otherwise
  exploit (use the currently-best ordering). Decay epsilon over time.

Studying which routing policy works best for semantic operators is itself a contribution -- nobody has
studied this.

3. Non-stationary data

Your current 25 configs assume uniform selectivity across all 1,000 rows. But real data can have clusters --
the first 500 rows might be mostly English movies, and the last 500 might be mostly non-English. A fixed-
sample approach samples upfront and commits, potentially getting it wrong for the second half. The eddy
adapts naturally because it continuously updates its estimates.

This is a very strong experimental angle: construct a "shifted" dataset where selectivity changes midway
through, and show that the eddy handles it while fixed-sample fails.

────────────────────────────────────────────────────────────────────────────────────────────────────────────

What the Architecture Would Look Like

                        ┌─────────────┐
                        │  Data Source │
                        │ (1000 rows)  │
                        └──────┬──────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    EDDY ROUTER       │
                    │                      │
                    │  Running statistics: │
                    │  - lang_filter sel   │
                    │  - genre_filter sel  │
                    │  - per-tuple costs   │
                    │                      │
                    │  Routing policy:     │
                    │  Thompson / UCB /    │
                    │  Lottery             │
                    └───┬─────────┬───────┘
                        │         │
              ┌─────────▼──┐  ┌──▼──────────┐
              │ lang_filter │  │ genre_filter │
              │ (title→LLM) │  │ (plot→LLM)  │
              └─────────┬──┘  └──┬──────────┘
                        │         │
                        ▼         ▼
                  Results flow back to eddy
                  → route to next operator
                  → or discard (failed filter)
                  → or emit (passed all filters)

For each incoming tuple, the eddy:

  • Consults its routing policy to decide: lang_filter first or genre_filter first?
  • Sends the tuple to the chosen operator (LLM call).
  • If the tuple fails, discard it. If it passes, route to the other operator.
  • Updates running statistics (selectivity, tokens used, latency) for the operator that just ran.
  • Adjusts routing policy based on updated statistics.

────────────────────────────────────────────────────────────────────────────────────────────────────────────

What the Experiments Would Show

You'd compare four methods across your 25 configs:

Method                             Planning Cost        Adapts?  Accuracy
Local-only                         0 LLM calls          No       ~76%
Fixed-sample (current)             40 LLM calls wasted  No       ~96%
Eddy (new)                         0 wasted LLM calls   Yes      ~96-100%
Oracle                             Requires labels      No       ~92%

Key metrics to report:

  • Convergence speed: After how many tuples does the eddy "lock in" the correct ordering? (Probably 15-30,
  based on your ablation showing 20 samples suffice.)
  • Total cost savings: Eddy vs. fixed-sample, accounting for the 40 wasted sample calls.
  • Non-stationary robustness: Performance on shifted/clustered data where fixed-sample breaks.
  • Routing policy comparison: Thompson vs. UCB vs. lottery vs. epsilon-greedy -- which converges fastest
  with least waste?

────────────────────────────────────────────────────────────────────────────────────────────────────────────

Why This Is Conference-Worthy

The contribution is clean and well-defined:

  • Novel formulation: Eddies have never been applied to LLM-backed semantic operators. The cost structure
  (expensive, variable-latency, accuracy-dependent) is fundamentally different from traditional operators.
  • Practical impact: Eliminates the "wasted sample" overhead that every sample-based system (AERO,
  Palimpzest) pays.
  • Rich experimental story: 25 selectivity configs + non-stationary data + routing policy comparison +
  ablation on convergence speed.
  • Strong theoretical grounding: Connects to a 25-year lineage of adaptive query processing research that
  DB reviewers respect.

The paper's narrative becomes: "SemOrder showed that local cost models work but fail when selectivity 
matters. Sample-based approaches fix this but waste LLM calls on planning. We unify execution and 
optimization via an Eddies-style adaptive router, achieving the accuracy of sample-based methods with zero 
planning overhead."