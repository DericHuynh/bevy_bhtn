# bevy_bhtn — API Redesign Proposal (Revised)

> This revision folds the design review into the original proposal: factual
> corrections about the current planner, the ranking × backtracking design,
> scoped-down partial order, `CostBounded` instead of full A*, ECS-idiomatic
> per-agent overrides, and a planner/ECS trace seam. Phasing is explicit.

## Research Summary

Orientation on the HTN landscape that motivates each decision:

| System | Search style | Method selection | Subtask ordering |
|---|---|---|---|
| SHOP / SHOP2 | Forward, left-to-right DFS | Declaration order; sort-by heuristic | Total-order (SHOP) / Partial-order (SHOP2) |
| PANDA / pandaPI | Forward progression + POCL | Heuristic (TDG-based) or user-ranked | Partial-order constraints |
| Fluid HTN (C#) | Forward DFS | First-valid, extensible to UtilitySelect / RandomSelect | Total-order |
| Unified-Planning | Pluggable engines | Per-method subtask ordering API | Partial-order + STN constraints |
| HyperTensioN | Forward, multi-stage compiled | Ordered methods, compiler optimisations | Total-order |

**Monte Carlo / Ant Colony.** MCTS treats decomposition choice points as
UCT-selected tree nodes with rollouts (Wichlacz et al., SoCS 2020; Pyhop-m,
KBS 2021) — anytime, budget-friendly, stochastic. ACO maintains
pheromone/heuristic tables over compound→method edges with evaporation and
cost-proportional deposit (Elkawkagy et al.) — population-based, near-optimal,
heavier per tick. Both need **persistent cross-tick statistics** and **seeded
randomness** to be usable in game AI (see Axis 4/6).

Key gaps in the current API:

- Only one decomposition model: total-order, first-valid-branch.
- Branch selection is implicit (declaration order); no hook to override it.
- No subtask ordering constraints within a branch.
- No cost / utility signals; no cost-aware search.
- No interruptible / partial-plan semantics at domain-author level.
- No per-agent search granularity.

## Design Goals

1. **Additive, not breaking.** The existing `charge_domain` compiles unchanged;
   new features are opt-in and inert under the default policy.
2. **Layered expressiveness.** Common cases stay two-liners.
3. **ECS-native.** Closures receive component references exactly as before;
   per-agent variation uses *components*, not world-querying predicates.
4. **Composable search.** Method selection and subtask ordering are separate
   axes; either can be configured independently.
5. **Planner stays ECS-free.** The core never touches `Events`, `World`, or
   Bevy types beyond `bevy_ecs::Component` bounds; the driver bridges.

---

## Axis 1 — Branch / Method Selection Policies

Separates *which branch to try* from *how subtasks inside that branch are
ordered* (Axis 2).

```rust
/// How a compound task's valid branches are ranked before the planner
/// descends. Applied at the compound-task level, after precondition
/// evaluation: only *valid* branches are ranked. The default is FirstMatch.
#[derive(Clone)]
pub enum SelectionPolicy {
    /// Branches are tried in declaration order (current behaviour).
    FirstMatch,

    /// All valid branches are scored by their `utility` closure; the highest
    /// wins. Ties break by declaration order.
    HighestUtility,

    /// Valid branches are sampled proportional to their utility scores.
    /// Useful for stochastic / exploratory NPC behaviour. The sampled order
    /// is snapshotted per decomposition (see "Ranking × backtracking").
    WeightedRandom { seed: u64 },

    /// The caller supplies the comparator at domain-build time.
    Custom(Arc<dyn BranchRanker>),
}

/// Ranks *valid* branch candidates. Implementations must be deterministic
/// for a given `(candidates, state)` pair unless the policy snapshots its
/// order (WeightedRandom does).
pub trait BranchRanker: Send + Sync {
    /// Appends the candidate indices to `out` in preferred order.
    /// Scratch-buffer signature: no per-node allocation.
    fn rank(&self, candidates: &[BranchCandidate], state: &PlanState, out: &mut Vec<u32>);
}

/// One valid branch offered to a ranker.
pub struct BranchCandidate<'a> {
    /// The branch's declaration index (its MTR identity).
    pub index: u32,
    /// The branch's declared name, if any (Axis 5).
    pub name: Option<&'static str>,
    /// The branch's declared utility, if any (Axis 3).
    pub utility: Option<f32>,
    /// The branch's subtask list (for structural heuristics).
    pub subtasks: &'a [u32],
}
```

Usage:

```rust
fn engage(task: &mut TaskBuilder) {
    // Override the default FirstMatch for this task only.
    task.select(SelectionPolicy::HighestUtility);

    task.branch()
        .named("snipe")
        .precondition(|a: &Ammo| a.sniper > 0)
        .utility_fn(|d: &Distance| (100.0 - d.0 as f32).max(0.0))
        .then(snipe);

    task.branch()
        .named("assault")
        .precondition(|a: &Ammo| a.rifle > 0)
        .utility(40.0)
        .then(assault);

    task.branch()
        .named("melee_fallback") // no precondition = always valid
        .utility(10.0)
        .then(melee);
}
```

### Ranking × backtracking (the load-bearing design point)

The current backtrack restores `skip_next = idx + 1` and re-runs
`find_method` — declaration order makes "try the next one" trivial. With a
ranking policy, the planner must try methods in **ranked order**, which
requires:

1. **Deterministic re-ranking is sound** for `HighestUtility`/`Custom`:
   backtracking restores the exact scratchpad state at the commitment point,
   so re-ranking reproduces the same order. `find_method` becomes
   `find_ranked_method(&state, skip, policy)`: evaluate all preconditions,
   rank the valid ones, return the `skip`-th in ranked order.
2. **`WeightedRandom` must snapshot its sampled order** in the decomposition
   frame (`ranked_order: SmallVec<[u32; 4]>`, alongside the existing queue
   suffix) — re-sampling on backtrack would break completeness (the failed
   branch could be re-sampled; others never tried). The frame field is only
   populated for non-`FirstMatch` policies, so the default path stays
   allocation-free.
3. **`skip` semantics change** from "declaration index + 1" to "position in
   the ranked order + 1". The MTR keeps recording *declaration* indices (its
   identity is stable for debugging and `is_preferred_over`), but the frame's
   resume cursor is a rank position.

### Ranking × look-ahead sweep ordering

The sweep prunes methods before commitment. With ranking, the per-commitment
sequence becomes:

1. Evaluate every method's preconditions (cheap, compiled closures).
2. Rank the **valid** methods.
3. Sweep in ranked order — pay the sweep's cost first for the branch ranking
   would actually take, and stop at the first survivor.
4. A sweep **pin** (unique surviving method) overrides ranking entirely: if
   only one method can possibly apply, there is nothing to rank.

This ordering means the sweep's cost is spent on likely branches first, and
the existing pin machinery is untouched.

---

## Axis 2 — Subtask Ordering within a Branch

**Scope correction:** full Partial-Order Causal-Link planning (PANDA-style) is
out of scope. What ships is **search-time scheduling inside the existing
DFS**: at an unordered or constrained set, the planner branches over *ready*
subtasks (preconditions hold + all `before` predecessors scheduled), and
backtracks over alternatives. The chosen linearization is a flat step
program, so **the compiled `Plan`, the executor, and the ECS driver are
completely unchanged** — only decomposition gains branching.

```rust
impl MethodBuilder<'_> {
    // ── Total-order helpers (existing, unchanged) ──────────────────────────

    /// Append a subtask at the end of the current total order.
    pub fn then<F: TaskFn>(&mut self, task: F) -> &mut Self;

    // ── Unordered / constrained subtasks (new) ─────────────────────────────

    /// Add a subtask with no ordering commitment relative to other unordered
    /// subtasks. Returns a handle for `before` constraints.
    pub fn subtask<F: TaskFn>(&mut self, task: F) -> SubtaskHandle;

    /// Require that `before` completes before `after` starts.
    /// Called multiple times to build a DAG over this branch's subtasks.
    pub fn before(&mut self, before: SubtaskHandle, after: SubtaskHandle) -> &mut Self;

    /// All subtasks in the set may execute in any order. Sugar for
    /// `subtask` + no `before` constraints.
    pub fn any_order<F: TaskFn>(&mut self, tasks: impl IntoIterator<Item = F>) -> &mut Self;
}
```

Example — partial-order "fetch ingredients" branch:

```rust
fn prepare_meal(task: &mut TaskBuilder) {
    task.branch().named("full_prep").constrain(|b| {
        // fetch_protein, fetch_carbs, fetch_veg can happen in any order.
        let protein = b.subtask(fetch_protein);
        let carbs = b.subtask(fetch_carbs);
        let veg = b.subtask(fetch_veg);
        // cook must come after all three fetches.
        let cook = b.subtask(cook_meal);
        b.before(protein, cook)
            .before(carbs, cook)
            .before(veg, cook);
        // `then` still appends to the tail: plate runs after everything.
        b.then(plate_dish);
    });
}
```

Scheduling semantics at decomposition time: repeatedly pick any *ready*
subtask (preconditions hold against the current scratchpad; `before`
predecessors already scheduled). The DFS branches over ready choices, so
genuine ordering conflicts are resolved by backtracking, not by a fixed
permutation.

**Conservative extensions** that keep the existing analyses sound:

- **Summaries**: `possible_writes`/`guaranteed_writes` of a set are
  order-independent unions (each member executes exactly once).
  `required_fields` uses a conservative under-approximation (a component is
  required only if read by a member that no set member can write).
- **Look-ahead sweep**: set members are checked optimistically (preconditions
  against the current state *or* the optimistic overlay = "maybe"); pruning
  weakens but stays sound.
- **`min_yield`**: the sum over members (order-independent).
- **MTR**: records the method index only; the chosen permutation is *not*
  recorded (plan-repair over permutations is out of scope).

---

## Axis 3 — Cost & Utility Signals

Cost and utility are orthogonal:

- **utility** — branch-selection score (higher = preferred); lives on
  branches (Axis 1).
- **cost** — per-primitive weight fed to cost-aware search (Axis 4).

```rust
impl TaskBuilder {
    /// Constant action cost (fed to CostBounded search / cost heuristics).
    /// Inert under the default strategy.
    pub fn cost(&mut self, c: f32) -> &mut Self;

    /// Dynamic cost sampled from the scratchpad at plan time.
    pub fn cost_fn<F, Args>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&PlanState) -> f32 + Send + Sync + 'static;
}

impl MethodBuilder<'_> {
    /// Static utility score for HighestUtility / WeightedRandom selection.
    pub fn utility(&mut self, u: f32) -> &mut Self;

    /// Dynamic utility scored from components at branch-evaluation time.
    /// Compiled with the same slot-offset machinery as preconditions.
    pub fn utility_fn<F, Args>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&PlanState) -> f32 + Send + Sync + 'static;
}
```

> **Honesty note.** A branch's *true* cost is the sum over its full
> refinement — unknowable statically through recursion. Static/dynamic cost
> annotations are **estimates** used for ordering and bounding; only
> `CostBounded` (Axis 4) reasons about the *accumulated* cost of the plan it
> actually builds.

---

## Axis 4 — Planner / Search Strategy

**Correction:** the current planner already backtracks completely on
downstream failure — there is no greedy-no-backtrack mode to contrast with.
`DepthFirstGreedy`/`DepthFirstComplete` are therefore collapsed into one
`DepthFirst` (the status quo), with an optional new fail-fast mode, and full
A\* is replaced by **cost-bounded branch-and-bound DFS**, which reuses the
entire append-only/rollback machinery.

```rust
/// The search algorithm used to expand the task network.
pub enum HtnSearchStrategy {
    /// Left-to-right DFS with full MTR backtracking and look-ahead pruning.
    /// The default; what the planner does today.
    DepthFirst,

    /// DFS that abandons a branch on first downstream failure and returns the
    /// partial plan immediately (no backtracking). Cheaper per tick, less
    /// complete — for tight per-frame budgets.
    DepthFirstFailFast,

    /// Depth-first with branch-and-bound on accumulated primitive cost:
    /// keeps the cheapest complete plan found within the sanity budget and
    /// prunes branches whose `g + Σ min_cost(remaining)` exceeds it.
    /// Requires `cost`/`cost_fn` annotations (unannotated primitives count 0).
    /// Anytime and deterministic; finds the cost-optimal plan when the budget
    /// suffices to exhaust the space.
    CostBounded,

    /// Caller-supplied strategy. The strategy object owns any persistent
    /// state (MCTS statistics, ACO pheromone tables) — wrap it in `Arc` and
    /// share it via `HtnConfig` or a per-agent component so a population of
    /// agents shares one table.
    Custom(Arc<dyn Searcher>),
}

pub trait Searcher: Send + Sync {
    /// Search the domain from `state`. Returns the best plan found (None if
    /// nothing decomposes). Implementations own their own statistics and
    /// randomness; they must be internally synchronized (`&self`).
    fn search(&self, domain: &HtnDomain, state: &PlanState) -> Option<Plan>;
}
```

Deliberately **deferred** (documented so the door stays open):

- `AStarHeuristic` — best-first over decomposition frontiers needs a
  `PlanState` per open node and a different search machine; `CostBounded`
  delivers most of the value inside the current one.
- First-class MCTS / ACO variants — admitted via `Custom(Searcher)`. A
  `MethodStats` table type (compound→method → Q/N or pheromone, with
  evaporation helpers) may be provided as a building block; the strategies
  themselves stay downstream until there is demand. Both require **seeded,
  per-agent randomness** so replans are stable (agents must not flip-flop
  plans every tick).

### Per-agent configuration

**Correction:** the original `with_agent_strategy(predicate, strategy)` runs
world-querying predicates per agent per tick and fights the driver's borrow
structure. The Bevy-idiomatic mechanism is a **component**:

```rust
/// Per-agent search override. Agents opt in by inserting this component;
/// the driver reads it and overrides `HtnConfig`'s strategy for that entity.
#[derive(Component, Clone)]
pub struct SearchOverride {
    pub strategy: HtnSearchStrategy,
    /// Overrides the global sanity budget for this agent.
    pub sanity_limit: Option<usize>,
}
```

`HtnConfig` gains builder methods mirroring its existing fields:

```rust
impl HtnConfig {
    pub fn with_strategy(mut self, s: HtnSearchStrategy) -> Self;
    pub fn with_sanity_limit(mut self, d: usize) -> Self;
    pub fn with_lookahead(mut self, enabled: bool) -> Self;
}
```

(`lookahead` stays orthogonal to the strategy — it prunes inside any
depth-first variant and is not a strategy itself.)

---

## Axis 5 — Named Branches & Debugging

Branches are currently anonymous (the fn-graph rewrite dropped the old DSL's
method names). Names return as optional, interned metadata:

```rust
// Domain authoring:
task.branch().named("solar_gather").precondition(|b: &Battery| b.0 < 3).then(gather);
```

**Trace seam correction:** the planner core is ECS-free, so it never touches
`Events`. The planner writes into a caller-owned buffer; the *driver* bridges
to Bevy's event system:

```rust
// Planner side (headless):
pub struct DecompositionTrace {
    /// Declaration index of the compound task.
    pub compound: u32,
    /// Declaration index of the branch that was selected.
    pub branch: u32,
    /// Branch name, when the branch was named.
    pub branch_name: Option<&'static str>,
    pub outcome: TraceOutcome,
}

pub enum TraceOutcome {
    Selected,
    PrecondFailed,
    Backtracked,
}

impl HtnPlanner<'_> {
    /// Install a trace buffer. Tracing is per *commitment* (one event per
    /// selected/failed branch), not per precondition attempt — per-attempt
    /// tracing would flood on combinatorial domains.
    pub fn set_trace(&mut self, buf: &mut Vec<DecompositionTrace>);
}

// Driver side (ECS): when `HtnConfig.debug_trace` is set, the planner's
// buffer is drained into `Events<DecompositionTrace>` after each plan.
```

---

## Axis 6 — Goals

**Correction:** `GoalBuilder::condition` is dropped. The backward planner
reasons over effect *write slots*; an opaque terminal predicate cannot be
reasoned backward — it would require a forward goal-directed search (a
different algorithm). It can return as post-plan validation later.

```rust
impl GoalBuilder<'_> {
    /// The desired post-state expressed as effects (existing API, unchanged).
    pub fn effect<E, Args>(&mut self, e: E) -> &mut Self;

    /// Priority among multiple registered goals; higher = evaluated first
    /// by multi-goal callers.
    pub fn priority(self, p: i32) -> &mut Self;
}
```

---

## Updated `charge_domain` — compatibility demo

```rust
fn charge_domain() -> HtnDomain {
    fn charge(task: &mut TaskBuilder) {
        // FirstMatch is the implicit default; `select` is optional but makes
        // intent explicit in review.
        task.select(SelectionPolicy::FirstMatch);

        task.branch()
            .named("battery_full")
            .precondition(|battery: &Battery| battery.0 >= 3);

        task.branch()
            .named("charge_cycle")
            .precondition(|battery: &Battery| battery.0 < 3)
            .then(gather)
            .then(charge);
    }

    fn gather(task: &mut TaskBuilder) {
        task.precondition(|battery: &Battery| battery.0 < 3)
            .cost(1.0) // silently ignored by DepthFirst
            .effect(|battery: &mut Battery| battery.0 += 1)
            .action(|cmds: &mut EntityCommands| {
                cmds.insert(Gathered);
            });
    }

    HtnDomain::from_root(charge).build().expect("well-formed")
}
```

No change from the caller's side; existing tests stay green.

## Updated `adaptive_recharge_domain` — utility + unordered set

```rust
fn adaptive_recharge_domain() -> HtnDomain {
    fn choose_method(task: &mut TaskBuilder) {
        task.select(SelectionPolicy::HighestUtility);

        task.branch()
            .named("dock_station")
            .precondition(|d: &DistanceToStation| d.0 < 10)
            .utility_fn(|d: &DistanceToStation| (10 - d.0).max(0) as f32 * 2.0)
            .then(dock_and_charge);

        task.branch()
            .named("solar_gather")
            .utility(1.0)
            .then(gather);
    }

    fn prepare_dock(task: &mut TaskBuilder) {
        task.branch()
            .named("parallel_prep")
            // Three independent sub-steps, any order; the planner schedules
            // whichever is ready, backtracking over alternatives.
            .any_order([lower_antenna, close_solar_panels, lock_wheels])
            .then(dock_and_charge);
    }

    fn dock_and_charge(task: &mut TaskBuilder) {
        task.cost(0.5).effect(|b: &mut Battery| b.0 = 3);
    }

    fn gather(task: &mut TaskBuilder) {
        task.cost(2.0).effect(|b: &mut Battery| b.0 += 1);
    }

    HtnDomain::from_root(choose_method).build().unwrap()
}
```

## Test updates

```rust
#[test]
fn utility_selects_dock_over_gather_when_close() {
    let mut world = World::new();
    // HighestUtility is a branch-ranking policy baked into the domain; the
    // search strategy governs how the tree is expanded, not ranked.
    world.insert_resource(
        HtnConfig::new(adaptive_recharge_domain())
            .with_strategy(HtnSearchStrategy::DepthFirst),
    );
    let entity = world
        .spawn((Battery(0), DistanceToStation(2), HtnAgent::default()))
        .id();

    htn_ai_system(&mut world);

    // dock_and_charge is a single primitive → plan length 1.
    let agent = world.get::<HtnAgent>(entity).unwrap();
    assert_eq!(agent.plan.as_ref().map(|p| p.len()), Some(1));
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);
}

#[test]
fn cost_bounded_search_prefers_the_cheap_path() {
    let mut world = World::new();
    world.insert_resource(
        HtnConfig::new(adaptive_recharge_domain())
            .with_strategy(HtnSearchStrategy::CostBounded),
    );
    let entity = world
        .spawn((Battery(0), DistanceToStation(15), HtnAgent::default()))
        .id();

    // Station is far (dock precondition fails), falls back to gather × 3.
    for _ in 0..3 {
        htn_ai_system(&mut world);
    }
    assert_eq!(world.get::<Battery>(entity).unwrap().0, 3);
}

#[test]
fn unordered_branch_satisfies_the_goal() {
    // any_order members may be scheduled in any ready order; every
    // linearization is a valid flat plan.
    let domain = adaptive_recharge_domain();
    let state = PlanState::build(&domain.components)
        .set(Battery(0))
        .set(DistanceToStation(2))
        .finish();
    let mut planner = HtnPlanner::new(&domain);
    let plan = planner.plan("prepare_dock", &state);
    assert_eq!(plan.len(), 4); // 3 prep + dock
}

#[test]
fn weighted_random_snapshots_its_order_across_backtracking() {
    // A domain where the sampled branch fails downstream: the planner must
    // exhaust the sampled order (not re-sample) before backtracking past the
    // choice point. Pinned by counting ranker invocations.
    //
    // (Body: Custom ranker that logs calls; assert the number of `rank`
    // calls at the choice point equals the number of branches, not more.)
    # todo!()
}

#[test]
fn decomposition_trace_reports_branch_selection() {
    let domain = charge_domain();
    let state = PlanState::build(&domain.components).set(Battery(0)).finish();
    let mut planner = HtnPlanner::new(&domain);
    let mut trace = Vec::new();
    planner.set_trace(&mut trace);

    planner.plan("charge", &state);

    assert!(trace.iter().any(|t| t.branch_name == Some("charge_cycle")
        && matches!(t.outcome, TraceOutcome::Selected)));
}
```

## Summary of changes

| Area | Change | Motivation |
|---|---|---|
| `MethodBuilder::named` | Attach a debug name to a branch | Tracing, logging, tooling |
| `MethodBuilder::utility / utility_fn` | Score a branch for HighestUtility / WeightedRandom | Utility-driven AI (game AI standard) |
| `MethodBuilder::subtask + before + any_order` | Unordered/constrained subtask sets, scheduled by the DFS | SHOP2-style expressiveness **without** a POCL planner |
| `TaskBuilder::cost / cost_fn` | Per-primitive cost (estimates) | Cost-aware search |
| `TaskBuilder::select` + `SelectionPolicy` | FirstMatch, HighestUtility, WeightedRandom, Custom | Branch ranking as a first-class concept |
| `DecompositionFrame.ranked_order` | Snapshot ranked/sampled branch order per commitment | **Soundness**: ranking × backtracking |
| `HtnSearchStrategy` | DepthFirst, DepthFirstFailFast, CostBounded, Custom(Searcher) | Honest mapping onto the current planner; anytime cost search |
| `SearchOverride` component | Per-entity strategy/sanity override | Multi-agent heterogeneous planning, ECS-idiomatic |
| `Searcher` trait + `MethodStats` table | Escape hatch owning persistent stats/randomness | Admits MCTS/ACO downstream without first-class variants |
| `HtnPlanner::set_trace` + driver→`Events` bridge | Headless trace buffer, ECS-side forwarding | Debug/profiling without coupling the planner to Bevy |
| `GoalBuilder::priority` | Multi-goal ordering | Multi-goal domains |
| *(dropped)* `GoalBuilder::condition` | — | Opaque predicates cannot be reasoned backward |
| *(dropped)* `DepthFirstGreedy`/`DepthFirstComplete` split | — | The current planner already backtracks fully |
| *(deferred)* `AStarHeuristic`, POCL, first-class MCTS/ACO | — | Different search machines; `CostBounded` + `Custom` cover the near-term value |

## Implementation phasing

1. **Phase 1 (fully additive):** `named`, `utility/utility_fn` +
   `SelectionPolicy::{FirstMatch, HighestUtility, WeightedRandom}` with
   frame-snapshotted ranked orders, `cost/cost_fn` (inert), trace buffer +
   driver bridge, `SearchOverride` component, `Searcher` trait.
2. **Phase 2:** `CostBounded` branch-and-bound DFS + a `min_cost` summary
   (same fixpoint machinery as `min_yield`).
3. **Phase 3:** `subtask`/`before`/`any_order` DFS-time scheduling with the
   conservative summary/sweep extensions.
4. **Deferred:** `AStarHeuristic`, POCL, first-class MCTS/ACO,
   `GoalBuilder::condition`.

## Benchmarking note

Cross-run benchmark comparisons are unreliable on this machine (thermal
drift; `REPLAN_CYCLES` has also been varied between runs). Only within-run
A/B groups are trusted. When Axis 1 lands, add a `selection` A/B group to
`benches/lookahead.rs` (FirstMatch vs HighestUtility on a ranking-sensitive
domain) so policy overhead is measured the same way the look-ahead win is.
