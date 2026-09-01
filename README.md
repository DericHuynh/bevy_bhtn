# bevy_bhtn

A headless **Hierarchical Task Network** planner for Bevy AI, with an idiomatic
`bevy_ecs` execution layer. Tasks are **plain Rust functions** whose zero-sized
fn-item types are the graph identity — no marker structs, no string ids, no
reflection. At startup the root function is recorded and **baked** into a flat
indexed graph; the runtime planner searches the arrays directly (O(1) node
lookups, allocation-light backtracking).

State is **ordinary Bevy components**: preconditions and effects are
monomorphized closures over component references, simulated on a dense
[`PlanState`] scratchpad. The planner runs both **forward** (MTR backtracking
with look-ahead pruning) and **backward / goal-state** planning.

```toml
[dependencies]
bevy_bhtn = "0.2"
bevy_ecs = "0.18"
```

## Why

Classic HTN planners make you describe your domain in strings, tables, or a
bes DSL. Here the domain *is* your code:

- **A task is a function.** `fn shoot(task: &mut TaskBuilder)` — its
  `TypeId` is the graph node, `type_name` is its debug name. `.then(shoot)`
  records an edge; recursion and repeated references become plain graph edges.
- **State is components.** Preconditions read `&T`, effects mutate `&mut T` —
  annotated closure parameters captured at build time into type-erased
  checkers/mutators over a byte-pool scratchpad. No `bevy_reflect`, no
  per-component boxes, no `dyn Any` downcasts.
- **The planner never touches the `World`.** The whole search runs on an
  isolated snapshot; the ECS driver commits only what the effects declare as
  writes back to the real entity.

## Quick tour

### Tasks and planning

```rust
use bevy_bhtn::prelude::*;

#[derive(Component, Clone, Default, Debug)]
struct Ammo(pub u32);

fn reload(task: &mut TaskBuilder) {
    task.effect(|ammo: &mut Ammo| ammo.0 = 30);
}

fn engage(task: &mut TaskBuilder) {
    // First branch: terminal (no `.then`) — "done" when ammo is fine.
    task.branch().precondition(|ammo: &Ammo| ammo.0 > 0);
    // Second branch: reload, then try again (recursion is just an edge).
    task.branch()
        .precondition(|ammo: &Ammo| ammo.0 == 0)
        .then(reload)
        .then(engage);
}

let domain = HtnDomain::from_root(engage).build().unwrap();
let state = PlanState::build(&domain.components).finish();
let mut planner = HtnPlanner::new(&domain);
let plan = planner.plan("engage", &state);
assert_eq!(plan.task_names(), ["reload"]);
```

### Executing in Bevy

```rust
use bevy_bhtn::ecs::{htn_ai_system, HtnAgent, HtnConfig};

let mut world = World::new();
world.insert_resource(HtnConfig::new(domain));
let entity = world.spawn((Ammo(0), HtnAgent::default())).id();

htn_ai_system(&mut world); // one AI tick: plan → validate → execute one step
assert_eq!(world.get::<Ammo>(entity).unwrap().0, 30);
```

Per agent per tick the driver: plans if planless (extracting a scratchpad from
the entity's components), re-validates the next step's preconditions against
the real world (drift ⇒ drop the plan, replan next tick), dispatches the
step's action commands, commits its effects to the real components, and
advances the cursor.

### Effects that mix reads and writes

`&mut T` marks a component the effect **writes** (journaled for rollback,
committed to the real entity); `&T` marks a **read-only** one — never
journaled, never committed. Any mix at any arity up to 8:

```rust
task.effect(
    |pockets: &mut PocketContents, goal: &CraftGoal, book: &RecipeBook| {
        if let Some(recipe) = book.recipe_for(goal.0) {
            for input in &recipe.inputs {
                pockets.remove(*input);
            }
            pockets.add(goal.0);
        }
    },
);
```

Only what the closure actually mutates costs anything.

### Backward (goal-state) planning

```rust
fn craft_spear(goal: &mut GoalBuilder) {
    goal.effect(|pockets: &mut PocketContents| pockets.add(ItemKind::Spear));
}

let domain = HtnDomain::from_root(behave)
    .goal(craft_spear)
    .build()
    .unwrap();
let plan = BackPlanner::new(&domain).plan("craft_spear", &state)?;
```

## The toolkit

- **Selection policies** — per compound task: `FirstMatch` (default),
  `HighestUtility`, `WeightedRandom` (deterministic, backtracking-safe), or a
  custom ranker.
- **Search strategies** — per config or per agent: `DepthFirst`,
  `DepthFirstFailFast`, `CostBounded` (branch-and-bound over annotated step
  costs, anytime), or a custom `Searcher`.
- **Partially-ordered subtasks** — `.subtask(f)` / `.before(a, b)` /
  `.any_order((a, b, c))` declare member sets; the search schedules
  linearizations and backtracks over them.
- **Look-ahead pruning** (Olz & Bercher, SoCS 2023) — before committing a
  method, a sweep over its subtask sequence proves dead ends and pins
  inevitable refinements, without decomposing anything.
- **Inferred summaries** (Olz/Biundo/Bercher, AAAI 2021 / JAIR 2025) — bake-
  time `required_fields` / `possible_writes` / `guaranteed_writes` per task,
  plus structure flags (`terminating`, `min_yield`, `min_cost`, `recursive`,
  `tail_recursive`) that bound non-terminating domains.
- **Tracing** — `plan_traced` emits one `DecompositionTrace` per branch
  decision (`Selected` / `PrecondFailed` / `Backtracked`); the driver bridges
  them to `Messages<DecompositionTrace>` when `HtnConfig::debug_trace` is set.

## Performance

The hot loop works on `usize` task indices over flat `Vec`s; [`Plan`] is a
compiled step program (a flat array walk — no name lookups at execution).
Backtracking is allocation-free: append-only plan/MTR restored by truncation,
queue suffixes snapshotted inline, and a rollback journal that deep-clones
only the slots each effect writes (plain-data slots clone as a `memcpy`).
Task names are interned [`Ustr`]s; the driver reuses one scratchpad and
zero steady-state allocations.

Benchmarks (Criterion, plan → execute-one-step → replan ×10 per iteration):

```
cargo bench -p bevy_bhtn
```

- `ai_throughput` — flat miner domain through a real Bevy schedule (10k–200k agents)
- `deep_ai` — deep outpost domain with relationship reads
- `lookahead` — look-ahead on/off A/B (exponential-backtrack and doomed-recursion wins)
- `wide_sets` — unordered-set scheduling cost vs. chains and recursion

## Testing

162 tests (151 integration + 5 unit + 6 doc), including byte-pool safety
invariants (drop-exactly-once across clone/rollback paths) and end-to-end
CDDA-like simulations that pin exact deterministic tick counts:

```
cargo test -p bevy_bhtn
```

## Status

Pre-1.0; the API may still shift. Built for a CDDA-style simulation
(`cdda_sim` may depend on this crate; the planner core itself is
ECS-driver-free and depends only on `bevy_ecs`).

## License

Apache-2.0 — see [LICENSE](LICENSE).
