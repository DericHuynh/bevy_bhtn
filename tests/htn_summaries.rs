//! Tests for bake-time inferred task/method summaries ([`TaskSummary`]) and
//! the forward planner's look-ahead pruning.
//!
//! The summaries adapt Olz, Biundo & Bercher's compound-task precondition/effect
//! inference (AAAI 2021, JAIR 2025) from propositional facts to **component
//! slots**; the look-ahead adapts Olz & Bercher's SoCS 2023 sweep. These tests
//! pin the soundness directions that matter:
//!
//! - `required_fields` under-approximates (only claim a slot when *every*
//!   refinement reads it before any writes it);
//! - `possible_writes` over-approximates (safe for optimistic propagation);
//! - `guaranteed_writes` under-approximates (effects only — `expected_effects`
//!   are hoped, not guaranteed);
//! - look-ahead pruning never changes *which* plan is found, only how fast it
//!   is found — except at the sanity limit, where pruning turns a doomed
//!   exhaustive descent into a successful alternative method.
//!
//! Domains are task-function graphs built inline (nested `fn`s inside each
//! domain constructor; domains whose summaries are pinned keep their task
//! functions in a sibling `*_tasks` module so tests can name them for
//! type-based lookup; the parameterized gate/chain generators are baked as
//! macro-generated static graphs). Summaries range over component slot
//! indices, resolved via `domain.components.get::<T>()`.
//!
//! Domains whose forward plans are exercised always plan their domain root
//! (resolved by task index, like [`common::HtnTestBed::plan_forward`]).

mod common;
use common::{Food, Fuel, Gold, Noise};

use bevy_bhtn::planner::HtnPlanner;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::{FieldSet, HtnDomain, Task, TaskBuilder, TaskFn, TaskSummary};
use bevy_ecs::prelude::*;

/// A task fn's item type cannot be named directly, so the lookup-by-type API
/// is reached through this inference helper: the fn value pins `F` to the
/// fn item's unique type, resolved through the baked `TypeId` index.
fn summary_of<F: TaskFn>(domain: &HtnDomain, _f: F) -> Option<&TaskSummary> {
    domain.task_summary(_f)
}

// ---------------------------------------------------------------------------
// Components (one per former state field)
// ---------------------------------------------------------------------------

/// Whether the miner is at base (former `at_base` field).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct AtBase(pub bool);
/// A generic work counter (former `count` field).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Count(pub i32);
/// The gamble's luck flag (former `luck` field).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Luck(pub bool);
/// Sweep-state coordinates and identifier operands (former `x`/`y`/`a`/`b`).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct X(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Y(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct A(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct B(pub i32);
/// The occurrence-pin test's phase counter (former `phase` field).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Phase(pub i32);
/// The occurrence-pin test's completion flag (former `done` field).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Done(pub bool);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Render a [`FieldSet`] as a sorted list of component slot names (robust to
/// the registry's registration order).
fn slot_names(domain: &HtnDomain, set: &FieldSet) -> Vec<&'static str> {
    let mut names: Vec<&'static str> = set
        .indices()
        .map(|i| domain.components.name_of(i))
        .collect();
    names.sort_unstable();
    names
}

/// The component slot index of `T` in `domain`'s registry.
fn slot_of<T: 'static>(domain: &HtnDomain) -> usize {
    domain
        .components
        .get::<T>()
        .unwrap_or_else(|| panic!("component not registered in domain"))
}

/// A default scratchpad over the domain's components.
fn default_state(domain: &HtnDomain) -> PlanState {
    PlanState::build(&domain.components).finish()
}

// ---------------------------------------------------------------------------
// FieldSet unit tests
// ---------------------------------------------------------------------------

#[test]
fn field_set_bit_operations() {
    let mut a = FieldSet::new(100);
    assert!(a.is_empty());
    assert_eq!(a.count(), 0);

    a.insert(0);
    a.insert(64);
    a.insert(99);
    assert!(a.contains(0));
    assert!(a.contains(64));
    assert!(a.contains(99));
    assert!(!a.contains(1));
    assert!(!a.contains(63));
    assert_eq!(a.count(), 3);
    assert_eq!(a.indices().collect::<Vec<_>>(), vec![0, 64, 99]);

    // Out-of-universe inserts/contains are no-ops/false, not panics.
    let tiny = FieldSet::new(2);
    assert!(!tiny.contains(70));
    let mut tiny_mut = FieldSet::new(2);
    tiny_mut.insert(70);
    assert_eq!(tiny_mut.count(), 0);

    // remove / clear
    let mut b = a.clone();
    b.remove(64);
    assert!(!b.contains(64));
    assert_eq!(b.count(), 2);
    b.clear();
    assert!(b.is_empty());

    // union / intersect / subtract / subset
    let mut u = FieldSet::new(100);
    u.insert(1);
    u.union_with(&a);
    assert!(u.contains(0) && u.contains(1) && u.contains(64) && u.contains(99));

    let mut i = u.clone();
    i.intersect_with(&a);
    assert!(i.is_subset_of(&a) && i.is_subset_of(&u));
    assert_eq!(i.count(), 3);

    let mut s = u.clone();
    s.subtract(&a);
    assert_eq!(s.count(), 1);
    assert!(s.contains(1));

    // Subset against a larger universe still works word-wise.
    assert!(a.is_subset_of(&u));
    let mut bigger = FieldSet::new(200);
    bigger.insert(150);
    assert!(!a.is_subset_of(&bigger));
}

// ---------------------------------------------------------------------------
// Summary domains
// ---------------------------------------------------------------------------

mod work_tasks {
    use super::*;

    pub fn work(task: &mut TaskBuilder) {
        task.branch().then(dig).then(haul);
        task.branch().then(sell);
    }
    pub fn dig(task: &mut TaskBuilder) {
        task.precondition(|fuel: &Fuel| fuel.0 > 0)
            .effect(|gold: &mut Gold| gold.0 += 1)
            .effect(|fuel: &mut Fuel| fuel.0 -= 1);
    }
    pub fn haul(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 0)
            .effect(|at_base: &mut AtBase| at_base.0 = true);
    }
    pub fn sell(task: &mut TaskBuilder) {
        task.precondition(|at_base: &AtBase| at_base.0)
            .effect(|gold: &mut Gold| gold.0 -= 1);
    }
}

fn work_domain() -> HtnDomain {
    HtnDomain::from_root(work_tasks::work)
        .build()
        .expect("work domain is well-formed")
}

#[test]
fn summaries_pin_flat_domain() {
    let domain = work_domain();

    // Work: possible = union over both methods; guaranteed = intersection
    // (dig writes all three, sell only gold); required = intersection of the
    // methods' sequence requirements (dig needs fuel, sell needs at_base) = ∅.
    let work = summary_of(&domain, work_tasks::work).expect("work summary");
    assert!(slot_names(&domain, &work.required_fields).is_empty());
    assert_eq!(
        slot_names(&domain, &work.possible_writes),
        vec!["AtBase", "Fuel", "Gold"]
    );
    assert_eq!(slot_names(&domain, &work.guaranteed_writes), vec!["Gold"]);

    // Dig: reads fuel before writing it; writes gold and fuel — pinned by
    // slot index, not by name.
    let dig = summary_of(&domain, work_tasks::dig).expect("dig summary");
    assert_eq!(slot_names(&domain, &dig.required_fields), vec!["Fuel"]);
    assert!(dig.required_fields.contains(slot_of::<Fuel>(&domain)));
    assert!(!dig.required_fields.contains(slot_of::<Gold>(&domain)));
    assert_eq!(
        slot_names(&domain, &dig.possible_writes),
        vec!["Fuel", "Gold"]
    );
    assert_eq!(
        slot_names(&domain, &dig.guaranteed_writes),
        vec!["Fuel", "Gold"]
    );

    // Haul: requires gold, writes at_base.
    let haul = summary_of(&domain, work_tasks::haul).expect("haul summary");
    assert_eq!(slot_names(&domain, &haul.required_fields), vec!["Gold"]);
    assert_eq!(slot_names(&domain, &haul.possible_writes), vec!["AtBase"]);
    assert_eq!(slot_names(&domain, &haul.guaranteed_writes), vec!["AtBase"]);

    // Sell: requires at_base, writes gold.
    let sell = summary_of(&domain, work_tasks::sell).expect("sell summary");
    assert_eq!(slot_names(&domain, &sell.required_fields), vec!["AtBase"]);
    assert_eq!(slot_names(&domain, &sell.possible_writes), vec!["Gold"]);
    assert_eq!(slot_names(&domain, &sell.guaranteed_writes), vec!["Gold"]);

    // Method-level possible writes: dig's chain writes everything, sell only
    // gold. These drive the look-ahead's optimistic propagation. The per-method
    // sets are bake-internal now, so pin the equivalent public data: a
    // total-order method's possible writes are exactly the union of its member
    // primitives' write slots.
    let Some(Task::Compound(work_task)) = domain.get_task("work") else {
        panic!("work must be compound");
    };
    let method_write_names = |method: usize| -> Vec<&'static str> {
        let mut names: Vec<&'static str> = work_task.methods[method]
            .subtasks
            .iter()
            .filter_map(|&s| match &domain.tasks[s as usize] {
                Task::Primitive(p) => Some(p),
                _ => None,
            })
            .flat_map(|p| p.write_slots())
            .map(|i| domain.components.name_of(i))
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    };
    assert_eq!(
        method_write_names(0),
        vec!["AtBase", "Fuel", "Gold"],
        "dig method writes"
    );
    assert_eq!(method_write_names(1), vec!["Gold"], "sell method writes");
}

mod recursion_tasks {
    use super::*;

    // Synthetic root so every top-level task is baked (from_root records the
    // transitive .then graph only); tests address the individual tasks by type.
    pub fn root(task: &mut TaskBuilder) {
        task.branch().then(loop_);
        task.branch().then(descend);
        task.branch().then(spiral);
    }
    pub fn loop_(task: &mut TaskBuilder) {
        task.branch().then(tick);
        task.branch().then(loop_).then(tick);
    }
    pub fn tick(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn descend(task: &mut TaskBuilder) {
        task.branch().then(descend).then(eat);
        task.branch().then(eat);
    }
    pub fn eat(task: &mut TaskBuilder) {
        task.precondition(|food: &Food| food.0 > 0)
            .effect(|food: &mut Food| food.0 -= 1);
    }
    pub fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral).then(tick);
    }
}

fn recursion_domain() -> HtnDomain {
    HtnDomain::from_root(recursion_tasks::root)
        .build()
        .expect("recursion domain is well-formed")
}

#[test]
fn summaries_pin_recursive_domains() {
    let domain = recursion_domain();

    // Loop terminates (base branch `tick`): every refinement is Tick^k, k>=1.
    let loop_summary = summary_of(&domain, recursion_tasks::loop_).expect("loop summary");
    assert!(slot_names(&domain, &loop_summary.required_fields).is_empty());
    assert_eq!(
        slot_names(&domain, &loop_summary.possible_writes),
        vec!["Count"]
    );
    assert_eq!(
        slot_names(&domain, &loop_summary.guaranteed_writes),
        vec!["Count"]
    );

    // Descend: every refinement is Eat^k, k>=1 — food is read before anything
    // writes it in every refinement, so it survives the recursion as required.
    let descend = summary_of(&domain, recursion_tasks::descend).expect("descend summary");
    assert_eq!(slot_names(&domain, &descend.required_fields), vec!["Food"]);
    assert_eq!(slot_names(&domain, &descend.possible_writes), vec!["Food"]);
    assert_eq!(
        slot_names(&domain, &descend.guaranteed_writes),
        vec!["Food"]
    );

    // Spiral can only refine forever (no base branch): it has no finite
    // refinements, so nothing is required (the inference papers' "undef"
    // convention, conservatively mapped to empty). Possible writes stay an
    // over-approximation.
    let spiral = summary_of(&domain, recursion_tasks::spiral).expect("spiral summary");
    assert!(slot_names(&domain, &spiral.required_fields).is_empty());
    assert_eq!(slot_names(&domain, &spiral.possible_writes), vec!["Count"]);
}

mod gamble_tasks {
    use super::*;

    pub fn gamble(task: &mut TaskBuilder) {
        task.branch().then(hope);
    }
    pub fn hope(task: &mut TaskBuilder) {
        task.expected(|luck: &mut Luck| luck.0 = true);
    }
}

fn gamble_domain() -> HtnDomain {
    HtnDomain::from_root(gamble_tasks::gamble)
        .build()
        .expect("gamble domain is well-formed")
}

#[test]
fn expected_effects_are_possible_but_not_guaranteed() {
    let domain = gamble_domain();

    let hope = summary_of(&domain, gamble_tasks::hope).expect("hope summary");
    assert_eq!(slot_names(&domain, &hope.possible_writes), vec!["Luck"]);
    assert!(slot_names(&domain, &hope.guaranteed_writes).is_empty());

    let gamble = summary_of(&domain, gamble_tasks::gamble).expect("gamble summary");
    assert_eq!(slot_names(&domain, &gamble.possible_writes), vec!["Luck"]);
    assert!(slot_names(&domain, &gamble.guaranteed_writes).is_empty());
}

// ---------------------------------------------------------------------------
// Look-ahead pruning
// ---------------------------------------------------------------------------

/// The doomed method's tail task requires `gold > 100`, and nothing in its
/// sequence (including the non-terminating `spiral2`) can ever write `gold` —
/// the sweep proves this without decomposing anything, so the planner commits
/// to `safe` directly.
fn doomed_domain() -> HtnDomain {
    fn act(task: &mut TaskBuilder) {
        task.branch().then(prime).then(spiral2).then(impossible);
        task.branch().then(safe);
    }
    fn spiral2(task: &mut TaskBuilder) {
        task.branch().then(spiral2);
    }
    fn prime(task: &mut TaskBuilder) {
        task.effect(|at_base: &mut AtBase| at_base.0 = true);
    }
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    HtnDomain::from_root(act)
        .build()
        .expect("doomed domain is well-formed")
}

#[test]
fn lookahead_beats_sanity_limit_on_doomed_method() {
    use common::HtnTestBed;

    let bed = HtnTestBed::new(doomed_domain());
    let start = PlanState::build(&bed.domain().components)
        .set(Gold(0))
        .finish();
    // Without the look-ahead, `act`'s doomed branch descends into spiral2's
    // unbounded recursion until the sanity limit fires and returns the partial
    // plan ["prime"]. With it, the doomed method is refuted at the frame.
    assert_eq!(bed.plan_forward(&start), vec!["safe"]);
}

#[test]
fn lookahead_keeps_plans_identical_when_backtracking_suffices() {
    use common::HtnTestBed;

    let bed = HtnTestBed::new(work_domain());

    // dig is refuted by the sweep (fuel known 0, nothing writes it before
    // dig); plain backtracking would find the same plan.
    let start = PlanState::build(&bed.domain().components)
        .set(Fuel(0))
        .set(Gold(3))
        .set(AtBase(true))
        .finish();
    assert_eq!(bed.plan_forward(&start), vec!["sell"]);

    // dig succeeds through the sweep: dig's increment keeps gold *known*
    // (deterministic relative write on a known value), so haul's gold > 0
    // check is evaluated exactly against the propagated state.
    let start = PlanState::build(&bed.domain().components)
        .set(Fuel(5))
        .finish();
    assert_eq!(bed.plan_forward(&start), vec!["dig", "haul"]);
}

/// A method whose preconditions can't be evaluated yet (they read a component
/// an earlier compound task *might* write) must not be refuted by the sweep —
/// unknown components are "maybe", never "no".
fn maybe_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(gamble_task).then(check_luck);
        task.branch().then(safe);
    }
    fn gamble_task(task: &mut TaskBuilder) {
        task.expected(|luck: &mut Luck| luck.0 = true);
    }
    fn check_luck(task: &mut TaskBuilder) {
        task.precondition(|luck: &Luck| luck.0);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    HtnDomain::from_root(root)
        .build()
        .expect("maybe domain is well-formed")
}

#[test]
fn lookahead_treats_unknown_fields_as_maybe() {
    use common::HtnTestBed;

    let bed = HtnTestBed::new(maybe_domain());
    let start = PlanState::build(&bed.domain().components)
        .set(Luck(false))
        .finish();
    // check_luck's precondition reads `luck`, which gamble_task's expected
    // effect (applied exactly by the sweep, since the planner applies expected
    // effects during search too) sets to true — the sequence survives and the
    // gamble is planned rather than pruned.
    assert_eq!(bed.plan_forward(&start), vec!["gamble_task", "check_luck"]);
}

// ---------------------------------------------------------------------------
// Backtracking queue restoration (observable with the look-ahead off)
// ---------------------------------------------------------------------------

/// With the look-ahead off, plain MTR backtracking must still restore the task
/// queue when unwinding: a failure in a method's *middle* subtask used to
/// leave the queue stale (siblings from the abandoned choice were executed,
/// and the fallback method was never reached).
fn mid_failure_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(fine).then(doomed).then(collateral);
        task.branch().then(rescue);
    }
    fn fine(task: &mut TaskBuilder) {
        task.effect(|noise: &mut Noise| noise.0 = true);
    }
    fn doomed(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn collateral(task: &mut TaskBuilder) {
        task.effect(|noise: &mut Noise| noise.0 = false);
    }
    fn rescue(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    HtnDomain::from_root(root)
        .build()
        .expect("mid-failure domain is well-formed")
}

#[test]
fn backtracking_restores_queue_after_mid_sequence_failure() {
    let domain = mid_failure_domain();
    let state = default_state(&domain);

    // Look-ahead on: the broken branch is refuted at the frame (doomed needs
    // gold > 100 and nothing in its sequence writes gold) — plan goes straight
    // to rescue.
    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(true);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names(),
        ["rescue"],
        "look-ahead should refute the broken branch without entering it"
    );

    // Look-ahead off: plain backtracking must unwind the broken branch
    // cleanly — the abandoned branch's collateral must NOT leak into the
    // plan, and the fallback method must be reached.
    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(false);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names(),
        ["rescue"],
        "plain backtracking must discard the abandoned branch and fall through"
    );
}

/// The *lost suffix* variant of the queue-restore bug: the failure happens at
/// a task that is itself the tail of the sequence, so the queue is empty at
/// failure time. Backtracking through the ancestor frames must re-attach the
/// ancestor's suffix (here `final`, queued behind `gate`) so that (a) `final`
/// is re-attempted after `gate`'s second method and (b) once `gate` is
/// exhausted, the search unwinds to `root` and finds `direct`. The old
/// code — which never restored the queue — consumed `final` on the first
/// failed attempt and terminated with the partial plan [junk_a, junk_b].
fn lost_suffix_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(gate).then(final_);
        task.branch().then(ok);
    }
    fn gate(task: &mut TaskBuilder) {
        task.branch().then(junk_a);
        task.branch().then(junk_b);
    }
    fn junk_a(task: &mut TaskBuilder) {
        task.effect(|noise: &mut Noise| noise.0 = true);
    }
    fn junk_b(task: &mut TaskBuilder) {
        task.effect(|noise: &mut Noise| noise.0 = false);
    }
    fn final_(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 100);
    }
    fn ok(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    HtnDomain::from_root(root)
        .build()
        .expect("lost-suffix domain is well-formed")
}

#[test]
fn backtracking_restores_ancestor_suffix_after_tail_failure() {
    let domain = lost_suffix_domain();
    let state = default_state(&domain);

    // Look-ahead on: the doomed branch is refuted at the frame (final needs
    // gold > 100 and nothing in the sequence writes gold).
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names(),
        ["ok"],
        "look-ahead should refute the doomed branch without entering it"
    );

    // Look-ahead off: the search must enumerate gate.a, gate.b, then unwind
    // past the consumed `final` and still reach `direct`.
    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(false);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names(),
        ["ok"],
        "backtracking must restore the ancestor suffix consumed by the failed tail"
    );
}

// ---------------------------------------------------------------------------
// Slice 4: termination-flag dead-ends
// ---------------------------------------------------------------------------
/// Pure infinite recursion with **no impossible precondition anywhere** — the
/// old sweep had nothing to refute (every method applies, no condition ever
/// definitely fails), so the doomed method burned the whole sanity budget.
/// The bake-time `terminating` flag refutes it in one sweep.
mod pure_recursion_tasks {
    use super::*;

    pub fn root(task: &mut TaskBuilder) {
        task.branch().then(prime).then(spiral);
        task.branch().then(ok);
    }
    pub fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral).then(tick);
    }
    pub fn tick(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn prime(task: &mut TaskBuilder) {
        task.effect(|noise: &mut Noise| noise.0 = true);
    }
    pub fn ok(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
}

fn pure_recursion_domain() -> HtnDomain {
    HtnDomain::from_root(pure_recursion_tasks::root)
        .build()
        .expect("pure-recursion domain is well-formed")
}

#[test]
fn terminating_flag_refutes_pure_infinite_recursion() {
    let domain = pure_recursion_domain();
    let state = default_state(&domain);

    // The flag itself: spiral can only refine forever.
    assert!(
        !summary_of(&domain, pure_recursion_tasks::spiral)
            .unwrap()
            .terminating
    );
    assert_eq!(
        summary_of(&domain, pure_recursion_tasks::spiral)
            .unwrap()
            .min_yield,
        usize::MAX
    );

    // Look-ahead on: the doomed branch is refuted at the frame without
    // recursing.
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(planner.plan_index(domain.root, &state).task_names(), ["ok"]);

    // Look-ahead off: plain backtracking burns the sanity budget and returns
    // the partial plan (the documented fallback semantics).
    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(false);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names(),
        ["prime"]
    );
}

/// The non-terminating task is buried one level deep: the sweep must refute
/// via `probe`'s own flag (its only refinement path is non-terminating)
/// without decomposing anything.
mod nested_recursion_tasks {
    use super::*;

    pub fn root(task: &mut TaskBuilder) {
        task.branch().then(probe).then(wrap);
        task.branch().then(ok);
    }
    pub fn probe(task: &mut TaskBuilder) {
        task.branch().then(spiral);
    }
    pub fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral).then(tick);
    }
    pub fn tick(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn wrap(task: &mut TaskBuilder) {
        task.effect(|noise: &mut Noise| noise.0 = false);
    }
    pub fn ok(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
}

fn nested_recursion_domain() -> HtnDomain {
    HtnDomain::from_root(nested_recursion_tasks::root)
        .build()
        .expect("nested-recursion domain is well-formed")
}

#[test]
fn terminating_flag_refutes_through_nested_compounds() {
    let domain = nested_recursion_domain();
    let state = default_state(&domain);

    assert!(
        !summary_of(&domain, nested_recursion_tasks::probe)
            .unwrap()
            .terminating
    );

    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(planner.plan_index(domain.root, &state).task_names(), ["ok"]);

    // Off: the doomed branch is entered; no primitive ever executes before
    // the budget burns, so the partial plan is empty.
    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(false);
    assert!(planner
        .plan_index(domain.root, &state)
        .task_names()
        .is_empty());
}

/// A non-terminating *method alternative*: the risky method is tried first,
/// refuted by the sweep at its own commitment, and the viable one is taken —
/// where the old planner burned the budget inside `risky` and returned an
/// empty partial plan.
mod mixed_termination_tasks {
    use super::*;

    pub fn pick(task: &mut TaskBuilder) {
        task.branch().then(risky);
        task.branch().then(ok);
    }
    pub fn risky(task: &mut TaskBuilder) {
        task.branch().then(spiral);
    }
    pub fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral).then(tick);
    }
    pub fn tick(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn ok(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
}

fn mixed_termination_domain() -> HtnDomain {
    HtnDomain::from_root(mixed_termination_tasks::pick)
        .build()
        .expect("mixed-termination domain is well-formed")
}

#[test]
fn non_terminating_method_skipped_among_viable_ones() {
    let domain = mixed_termination_domain();
    let state = default_state(&domain);

    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(planner.plan_index(domain.root, &state).task_names(), ["ok"]);

    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(false);
    assert!(planner
        .plan_index(domain.root, &state)
        .task_names()
        .is_empty());
}

/// Over-refutation guard: *terminating* recursion must keep its flag and keep
/// planning — a wrong `terminating = false` here would collapse the plan.
mod terminating_recursion_tasks {
    use super::*;

    pub fn root(task: &mut TaskBuilder) {
        task.branch().then(loop_).then(check);
    }
    pub fn loop_(task: &mut TaskBuilder) {
        task.branch().then(tick);
        task.branch().then(loop_).then(tick);
    }
    pub fn tick(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn check(task: &mut TaskBuilder) {
        task.precondition(|count: &Count| count.0 > 0);
    }
}

fn terminating_recursion_domain() -> HtnDomain {
    HtnDomain::from_root(terminating_recursion_tasks::root)
        .build()
        .expect("terminating-recursion domain is well-formed")
}

#[test]
fn terminating_recursion_not_flagged() {
    let domain = terminating_recursion_domain();
    let state = default_state(&domain);

    let summary = summary_of(&domain, terminating_recursion_tasks::loop_).unwrap();
    assert!(summary.terminating);
    assert_eq!(summary.min_yield, 1);

    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names(),
        ["tick", "check"]
    );
}

// ---------------------------------------------------------------------------
// Slice 5: per-condition precomputed read indices
// ---------------------------------------------------------------------------

// Task-function identity is the function item itself, so the parameterized
// gate chain is baked as a macro-generated static graph: 12 gates, identical
// subtask shapes to the old builder chain, `root` declared first so it stays
// the default forward root.

/// Generates the gate/junk task triples of the doomed chain.
macro_rules! gate_tasks {
    () => {};
    ($gate:ident $junk_a:ident $junk_b:ident $($rest:tt)*) => {
        pub fn $gate(task: &mut TaskBuilder) {
            task.branch().then($junk_a);
            task.branch().then($junk_b);
        }
        pub fn $junk_a(task: &mut TaskBuilder) {
            task.effect(|noise: &mut Noise| noise.0 = true);
        }
        pub fn $junk_b(task: &mut TaskBuilder) {
            task.effect(|noise: &mut Noise| noise.0 = false);
        }
        gate_tasks!($($rest)*);
    };
}

/// Bakes the whole 12-gate domain as a module of task functions plus a
/// constructor fn named after the module.
macro_rules! gates_domain_mod {
    ($mod_name:ident, [$($gate:ident),*], $($triples:tt)*) => {
        mod $mod_name {
            use super::{Gold, Noise, X, Y};
            use bevy_bhtn::TaskBuilder;

            pub fn root(task: &mut TaskBuilder) {
                let mut doomed = task.branch();
                doomed.then(x_writer);
                $(doomed.then($gate);)*
                doomed.then(checker);
                task.branch().then(ok);
            }
            pub fn x_writer(task: &mut TaskBuilder) {
                task.branch().then(set_x);
            }
            pub fn set_x(task: &mut TaskBuilder) {
                task.effect(|x: &mut X| x.0 = 1);
            }
            pub fn checker(task: &mut TaskBuilder) {
                task.precondition(|x: &X| x.0 == 1)
                    .precondition(|y: &Y| y.0 == 5);
            }
            pub fn ok(task: &mut TaskBuilder) {
                task.effect(|gold: &mut Gold| gold.0 = 1);
            }
            gate_tasks!($($triples)*);
        }

        fn $mod_name() -> HtnDomain {
            HtnDomain::from_root($mod_name::root)
                .build()
                .expect("gates domain is well-formed")
        }
    };
}

gates_domain_mod!(
    gates12,
    [
        gate0, gate1, gate2, gate3, gate4, gate5, gate6, gate7, gate8, gate9, gate10, gate11
    ],
    gate0 junk_a0 junk_b0
    gate1 junk_a1 junk_b1
    gate2 junk_a2 junk_b2
    gate3 junk_a3 junk_b3
    gate4 junk_a4 junk_b4
    gate5 junk_a5 junk_b5
    gate6 junk_a6 junk_b6
    gate7 junk_a7 junk_b7
    gate8 junk_a8 junk_b8
    gate9 junk_a9 junk_b9
    gate10 junk_a10 junk_b10
    gate11 junk_a11 junk_b11
);

/// Two preconditions on one primitive with **different** known/unknown status:
/// `x` is unknown (a compound might write it) but `y` is known-false. The
/// per-condition read indices must prune via `y == 5` alone — a union-based
/// regression ("any read unknown → maybe") would treat the whole primitive as
/// maybe, commit the doomed method, and burn the sanity budget on the gates.
fn gates_with_unknown_x_domain(gates: usize) -> HtnDomain {
    assert_eq!(gates, 12, "the baked gate graph is fixed at 12 gates");
    gates12()
}

#[test]
fn per_condition_reads_prune_despite_unknown_sibling_field() {
    let domain = gates_with_unknown_x_domain(12);
    let state = default_state(&domain);

    // On: `y == 5` is definitely false (y known 0) even though `x == 1` is
    // maybe — the doomed method is refuted at the frame.
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(planner.plan_index(domain.root, &state).task_names(), ["ok"]);

    // Off: the doomed method is entered (first task set_x) and burns the
    // budget on the gate enumeration.
    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(false);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names()[0],
        "set_x"
    );
}

/// An identifier condition (`a == b`) with one unknown operand must be
/// "maybe", never evaluated against the stale clone value — the gamble path
/// succeeds at runtime and must be planned.
fn identifier_maybe_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(a_writer).then(cmp);
        task.branch().then(safe);
    }
    fn a_writer(task: &mut TaskBuilder) {
        task.branch().then(set_a);
    }
    fn set_a(task: &mut TaskBuilder) {
        task.effect(|a: &mut A| a.0 = 5);
    }
    fn cmp(task: &mut TaskBuilder) {
        task.precondition(|a: &A, b: &B| a.0 == b.0);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    HtnDomain::from_root(root)
        .build()
        .expect("identifier-maybe domain is well-formed")
}

#[test]
fn identifier_condition_with_unknown_field_is_maybe() {
    let domain = identifier_maybe_domain();
    // a=0 (a stale clone would compare 0 == 5 → false), b=5; set_a makes them
    // equal at runtime.
    let state = PlanState::build(&domain.components).set(B(5)).finish();

    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names(),
        ["set_a", "cmp"]
    );
}

/// A negated condition on an unknown field must likewise be "maybe": the
/// stale clone value (a = 5) would fail `a != 5`, but the runtime value (7)
/// passes.
fn notted_maybe_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch().then(a_writer).then(cmp);
        task.branch().then(safe);
    }
    fn a_writer(task: &mut TaskBuilder) {
        task.branch().then(set_a);
    }
    fn set_a(task: &mut TaskBuilder) {
        task.effect(|a: &mut A| a.0 = 7);
    }
    fn cmp(task: &mut TaskBuilder) {
        task.precondition(|a: &A| a.0 != 5);
    }
    fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut Gold| gold.0 = 1);
    }
    HtnDomain::from_root(root)
        .build()
        .expect("notted-maybe domain is well-formed")
}

#[test]
fn notted_condition_on_unknown_field_is_maybe() {
    let domain = notted_maybe_domain();
    let state = PlanState::build(&domain.components).set(A(5)).finish();

    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        planner.plan_index(domain.root, &state).task_names(),
        ["set_a", "cmp"]
    );
}

// ---------------------------------------------------------------------------
// Occurrence-scoped pins
// ---------------------------------------------------------------------------

/// The same compound task (`twice`) is decomposed **twice** in one method
/// sequence, under different states: the first occurrence can only use its
/// `phase == 0` method, the second only its `phase == 1` method. The sweep
/// legitimately pins both occurrences — but each pin must travel with its own
/// occurrence. Keying pins by task index (the original implementation) applied
/// the *last* pin to both occurrences, exhausting the first illegally and
/// collapsing the whole plan.
fn occurrence_pin_domain() -> HtnDomain {
    fn root(task: &mut TaskBuilder) {
        task.branch()
            .then(twice)
            .then(bump)
            .then(twice)
            .then(verify);
    }
    fn twice(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|phase: &Phase| phase.0 == 0)
            .then(step0);
        task.branch()
            .precondition(|phase: &Phase| phase.0 == 1)
            .then(step1);
    }
    fn step0(task: &mut TaskBuilder) {
        task.effect(|done: &mut Done| done.0 = true);
    }
    fn step1(task: &mut TaskBuilder) {
        task.effect(|done: &mut Done| done.0 = true);
    }
    fn bump(task: &mut TaskBuilder) {
        task.effect(|phase: &mut Phase| phase.0 = 1);
    }
    fn verify(task: &mut TaskBuilder) {
        task.precondition(|done: &Done| done.0);
    }
    HtnDomain::from_root(root)
        .build()
        .expect("occurrence-pin domain is well-formed")
}

#[test]
fn pins_apply_per_occurrence_not_per_task_index() {
    use common::HtnTestBed;

    let bed = HtnTestBed::new(occurrence_pin_domain());
    let start = default_state(bed.domain());

    // The sweep pins occurrence 1 to `phase zero` and occurrence 2 to
    // `phase one`; both pins must hold simultaneously.
    assert_eq!(
        bed.plan_forward(&start),
        vec!["step0", "bump", "step1", "verify"],
        "each twice occurrence must use its own pinned method"
    );

    // Without the look-ahead there are no pins at all; plain backtracking
    // must find the same plan.
    let domain = occurrence_pin_domain();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_lookahead(false);
    assert_eq!(
        planner.plan_index(domain.root, &start).task_names(),
        ["step0", "bump", "step1", "verify"],
        "plain backtracking must agree with the pinned plan"
    );
}
