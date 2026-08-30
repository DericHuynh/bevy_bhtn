//! Tests for parse-time inferred task/method summaries ([`TaskSummary`]) and
//! the forward planner's look-ahead pruning.
//!
//! The summaries adapt Olz, Biundo & Bercher's compound-task precondition/effect
//! inference (AAAI 2021, JAIR 2025) from propositional facts to typed state
//! fields; the look-ahead adapts Olz & Bercher's SoCS 2023 sweep. These tests
//! pin the soundness directions that matter:
//!
//! - `required_fields` under-approximates (only claim a field when *every*
//!   refinement reads it before any writes it);
//! - `possible_writes` over-approximates (safe for optimistic propagation);
//! - `guaranteed_writes` under-approximates (effects only — `expected_effects`
//!   are hoped, not guaranteed);
//! - look-ahead pruning never changes *which* plan is found, only how fast it
//!   is found — except at the sanity limit, where pruning turns a doomed
//!   exhaustive descent into a successful alternative method.

mod common;
use common::HtnTestBed;

use bevy_bhtn::{FieldSet, HtnDomain};
use bevy_reflect::Reflect;
use bevy_reflect::TypeRegistry;

// ---------------------------------------------------------------------------
// Helpers + state
// ---------------------------------------------------------------------------

#[derive(Reflect, Clone, Debug, Default)]
struct MinerState {
    gold: i32,
    fuel: i32,
    at_base: bool,
    count: i32,
    food: i32,
    luck: bool,
}

fn register_miner(registry: &mut TypeRegistry) {
    registry.register::<MinerState>();
}

/// State for the per-condition-read-index tests (slice 5): `x`/`y` exercise
/// mixed known/unknown preconditions, `a`/`b` identifier conditions.
#[derive(Reflect, Clone, Debug, Default)]
struct SweepState {
    x: i32,
    y: i32,
    a: i32,
    b: i32,
    gold: i32,
    noise: bool,
    count: i32,
}

fn register_sweep(registry: &mut TypeRegistry) {
    registry.register::<SweepState>();
}

/// Render a [`FieldSet`] as a sorted list of field names (robust to the
/// domain's interning order).
fn field_names(domain: &HtnDomain, set: &FieldSet) -> Vec<String> {
    let mut names: Vec<String> = set
        .indices()
        .map(|i| domain.fields[i].to_string())
        .collect();
    names.sort();
    names
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

const WORK_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Work" {
    method "dig" {
        subtasks: [Dig, Haul]
    }
    method "sell" {
        subtasks: [Sell]
    }
}

primitive_task "Dig" {
    operator: NoopOperator
    preconditions: [fuel > 0]
    effects: [gold += 1, fuel -= 1]
}

primitive_task "Haul" {
    operator: NoopOperator
    preconditions: [gold > 0]
    effects: [at_base = true]
}

primitive_task "Sell" {
    operator: NoopOperator
    preconditions: [at_base == true]
    effects: [gold -= 1]
}
"#;

#[test]
fn summaries_pin_flat_domain() {
    let bed = HtnTestBed::new(WORK_HTN, "Work", register_miner);
    let domain = bed.domain();

    // Work: possible = union over both methods; guaranteed = intersection
    // (dig writes all three, sell only gold); required = intersection of the
    // methods' sequence requirements (dig needs fuel, sell needs at_base) = ∅.
    let work = domain.task_summary("Work").expect("Work summary");
    assert!(field_names(domain, &work.required_fields).is_empty());
    assert_eq!(
        field_names(domain, &work.possible_writes),
        vec!["at_base", "fuel", "gold"]
    );
    assert_eq!(field_names(domain, &work.guaranteed_writes), vec!["gold"]);

    // Dig: reads fuel before writing it; writes gold and fuel.
    let dig = domain.task_summary("Dig").expect("Dig summary");
    assert_eq!(field_names(domain, &dig.required_fields), vec!["fuel"]);
    assert_eq!(
        field_names(domain, &dig.possible_writes),
        vec!["fuel", "gold"]
    );
    assert_eq!(
        field_names(domain, &dig.guaranteed_writes),
        vec!["fuel", "gold"]
    );

    // Haul: requires gold, writes at_base.
    let haul = domain.task_summary("Haul").expect("Haul summary");
    assert_eq!(field_names(domain, &haul.required_fields), vec!["gold"]);
    assert_eq!(field_names(domain, &haul.possible_writes), vec!["at_base"]);
    assert_eq!(
        field_names(domain, &haul.guaranteed_writes),
        vec!["at_base"]
    );

    // Sell: requires at_base, writes gold.
    let sell = domain.task_summary("Sell").expect("Sell summary");
    assert_eq!(field_names(domain, &sell.required_fields), vec!["at_base"]);
    assert_eq!(field_names(domain, &sell.possible_writes), vec!["gold"]);
    assert_eq!(field_names(domain, &sell.guaranteed_writes), vec!["gold"]);

    // Method-level possible writes: dig's chain writes everything, sell only
    // gold. These drive the look-ahead's optimistic propagation.
    let dig_writes = domain
        .method_possible_writes("Work", 0)
        .expect("dig method writes");
    assert_eq!(
        field_names(domain, dig_writes),
        vec!["at_base", "fuel", "gold"]
    );
    let sell_writes = domain
        .method_possible_writes("Work", 1)
        .expect("sell method writes");
    assert_eq!(field_names(domain, sell_writes), vec!["gold"]);
}

const RECURSION_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Loop" {
    method "done" {
        subtasks: [Tick]
    }
    method "again" {
        subtasks: [Loop, Tick]
    }
}

primitive_task "Tick" {
    operator: NoopOperator
    effects: [count += 1]
}

compound_task "Descend" {
    method "step" {
        subtasks: [Descend, Eat]
    }
    method "stop" {
        subtasks: [Eat]
    }
}

primitive_task "Eat" {
    operator: NoopOperator
    preconditions: [food > 0]
    effects: [food -= 1]
}

compound_task "Spiral" {
    method "only" {
        subtasks: [Spiral, Tick]
    }
}
"#;

#[test]
fn summaries_pin_recursive_domains() {
    let bed = HtnTestBed::new(RECURSION_HTN, "Loop", register_miner);
    let domain = bed.domain();

    // Loop terminates (base method `done`): every refinement is Tick^k, k>=1.
    let loop_summary = domain.task_summary("Loop").expect("Loop summary");
    assert!(field_names(domain, &loop_summary.required_fields).is_empty());
    assert_eq!(
        field_names(domain, &loop_summary.possible_writes),
        vec!["count"]
    );
    assert_eq!(
        field_names(domain, &loop_summary.guaranteed_writes),
        vec!["count"]
    );

    // Descend: every refinement is Eat^k, k>=1 — food is read before anything
    // writes it in every refinement, so it survives the recursion as required.
    let descend = domain.task_summary("Descend").expect("Descend summary");
    assert_eq!(field_names(domain, &descend.required_fields), vec!["food"]);
    assert_eq!(field_names(domain, &descend.possible_writes), vec!["food"]);
    assert_eq!(
        field_names(domain, &descend.guaranteed_writes),
        vec!["food"]
    );

    // Spiral can only refine forever (no base method): it has no finite
    // refinements, so nothing is required (the inference papers' "undef"
    // convention, conservatively mapped to empty). Possible writes stay an
    // over-approximation.
    let spiral = domain.task_summary("Spiral").expect("Spiral summary");
    assert!(field_names(domain, &spiral.required_fields).is_empty());
    assert_eq!(field_names(domain, &spiral.possible_writes), vec!["count"]);
}

const GAMBLE_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Gamble" {
    method "hope" {
        subtasks: [Hope]
    }
}

primitive_task "Hope" {
    operator: NoopOperator
    expected_effects: [luck = true]
}
"#;

#[test]
fn expected_effects_are_possible_but_not_guaranteed() {
    let bed = HtnTestBed::new(GAMBLE_HTN, "Gamble", register_miner);
    let domain = bed.domain();

    let hope = domain.task_summary("Hope").expect("Hope summary");
    assert_eq!(field_names(domain, &hope.possible_writes), vec!["luck"]);
    assert!(field_names(domain, &hope.guaranteed_writes).is_empty());

    let gamble = domain.task_summary("Gamble").expect("Gamble summary");
    assert_eq!(field_names(domain, &gamble.possible_writes), vec!["luck"]);
    assert!(field_names(domain, &gamble.guaranteed_writes).is_empty());
}

// ---------------------------------------------------------------------------
// Look-ahead pruning
// ---------------------------------------------------------------------------

/// The doomed method's tail task requires `gold > 100`, and nothing in its
/// sequence (including the non-terminating `Spiral2`) can ever write `gold` —
/// the sweep proves this without decomposing anything, so the planner commits
/// to `safe` directly.
const DOOMED_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Act" {
    method "doomed" {
        subtasks: [Prime, Spiral2, Impossible]
    }
    method "safe" {
        subtasks: [Safe]
    }
}

compound_task "Spiral2" {
    method "only" {
        subtasks: [Spiral2]
    }
}

primitive_task "Prime" {
    operator: NoopOperator
    effects: [at_base = true]
}

primitive_task "Impossible" {
    operator: NoopOperator
    preconditions: [gold > 100]
}

primitive_task "Safe" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn lookahead_beats_sanity_limit_on_doomed_method() {
    let bed = HtnTestBed::new(DOOMED_HTN, "Act", register_miner);
    let start = MinerState {
        gold: 0,
        ..Default::default()
    };
    // Without the look-ahead, `doomed` descends into Spiral2's unbounded
    // recursion until the sanity limit fires and returns the partial
    // plan ["Prime"]. With it, the doomed method is refuted at the frame.
    assert_eq!(bed.plan_forward(&start), vec!["Safe"]);
}

#[test]
fn lookahead_keeps_plans_identical_when_backtracking_suffices() {
    let bed = HtnTestBed::new(WORK_HTN, "Work", register_miner);

    // dig is refuted by the sweep (fuel known 0, nothing writes it before
    // Dig); plain backtracking would find the same plan.
    let start = MinerState {
        fuel: 0,
        gold: 3,
        at_base: true,
        ..Default::default()
    };
    assert_eq!(bed.plan_forward(&start), vec!["Sell"]);

    // dig succeeds through the sweep: Dig's increment keeps gold *known*
    // (deterministic relative write on a known value), so Haul's gold > 0
    // check is evaluated exactly against the propagated state.
    let start = MinerState {
        fuel: 5,
        ..Default::default()
    };
    assert_eq!(bed.plan_forward(&start), vec!["Dig", "Haul"]);
}

/// A method whose preconditions can't be evaluated yet (they read a field an
/// earlier compound task *might* write) must not be refuted by the sweep —
/// unknown fields are "maybe", never "no".
const MAYBE_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "via-gamble" {
        subtasks: [GambleTask, CheckLuck]
    }
    method "fallback" {
        subtasks: [Safe]
    }
}

primitive_task "GambleTask" {
    operator: NoopOperator
    expected_effects: [luck = true]
}

primitive_task "CheckLuck" {
    operator: NoopOperator
    preconditions: [luck == true]
}

primitive_task "Safe" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn lookahead_treats_unknown_fields_as_maybe() {
    let bed = HtnTestBed::new(MAYBE_HTN, "Root", register_miner);
    let start = MinerState {
        luck: false,
        ..Default::default()
    };
    // CheckLuck's precondition reads `luck`, which GambleTask's expected
    // effect (applied exactly by the sweep, since the planner applies expected
    // effects during search too) sets to true — the sequence survives and the
    // gamble is planned rather than pruned.
    assert_eq!(bed.plan_forward(&start), vec!["GambleTask", "CheckLuck"]);
}

// ---------------------------------------------------------------------------
// Backtracking queue restoration (observable with the look-ahead off)
// ---------------------------------------------------------------------------

/// With the look-ahead off, plain MTR backtracking must still restore the task
/// queue when unwinding: a failure in a method's *middle* subtask used to
/// leave the queue stale (siblings from the abandoned choice were executed,
/// and the fallback method was never reached).
const MID_FAILURE_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "broken" {
        subtasks: [Fine, Doomed, Collateral]
    }
    method "fallback" {
        subtasks: [Rescue]
    }
}

primitive_task "Fine" {
    operator: NoopOperator
    effects: [noise = true]
}

primitive_task "Doomed" {
    operator: NoopOperator
    preconditions: [gold > 100]
}

primitive_task "Collateral" {
    operator: NoopOperator
    effects: [noise = false]
}

primitive_task "Rescue" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn backtracking_restores_queue_after_mid_sequence_failure() {
    let domain = bevy_bhtn::parse_htn(MID_FAILURE_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_miner(&mut registry);
    let state = MinerState::default();

    // Look-ahead on: `broken` is refuted at the frame (Doomed needs gold > 100
    // and nothing in its sequence writes gold) — plan goes straight to Rescue.
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    planner.set_lookahead(true);
    assert_eq!(
        planner.plan("Root", &state).task_names(),
        ["Rescue"],
        "look-ahead should refute `broken` without entering it"
    );

    // Look-ahead off: plain backtracking must unwind `broken` cleanly — the
    // abandoned branch's Collateral must NOT leak into the plan, and the
    // fallback method must be reached.
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    planner.set_lookahead(false);
    assert_eq!(
        planner.plan("Root", &state).task_names(),
        ["Rescue"],
        "plain backtracking must discard the abandoned branch and fall through"
    );
}

/// The *lost suffix* variant of the queue-restore bug: the failure happens at
/// a task that is itself the tail of the sequence, so the queue is empty at
/// failure time. Backtracking through the ancestor frames must re-attach the
/// ancestor's suffix (here `Final`, queued behind `Gate`) so that (a) `Final`
/// is re-attempted after `Gate`'s second method and (b) once `Gate` is
/// exhausted, the search unwinds to `Root` and finds `direct`. The old
/// code — which never restored the queue — consumed `Final` on the first
/// failed attempt and terminated with the partial plan [JunkA, JunkB].
const LOST_SUFFIX_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "doomed" {
        subtasks: [Gate, Final]
    }
    method "direct" {
        subtasks: [Ok]
    }
}

compound_task "Gate" {
    method "a" {
        subtasks: [JunkA]
    }
    method "b" {
        subtasks: [JunkB]
    }
}

primitive_task "JunkA" {
    operator: NoopOperator
    effects: [noise = true]
}

primitive_task "JunkB" {
    operator: NoopOperator
    effects: [noise = false]
}

primitive_task "Final" {
    operator: NoopOperator
    preconditions: [gold > 100]
}

primitive_task "Ok" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn backtracking_restores_ancestor_suffix_after_tail_failure() {
    let domain = bevy_bhtn::parse_htn(LOST_SUFFIX_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_miner(&mut registry);
    let state = MinerState::default();

    // Look-ahead on: `doomed` is refuted at the frame (Final needs gold > 100
    // and nothing in the sequence writes gold).
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    assert_eq!(
        planner.plan("Root", &state).task_names(),
        ["Ok"],
        "look-ahead should refute `doomed` without entering it"
    );

    // Look-ahead off: the search must enumerate Gate.a, Gate.b, then unwind
    // past the consumed `Final` and still reach `direct`.
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    planner.set_lookahead(false);
    assert_eq!(
        planner.plan("Root", &state).task_names(),
        ["Ok"],
        "backtracking must restore the ancestor suffix consumed by the failed tail"
    );
}

// ---------------------------------------------------------------------------
// Slice 4: termination-flag dead-ends
// ---------------------------------------------------------------------------
/// Pure infinite recursion with **no impossible precondition anywhere** — the
/// old sweep had nothing to refute (every method applies, no condition ever
/// definitely fails), so the doomed method burned the whole sanity budget.
/// The parse-time `terminating` flag refutes it in one sweep.
const PURE_RECURSION_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "doomed" {
        subtasks: [Prime, Spiral]
    }
    method "direct" {
        subtasks: [Ok]
    }
}

compound_task "Spiral" {
    method "only" {
        subtasks: [Spiral, Tick]
    }
}

primitive_task "Tick" {
    operator: NoopOperator
    effects: [count += 1]
}

primitive_task "Prime" {
    operator: NoopOperator
    effects: [noise = true]
}

primitive_task "Ok" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn terminating_flag_refutes_pure_infinite_recursion() {
    let domain = bevy_bhtn::parse_htn(PURE_RECURSION_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_sweep(&mut registry);
    let state = SweepState::default();

    // The flag itself: Spiral can only refine forever.
    assert!(!domain.task_summary("Spiral").unwrap().terminating);
    assert_eq!(domain.task_summary("Spiral").unwrap().min_yield, usize::MAX);

    // Look-ahead on: `doomed` is refuted at the frame without recursing.
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    assert_eq!(planner.plan("Root", &state).task_names(), ["Ok"]);

    // Look-ahead off: plain backtracking burns the sanity budget and returns
    // the partial plan (the documented fallback semantics).
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    planner.set_lookahead(false);
    assert_eq!(planner.plan("Root", &state).task_names(), ["Prime"]);
}

/// The non-terminating task is buried one level deep: the sweep must refute
/// via `Probe`'s own flag (its only refinement path is non-terminating)
/// without decomposing anything.
const NESTED_RECURSION_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "doomed" {
        subtasks: [Probe, Wrap]
    }
    method "direct" {
        subtasks: [Ok]
    }
}

compound_task "Probe" {
    method "only" {
        subtasks: [Spiral]
    }
}

compound_task "Spiral" {
    method "only" {
        subtasks: [Spiral, Tick]
    }
}

primitive_task "Tick" {
    operator: NoopOperator
    effects: [count += 1]
}

primitive_task "Wrap" {
    operator: NoopOperator
    effects: [noise = false]
}

primitive_task "Ok" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn terminating_flag_refutes_through_nested_compounds() {
    let domain = bevy_bhtn::parse_htn(NESTED_RECURSION_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_sweep(&mut registry);
    let state = SweepState::default();

    assert!(!domain.task_summary("Probe").unwrap().terminating);

    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    assert_eq!(planner.plan("Root", &state).task_names(), ["Ok"]);

    // Off: the doomed branch is entered; no primitive ever executes before
    // the budget burns, so the partial plan is empty.
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    planner.set_lookahead(false);
    assert!(planner.plan("Root", &state).task_names().is_empty());
}

/// A non-terminating *method alternative*: the risky method is tried first,
/// refuted by the sweep at its own commitment, and the viable one is taken —
/// where the old planner burned the budget inside `risky` and returned an
/// empty partial plan.
const MIXED_TERMINATION_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Pick" {
    method "risky" {
        subtasks: [Spiral]
    }
    method "safe" {
        subtasks: [Ok]
    }
}

compound_task "Spiral" {
    method "only" {
        subtasks: [Spiral, Tick]
    }
}

primitive_task "Tick" {
    operator: NoopOperator
    effects: [count += 1]
}

primitive_task "Ok" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn non_terminating_method_skipped_among_viable_ones() {
    let domain = bevy_bhtn::parse_htn(MIXED_TERMINATION_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_sweep(&mut registry);
    let state = SweepState::default();

    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    assert_eq!(planner.plan("Pick", &state).task_names(), ["Ok"]);

    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    planner.set_lookahead(false);
    assert!(planner.plan("Pick", &state).task_names().is_empty());
}

/// Over-refutation guard: *terminating* recursion must keep its flag and keep
/// planning — a wrong `terminating = false` here would collapse the plan.
const TERMINATING_RECURSION_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "m" {
        subtasks: [Loop, Check]
    }
}

compound_task "Loop" {
    method "done" {
        subtasks: [Tick]
    }
    method "again" {
        subtasks: [Loop, Tick]
    }
}

primitive_task "Tick" {
    operator: NoopOperator
    effects: [count += 1]
}

primitive_task "Check" {
    operator: NoopOperator
    preconditions: [count > 0]
}
"#;

#[test]
fn terminating_recursion_not_flagged() {
    let domain = bevy_bhtn::parse_htn(TERMINATING_RECURSION_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_sweep(&mut registry);
    let state = SweepState::default();

    let summary = domain.task_summary("Loop").unwrap();
    assert!(summary.terminating);
    assert_eq!(summary.min_yield, 1);

    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    assert_eq!(planner.plan("Root", &state).task_names(), ["Tick", "Check"]);
}

// ---------------------------------------------------------------------------
// Slice 5: per-condition precomputed read indices
// ---------------------------------------------------------------------------

/// Two preconditions on one primitive with **different** known/unknown status:
/// `x` is unknown (a compound might write it) but `y` is known-false. The
/// per-condition read indices must prune via `y == 5` alone — a union-based
/// regression ("any read unknown → maybe") would treat the whole primitive as
/// maybe, commit the doomed method, and burn the sanity budget on the gates.
fn gates_with_unknown_x_domain(gates: usize) -> String {
    let mut s = String::from(
        "schema {\n    version: 0.1.0\n}\n\n\
         compound_task \"Root\" {\n\
         \x20   method \"doomed\" {\n        subtasks: [XWriter",
    );
    for i in 0..gates {
        s.push_str(&format!(", Gate{i}"));
    }
    s.push_str(", Checker]\n    }\n");
    s.push_str(
        "    method \"direct\" {\n        subtasks: [Ok]\n    }\n}\n\n\
         compound_task \"XWriter\" {\n\
         \x20   method \"only\" {\n        subtasks: [SetX]\n    }\n}\n\n\
         primitive_task \"SetX\" {\n\
         \x20   operator: NoopOperator\n\
         \x20   effects: [x = 1]\n}\n\n",
    );
    for i in 0..gates {
        s.push_str(&format!(
            "compound_task \"Gate{i}\" {{\n\
             \x20   method \"a\" {{\n        subtasks: [JunkA{i}]\n    }}\n\
             \x20   method \"b\" {{\n        subtasks: [JunkB{i}]\n    }}\n}}\n\n\
             primitive_task \"JunkA{i}\" {{\n\
             \x20   operator: NoopOperator\n\
             \x20   effects: [noise = true]\n}}\n\n\
             primitive_task \"JunkB{i}\" {{\n\
             \x20   operator: NoopOperator\n\
             \x20   effects: [noise = false]\n}}\n\n"
        ));
    }
    s.push_str(
        "primitive_task \"Checker\" {\n\
         \x20   operator: NoopOperator\n\
         \x20   preconditions: [x == 1, y == 5]\n}\n\n\
         primitive_task \"Ok\" {\n\
         \x20   operator: NoopOperator\n\
         \x20   effects: [gold = 1]\n}\n",
    );
    s
}

#[test]
fn per_condition_reads_prune_despite_unknown_sibling_field() {
    let domain = bevy_bhtn::parse_htn(&gates_with_unknown_x_domain(12)).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_sweep(&mut registry);
    let state = SweepState::default();

    // On: `y == 5` is definitely false (y known 0) even though `x == 1` is
    // maybe — the doomed method is refuted at the frame.
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    assert_eq!(planner.plan("Root", &state).task_names(), ["Ok"]);

    // Off: the doomed method is entered (first task SetX) and burns the
    // budget on the gate enumeration.
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    planner.set_lookahead(false);
    assert_eq!(planner.plan("Root", &state).task_names()[0], "SetX");
}

/// An identifier condition (`a == b`) with one unknown operand must be
/// "maybe", never evaluated against the stale clone value — the gamble path
/// succeeds at runtime and must be planned.
const IDENTIFIER_MAYBE_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "gamble" {
        subtasks: [AWriter, Cmp]
    }
    method "fallback" {
        subtasks: [Safe]
    }
}

compound_task "AWriter" {
    method "only" {
        subtasks: [SetA]
    }
}

primitive_task "SetA" {
    operator: NoopOperator
    effects: [a = 5]
}

primitive_task "Cmp" {
    operator: NoopOperator
    preconditions: [a == b]
}

primitive_task "Safe" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn identifier_condition_with_unknown_field_is_maybe() {
    let domain = bevy_bhtn::parse_htn(IDENTIFIER_MAYBE_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_sweep(&mut registry);
    // a=0 (stale clone would compare 0 == 5 → false), b=5; SetA makes them
    // equal at runtime.
    let state = SweepState {
        b: 5,
        ..Default::default()
    };

    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    assert_eq!(planner.plan("Root", &state).task_names(), ["SetA", "Cmp"]);
}

/// A negated condition on an unknown field must likewise be "maybe": the
/// stale clone value (a = 5) would fail `a != 5`, but the runtime value (7)
/// passes.
const NOTTED_MAYBE_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "gamble" {
        subtasks: [AWriter, Cmp]
    }
    method "fallback" {
        subtasks: [Safe]
    }
}

compound_task "AWriter" {
    method "only" {
        subtasks: [SetA]
    }
}

primitive_task "SetA" {
    operator: NoopOperator
    effects: [a = 7]
}

primitive_task "Cmp" {
    operator: NoopOperator
    preconditions: [a != 5]
}

primitive_task "Safe" {
    operator: NoopOperator
    effects: [gold = 1]
}
"#;

#[test]
fn notted_condition_on_unknown_field_is_maybe() {
    let domain = bevy_bhtn::parse_htn(NOTTED_MAYBE_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_sweep(&mut registry);
    let state = SweepState {
        a: 5,
        ..Default::default()
    };

    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    assert_eq!(planner.plan("Root", &state).task_names(), ["SetA", "Cmp"]);
}

// ---------------------------------------------------------------------------
// Occurrence-scoped pins
// ---------------------------------------------------------------------------

#[derive(Reflect, Clone, Debug, Default)]
struct PinState {
    phase: i32,
    done: bool,
}

fn register_pin(registry: &mut TypeRegistry) {
    registry.register::<PinState>();
}

/// The same compound task (`Twice`) is decomposed **twice** in one method
/// sequence, under different states: the first occurrence can only use its
/// `phase == 0` method, the second only its `phase == 1` method. The sweep
/// legitimately pins both occurrences — but each pin must travel with its own
/// occurrence. Keying pins by task index (the original implementation) applied
/// the *last* pin to both occurrences, exhausting the first illegally and
/// collapsing the whole plan.
const OCCURRENCE_PIN_HTN: &str = r#"
schema {
    version: 0.1.0
}

compound_task "Root" {
    method "twice" {
        subtasks: [Twice, Bump, Twice, Verify]
    }
}

compound_task "Twice" {
    method "phase zero" {
        preconditions: [phase == 0]
        subtasks: [Step0]
    }
    method "phase one" {
        preconditions: [phase == 1]
        subtasks: [Step1]
    }
}

primitive_task "Step0" {
    operator: NoopOperator
    effects: [done = true]
}

primitive_task "Step1" {
    operator: NoopOperator
    effects: [done = true]
}

primitive_task "Bump" {
    operator: NoopOperator
    effects: [phase = 1]
}

primitive_task "Verify" {
    operator: NoopOperator
    preconditions: [done == true]
}
"#;

#[test]
fn pins_apply_per_occurrence_not_per_task_index() {
    let bed = HtnTestBed::new(OCCURRENCE_PIN_HTN, "Root", register_pin);
    let start = PinState::default();

    // The sweep pins occurrence 1 to `phase zero` and occurrence 2 to
    // `phase one`; both pins must hold simultaneously.
    assert_eq!(
        bed.plan_forward(&start),
        vec!["Step0", "Bump", "Step1", "Verify"],
        "each Twice occurrence must use its own pinned method"
    );

    // Without the look-ahead there are no pins at all; plain backtracking
    // must find the same plan.
    let domain = bevy_bhtn::parse_htn(OCCURRENCE_PIN_HTN).expect("parses");
    let mut registry = bevy_reflect::TypeRegistry::default();
    register_pin(&mut registry);
    let mut planner = bevy_bhtn::planner::HtnPlanner::new(&domain, &registry);
    planner.set_lookahead(false);
    assert_eq!(
        planner.plan("Root", &start).task_names(),
        ["Step0", "Bump", "Step1", "Verify"],
        "plain backtracking must agree with the pinned plan"
    );
}
