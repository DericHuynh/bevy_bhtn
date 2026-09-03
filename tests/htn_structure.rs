//! Tests for the bake-time domain structure analysis (slice of the
//! summaries work): per-task termination, minimum yield, and the
//! recursion-shape flags (recursive / self-embedding / tail-recursive), plus
//! the planner's min-yield budget pruning.
//!
//! The flags follow Toad's classification (JAIR 2024): a task is
//! *right-generating* when it can appear at a non-last position of its own
//! refinement (→ not tail-recursive), *left-generating* when it can appear at
//! a non-first position, and *self-embedding* when both hold (the
//! context-free core of a domain). Min yield is the shortest primitive
//! sequence any refinement produces — a lower bound on decomposition work,
//! used by the look-ahead sweep to refute methods that cannot finish within
//! the planner's step budget.
//!
//! Domains are task-function graphs built inline; the parameterized deep-chain
//! generator is baked as a macro-generated static graph (task-function
//! identity needs static graphs, so the two tested depths are pre-baked).

mod common;
use common::{Gold, Noise};

use bevy_bhtn::planner::{HtnPlanner, Plan};
use bevy_bhtn::state::PlanState;
use bevy_bhtn::{HtnDomain, TaskBuilder, TaskFn, TaskSummary};
use bevy_ecs::prelude::*;

/// A task fn's item type cannot be named directly, so the lookup-by-type API
/// is reached through these inference helpers: the fn value pins `F` to the
/// fn item's unique type, resolved through the baked `TypeId` index.
fn plan_of<F: TaskFn>(planner: &mut HtnPlanner<'_>, _f: F, state: &PlanState) -> Plan {
    planner.plan(_f, state)
}

fn summary_of<F: TaskFn>(domain: &HtnDomain, _f: F) -> Option<&TaskSummary> {
    domain.task_summary(_f)
}

/// A generic work counter (former `count` field).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct Count(pub i32);

/// One domain exercising every flag: flat work, terminating tail-recursive
/// and self-embedding recursion, and non-terminating recursion. The task
/// functions live in a module so tests can name them for type-based lookup.
mod structure_tasks {
    use super::*;

    // Synthetic root so every top-level task is baked (from_root records the
    // transitive .then graph only); tests plan the individual tasks by type.
    pub fn root(task: &mut TaskBuilder) {
        task.branch().then(work);
        task.branch().then(loop_);
        task.branch().then(spiral);
        task.branch().then(descend);
        task.branch().then(tail);
        task.branch().then(emb);
    }
    pub fn work(task: &mut TaskBuilder) {
        task.branch().then(dig).then(haul);
        task.branch().then(sell);
    }
    pub fn dig(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 > 0)
            .effect(|count: &mut Count| count.0 += 1);
    }
    pub fn haul(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn sell(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn loop_(task: &mut TaskBuilder) {
        task.branch().then(tick);
        task.branch().then(loop_).then(tick);
    }
    pub fn tick(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral).then(tick);
    }
    pub fn descend(task: &mut TaskBuilder) {
        task.branch().then(descend).then(eat);
        task.branch().then(eat);
    }
    pub fn eat(task: &mut TaskBuilder) {
        task.effect(|count: &mut Count| count.0 += 1);
    }
    pub fn tail(task: &mut TaskBuilder) {
        task.branch().then(tick);
        task.branch().then(tick).then(tail);
    }
    pub fn emb(task: &mut TaskBuilder) {
        task.branch().then(tick);
        task.branch().then(emb).then(tick).then(emb);
    }
}

fn structure_domain() -> HtnDomain {
    HtnDomain::from_root(structure_tasks::root)
        .build()
        .expect("structure domain is well-formed")
}

#[test]
fn structure_analysis_pins_flags_and_min_yield() {
    let domain = structure_domain();

    // Flat task: cheapest method is `sell` (one primitive); no recursion, so
    // tail-recursive holds vacuously.
    let work = summary_of(&domain, structure_tasks::work).expect("summary");
    assert_eq!(work.min_yield, 1);
    assert!(work.terminating);
    assert!(!work.recursive);
    assert!(!work.self_embedding);
    assert!(work.tail_recursive);

    // Terminating recursion with the recursive task always non-last
    // (⟨loop_, tick⟩): right-generating → not tail-recursive, but material
    // never precedes it → not self-embedding.
    let loop_summary = summary_of(&domain, structure_tasks::loop_).expect("summary");
    assert!(loop_summary.terminating);
    assert_eq!(loop_summary.min_yield, 1);
    assert!(loop_summary.recursive);
    assert!(!loop_summary.self_embedding);
    assert!(!loop_summary.tail_recursive);

    // Non-terminating recursion: no finite refinement at all.
    let spiral = summary_of(&domain, structure_tasks::spiral).expect("summary");
    assert!(!spiral.terminating);
    assert_eq!(spiral.min_yield, usize::MAX);
    assert!(spiral.recursive);

    // Same shape as loop_ via a different method set.
    let descend = summary_of(&domain, structure_tasks::descend).expect("summary");
    assert!(descend.terminating);
    assert_eq!(descend.min_yield, 1);
    assert!(descend.recursive);
    assert!(!descend.self_embedding);
    assert!(!descend.tail_recursive);

    // Tail recursion (⟨tick, tail⟩): the recursive task is always last →
    // tail-recursive, left-generating only → not self-embedding.
    let tail = summary_of(&domain, structure_tasks::tail).expect("summary");
    assert!(tail.terminating);
    assert_eq!(tail.min_yield, 1);
    assert!(tail.recursive);
    assert!(!tail.self_embedding);
    assert!(tail.tail_recursive);

    // Self-embedding (⟨emb, tick, emb⟩): material on BOTH sides — the
    // context-free core. This is Toad's exact-translation boundary.
    let emb = summary_of(&domain, structure_tasks::emb).expect("summary");
    assert!(emb.terminating);
    assert_eq!(emb.min_yield, 1);
    assert!(emb.recursive);
    assert!(emb.self_embedding);
    assert!(!emb.tail_recursive);
}

// ---------------------------------------------------------------------------
// Deep-chain generator (macro-baked static graphs)
// ---------------------------------------------------------------------------

/// Generates one chain link + its primitive. The token stream is
/// `c0 p0 c1 c1 p1 c2 … cN-1 pN-1` — each link's `next` is the following
/// link's own name, and the final two tokens are the base link.
macro_rules! chain_tasks {
    ($last_link:ident $last_prim:ident) => {
        pub fn $last_link(task: &mut TaskBuilder) {
            task.branch().then($last_prim);
        }
        pub fn $last_prim(task: &mut TaskBuilder) {
            task.effect(|noise: &mut Noise| noise.0 = true);
        }
    };
    ($link:ident $prim:ident $next:ident $($rest:tt)*) => {
        pub fn $link(task: &mut TaskBuilder) {
            task.branch().then($prim).then($next);
        }
        pub fn $prim(task: &mut TaskBuilder) {
            task.effect(|noise: &mut Noise| noise.0 = true);
        }
        chain_tasks!($($rest)*);
    };
}

/// Bakes a deep-chain domain module: `root` (deep method → `c0`, quick method
/// → `ok`) plus the linear chain, and a constructor fn named after the module.
macro_rules! deep_chain_mod {
    ($mod_name:ident, $($tokens:tt)*) => {
        mod $mod_name {
            use super::{Gold, Noise};
            use bevy_bhtn::TaskBuilder;

            pub fn root(task: &mut TaskBuilder) {
                task.branch().then(c0);
                task.branch().then(ok);
            }
            pub fn ok(task: &mut TaskBuilder) {
                task.effect(|gold: &mut Gold| gold.0 = 1);
            }
            chain_tasks!($($tokens)*);
        }

        fn $mod_name() -> HtnDomain {
            HtnDomain::from_root($mod_name::root)
                .build()
                .expect("deep-chain domain is well-formed")
        }
    };
}

deep_chain_mod!(
    chain30,
    c0 p0 c1
    c1 p1 c2
    c2 p2 c3
    c3 p3 c4
    c4 p4 c5
    c5 p5 c6
    c6 p6 c7
    c7 p7 c8
    c8 p8 c9
    c9 p9 c10
    c10 p10 c11
    c11 p11 c12
    c12 p12 c13
    c13 p13 c14
    c14 p14 c15
    c15 p15 c16
    c16 p16 c17
    c17 p17 c18
    c18 p18 c19
    c19 p19 c20
    c20 p20 c21
    c21 p21 c22
    c22 p22 c23
    c23 p23 c24
    c24 p24 c25
    c25 p25 c26
    c26 p26 c27
    c27 p27 c28
    c28 p28 c29
    c29 p29
);

deep_chain_mod!(
    chain5,
    c0 p0 c1
    c1 p1 c2
    c2 p2 c3
    c3 p3 c4
    c4 p4
);

/// A `depth`-deep compound chain: `min_yield(c0) = depth`. With a step budget
/// of 10 the method cannot possibly finish, and the min-yield check must
/// refute it at the frame — where the old planner (without the check) burned
/// the budget executing the chain's prefix and returned a partial plan.
///
/// Task-function identity is static, so only the two tested depths (5 and 30)
/// are baked; the generator keeps its old signature and dispatches on them.
fn deep_chain_domain(depth: usize) -> HtnDomain {
    match depth {
        5 => chain5(),
        30 => chain30(),
        _ => panic!("only depths 5 and 30 are baked as static task graphs"),
    }
}

#[test]
fn min_yield_refutes_method_that_cannot_finish_within_budget() {
    let domain = deep_chain_domain(30);
    let state = PlanState::build(&domain.components).finish();

    assert_eq!(summary_of(&domain, chain30::c0).unwrap().min_yield, 30);

    // On: the deep method needs ≥ 30 steps but only ~9 remain — refuted at
    // the frame, and the quick method plans.
    let mut planner = HtnPlanner::new(&domain);
    planner.set_sanity_limit(10);
    assert_eq!(
        plan_of(&mut planner, chain30::root, &state).task_names(),
        ["ok"]
    );

    // Off: plain backtracking enters the chain and burns the budget on its
    // prefix (the documented fallback semantics).
    let mut planner = HtnPlanner::new(&domain);
    planner.set_sanity_limit(10);
    planner.set_lookahead(false);
    let plan = plan_of(&mut planner, chain30::root, &state);
    let names = plan.task_names();
    assert_eq!(names[0], "p0");
    assert!(!names.contains(&"ok".into()));
}

#[test]
fn min_yield_does_not_refute_within_budget() {
    // Same shape, 5 deep, default budget: the chain must plan to completion —
    // a wrong min-yield computation (e.g. counting compounds twice, or
    // flagging recursive tasks as infinite) would over-refute here.
    let domain = deep_chain_domain(5);
    let state = PlanState::build(&domain.components).finish();

    assert_eq!(summary_of(&domain, chain5::c0).unwrap().min_yield, 5);

    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, chain5::root, &state).task_names(),
        ["p0", "p1", "p2", "p3", "p4"]
    );

    // And the recursive domains from the structure domain keep planning under
    // the default budget (their min yields are small despite recursion).
    let domain = structure_domain();
    let state = PlanState::build(&domain.components).finish();
    let mut planner = HtnPlanner::new(&domain);
    assert_eq!(
        plan_of(&mut planner, structure_tasks::loop_, &state).task_names(),
        ["tick"]
    );
}
