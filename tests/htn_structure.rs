//! Tests for the parse-time domain structure analysis (slice of the
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

mod common;

use bevy_reflect::Reflect;
use bevy_reflect::TypeRegistry;
use cdda_htn::planner::HtnPlanner;

#[derive(Reflect, Clone, Debug, Default)]
struct ChainState {
    count: i32,
    gold: i32,
    noise: bool,
}

fn register_chain(registry: &mut TypeRegistry) {
    registry.register::<ChainState>();
}

/// One domain exercising every flag: flat work, terminating tail-recursive
/// and self-embedding recursion, and non-terminating recursion.
const STRUCTURE_HTN: &str = r#"
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
    preconditions: [gold > 0]
    effects: [count += 1]
}

primitive_task "Haul" {
    operator: NoopOperator
    effects: [count += 1]
}

primitive_task "Sell" {
    operator: NoopOperator
    effects: [count += 1]
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

compound_task "Spiral" {
    method "only" {
        subtasks: [Spiral, Tick]
    }
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
    effects: [count += 1]
}

compound_task "Tail" {
    method "base" {
        subtasks: [Tick]
    }
    method "recurse" {
        subtasks: [Tick, Tail]
    }
}

compound_task "Emb" {
    method "base" {
        subtasks: [Tick]
    }
    method "cycle" {
        subtasks: [Emb, Tick, Emb]
    }
}
"#;

#[test]
fn structure_analysis_pins_flags_and_min_yield() {
    let domain = cdda_htn::parse_htn(STRUCTURE_HTN).expect("parses");
    let summary = |name: &str| domain.task_summary(name).expect("summary");

    // Flat task: cheapest method is `sell` (one primitive); no recursion, so
    // tail-recursive holds vacuously.
    let work = summary("Work");
    assert_eq!(work.min_yield, 1);
    assert!(work.terminating);
    assert!(!work.recursive);
    assert!(!work.self_embedding);
    assert!(work.tail_recursive);

    // Terminating recursion with the recursive task always non-last
    // (⟨Loop, Tick⟩): right-generating → not tail-recursive, but material
    // never precedes it → not self-embedding.
    let loop_summary = summary("Loop");
    assert!(loop_summary.terminating);
    assert_eq!(loop_summary.min_yield, 1);
    assert!(loop_summary.recursive);
    assert!(!loop_summary.self_embedding);
    assert!(!loop_summary.tail_recursive);

    // Non-terminating recursion: no finite refinement at all.
    let spiral = summary("Spiral");
    assert!(!spiral.terminating);
    assert_eq!(spiral.min_yield, usize::MAX);
    assert!(spiral.recursive);

    // Same shape as Loop via a different method set.
    let descend = summary("Descend");
    assert!(descend.terminating);
    assert_eq!(descend.min_yield, 1);
    assert!(descend.recursive);
    assert!(!descend.self_embedding);
    assert!(!descend.tail_recursive);

    // Tail recursion (⟨Tick, Tail⟩): the recursive task is always last →
    // tail-recursive, left-generating only → not self-embedding.
    let tail = summary("Tail");
    assert!(tail.terminating);
    assert_eq!(tail.min_yield, 1);
    assert!(tail.recursive);
    assert!(!tail.self_embedding);
    assert!(tail.tail_recursive);

    // Self-embedding (⟨Emb, Tick, Emb⟩): material on BOTH sides — the
    // context-free core. This is Toad's exact-translation boundary.
    let emb = summary("Emb");
    assert!(emb.terminating);
    assert_eq!(emb.min_yield, 1);
    assert!(emb.recursive);
    assert!(emb.self_embedding);
    assert!(!emb.tail_recursive);
}

/// A 30-deep compound chain: `min_yield(C0) = 30`. With a step budget of 10
/// the method cannot possibly finish, and the min-yield check must refute it
/// at the frame — where the old planner (without the check) burned the budget
/// executing the chain's prefix and returned a partial plan.
fn deep_chain_domain(depth: usize) -> String {
    let mut s = String::from(
        "schema {\n    version: 0.1.0\n}\n\n\
         compound_task \"Root\" {\n\
         \x20   method \"deep\" {\n        subtasks: [C0]\n    }\n\
         \x20   method \"quick\" {\n        subtasks: [Ok]\n    }\n}\n\n",
    );
    for i in 0..depth {
        let body = if i + 1 == depth {
            format!("P{i}")
        } else {
            format!("P{i}, C{}", i + 1)
        };
        s.push_str(&format!(
            "compound_task \"C{i}\" {{\n    method \"only\" {{\n        subtasks: [{body}]\n    }}\n}}\n\n"
        ));
        s.push_str(&format!(
            "primitive_task \"P{i}\" {{\n    operator: NoopOperator\n    effects: [noise = true]\n}}\n\n"
        ));
    }
    s.push_str("primitive_task \"Ok\" {\n    operator: NoopOperator\n    effects: [gold = 1]\n}\n");
    s
}

#[test]
fn min_yield_refutes_method_that_cannot_finish_within_budget() {
    let domain = cdda_htn::parse_htn(&deep_chain_domain(30)).expect("parses");
    let mut registry = TypeRegistry::default();
    register_chain(&mut registry);
    let state = ChainState::default();

    assert_eq!(domain.task_summary("C0").unwrap().min_yield, 30);

    // On: the deep method needs ≥ 30 steps but only ~9 remain — refuted at
    // the frame, and the quick method plans.
    let mut planner = HtnPlanner::new(&domain, &registry);
    planner.set_sanity_limit(10);
    assert_eq!(planner.plan("Root", &state).task_names(), ["Ok"]);

    // Off: plain backtracking enters the chain and burns the budget on its
    // prefix (the documented fallback semantics).
    let mut planner = HtnPlanner::new(&domain, &registry);
    planner.set_sanity_limit(10);
    planner.set_lookahead(false);
    let plan = planner.plan("Root", &state);
    let names = plan.task_names();
    assert_eq!(names[0], "P0");
    assert!(!names.contains(&"Ok".into()));
}

#[test]
fn min_yield_does_not_refute_within_budget() {
    // Same shape, 5 deep, default budget: the chain must plan to completion —
    // a wrong min-yield computation (e.g. counting compounds twice, or
    // flagging recursive tasks as infinite) would over-refute here.
    let domain = cdda_htn::parse_htn(&deep_chain_domain(5)).expect("parses");
    let mut registry = TypeRegistry::default();
    register_chain(&mut registry);
    let state = ChainState::default();

    assert_eq!(domain.task_summary("C0").unwrap().min_yield, 5);

    let mut planner = HtnPlanner::new(&domain, &registry);
    assert_eq!(
        planner.plan("Root", &state).task_names(),
        ["P0", "P1", "P2", "P3", "P4"]
    );

    // And the recursive domains from STRUCTURE_HTN keep planning under the
    // default budget (their min yields are small despite recursion).
    let domain = cdda_htn::parse_htn(STRUCTURE_HTN).expect("parses");
    let mut registry = TypeRegistry::default();
    register_chain(&mut registry);
    let mut planner = HtnPlanner::new(&domain, &registry);
    assert_eq!(
        planner.plan("Loop", &ChainState::default()).task_names(),
        ["Tick"]
    );
}
