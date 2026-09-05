//! GTN (goal-task network) extensions compiled into the plain HTN network.
//!
//! [`Alford et al. (IJCAI-16)`] define three extensions over plain HTN —
//! goal decomposition, task insertion, and task sharing — and prove they are
//! all *plan-preserving translations* back to plain HTN: no planner surgery
//! is required, only network construction. This module implements the two
//! constructions that fit bevy_bhtn's design as **bake-time transforms** on
//! the recording (`DomainBuilder`); the planner, look-ahead, summaries, and
//! driver consume the compiled network exactly as they would a hand-written
//! domain.
//!
//! # Task sharing — the `fin_o` construction (Theorem 4.8)
//!
//! Sharing means "this primitive runs at most once per plan": two tasks that
//! both need `equip_boots` should not run it twice. Construction 4.8 compiles
//! each shared primitive `o` into:
//!
//! - a marker component slot ([`SharedMarks`], one boolean per shared task);
//! - an **expected** effect appended to `o` itself that sets the marker
//!   (expected effects are applied during search but never committed to the
//!   real entity — the marker is planning-simulation state, exactly like the
//!   crate's `.expected(...)` contract);
//! - a wrapper compound `gtn/shared:o` with two methods, **in this order**:
//!   1. `done` — empty body, precondition "marker set **or** `o`'s own
//!      precondition gate has closed" (the shared case: `o` already ran —
//!      in this plan, or for real in an earlier one);
//!   2. `do` — body `[o]` (the first occurrence).
//!
//! Every method body that referenced `o` now references the wrapper. Under
//! first-match selection the first occurrence takes `do` (the marker is
//! unset, the gate open), sets the marker in simulation, and every later
//! occurrence takes the empty `done` — the plan contains `o` exactly once,
//! under its normal user-facing name. The gate clause is what keeps this
//! stable across the driver's drift-replans: a fresh plan's marker is
//! reset, but a committed real-world gate (e.g. `packed == true`) persists,
//! so re-planning takes `done` instead of re-running `o`. Ungated
//! primitives (no preconditions) therefore share only within a single plan.
//!
//! **Execution contract.** The marker never reaches the real world, so when
//! the driver re-validates a `done` step it fails and the agent replans from
//! the true state. This is benign *iff* the shared primitive follows the
//! natural "done once" pattern: after `o` really executes, its own
//! preconditions no longer hold (e.g. `!equipped` fails once equipped), so
//! the replanned route cannot re-run it. Sharing a repeatable action (one
//! whose preconditions stay true) is a modeling error — its real effects
//! would apply again.
//!
//! # Task insertion — gap compilation
//!
//! Insertion (GTN_I) lets any applicable primitive run between the steps of
//! any method — plan repair. [`DomainBuilder::with_insertion`](crate::domain::DomainBuilder::with_insertion)
//! adds one synthetic compound
//! `gtn/insert` with the **empty stop method first** and one
//! `[candidate, insert]` method per domain primitive, then splices
//! `[insert, t1, insert, …, tn, insert]` into every total-order method body.
//!
//! The stop-first order is the load-bearing choice: under first-match
//! selection the search commits "no insertion" and only explores insertions
//! on backtrack — the plain plan is still found first, and insertion acts as
//! *repair*, not as plan pollution. (The reverse order would maximally
//! pollute every plan; GTN_I's full interleaving semantics is NEXPTIME-complete
//! precisely because the search cannot be spared this.) Candidates reference
//! the *shared* wrapper for shared primitives, so inserted steps respect
//! sharing. Partially-ordered bodies are left untouched (their scheduling
//! edges are position-keyed; splicing gaps would shift every member).
//!
//! # What is *not* here
//!
//! Goal decomposition (HGN-style inline subgoals) would need heterogeneous
//! method bodies (`MethodItem::Task | Subgoal`) — a real planner change, not
//! a compilation. The paper's general translation (Construction 4.1) exists
//! on paper, but with opaque closure preconditions the value-based parts
//! cannot be compiled faithfully; the two constructions above only rely on
//! the component-slot model, which is why they compile cleanly.
//!
//! [`Alford et al. (IJCAI-16)`]: https://doi.org/10.5555/3060852.3060867

use std::collections::HashMap;

use bevy_ecs::prelude::Component;
use smallvec::{smallvec, SmallVec};

use crate::error::HtnResult;
use crate::state::PlanState;
use crate::tasks::{Effect, MethodProto, Precondition, Recorder, SubtaskRef, TaskProto};

/// Per-share completion markers, indexed by the shared task's ordinal (the
/// order [`share_task`](crate::HtnDomain::from_root)-side requests were made).
///
/// One registry slot serves every shared task. The flags live only in the
/// planning scratchpad: the marker effects are *expected* effects, so they are
/// simulated during search and never committed to the real entity.
#[derive(Component, Clone, Default, Debug, PartialEq)]
pub struct SharedMarks(pub Vec<bool>);

/// Leak a synthetic task's name — baked domains are built once, and task
/// names are `&'static str` throughout.
fn leak_name(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

/// Compile the requested task sharings into the recording. Called from
/// `DomainBuilder::build` before validation; returns the map from each
/// shared primitive's ref to its wrapper (insertion routes candidates
/// through it). See the [module docs](self).
pub(crate) fn apply_sharing(
    rec: &mut Recorder,
) -> HtnResult<HashMap<SubtaskRef, (SubtaskRef, &'static str)>> {
    if rec.shares.is_empty() {
        return Ok(HashMap::new());
    }
    let requests = std::mem::take(&mut rec.shares);
    let marks_slot = rec.registry.index::<SharedMarks>();
    // original task ref -> (shared wrapper ref, wrapper name)
    let mut wrapped: HashMap<SubtaskRef, (SubtaskRef, &'static str)> = HashMap::new();

    for (share_id, req) in requests.iter().enumerate() {
        let Some(&orig_idx) = rec.index_of.get(req) else {
            return Err(crate::error::HtnError::builder(
                "share_task: task function not recorded (was it referenced via `.then`?)"
                    .to_string(),
            ));
        };
        let orig_name = rec.tasks[orig_idx].1;
        if !matches!(rec.tasks[orig_idx].2, TaskProto::Primitive { .. }) {
            return Err(crate::error::HtnError::builder(format!(
                "share_task: `{orig_name}` is not a primitive task"
            )));
        }

        let id = share_id;
        // The completion marker rides the shared primitive itself as an
        // *expected* effect — simulated during search, never committed to the
        // world. Plans therefore keep the user-facing primitive name and
        // contain the shared step exactly once.
        if let TaskProto::Primitive {
            expected_effects, ..
        } = &mut rec.tasks[orig_idx].2
        {
            expected_effects.push(Effect::new(
                smallvec![marks_slot],
                Box::new(move |state: &mut PlanState| {
                    let marks = &mut state.get_mut::<SharedMarks>(marks_slot).0;
                    if marks.len() <= id {
                        marks.resize(id + 1, false);
                    }
                    marks[id] = true;
                }),
            ));
        }

        // The wrapper compound: `done` (marker set) first, then `do`.
        //
        // `done` applies when the marker is set — this plan already ran `o` —
        // OR when `o`'s own gate reports "already done" (any of its
        // preconditions fails). The second clause is what keeps the wrapper
        // consistent across the driver's drift-replans: the marker resets
        // with every fresh plan, but a committed real-world gate (e.g.
        // `packed == true`) persists, so re-planning takes the empty `done`
        // instead of re-running `o` or failing the branch. Ungated
        // primitives therefore share only within a single plan.
        let comp_ref = SubtaskRef::Synthetic(rec.next_synthetic);
        rec.next_synthetic += 1;
        let comp_name = leak_name(format!("gtn/shared:{orig_name}"));
        let o_preconds: Vec<Precondition> = match &rec.tasks[orig_idx].2 {
            TaskProto::Primitive { preconditions, .. } => preconditions.clone(),
            _ => unreachable!("checked above"),
        };
        let mut done_reads: SmallVec<[usize; 4]> = smallvec![marks_slot];
        for p in &o_preconds {
            for &r in p.reads() {
                if !done_reads.contains(&r) {
                    done_reads.push(r);
                }
            }
        }
        let done_pre = Precondition::new(
            done_reads,
            std::sync::Arc::new(move |state: &PlanState| {
                if state
                    .get::<SharedMarks>(marks_slot)
                    .0
                    .get(id)
                    .copied()
                    .unwrap_or(false)
                {
                    return true;
                }
                // "Already done" in the real world: the shared primitive's
                // own precondition gate has closed.
                o_preconds.iter().any(|c| !c.evaluate(state))
            }),
        );
        let mom = MethodProto {
            name: Some(leak_name("done".to_string())),
            utility: None,
            preconditions: vec![done_pre],
            subtasks: Vec::new(),
            unordered: false,
            edges: Vec::new(),
            pause_positions: Vec::new(),
        };
        let moa = MethodProto {
            name: Some(leak_name("do".to_string())),
            utility: None,
            preconditions: Vec::new(),
            subtasks: vec![(*req, orig_name, true)],
            unordered: false,
            edges: Vec::new(),
            pause_positions: Vec::new(),
        };
        let comp = TaskProto::Compound {
            methods: vec![mom, moa],
            policy: crate::domain::SelectionPolicy::default(),
        };

        rec.tasks.push((comp_ref, comp_name, comp));
        rec.index_of.insert(comp_ref, rec.tasks.len() - 1);
        wrapped.insert(*req, (comp_ref, comp_name));
    }

    // Rewrite every non-synthetic method body: references to a shared
    // primitive become references to its wrapper.
    for (tref, _, proto) in rec.tasks.iter_mut() {
        if matches!(tref, SubtaskRef::Synthetic(_)) {
            continue;
        }
        let TaskProto::Compound { methods, .. } = proto else {
            continue;
        };
        for m in methods.iter_mut() {
            for (r, name, _) in m.subtasks.iter_mut() {
                if let Some((comp_ref, comp_name)) = wrapped.get(r) {
                    *r = *comp_ref;
                    *name = comp_name;
                }
            }
        }
    }
    Ok(wrapped)
}

/// Compile task insertion into the recording. Called from
/// `DomainBuilder::build` after [`apply_sharing`]; see the [module docs](self).
pub(crate) fn apply_insertion(
    rec: &mut Recorder,
    wrapped: &HashMap<SubtaskRef, (SubtaskRef, &'static str)>,
) -> HtnResult<()> {
    if !rec.insertion || rec.insertables.is_empty() {
        return Ok(());
    }

    // Insertion candidates: the explicitly registered insertable tasks only
    // (an ungated free-for-all is an unbounded insertion well), routed
    // through their shared wrapper when one exists.
    let mut candidates: Vec<(SubtaskRef, &'static str)> = Vec::new();
    for tref in &rec.insertables {
        let Some(&idx) = rec.index_of.get(tref) else {
            continue;
        };
        if !matches!(rec.tasks[idx].2, TaskProto::Primitive { .. }) {
            return Err(crate::error::HtnError::builder(format!(
                "insertable: `{}` is not a primitive task",
                rec.tasks[idx].1
            )));
        }
        match wrapped.get(tref) {
            Some((cref, cname)) => candidates.push((*cref, *cname)),
            None => candidates.push((*tref, rec.tasks[idx].1)),
        }
    }

    let insert_ref = SubtaskRef::Synthetic(rec.next_synthetic);
    rec.next_synthetic += 1;
    let insert_name = leak_name("gtn/insert".to_string());

    // Stop method FIRST: first-match commits "no insertion"; the insertion
    // methods are only offered on backtrack (repair semantics).
    let mut methods = vec![MethodProto {
        name: Some(leak_name("stop".to_string())),
        utility: None,
        preconditions: Vec::new(),
        subtasks: Vec::new(),
        unordered: false,
        edges: Vec::new(),
        pause_positions: Vec::new(),
    }];
    for (cref, cname) in candidates {
        methods.push(MethodProto {
            name: Some(leak_name(format!("use:{cname}"))),
            utility: None,
            preconditions: Vec::new(),
            subtasks: vec![(cref, cname, true), (insert_ref, insert_name, true)],
            unordered: false,
            edges: Vec::new(),
            pause_positions: Vec::new(),
        });
    }
    let insert = TaskProto::Compound {
        methods,
        policy: crate::domain::SelectionPolicy::default(),
    };

    // Splice gaps into every non-synthetic *total-order* body. Partially
    // ordered bodies keep their member set: their scheduling edges are
    // position-keyed and splicing would shift every member.
    for (tref, _, proto) in rec.tasks.iter_mut() {
        if matches!(tref, SubtaskRef::Synthetic(_)) {
            continue;
        }
        let TaskProto::Compound { methods, .. } = proto else {
            continue;
        };
        for m in methods.iter_mut() {
            if m.unordered || !m.edges.is_empty() {
                continue;
            }
            if m.subtasks.is_empty() {
                continue;
            }
            let old = std::mem::take(&mut m.subtasks);
            m.subtasks.push((insert_ref, insert_name, true));
            for (r, name, is_then) in old {
                m.subtasks.push((r, name, is_then));
                m.subtasks.push((insert_ref, insert_name, true));
            }
        }
    }

    rec.tasks.push((insert_ref, insert_name, insert));
    rec.index_of.insert(insert_ref, rec.tasks.len() - 1);
    Ok(())
}
