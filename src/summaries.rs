//! Bake-time inferred summaries of tasks and methods.
//!
//! Compound tasks and their methods declare no preconditions or effects of
//! their own — their interaction with world state is implicit in their
//! decomposition tree. Following Olz, Biundo & Bercher (AAAI 2021, JAIR 2025),
//! we infer **state-independent summaries** for every task at bake time so the
//! planners (and any look-ahead pruning) can reason about compound tasks
//! without decomposing them.
//!
//! The classic algorithms work over propositional facts; this crate's state is
//! a set of independent ECS components, so the adaptation works over
//! **component slots** (the [`ComponentRegistry`](crate::state::ComponentRegistry)
//! indices): a primitive "reads" the components its preconditions inspect and
//! "writes" the components its effects mutate. Three sets are inferred per
//! task:
//!
//! - **required components** (executability-relaxed preconditions,
//!   under-approximation): every refinement of the task reads the component
//!   before any refinement writes it. Sound to treat as "this task needs a
//!   suitable value for the component".
//! - **possible writes** (over-approximation): some refinement writes the
//!   component. Sound to treat as "the component's value may change".
//! - **guaranteed writes** (under-approximation): every refinement writes the
//!   component. Sound to treat as "the component will definitely be assigned".
//!
//! All three are computed as fixpoints over the decomposition graph (mirroring
//! the paper's `EmptyRefinements` + method-shortening +
//! decomposition-reachability pipeline): possible and guaranteed writes as
//! least fixpoints, and required components as a greatest fixpoint (a safety
//! property over all refinements) guarded by least-fixpoint termination and
//! f-vanishing checks — so recursive domains converge correctly.
//!
//! Summaries are **sound approximations, not exact semantics**: a pruned
//! component is provably (relaxed-)absent, but a kept component isn't
//! guaranteed present. Exact inference is PSPACE/EXPTIME-complete; these
//! polynomial sets matched the exact ones on nearly all IPC 2020 benchmark
//! tasks.

use crate::domain::HtnDomain;
use crate::order::SubtaskOrder;
use crate::state::FieldSet;

use crate::domain::Task;

// ---------------------------------------------------------------------------
// TaskSummary
// ---------------------------------------------------------------------------

/// The inferred, state-independent summary of one task (see [module docs]).
///
/// Beyond the three component sets, this carries the domain-structure analysis
/// (SCC/reachability over the decomposition graph, in the spirit of Toad's
/// self-embedding criterion and Alford's stratification): all of it is
/// computed once at bake time and read O(1) by the planners.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskSummary {
    /// Components every refinement reads before any refinement writes them
    /// (executability-relaxed preconditions; under-approximation).
    ///
    /// Introspection only — it cannot refute anything at search time. In
    /// classical HTN a required fact that is false refutes the task; here
    /// slots are always initialized and the reads are opaque closures, so
    /// there is no stored value predicate to check against the state, and an
    /// unknown slot is "maybe", never "no" (the sweep's own convention).
    pub required_fields: FieldSet,
    /// Components some refinement writes (over-approximation).
    pub possible_writes: FieldSet,
    /// Components every refinement writes (under-approximation).
    pub guaranteed_writes: FieldSet,
    /// Whether the task has at least one finite refinement. A task that can
    /// only refine forever can never complete, so any method whose sequence
    /// contains it is refutable without decomposing anything.
    pub terminating: bool,
    /// Length of the shortest primitive sequence some refinement produces
    /// (`usize::MAX` when [`Self::terminating`] is false). A lower bound on
    /// the decomposition work any refinement of this task requires.
    pub min_yield: usize,
    /// Total primitive cost of the cheapest refinement (`f32::INFINITY` when
    /// [`Self::terminating`] is false). Primitives contribute their declared
    /// constant cost (`.cost(c)`); dynamic `cost_fn` costs are opaque at bake
    /// time and conservatively count 0 — so this is a sound lower bound.
    pub min_cost: f32,
    /// Whether the task can (transitively) decompose into itself.
    ///
    /// Structure metadata (Toad-style analysis): introspection and docs —
    /// the search handles non-termination directly via `terminating` /
    /// `min_yield` budget refutation. Note acyclicity does NOT bound the
    /// search space (an acyclic domain can still enumerate exponentially —
    /// see the gate fixture), so these flags are no substitute for the
    /// sanity limit.
    pub recursive: bool,
    /// Whether the task can appear at a **non-last** position of its own
    /// refinement (right-generating, in Toad's terms). Non-tail recursion.
    /// Structure metadata — see [`Self::recursive`].
    pub self_embedding: bool,
    /// Whether the task can appear with material on **both** sides of itself
    /// in its own refinement (left- and right-generating). Such tasks are the
    /// context-free (non-regular) core of a domain — Toad's exact-translation
    /// criterion. Structure metadata — see [`Self::recursive`].
    pub tail_recursive: bool,
}

// ---------------------------------------------------------------------------
// Computation
// ---------------------------------------------------------------------------

/// Compute and store summaries for every task in `domain`.
///
/// Called from [`HtnDomain::build`](crate::domain::HtnDomain::build); fills
/// one [`TaskSummary`] per task index plus each compound method's
/// `possible_writes` (the union of its subtasks' possible writes), which the
/// forward planner's look-ahead sweep uses for optimistic state propagation.
pub(crate) fn compute_summaries(domain: &mut HtnDomain) {
    let n = domain.tasks.len();
    // The component registry was populated during recording: every component
    // any precondition reads or any effect writes is already registered, so
    // the universe is fixed and slot indices are stable.
    let nf = domain.components.len();

    // ---- 1. Base sets for primitive tasks ---------------------------------
    // reads: components inspected by preconditions (conditions evaluate before
    // effects apply, so every read is required within the task itself).
    // possible writes: effects + expected effects (the planner applies both
    // during search). guaranteed writes: effects only — expected effects are
    // hoped, not guaranteed.
    let mut reads: Vec<FieldSet> = (0..n).map(|_| FieldSet::new(nf)).collect();
    let mut pw: Vec<FieldSet> = (0..n).map(|_| FieldSet::new(nf)).collect();
    let mut gw: Vec<FieldSet> = (0..n).map(|_| FieldSet::new(nf)).collect();
    for (i, task) in domain.tasks.iter().enumerate() {
        let Task::Primitive(p) = task else {
            continue;
        };
        for c in &p.preconditions {
            for &r in c.reads() {
                reads[i].insert(r);
            }
        }
        for e in p.effects.iter().chain(p.expected_effects.iter()) {
            for &w in e.writes() {
                pw[i].insert(w);
            }
        }
        for e in &p.effects {
            for &w in e.writes() {
                gw[i].insert(w);
            }
        }
    }

    // Subtask index accessor: baked methods hold direct task indices.
    let sub_index = |_m: &crate::domain::Method, sub: u32| -> usize { sub as usize };

    // ---- 2. Termination: least fixpoint of "has a finite refinement" ------
    // T(c) = ∃ method whose every subtask terminates. A task that can only
    // refine forever has no finite refinements at all — nothing it "does" in
    // any refinement sense, so every consumer (required components, min yield,
    // look-ahead dead-ends) treats it as empty/incompletable.
    let term = {
        let mut term = vec![false; n];
        loop {
            let mut changed = false;
            for (i, task) in domain.tasks.iter().enumerate() {
                let v = match task {
                    Task::Primitive(_) | Task::Goal(_) => true,
                    Task::Compound(c) => c
                        .methods
                        .iter()
                        .any(|m| m.subtasks.iter().all(|&sub| term[sub_index(m, sub)])),
                };
                if v && !term[i] {
                    term[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        term
    };

    // ---- 3. Possible writes: least fixpoint of ∪ over method subtasks -----
    // Over-approximation: some refinement of c writes f iff some method's
    // subtask chain can write f. Recursion converges to the union over all
    // reachable primitive descendants.
    loop {
        let mut changed = false;
        for (i, task) in domain.tasks.iter().enumerate() {
            let Task::Compound(c) = task else {
                continue;
            };
            let mut set = FieldSet::new(nf);
            for m in &c.methods {
                for &sub in &m.subtasks {
                    set.union_with(&pw[sub_index(m, sub)]);
                }
            }
            if set != pw[i] {
                pw[i] = set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // ---- 4. Guaranteed writes: least fixpoint of ∩ over methods -----------
    // f is written by *every* refinement of the sequence ⟨t1..tn⟩ iff some
    // single ti writes f in every refinement of ti (each refinement is the
    // concatenation of one refinement per subtask). So per method:
    // seq_gw = ∪_i gw(ti), and per task: gw(c) = ∩_m seq_gw(m).
    // A method with an empty subtask list yields an empty seq (an empty
    // refinement writes nothing), correctly zeroing the task's set.
    let mut seq = FieldSet::new(nf);
    loop {
        let mut changed = false;
        for (i, task) in domain.tasks.iter().enumerate() {
            let Task::Compound(c) = task else {
                continue;
            };
            let mut acc: Option<FieldSet> = None;
            for m in &c.methods {
                seq.clear();
                for &sub in &m.subtasks {
                    seq.union_with(&gw[sub_index(m, sub)]);
                }
                acc = Some(match acc {
                    None => seq.clone(),
                    Some(mut a) => {
                        a.intersect_with(&seq);
                        a
                    }
                });
            }
            let set = acc.unwrap_or_else(|| FieldSet::new(nf));
            if set != gw[i] {
                gw[i] = set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // ---- 4b. Per-method guaranteed writes ---------------------------------
    // The seq_gw computed inside the fixpoint above, stored on each baked
    // method: the backward planner uses it as the under-approximation that
    // lets compound methods participate in reverse chaining.
    for task in domain.tasks.iter_mut() {
        let Task::Compound(c) = task else { continue };
        for m in &mut c.methods {
            let mut seq = FieldSet::new(nf);
            for &sub in &m.subtasks {
                seq.union_with(&gw[sub as usize]);
            }
            m.guaranteed_writes = seq;
        }
    }

    // ---- 5. Min yield: least fixpoint of min over methods -----------------
    // The shortest primitive sequence some refinement produces: a primitive
    // yields itself (1); a compound takes the cheapest method's subtask sum.
    // Descending fixpoint from `usize::MAX` (saturating arithmetic makes
    // non-terminating cycles stay at MAX naturally); non-terminating tasks
    // are forced to MAX afterwards as a belt-and-braces guard.
    let mut min_yield: Vec<usize> = vec![usize::MAX; n];
    for (i, task) in domain.tasks.iter().enumerate() {
        if matches!(task, Task::Primitive(_) | Task::Goal(_)) {
            min_yield[i] = 1;
        }
    }
    loop {
        let mut changed = false;
        for (i, task) in domain.tasks.iter().enumerate() {
            let Task::Compound(c) = task else {
                continue;
            };
            let mut best = usize::MAX;
            for m in &c.methods {
                let mut sum = 0usize;
                for &sub in &m.subtasks {
                    sum = sum.saturating_add(min_yield[sub_index(m, sub)]);
                    if sum == usize::MAX {
                        break;
                    }
                }
                best = best.min(sum);
            }
            if best < min_yield[i] {
                min_yield[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for i in 0..n {
        if !term[i] {
            min_yield[i] = usize::MAX;
        }
    }

    // ---- 5b. Min cost: least fixpoint of min over methods -----------------
    // The cheapest total primitive cost some refinement produces: a primitive
    // yields its declared constant cost (dynamic `cost_fn` costs are opaque
    // at bake time and count 0 — a sound lower bound); a compound takes the
    // cheapest method's subtask sum. Same fixpoint shape as min yield, over
    // f32 with `INFINITY` as the non-terminating sentinel (f32 addition
    // saturates to infinity naturally). Non-terminating tasks are forced to
    // INFINITY afterwards, mirroring min yield.
    let mut min_cost: Vec<f32> = vec![f32::INFINITY; n];
    for (i, task) in domain.tasks.iter().enumerate() {
        if let Task::Primitive(p) = task {
            min_cost[i] = p.static_cost.unwrap_or(0.0).max(0.0);
        } else if matches!(task, Task::Goal(_)) {
            min_cost[i] = 0.0;
        }
    }
    loop {
        let mut changed = false;
        for (i, task) in domain.tasks.iter().enumerate() {
            let Task::Compound(c) = task else {
                continue;
            };
            let mut best = f32::INFINITY;
            for m in &c.methods {
                let mut sum = 0.0f32;
                for &sub in &m.subtasks {
                    sum += min_cost[sub_index(m, sub)];
                    if sum.is_infinite() {
                        break;
                    }
                }
                best = best.min(sum);
            }
            if best < min_cost[i] {
                min_cost[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for i in 0..n {
        if !term[i] {
            min_cost[i] = f32::INFINITY;
        }
    }

    // ---- 6. Required components (executability-relaxed preconditions) -----
    // Per component f, a refinement "first touches" f at its first primitive
    // that reads or writes f (a method-precondition read touches at
    // decomposition time, before any subtask runs). f is required for task c
    // iff **every finite refinement** of c first touches f with a READ. That
    // is a safety property over all refinements, so it is computed as a
    // **greatest** fixpoint of exact identities, guarded by two least
    // fixpoints:
    //
    //   T   — c has at least one finite refinement (least fixpoint). Tasks
    //       that can only refine forever have no refinements at all; following
    //       the inference papers' standing assumption (and the "undef"
    //       convention), their requirement set is empty.
    //   E_f — c has a refinement containing no primitive that touches f
    //       (least fixpoint): that refinement never reads f, so f is not
    //       required.
    //   R_f — primitive: reads f (conditions evaluate before effects, so the
    //       read precedes the write within the task). compound: terminating,
    //       not f-vanishing, and every method's subtask sequence first
    //       touches f with a read:
    //       R(seq) = R(u1) ∨ (E_f(u1) ∧ R(rest)); a vanishing subtask is
    //       skipped (its refinement may contain no f-touching primitive), a
    //       relevant non-reading subtask blocks, and an empty relevant
    //       sequence is a refinement that never touches f.
    //
    // The greatest fixpoint is exact for terminating tasks; the T guard keeps
    // the claim conservative (under-approximating) for non-terminating ones.

    let mut req: Vec<FieldSet> = (0..n).map(|_| FieldSet::new(nf)).collect();
    let mut evan: Vec<bool> = vec![false; n];
    let mut r: Vec<bool> = vec![false; n];
    for fi in 0..nf {
        // Does primitive i touch f at all (read or write)?
        let touches = |i: usize| reads[i].contains(fi) || pw[i].contains(fi);

        // E_f: least fixpoint — a refinement with no f-touching primitive.
        evan.iter_mut().for_each(|v| *v = false);
        loop {
            let mut changed = false;
            for (i, task) in domain.tasks.iter().enumerate() {
                let v = match task {
                    Task::Primitive(_) => !touches(i),
                    Task::Goal(_) => true,
                    Task::Compound(c) => c
                        .methods
                        .iter()
                        .any(|m| m.subtasks.iter().all(|&sub| evan[sub_index(m, sub)])),
                };
                if v && !evan[i] {
                    evan[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // R_f: greatest fixpoint, starting from optimistic truth.
        for (i, task) in domain.tasks.iter().enumerate() {
            r[i] = match task {
                // Conditions evaluate before effects, so a read precedes the
                // task's own writes.
                Task::Primitive(_) => reads[i].contains(fi),
                _ => true,
            };
        }
        loop {
            let mut changed = false;
            for (i, task) in domain.tasks.iter().enumerate() {
                let Task::Compound(c) = task else {
                    continue;
                };
                let mut v = term[i] && !evan[i];
                for m in &c.methods {
                    // R_seq(m): does every refinement via this method first
                    // touch f with a read?
                    let mut seq = m.preconditions.iter().any(|c| c.reads().contains(&fi));
                    if !seq {
                        match &m.order {
                            SubtaskOrder::Total => {
                                for &sub in &m.subtasks {
                                    let j = sub_index(m, sub);
                                    if r[j] {
                                        seq = true;
                                        break;
                                    }
                                    if !evan[j] {
                                        // First relevant subtask doesn't start
                                        // with a read: the touch is a write.
                                        break;
                                    }
                                    // Vanishing: the first touch may come later.
                                }
                            }
                            SubtaskOrder::Partial { .. } => {
                                // Set semantics, conservative under-
                                // approximation: f is required iff some member
                                // starts with a read of f and **no** member can
                                // write f — then every linearization first
                                // touches f with that read. If any member can
                                // write f, some linearization touches f with a
                                // write first, so f is not required.
                                seq = m.subtasks.iter().any(|&sub| r[sub_index(m, sub)])
                                    && m.subtasks
                                        .iter()
                                        .all(|&sub| !pw[sub_index(m, sub)].contains(fi));
                            }
                        }
                    }
                    if !seq {
                        v = false;
                        break;
                    }
                }
                if v != r[i] {
                    r[i] = v;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        for (i, _) in r.iter().enumerate() {
            if r[i] {
                req[i].insert(fi);
            }
        }
    }

    // ---- 7. Structure flags: reachability over the decomposition graph ----
    // Edges are compound → compound-subtask, tagged by position: an edge is
    // *non-last* when the subtask is not its method's final subtask (c can
    // appear with material after it) and *non-first* when it is not the first
    // (material before it). Following Toad's criterion:
    //   right-generating  — c reaches itself through a non-last closing edge
    //   (c appears non-last somewhere in its own refinement → not
    //   tail-recursive); left-generating — through a non-first closing edge;
    //   self-embedding — both (the context-free core of the domain).
    let mut adj: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    let mut nl_pred: Vec<Vec<bool>> = vec![vec![false; n]; n];
    let mut nf_pred: Vec<Vec<bool>> = vec![vec![false; n]; n];
    for (i, task) in domain.tasks.iter().enumerate() {
        let Task::Compound(c) = task else {
            continue;
        };
        for m in &c.methods {
            let last = m.subtasks.len().saturating_sub(1);
            // In a partially-ordered set every member can have material
            // before or after it, regardless of its declaration position.
            let set_semantics = m.order.is_partial();
            for (p, &sub) in m.subtasks.iter().enumerate() {
                let j = sub_index(m, sub);
                if matches!(domain.tasks[j], Task::Compound(_)) {
                    if !adj[i].contains(&j) {
                        adj[i].push(j);
                    }
                    if set_semantics || p != last {
                        nl_pred[j][i] = true;
                    }
                    if set_semantics || p != 0 {
                        nf_pred[j][i] = true;
                    }
                }
            }
        }
    }
    // reach[i][j]: j reachable from i via >= 1 edge.
    let mut reach: Vec<Vec<bool>> = vec![vec![false; n]; n];
    for i in 0..n {
        if adj[i].is_empty() {
            continue;
        }
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = adj[i].clone();
        for &j in &stack {
            seen[j] = true;
        }
        while let Some(j) = stack.pop() {
            for &k in &adj[j] {
                if !seen[k] {
                    seen[k] = true;
                    stack.push(k);
                }
            }
        }
        reach[i] = seen;
    }
    let mut recursive = vec![false; n];
    let mut self_embedding = vec![false; n];
    let mut tail_recursive = vec![true; n];
    for i in 0..n {
        recursive[i] = reach[i][i];
        let right_gen = (0..n).any(|j| reach[i][j] && nl_pred[i][j]);
        let left_gen = (0..n).any(|j| reach[i][j] && nf_pred[i][j]);
        self_embedding[i] = left_gen && right_gen;
        tail_recursive[i] = !right_gen;
    }

    // ---- 8. Per-method possible writes ------------------------------------
    // The union of the method's subtasks' possible writes — what the forward
    // planner's look-ahead sweep uses to propagate optimistic state. (Each
    // precondition's read slots live on the `Precondition` itself, captured
    // at build time.)
    for task in &mut domain.tasks {
        if let Task::Compound(c) = task {
            for m in &mut c.methods {
                let mut set = FieldSet::new(nf);
                let mut cost = 0.0f32;
                for &sub in &m.subtasks {
                    set.union_with(&pw[sub as usize]);
                    cost += min_cost[sub as usize];
                }
                m.possible_writes = set;
                m.min_cost = cost;
            }
        }
    }

    // ---- 9. Store ----------------------------------------------------------
    domain.summaries = (0..n)
        .map(|i| TaskSummary {
            required_fields: req[i].clone(),
            possible_writes: pw[i].clone(),
            guaranteed_writes: gw[i].clone(),
            terminating: term[i],
            min_yield: min_yield[i],
            min_cost: min_cost[i],
            recursive: recursive[i],
            self_embedding: self_embedding[i],
            tail_recursive: tail_recursive[i],
        })
        .collect();
}
