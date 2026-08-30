//! Parse-time inferred summaries of tasks and methods.
//!
//! Compound tasks and their methods declare no preconditions or effects of
//! their own — their interaction with world state is implicit in their
//! decomposition tree. Following Olz, Biundo & Bercher (AAAI 2021, JAIR 2025),
//! we infer **state-independent summaries** for every task at parse time so the
//! planners (and any look-ahead pruning) can reason about compound tasks
//! without decomposing them.
//!
//! The classic algorithms work over propositional facts; this crate's state is
//! a typed reflected struct, so the adaptation works over **state fields**:
//! a primitive "reads" the fields its conditions inspect and "writes" the
//! fields its effects assign. Three sets are inferred per task:
//!
//! - **required fields** (executability-relaxed preconditions,
//!   under-approximation): every refinement of the task reads the field before
//!   any refinement writes it. Sound to treat as "this task needs a suitable
//!   value for the field".
//! - **possible writes** (over-approximation): some refinement writes the
//!   field. Sound to treat as "the field's value may change".
//! - **guaranteed writes** (under-approximation): every refinement writes the
//!   field. Sound to treat as "the field will definitely be assigned".
//!
//! All three are computed as fixpoints over the decomposition graph (mirroring
//! the paper's `EmptyRefinements` + method-shortening +
//! decomposition-reachability pipeline): possible and guaranteed writes as
//! least fixpoints, and required fields as a greatest fixpoint (a safety
//! property over all refinements) guarded by least-fixpoint termination and
//! f-vanishing checks — so recursive domains converge correctly: e.g. a
//! terminating self-recursive task whose every refinement reads `food` before
//! writing it still infers `food` as required.
//!
//! Summaries are **sound approximations, not exact semantics**: a pruned
//! field is provably (relaxed-)absent, but a kept field isn't guaranteed
//! present. Exact inference is PSPACE/EXPTIME-complete; these polynomial sets
//! matched the exact ones on nearly all IPC 2020 benchmark tasks.

use std::collections::HashMap;

use ustr::Ustr;

use crate::domain::HtnDomain;
use crate::tasks::Task;

// ---------------------------------------------------------------------------
// FieldSet
// ---------------------------------------------------------------------------

/// A compact bitset over the domain's interned state-field indices (see
/// [`HtnDomain::fields`]).
///
/// All operations assume both sets share the same universe (the same domain's
/// field table). Field indices are dense and domains touch only a handful of
/// fields, so a `Vec<u64>` bitset keeps summary set operations to a few word
/// ops in the planner's hot path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FieldSet {
    bits: Vec<u64>,
}

impl FieldSet {
    /// An empty set over a universe of `universe` fields.
    pub fn new(universe: usize) -> Self {
        let words = universe.div_ceil(64);
        Self {
            bits: vec![0; words],
        }
    }

    /// Add field `idx` to the set.
    pub fn insert(&mut self, idx: usize) {
        let (word, bit) = (idx / 64, idx % 64);
        if word < self.bits.len() {
            self.bits[word] |= 1 << bit;
        }
    }

    /// Whether field `idx` is in the set.
    pub fn contains(&self, idx: usize) -> bool {
        let (word, bit) = (idx / 64, idx % 64);
        word < self.bits.len() && self.bits[word] & (1 << bit) != 0
    }

    /// Remove field `idx` from the set.
    pub fn remove(&mut self, idx: usize) {
        let (word, bit) = (idx / 64, idx % 64);
        if word < self.bits.len() {
            self.bits[word] &= !(1 << bit);
        }
    }

    /// Remove every field from the set.
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }

    /// Add every field of `other` to this set.
    pub fn union_with(&mut self, other: &Self) {
        for (w, o) in self.bits.iter_mut().zip(other.bits.iter()) {
            *w |= o;
        }
    }

    /// Keep only fields present in both sets.
    pub fn intersect_with(&mut self, other: &Self) {
        for (w, o) in self.bits.iter_mut().zip(other.bits.iter()) {
            *w &= o;
        }
    }

    /// Remove every field of `other` from this set.
    pub fn subtract(&mut self, other: &Self) {
        for (w, o) in self.bits.iter_mut().zip(other.bits.iter()) {
            *w &= !o;
        }
    }

    /// Whether every field of this set is also in `other`.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .all(|(a, b)| a & !b == 0)
    }

    /// Whether the set contains no fields.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|w| *w == 0)
    }

    /// The number of fields in the set.
    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate the field indices in the set, in ascending order.
    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter().enumerate().flat_map(|(w, bits)| {
            (0..64)
                .filter(move |bit| bits & (1 << bit) != 0)
                .map(move |bit| w * 64 + bit)
        })
    }
}

// ---------------------------------------------------------------------------
// TaskSummary
// ---------------------------------------------------------------------------

/// The inferred, state-independent summary of one task (see [module docs]).
///
/// Beyond the three field sets, this carries the domain-structure analysis
/// (SCC/reachability over the decomposition graph, in the spirit of Toad's
/// self-embedding criterion and Alford's stratification): all of it is
/// computed once at parse time and read O(1) by the planners.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskSummary {
    /// Fields every refinement reads before any refinement writes them
    /// (executability-relaxed preconditions; under-approximation).
    pub required_fields: FieldSet,
    /// Fields some refinement writes (over-approximation).
    pub possible_writes: FieldSet,
    /// Fields every refinement writes (under-approximation).
    pub guaranteed_writes: FieldSet,
    /// Whether the task has at least one finite refinement. A task that can
    /// only refine forever can never complete, so any method whose sequence
    /// contains it is refutable without decomposing anything.
    pub terminating: bool,
    /// Length of the shortest primitive sequence some refinement produces
    /// (`usize::MAX` when [`Self::terminating`] is false). A lower bound on
    /// the decomposition work any refinement of this task requires.
    pub min_yield: usize,
    /// Whether the task can (transitively) decompose into itself.
    pub recursive: bool,
    /// Whether the task can appear at a **non-last** position of its own
    /// refinement (right-generating, in Toad's terms). Non-tail recursion.
    pub self_embedding: bool,
    /// Whether the task can appear with material on **both** sides of itself
    /// in its own refinement (left- and right-generating). Such tasks are the
    /// context-free (non-regular) core of a domain — Toad's exact-translation
    /// criterion.
    pub tail_recursive: bool,
}

// ---------------------------------------------------------------------------
// Computation
// ---------------------------------------------------------------------------

/// Compute and store summaries for every task in `domain`.
///
/// Called from [`HtnDomain`]'s parse-time index rebuild; fills
/// [`HtnDomain::fields`], the internal field-index map, and one
/// [`TaskSummary`] per task index. Also fills each compound method's
/// `possible_writes` (the union of its subtasks' possible writes), which the
/// forward planner's look-ahead sweep uses for optimistic state propagation.
pub(crate) fn compute_summaries(domain: &mut HtnDomain) {
    // ---- 1. Intern every field name the domain touches --------------------
    let mut field_index: HashMap<Ustr, usize> = HashMap::new();
    let mut fields: Vec<Ustr> = Vec::new();
    let intern = |name: &str, field_index: &mut HashMap<Ustr, usize>, fields: &mut Vec<Ustr>| {
        let key = Ustr::from(name);
        *field_index.entry(key).or_insert_with(|| {
            fields.push(key);
            fields.len() - 1
        })
    };

    for task in &domain.tasks {
        match task {
            Task::Primitive(p) => {
                for c in &p.preconditions {
                    for f in c.read_fields() {
                        intern(f, &mut field_index, &mut fields);
                    }
                }
                for e in p.effects.iter().chain(p.expected_effects.iter()) {
                    intern(e.field(), &mut field_index, &mut fields);
                    if let Some(src) = e.source_field() {
                        intern(src, &mut field_index, &mut fields);
                    }
                }
            }
            Task::Compound(c) => {
                for m in &c.methods {
                    for c in &m.preconditions {
                        for f in c.read_fields() {
                            intern(f, &mut field_index, &mut fields);
                        }
                    }
                }
            }
            Task::Goal(g) => {
                for e in &g.effects {
                    intern(e.field(), &mut field_index, &mut fields);
                    if let Some(src) = e.source_field() {
                        intern(src, &mut field_index, &mut fields);
                    }
                }
            }
        }
    }

    let n = domain.tasks.len();
    let nf = fields.len();
    let idx = |name: &str| -> usize {
        field_index
            .get(&Ustr::from(name))
            .copied()
            .expect("field was interned above")
    };

    // ---- 2. Base sets for primitive tasks ---------------------------------
    // reads: fields inspected by preconditions (conditions evaluate before
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
            for f in c.read_fields() {
                reads[i].insert(idx(f));
            }
        }
        for e in p.effects.iter().chain(p.expected_effects.iter()) {
            pw[i].insert(idx(e.field()));
        }
        for e in &p.effects {
            gw[i].insert(idx(e.field()));
        }
    }

    // ---- 3. Termination: least fixpoint of "has a finite refinement" ------
    // T(c) = ∃ method whose every subtask terminates. A task that can only
    // refine forever has no finite refinements at all — nothing it "does" in
    // any refinement sense, so every consumer (required fields, min yield,
    // look-ahead dead-ends) treats it as empty/incompletable.
    let term = vec![false; n];
    let term = {
        let mut term = term;
        loop {
            let mut changed = false;
            for (i, task) in domain.tasks.iter().enumerate() {
                let v = match task {
                    Task::Primitive(_) | Task::Goal(_) => true,
                    Task::Compound(c) => c.methods.iter().any(|m| {
                        m.subtasks
                            .iter()
                            .all(|sub| domain.index_of.get(sub).map_or(true, |&j| term[j]))
                    }),
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

    // ---- 4. Possible writes: least fixpoint of ∪ over method subtasks -----
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
                for sub in &m.subtasks {
                    if let Some(&j) = domain.index_of.get(sub) {
                        set.union_with(&pw[j]);
                    }
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

    // ---- 5. Guaranteed writes: least fixpoint of ∩ over methods -----------
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
                for sub in &m.subtasks {
                    if let Some(&j) = domain.index_of.get(sub) {
                        seq.union_with(&gw[j]);
                    }
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

    // ---- 6. Min yield: least fixpoint of min over methods -----------------
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
                for sub in &m.subtasks {
                    if let Some(&j) = domain.index_of.get(sub) {
                        sum = sum.saturating_add(min_yield[j]);
                    }
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

    // ---- 7. Required fields (executability-relaxed preconditions) ---------
    // Per field f, a refinement "first touches" f at its first primitive that
    // reads or writes f (a method-precondition read touches at decomposition
    // time, before any subtask runs). f is required for task c iff **every
    // finite refinement** of c first touches f with a READ. That is a safety
    // property over all refinements, so it is computed as a **greatest**
    // fixpoint of exact identities, guarded by two least fixpoints:
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
    // (`term` was computed in step 3.)

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
                    Task::Compound(c) => c.methods.iter().any(|m| {
                        m.subtasks
                            .iter()
                            .all(|sub| domain.index_of.get(sub).map_or(true, |&j| evan[j]))
                    }),
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
                    let mut seq = m.preconditions.iter().any(|c| {
                        c.read_fields()
                            .any(|f| domain.field_index.get(&Ustr::from(f)) == Some(&fi))
                    });
                    for sub in &m.subtasks {
                        match domain.index_of.get(sub) {
                            Some(&j) => {
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
                            // Unknown subtask: the planner drops it too.
                            None => continue,
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

    // ---- 8. Structure flags: reachability over the decomposition graph ----
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
            for (p, sub) in m.subtasks.iter().enumerate() {
                if let Some(&j) = domain.index_of.get(sub) {
                    if matches!(domain.tasks[j], Task::Compound(_)) {
                        if !adj[i].contains(&j) {
                            adj[i].push(j);
                        }
                        if p != last {
                            nl_pred[j][i] = true;
                        }
                        if p != 0 {
                            nf_pred[j][i] = true;
                        }
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

    // ---- 9. Per-method possible writes + per-condition read indices -------
    // The union of the method's subtasks' possible writes — what the forward
    // planner's look-ahead sweep uses to propagate optimistic state — and the
    // field indices each precondition reads, precomputed so the sweep's
    // known/unknown checks need no per-call hash lookups.
    let cond_reads = |c: &crate::conditions::HtnCondition, field_index: &HashMap<Ustr, usize>| {
        let mut reads: [Option<usize>; 2] = [None, None];
        for (slot, f) in reads.iter_mut().zip(c.read_fields()) {
            *slot = field_index.get(&Ustr::from(f)).copied();
        }
        reads
    };
    for task in &mut domain.tasks {
        match task {
            Task::Primitive(p) => {
                p.prec_reads = p
                    .preconditions
                    .iter()
                    .map(|c| cond_reads(c, &field_index))
                    .collect();
            }
            Task::Compound(c) => {
                for m in &mut c.methods {
                    let mut set = FieldSet::new(nf);
                    for sub in &m.subtasks {
                        if let Some(&j) = domain.index_of.get(sub) {
                            set.union_with(&pw[j]);
                        }
                    }
                    m.possible_writes = set;
                    m.prec_reads = m
                        .preconditions
                        .iter()
                        .map(|c| cond_reads(c, &field_index))
                        .collect();
                }
            }
            Task::Goal(_) => {}
        }
    }

    // ---- 10. Store ---------------------------------------------------------
    domain.fields = fields;
    domain.field_index = field_index;
    domain.summaries = (0..n)
        .map(|i| TaskSummary {
            required_fields: req[i].clone(),
            possible_writes: pw[i].clone(),
            guaranteed_writes: gw[i].clone(),
            terminating: term[i],
            min_yield: min_yield[i],
            recursive: recursive[i],
            self_embedding: self_embedding[i],
            tail_recursive: tail_recursive[i],
        })
        .collect();
}
