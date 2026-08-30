# HTN Research Synthesis

Distilled from the seven papers in this directory. Focus: techniques that refine or extend
classic HTN planning, and how they map onto `cdda_htn` (forward MTR planner + backward
greedy goal planner, totally-ordered, reflection-based state).

Raw extracted text for two papers is kept alongside: `grounding.txt`, `inferring.txt`.

---

## 1. The papers and what each contributes

| Paper | Core contribution |
|---|---|
| Alford et al., *Relating Task and Goal Decomposition with Task Sharing* (IJCAI-16) | Unifies HTN, HGN (goal decomposition), and ANML-style task insertion/sharing under one formalism (GTN); proves plan-preserving translations back to plain HTN; settles expressivity/complexity of each extension. |
| Höller, *PANDA-λ at IPC 2023* | Landmark-guided progression search: RC (Relaxed Composition) heuristic family — a classical STRIPS relaxation of the decomposition process usable for Add/FF/LM-Cut — plus AND/OR landmarks, multi-fringe search. |
| Olz & Bercher, *Look-Ahead Pruning for HTN* (SoCS 2023) | Per-node look-ahead that detects **inevitable refinements** (all but one method provably infeasible → commit early, even for non-first tasks) and **dead-ends** (all methods infeasible → prune before heuristic). Powered PANDADealer, which won all three total-order IPC 2023 tracks. |
| Olz et al., *Inferring Preconditions and Effects of Compound Tasks* (JAIR 2025) | Compound tasks have no declared prec/eff; exact inference is PSPACE/EXPTIME-complete, but a **polynomial relaxation** (executability-relaxed preconditions `prec∅`, precondition-relaxed effects) is essentially exact in practice (100% on tasks, ~99.9% on methods for preconditions). Gives the soundness directions needed for pruning. |
| Behnke et al., *On Succinct Groundings of HTN Planning Problems* (AAAI 2020) | Grounding must be an approximation (relevance is undecidable); interleave instantiation with delete-relaxed + hierarchy (TDG) reachability, **iterate to fixpoint**; planning graph and TDG are the same fixed-point algorithm (GPG); parameter splitting shrinks groundings polynomially. |
| Höller, *The Toad System* (JAIR 2024) | Totally-ordered HTN = intersection of a context-free language (decomposition, Lʰ) and a regular language (state transitions, Lᶜ). Translate to classical planning via a finite automaton threaded through an FDR variable; **non-self-embedding** domains (78% of IPC 2020) translate *exactly*; h_FFA = precomputed per-FA-state distance-to-acceptance heuristic. |
| Yuan & Bercher, *Search Node-Specific Special-Case Heuristics* | Under progression search, nodes **monotonically acquire** structural properties (primitive, regular, acyclic, tail-recursive, totally-ordered); detect them per node (cheap TDG checks) and dispatch specialized cheaper/more-informed techniques. Tail-recursive/acyclic nodes dominate real searches. |

## 2. The through-lines

1. **Compound-task summaries are the enabling primitive.** The inference paper (and its
   AAAI 2021 predecessor) supply polynomial, sound summaries of compound tasks/methods:
   - `prec∅(c)` — executability-relaxed preconditions (**under**approximation: every
     refinement needs `f` before anything produces it).
   - possible effects (**over**approximation) and guaranteed effects (**under**approximation),
     with the `cor-eff` correction (subtract preconditions from relaxed positive effects).
   Everything else in this stack builds on them: look-ahead pruning uses them directly;
   RC-style heuristics and grounders use the same reachability machinery.

2. **Pruning beats heuristics for a DFS planner.** The single most proven practical win
   (IPC 2023 total-order sweep) is not a heuristic but the look-ahead: a linear sweep over
   the remaining task sequence, treating compound tasks via their relaxed summaries,
   propagating an optimistic state (union of possible adds, intersection of guaranteed
   deletes), detecting dead-ends and unique-method commitments before branching.

3. **Structure is usually benign.** 78% of real total-order domains are non-self-embedding
   (Toad), most searches become tail-recursive/acyclic within a few decompositions
   (Yuan & Bercher). Static per-domain analysis (SCCs of the task graph, stratification,
   min-yield lengths) is cheap at parse time and pays off per plan call.

4. **Goals vs. tasks are two views of one mechanism.** Alford et al. compile goal
   decomposition/release into ordinary methods (a "verify" method with the goal as
   precondition + per-action "do `o`, then still need the goal" methods). Goal-directed
   selection = an operator is relevant to a goal iff its effects cover a needed literal —
   exactly the criterion `BackPlanner`'s greedy set-cover already implements.

5. **Never ground naively; never prune once.** Succinct grounding interleaves
   instantiation with reachability, and the delete-relaxed planning graph + TDG pruning
   must be iterated to fixpoint (one-pass pruning leaves cascading dead instances).

## 3. Soundness cheat-sheet for using inferred summaries

| Set | Direction | Safe use |
|---|---|---|
| `prec∅(task/method)` | ⊆ exact preconditions | **Fail early** if `prec∅ ⊄ state` (prune branch) |
| possible effects (relaxed) | ⊇ exact possible effects | **Prune** if a needed fact can never be produced |
| guaranteed effects (relaxed, corrected) | ⊆ exact guaranteed effects | **Assert** the fact will change; filter subsequent condition checks optimistically |
| relaxed guaranteed positive effects | may include preconditions | subtract `prec∅` first (`cor-eff`) |

## 4. Mapping onto `cdda_htn`

Concrete, prioritized adoption path (all consistent with the crate's contracts —
parse-time precomputation in `HtnDomain`, `usize` indices, allocation-free backtracking):

1. ✅ **Parse-time inferred summaries** (enabler for everything else): per `Method`/`Task`,
   compute `prec∅` and relaxed possible/guaranteed effects via the Algorithm 5/6 fixpoint
   (`EmptyRefinements` → shorten methods from left/right → decomposition reachability),
   stored as interned `Ustr` sets or bitsets over a fact index. Polynomial, domain-level,
   state-independent — same philosophy as the existing `name -> index` map.
2. ✅ **Look-ahead dead-end check in `HtnPlanner`**: before method selection at a
   decomposition frame, check `prec∅(task) ⊆ state`; after choosing a method, sweep the
   remaining subtask sequence with optimistic effect propagation to fail at the frame
   instead of deep in recursion. Cheapest version (first-task methods only) is nearly free.
3. ✅ **Domain structure analysis**: SCCs of the task/method graph → per-task flags
   (recursive / tail-recursive / acyclic / self-embedding) and min-yield (shortest
   primitive sequence) for O(1) depth-bounded pruning. Store per-frame property bits
   (properties are monotone down the search tree — inheritable in `DecompositionFrame`).
4. ⬜ **BackPlanner upgrade**: use method-level guaranteed effects (underapproximation) so
   compound tasks can participate in reverse chaining, with `prec∅` giving the recursive
   subgoal set — a principled generalization of the current operator-only greedy chaining.
5. ⬜ **If a best-first mode is ever added**: RC-style classical relaxation ("methods are
   actions applicable when their subtasks are marked in the decomposition tree; goal =
   mark all current tasks") is the proven recipe for reusing classical heuristics;
   h_FFA-style precomputed distance tables give nearly free guidance.
6. **Caveats to respect**: plain HTN is semi-decidable (keep the sanity limit); don't
   memoize visited `(state)` without the remaining task context (Toad's graph-search
   incompleteness lesson); reject/warn on methods with no primitive refinement (the
   inference algorithms' correctness assumption); don't drop effectless actions when
   grounding (they constrain ordering).

## 5. Complexity quick reference

- Plain HTN (and GTN, HGN, HTN with sharing): semi-decidable in general.
- HTN + task insertion: NEXPTIME-complete (PSPACE-complete for goal-only domains).
- Insertion + sharing: PSPACE-hard, in NEXPTIME.
- Exact compound-task prec/eff membership: PSPACE-complete (acyclic/regular/tail-recursive
  tasks), EXPTIME-complete in general — hence the polynomial relaxation.
- "Is action `a` in some plan?": undecidable → grounding is always an approximation.
