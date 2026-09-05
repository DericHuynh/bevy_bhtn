//! Adversarial HTN planning (AHTN) — Ontañón & Buro, *Adversarial
//! Hierarchical-Task Network Planning for Complex Real-Time Games*
//! (IJCAI 2015), Algorithm 2 with alpha-beta pruning.
//!
//! AHTN integrates HTN decomposition into game tree search: each game-tree
//! node carries a decomposition frontier per player, and the search
//! alternates between **executing** already-decomposed primitive actions and
//! **decomposing** non-primitive tasks into methods. The branching factor is
//! bounded by the number of applicable *methods* per choice point — not by
//! raw action sequences — which is the paper's answer to the enormous
//! branching of adversarial domains.
//!
//! This module implements the paper's **base algorithm**: two players
//! (max/min), turn-based, perfect information, deterministic, zero-sum —
//! exactly the regime that fits a CDDA-style game (the paper's RTS
//! extensions — durative, simultaneous, and concurrent actions — are out of
//! scope here). Both players plan from the **same baked domain** against the
//! same [`PlanState`]: player max from the forward root (or any compound
//! registered via [`DomainBuilder::add_root`](crate::domain::DomainBuilder::add_root)),
//! player min from an extra root.
//!
//! # The algorithm
//!
//! Each node is `(state, max-queue, min-queue)` where a queue is the
//! player's remaining task occurrences (the paper's HTN + execution pointer,
//! flattened to the crate's compiled form). Per turn:
//!
//! - **Execute** — if the queue head is a primitive whose preconditions hold
//!   in the current state, apply its effects, advance the pointer, and hand
//!   the turn to the opponent (depth decreases by one: depth counts
//!   *primitive actions issued*, by either player).
//! - **Decompose** — if the head is a compound task, the player's choices
//!   are its applicable methods (declaration order; one method per branch).
//!   Committing a method replaces the head with its subtasks — the turn
//!   stays with the same player (the plan may need several decompositions
//!   before it yields an action), and depth is unchanged. Partially-ordered
//!   methods schedule their baked **first** topological order (the same
//!   commitment rule the forward planner uses).
//! - **Yield** — a player whose queue is empty (plan fully executed) passes;
//!   when both queues are empty, or `depth` reaches 0, the evaluation
//!   function is applied.
//!
//! Max maximizes the evaluation; min minimizes it. Alpha-beta pruning cuts
//! provably-dominated branches. The result is max's primitive plan (the
//! principal variation) and the root value.
//!
//! # Defined behavior
//!
//! - **Stuck players lose the branch**: if a player's next primitive fails
//!   its preconditions (an inconsistent plan — γ(s, a) = ⊥), or its head
//!   compound has no applicable method, or the decomposition budget is
//!   exhausted, that branch is valued at the worst value for that player
//!   (±∞). The paper instead *requires* domains where every task always has
//!   an applicable method — its recommended pattern (a fallback method whose
//!   precondition is the negation of the others', decomposing to a wait
//!   action and a recursive call) is pinned by a test here.
//! - **No viable plan** — if max is stuck in *every* branch, `search`
//!   returns `Ok(None)`.
//! - **Baked selection policies are ignored**: every applicable method is a
//!   searched branch; alpha-beta's ordering is declaration order.
//! - **Bounded**: the `depth` parameter caps primitive actions; a
//!   decomposition budget ([`Ahtn::with_decomposition_budget`], default
//!   1000) bounds pure-decomposition recursion, so even self-recursive
//!   domains terminate.
//! - **Deterministic**: no randomness; the same inputs yield the same plan
//!   and value.
//! - **The input state is never mutated**; the search works on clones.
//! - **Zero-sum**: `eval` returns max's payoff; min plays to minimize it.
//!   Non-zero-sum or multi-player (>2) settings are out of scope.

use std::collections::VecDeque;

use crate::domain::{HtnDomain, Task};
use crate::error::{HtnError, HtnResult};
use crate::state::PlanState;
use crate::tasks::TaskFn;

/// Which player's turn it is (index into the two queues).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Max,
    Min,
}

impl Side {
    fn other(self) -> Side {
        match self {
            Side::Max => Side::Min,
            Side::Min => Side::Max,
        }
    }

    /// The worst possible value for this player (what a stuck branch scores).
    fn worst(self) -> f32 {
        match self {
            Side::Max => f32::NEG_INFINITY,
            Side::Min => f32::INFINITY,
        }
    }
}

/// The result of an AHTN search: max's primitive plan (the principal
/// variation) and the root's minimax value.
#[derive(Debug, Clone, PartialEq)]
pub struct AhtnOutcome {
    /// Max's primitive task indices, in execution order (min's actions are
    /// interleaved between them but are not part of this plan).
    ///
    /// Deliberately **not** a [`Plan`](crate::planner::Plan): the principal
    /// variation is not a driver-executable program — min's actions happen
    /// between max's steps, and the driver's one-agent model has no
    /// counterpart for them. A game runs the PV through its own
    /// turn arbitration.
    pub plan: Vec<u32>,
    /// The root's minimax value (max's payoff under best play by both).
    pub value: f32,
}

/// An adversarial HTN planner over a baked domain.
///
/// Both players share the domain's component registry and state
/// representation; each decomposes from its own root task. See the
/// [module docs](self) for the algorithm and contracts.
#[derive(Clone, Copy, Debug)]
pub struct Ahtn<'a> {
    domain: &'a HtnDomain,
    max_decompositions: usize,
}

impl<'a> Ahtn<'a> {
    /// A planner over `domain` with the default decomposition budget
    /// (1000 method applications per search call).
    pub fn new(domain: &'a HtnDomain) -> Self {
        Self {
            domain,
            max_decompositions: 1000,
        }
    }

    /// Set the per-search decomposition budget (method applications across
    /// both players). Exhausting it counts as a stuck branch.
    #[must_use]
    pub fn with_decomposition_budget(mut self, max_decompositions: usize) -> Self {
        self.max_decompositions = max_decompositions;
        self
    }

    /// Run the adversarial search.
    ///
    /// `max_root` and `min_root` are the two players' root task functions,
    /// passed by value and resolved by their `TypeId`s through the baked type
    /// index (both must be compound tasks — register the opponent's via
    /// [`DomainBuilder::add_root`](crate::domain::DomainBuilder::add_root)). `eval`
    /// scores a state from max's perspective; `depth` bounds the number of
    /// primitive actions issued (by either player) before evaluation.
    ///
    /// Returns `Ok(None)` when max has no viable plan (every branch ends
    /// with max stuck); `Err` for unregistered roots or non-compound roots.
    pub fn search<MaxRoot: TaskFn, MinRoot: TaskFn>(
        &self,
        max_root: MaxRoot,
        min_root: MinRoot,
        state: &PlanState,
        eval: impl Fn(&PlanState) -> f32,
        depth: usize,
    ) -> HtnResult<Option<AhtnOutcome>> {
        let resolve = |type_path: &str, idx: Option<usize>| -> HtnResult<usize> {
            let idx = idx.ok_or_else(|| HtnError::UnregisteredTask {
                type_name: type_path.to_string(),
            })?;
            if !self.domain.tasks[idx].is_compound() {
                return Err(HtnError::builder(format!(
                    "adversarial root `{type_path}` must be a compound task"
                )));
            }
            Ok(idx)
        };
        let max_idx = resolve(
            std::any::type_name::<MaxRoot>(),
            self.domain.task_index_by_type(max_root.task_type_id()),
        )?;
        let min_idx = resolve(
            std::any::type_name::<MinRoot>(),
            self.domain.task_index_by_type(min_root.task_type_id()),
        )?;

        let mut queues = [VecDeque::from([max_idx]), VecDeque::from([min_idx])];
        let mut budget = self.max_decompositions;
        let (value, best) = self.node(
            Side::Max,
            state,
            &mut queues,
            &mut budget,
            f32::NEG_INFINITY,
            f32::INFINITY,
            depth,
            &eval,
        );
        if value == f32::NEG_INFINITY {
            return Ok(None);
        }
        Ok(Some(AhtnOutcome { plan: best, value }))
    }

    /// One turn of `side`. Returns the node's minimax value and max's
    /// primitive plan beneath this node (the principal variation).
    #[allow(clippy::too_many_arguments)]
    fn node(
        &self,
        side: Side,
        state: &PlanState,
        queues: &mut [VecDeque<usize>; 2],
        budget: &mut usize,
        mut alpha: f32,
        mut beta: f32,
        depth: usize,
        eval: &dyn Fn(&PlanState) -> f32,
    ) -> (f32, Vec<u32>) {
        let me = match side {
            Side::Max => 0,
            Side::Min => 1,
        };
        // Terminal: budget of actions exhausted, or both plans finished.
        if depth == 0 || (queues[0].is_empty() && queues[1].is_empty()) {
            return (eval(state), Vec::new());
        }
        let Some(&head) = queues[me].front() else {
            // This player's plan is fully executed: yield to the opponent.
            return self.node(
                side.other(),
                state,
                queues,
                budget,
                alpha,
                beta,
                depth,
                eval,
            );
        };
        match &self.domain.tasks[head] {
            Task::Primitive(p) => {
                // Consistency: an already-planned primitive that cannot
                // execute loses the branch for its player (γ(s, a) = ⊥).
                if !p.preconditions_met(state) {
                    return (side.worst(), Vec::new());
                }
                let mut next = state.clone();
                p.apply_effects(&mut next);
                queues[me].pop_front();
                let (v, sub) = self.node(
                    side.other(),
                    &next,
                    queues,
                    budget,
                    alpha,
                    beta,
                    depth - 1,
                    eval,
                );
                // Only max's primitives form the returned plan.
                match side {
                    Side::Max => {
                        let mut out = Vec::with_capacity(sub.len() + 1);
                        out.push(head as u32);
                        out.extend(sub);
                        (v, out)
                    }
                    Side::Min => (v, sub),
                }
            }
            Task::Compound(c) => {
                if *budget == 0 {
                    return (side.worst(), Vec::new());
                }
                let applicable: Vec<usize> = c
                    .methods
                    .iter()
                    .enumerate()
                    .filter(|(_i, m)| m.applicable(state))
                    .map(|(i, _)| i)
                    .collect();
                if applicable.is_empty() {
                    return (side.worst(), Vec::new());
                }
                match side {
                    Side::Max => {
                        let mut best = (f32::NEG_INFINITY, Vec::new());
                        for m in c.methods.iter() {
                            if !m.applicable(state) {
                                continue;
                            }
                            if *budget == 0 {
                                break;
                            }
                            *budget -= 1;
                            // Save BOTH queues: the child's subtree may
                            // execute the opponent's primitives (via the
                            // yield path), and every sibling must start
                            // from the identical frontier.
                            let saved_me = queues[me].clone();
                            let saved_other = queues[1 - me].clone();
                            let first = match &m.order {
                                crate::order::SubtaskOrder::Total => None,
                                crate::order::SubtaskOrder::Partial { first, .. } => {
                                    Some(first.as_slice())
                                }
                            };
                            self.commit(&mut queues[me], &m.subtasks, first);
                            let (v, sub) =
                                self.node(side, state, queues, budget, alpha, beta, depth, eval);
                            queues[me] = saved_me;
                            queues[1 - me] = saved_other;
                            if v > best.0 {
                                best = (v, sub);
                            }
                            alpha = alpha.max(best.0);
                            if alpha >= beta {
                                break; // beta cut
                            }
                        }
                        best
                    }
                    Side::Min => {
                        let mut best = (f32::INFINITY, Vec::new());
                        for m in c.methods.iter() {
                            if !m.applicable(state) {
                                continue;
                            }
                            if *budget == 0 {
                                break;
                            }
                            *budget -= 1;
                            let saved_me = queues[me].clone();
                            let saved_other = queues[1 - me].clone();
                            let first = match &m.order {
                                crate::order::SubtaskOrder::Total => None,
                                crate::order::SubtaskOrder::Partial { first, .. } => {
                                    Some(first.as_slice())
                                }
                            };
                            self.commit(&mut queues[me], &m.subtasks, first);
                            let (v, sub) =
                                self.node(side, state, queues, budget, alpha, beta, depth, eval);
                            queues[me] = saved_me;
                            queues[1 - me] = saved_other;
                            if v < best.0 {
                                best = (v, sub);
                            }
                            beta = beta.min(best.0);
                            if alpha >= beta {
                                break; // alpha cut
                            }
                        }
                        best
                    }
                }
            }
            // Goal tasks are back-planning targets, not forward steps.
            Task::Goal(_) => {
                queues[me].pop_front();
                self.node(side, state, queues, budget, alpha, beta, depth, eval)
            }
        }
    }

    /// Replace the queue head with the method's subtask occurrences: total
    /// orders run in declaration order; partially-ordered methods schedule
    /// their baked first topological order (the forward planner's commitment
    /// rule).
    fn commit(&self, queue: &mut VecDeque<usize>, subtasks: &[u32], first: Option<&[u8]>) {
        queue.pop_front();
        match first {
            Some(order) => {
                for &pos in order.iter().rev() {
                    queue.push_front(subtasks[pos as usize] as usize);
                }
            }
            None => {
                for &sub in subtasks.iter().rev() {
                    queue.push_front(sub as usize);
                }
            }
        }
    }
}
