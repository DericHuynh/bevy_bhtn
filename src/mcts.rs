//! Monte Carlo Tree Search (UCT) as a pluggable search strategy.
//!
//! [`MctsSearcher`] implements [`Searcher`](crate::selection::Searcher), so it plugs into
//! [`HtnSearchStrategy::Custom`](crate::selection::HtnSearchStrategy::Custom)
//! — globally via `HtnConfig::with_strategy`, or per agent via a
//! [`SearchOverride`](crate::selection::SearchOverride). No planner surgery:
//! the search runs entirely over the public baked network and the
//! [`PlanState`] scratchpad.
//!
//! The design follows Wichlacz, Höller, Torralba & Hoffmann, *Applying
//! Monte-Carlo Tree Search in HTN Planning* (SoCS 2020) — specifically their
//! JSHOP configuration `M(e,c,dfs,P)`: single-child elimination, node caching
//! with dead-end marking, depth-first rollouts, and upper-bound pruning.
//!
//! # The search regime
//!
//! A **state** is a decomposition frontier: the simulated scratchpad, the
//! queue of remaining task occurrences, the accumulated primitive cost, and
//! the primitive plan so far. Advancing a state applies *forced* moves inline
//! — an applicable primitive at the queue head, and a compound with exactly
//! one applicable method, are committed immediately (the paper's
//! single-child elimination) — and stops at the next **choice point**: a
//! compound with `k ≥ 2` applicable methods. Each iteration:
//!
//! 1. **Select** — descend by UCB1
//!    (`mean reward + C·√(ln N / n)`), skipping children proven to be dead
//!    ends or already complete (solved-labeling).
//! 2. **Expand** — commit one untried applicable method of the node's head
//!    compound (declaration order) and advance.
//! 3. **Roll out** — from the new node, a **depth-first** decomposition in
//!    declaration order with backtracking: it terminates with a plan, a
//!    proof that the node is a dead end, or a budget abort. DFS rollouts are
//!    what make HTN MCTS viable — shallow dead-ends (doomed methods) are
//!    backtracked instead of poisoning the reward statistics (a random
//!    rollout over a 20-link chain of binary choices would need ~2²⁰ samples
//!    to find the one plan; a DFS rollout finds it in one pass).
//! 4. **Backpropagate** — add the reward up the visited path, marking nodes
//!    dead when every method is tried and every child is dead.
//!
//! The returned plan is the **incumbent**: the cheapest complete plan any
//! rollout or expansion found.
//!
//! # Defined behavior
//!
//! - **Deterministic** given `(domain, state)`: no randomness anywhere — the
//!   tree policy is UCB1 over visit counts, the rollout policy is DFS in
//!   declaration order. The same state always yields the same plan (stable
//!   replans), and a population can share one `Arc<MctsSearcher>`.
//! - **The input state is never mutated**; the search works on clones.
//! - **Cost-aware rewards**: a complete plan of cost `c` scores
//!   `1 / (1 + c)` (costs are the primitives' declared `cost`/`cost_fn`,
//!   clamped ≥ 0), so UCT converges toward cheap plans — the paper's headline
//!   result. Failures reward 0.
//! - **Upper-bound pruning**: once an incumbent exists, rollouts prune any
//!   branch whose accumulated cost reaches it (only strictly cheaper plans
//!   matter). A pruned rollout is *inconclusive*, never a dead-end proof.
//! - **Only complete plans are returned**: if no rollout reached a finished
//!   decomposition within the iteration budget, `search` returns `None` (the
//!   driver stores an empty plan — the agent replans next tick, never
//!   wedged). A Monte Carlo estimate of an unfinished prefix is not a plan.
//! - **Baked selection policies are ignored**: every applicable method is an
//!   equal choice for tree expansion; the rollout uses declaration order.
//! - **Bounded**: rollouts carry a decomposition-step budget
//!   (`MctsSearcher::rollout_depth`); exhausting it is *inconclusive*
//!   (reward 0, no dead-end marking), so even non-terminating domains cost a
//!   bounded amount of work per iteration.

use std::collections::VecDeque;

use crate::domain::{HtnDomain, Task};
use crate::planner::{Plan, PlanStatus};
use crate::state::PlanState;

/// One search-tree node: a choice point (or terminal) with its simulated
/// frontier.
struct Node {
    state: PlanState,
    /// Remaining task occurrences after this node's choice point.
    queue: VecDeque<usize>,
    /// The compound task whose applicable methods are this node's choices.
    head: usize,
    /// The primitive plan accumulated so far.
    plan: Vec<u32>,
    /// Accumulated primitive cost of `plan`.
    g: f32,
    /// Applicable method indices not yet tried from this node.
    untried: Vec<usize>,
    children: Vec<usize>,
    parent: Option<usize>,
    visits: u32,
    reward: f32,
    /// The queue drained here: `plan` is a finished decomposition.
    complete: bool,
    /// Proven: no completion beneath this node (every method tried, every
    /// child dead). Solved-labeling's complement — never selected again.
    dead_end: bool,
}

/// Where advancing a state's forced moves stopped.
enum Stop {
    /// The queue drained: the plan is final, with its total cost.
    Complete(Vec<u32>, f32),
    /// A primitive's preconditions failed (or a compound has no applicable
    /// methods).
    Failed,
    /// The step budget ran out mid-advance (inconclusive).
    Aborted,
    /// A compound choice point: state, remaining queue, head task, plan,
    /// accumulated cost, and the applicable method indices (`k ≥ 2`).
    ChoicePoint(PlanState, VecDeque<usize>, usize, Vec<u32>, f32, Vec<usize>),
}

/// Apply forced moves from the queue head until a compound choice point with
/// `k ≥ 2` applicable methods, a terminal, a failure, or budget exhaustion.
/// Consumes `queue` and `plan`. Single-applicable compounds are committed
/// inline (the paper's single-child elimination).
fn advance(
    domain: &HtnDomain,
    mut state: PlanState,
    mut queue: VecDeque<usize>,
    mut plan: Vec<u32>,
    mut g: f32,
    fuel: &mut u32,
) -> Stop {
    loop {
        let Some(idx) = queue.pop_front() else {
            return Stop::Complete(plan, g);
        };
        match &domain.tasks[idx] {
            Task::Primitive(p) => {
                if !p.preconditions_met(&state) {
                    return Stop::Failed;
                }
                if *fuel == 0 {
                    return Stop::Aborted;
                }
                *fuel -= 1;
                g += p.cost.as_ref().map(|f| f(&state)).unwrap_or(0.0).max(0.0);
                p.apply_effects(&mut state);
                plan.push(idx as u32);
            }
            Task::Compound(c) => {
                let applicable: Vec<usize> = c
                    .methods
                    .iter()
                    .enumerate()
                    .filter(|(_i, m)| m.applicable(&state))
                    .map(|(i, _)| i)
                    .collect();
                match applicable.len() {
                    0 => return Stop::Failed,
                    1 => {
                        // Forced move: no choice, inline it.
                        if *fuel == 0 {
                            return Stop::Aborted;
                        }
                        *fuel -= 1;
                        for &sub in c.methods[applicable[0]].subtasks.iter().rev() {
                            queue.push_front(sub as usize);
                        }
                    }
                    _ => {
                        return Stop::ChoicePoint(state, queue, idx, plan, g, applicable);
                    }
                }
            }
            // Goal tasks are back-planning targets, not forward steps.
            Task::Goal(_) => continue,
        }
    }
}

/// The outcome of one depth-first rollout.
enum Rollout {
    /// A complete plan and its total cost.
    Plan(Vec<u32>, f32),
    /// Proven: no completion beneath this frontier.
    DeadEnd,
    /// The step budget ran out (inconclusive — never a dead-end proof).
    Unknown,
}

/// The decomposition frontier passed through `advance` and the rollout.
struct Frontier {
    state: PlanState,
    queue: VecDeque<usize>,
    plan: Vec<u32>,
    g: f32,
}

/// One depth-first rollout from a **choice point**: try the head compound's
/// applicable methods in declaration order, backtracking over dead ends,
/// until a plan, a proven dead end, or the budget runs out. `depth` bounds
/// the choice depth (recursion); `fuel` bounds total work.
fn dfs_rollout(
    domain: &HtnDomain,
    frontier: Frontier,
    head: usize,
    depth: usize,
    bound: Option<f32>,
    fuel: &mut u32,
) -> Rollout {
    // Upper-bound pruning: a prefix already at (or past) the incumbent can
    // only yield equal-or-worse plans. Inconclusive by design — the node may
    // still have equal-cost completions, so this is not a dead-end proof.
    if let Some(b) = bound {
        if frontier.g >= b {
            return Rollout::Unknown;
        }
    }
    if depth == 0 || *fuel == 0 {
        return Rollout::Unknown;
    }
    let Task::Compound(c) = &domain.tasks[head] else {
        unreachable!("rollout head is compound");
    };
    for m in c.methods.iter() {
        if !m.applicable(&frontier.state) {
            continue;
        }
        if *fuel == 0 {
            return Rollout::Unknown;
        }
        *fuel -= 1;
        let mut q = frontier.queue.clone();
        for &sub in m.subtasks.iter().rev() {
            q.push_front(sub as usize);
        }
        match advance(
            domain,
            frontier.state.clone(),
            q,
            frontier.plan.clone(),
            frontier.g,
            fuel,
        ) {
            Stop::Complete(p, cost) => return Rollout::Plan(p, cost),
            Stop::Failed => continue, // backtrack: next method
            Stop::Aborted => return Rollout::Unknown,
            Stop::ChoicePoint(state, queue, head2, plan, g, _app2) => {
                let f = Frontier {
                    state,
                    queue,
                    plan,
                    g,
                };
                match dfs_rollout(domain, f, head2, depth - 1, bound, fuel) {
                    Rollout::Plan(p, cost) => return Rollout::Plan(p, cost),
                    Rollout::DeadEnd => continue, // backtrack: next method
                    Rollout::Unknown => return Rollout::Unknown,
                }
            }
        }
    }
    // Every applicable method failed: a proven dead end.
    Rollout::DeadEnd
}

/// Monte Carlo Tree Search over HTN decompositions (UCT).
///
/// Stateless per call: every `search` builds a fresh tree under the iteration
/// budget. See the [module docs](self) for the regime and contracts.
#[derive(Clone, Copy, Debug)]
pub struct MctsSearcher {
    /// UCT iterations (expansions + rollouts) per `search` call.
    iterations: u32,
    /// UCB1 exploration constant (0 = pure exploitation; √2 is the classic
    /// default).
    exploration: f32,
    /// Decomposition-step budget for one rollout's forced moves and method
    /// commits (exhausting it is inconclusive, not a failure).
    rollout_depth: usize,
}

impl MctsSearcher {
    /// A searcher with the given iteration budget; the exploration constant
    /// defaults to √2 and the rollout budget to 1000 steps.
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations,
            exploration: std::f32::consts::SQRT_2,
            rollout_depth: 1000,
        }
    }

    /// Set the UCB1 exploration constant.
    #[must_use]
    pub fn with_exploration(mut self, exploration: f32) -> Self {
        self.exploration = exploration;
        self
    }

    /// Set the per-rollout decomposition-step budget.
    #[must_use]
    pub fn with_rollout_depth(mut self, rollout_depth: usize) -> Self {
        self.rollout_depth = rollout_depth;
        self
    }
}

impl Default for MctsSearcher {
    fn default() -> Self {
        Self::new(2000)
    }
}

/// Add `reward` to the node and every ancestor, marking dead ends on the way
/// up: a node is a proven dead end when it is not complete, every method has
/// been tried, and every child is dead (vacuously true for failed terminals).
fn backprop(nodes: &mut [Node], mut id: usize, reward: f32) {
    loop {
        {
            let n = &mut nodes[id];
            n.visits += 1;
            n.reward += reward;
        }
        let dead = {
            let n = &nodes[id];
            !n.complete && n.untried.is_empty() && n.children.iter().all(|&c| nodes[c].dead_end)
        };
        if dead {
            nodes[id].dead_end = true;
        }
        match nodes[id].parent {
            Some(p) => id = p,
            None => break,
        }
    }
}

impl crate::selection::Searcher for MctsSearcher {
    fn search(&self, domain: &HtnDomain, state: &PlanState) -> Option<Plan> {
        if self.iterations == 0 {
            return None;
        }
        // Advance from the root. A single-applicable root compound is
        // inlined by `advance`, so the whole plan may complete right here.
        let mut fuel = self.rollout_depth as u32;
        let (state, queue, head, plan, g, applicable) = match advance(
            domain,
            state.clone(),
            VecDeque::from([domain.root]),
            Vec::new(),
            0.0,
            &mut fuel,
        ) {
            Stop::Complete(plan, _) => {
                return Some(Plan::compiled(
                    plan,
                    Vec::new(),
                    PlanStatus::Complete,
                    // Custom searchers own their search: pause markers do
                    // not apply to them.
                    None,
                ));
            }
            // A failed or aborted root: the initial state already violates
            // something unfixable (or the budget is zero). (A non-compound
            // root is rejected at bake.)
            Stop::Failed | Stop::Aborted => return None,
            Stop::ChoicePoint(state, queue, head, plan, g, applicable) => {
                (state, queue, head, plan, g, applicable)
            }
        };

        let mut nodes: Vec<Node> = vec![Node {
            state,
            queue,
            head,
            plan,
            g,
            untried: applicable,
            children: Vec::new(),
            parent: None,
            visits: 0,
            reward: 0.0,
            complete: false,
            dead_end: false,
        }];
        // The incumbent: the cheapest complete plan found anywhere.
        let mut incumbent: Option<(f32, Vec<u32>)> = None;

        for _ in 0..self.iterations {
            // ---- 1. Select: descend by UCB1 over live children ----------
            let mut current = 0;
            loop {
                if !nodes[current].untried.is_empty() {
                    break;
                }
                let live: Vec<usize> = nodes[current]
                    .children
                    .iter()
                    .copied()
                    .filter(|&c| !nodes[c].dead_end && !nodes[c].complete)
                    .collect();
                if live.is_empty() {
                    // Everything beneath is proven dead or already complete:
                    // mark and record the loss (a no-op iteration).
                    backprop(&mut nodes, current, 0.0);
                    break;
                }
                let total = (nodes[current].visits as f32).max(1.0);
                let mut best_child = live[0];
                let mut best_ucb = f32::NEG_INFINITY;
                for &ci in &live {
                    let c = &nodes[ci];
                    let mean = c.reward / c.visits.max(1) as f32;
                    let ucb =
                        mean + self.exploration * (total.ln() / c.visits.max(1) as f32).sqrt();
                    if ucb > best_ucb {
                        best_ucb = ucb;
                        best_child = ci;
                    }
                }
                current = best_child;
            }
            if nodes[current].dead_end || nodes[current].complete {
                continue;
            }

            // ---- 2. Expand: commit one untried method --------------------
            let Some(&method_idx) = nodes[current].untried.first() else {
                // Dead-end choice point (no applicable methods at all): a
                // confirmed loss.
                backprop(&mut nodes, current, 0.0);
                continue;
            };
            nodes[current].untried.remove(0);
            let (cstate, cqueue, cplan, cg) = {
                let node = &nodes[current];
                let Task::Compound(c) = &domain.tasks[node.head] else {
                    unreachable!("node.head is a compound choice point");
                };
                let mut q = node.queue.clone();
                for &sub in c.methods[method_idx].subtasks.iter().rev() {
                    q.push_front(sub as usize);
                }
                (node.state.clone(), q, node.plan.clone(), node.g)
            };
            let mut fuel = self.rollout_depth as u32;
            let stop = advance(domain, cstate, cqueue, cplan, cg, &mut fuel);
            let child = match stop {
                Stop::Complete(plan, cost) => {
                    let id = nodes.len();
                    nodes.push(Node {
                        state: PlanState::default(),
                        queue: VecDeque::new(),
                        head: 0,
                        plan: plan.clone(),
                        g: cost,
                        untried: Vec::new(),
                        children: Vec::new(),
                        parent: Some(current),
                        visits: 0,
                        reward: 0.0,
                        complete: true,
                        dead_end: false,
                    });
                    if incumbent.as_ref().is_none_or(|(c, _)| cost < *c) {
                        incumbent = Some((cost, plan));
                    }
                    id
                }
                Stop::Failed => {
                    let id = nodes.len();
                    nodes.push(Node {
                        state: PlanState::default(),
                        queue: VecDeque::new(),
                        head: 0,
                        plan: Vec::new(),
                        g: 0.0,
                        untried: Vec::new(),
                        children: Vec::new(),
                        parent: Some(current),
                        visits: 0,
                        reward: 0.0,
                        complete: false,
                        // A confirmed loss: never selected, never rolled out
                        // (its stored frontier is a placeholder — rolling it
                        // out would "complete" trivially with an empty plan).
                        dead_end: true,
                    });
                    id
                }
                Stop::Aborted => {
                    // Inconclusive: the forced chain outran the budget. No
                    // child node (there is nothing to re-select); the method
                    // is recorded as tried with a loss.
                    backprop(&mut nodes, current, 0.0);
                    continue;
                }
                Stop::ChoicePoint(state, queue, head, plan, g, applicable) => {
                    let id = nodes.len();
                    nodes.push(Node {
                        state,
                        queue,
                        head,
                        plan,
                        g,
                        untried: applicable,
                        children: Vec::new(),
                        parent: Some(current),
                        visits: 0,
                        reward: 0.0,
                        complete: false,
                        dead_end: false,
                    });
                    id
                }
            };
            nodes[current].children.push(child);

            // ---- 3. Roll out from the new node ----------------------------
            // (A complete child wins outright; a failed frontier is a proven
            // loss; a choice point gets a DFS rollout.)
            let reward = if nodes[child].complete {
                1.0
            } else if nodes[child].dead_end {
                0.0
            } else {
                let bound = incumbent.as_ref().map(|(c, _)| *c);
                let f = Frontier {
                    state: nodes[child].state.clone(),
                    queue: nodes[child].queue.clone(),
                    plan: nodes[child].plan.clone(),
                    g: nodes[child].g,
                };
                match dfs_rollout(
                    domain,
                    f,
                    nodes[child].head,
                    self.rollout_depth,
                    bound,
                    &mut fuel,
                ) {
                    Rollout::Plan(plan, cost) => {
                        if incumbent.as_ref().is_none_or(|(c, _)| cost < *c) {
                            incumbent = Some((cost, plan));
                        }
                        1.0 / (1.0 + cost)
                    }
                    // A proven dead end marks the child; an aborted rollout
                    // is only a loss for this iteration.
                    Rollout::DeadEnd | Rollout::Unknown => 0.0,
                }
            };

            // ---- 4. Backpropagate ----------------------------------------
            backprop(&mut nodes, child, reward);
        }

        let (_, plan) = incumbent?;
        Some(Plan::compiled(
            plan,
            Vec::new(),
            PlanStatus::Complete,
            // Custom searchers own their search: pause markers do not apply.
            None,
        ))
    }
}
