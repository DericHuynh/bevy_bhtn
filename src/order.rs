//! Subtask ordering within a method, and the linearization machinery the
//! forward planner uses to schedule partially-ordered member sets.
//!
//! A method's members are either a **total order** (the `then` chain — the
//! default, one execution order) or a **partially-ordered set** (declared via
//! [`MethodBuilder::subtask`](crate::tasks::MethodBuilder::subtask) /
//! [`MethodBuilder::before`](crate::tasks::MethodBuilder::before) /
//! [`MethodBuilder::any_order`](crate::tasks::MethodBuilder::any_order)): the
//! members run in any topological order of the constraint DAG, each exactly
//! once. There is no POCL planner — the DFS simply branches over
//! linearizations: it commits to the first topological order and, on
//! downstream failure, retries the *same method* with the next one before
//! offering other methods. The chosen linearization is a flat step program,
//! so the compiled [`Plan`](crate::planner::Plan), the executor, and the ECS
//! driver are completely unchanged.
//!
//! Enumeration is deterministic (ascending member position at every choice),
//! so order 0 is the declaration order whenever the declaration order is
//! topological — the common case, which keeps plans stable across replans.
//! The number of orders is capped at [`LINEARIZATION_CAP`] at bake time; a
//! method whose constraint DAG admits more orders explores at most that many
//! (deterministically, declaration-order first).
//!
//! # Envelope (game-domain guidance)
//!
//! Because enumeration is lexicographic, a valid linearization is reachable
//! within the cap only when it sorts early. For an unconstrained set of n
//! members, n! orders share each leading member, so cross-member dependencies
//! (a member whose precondition another member's effect enables) are
//! reliably schedulable up to **~4 members** (4! = 24 ≤ 64) and become
//! unreachable beyond ~5–8 (pinned by `wide_set_envelope_buried_dependencies`;
//! measured in `benches/wide_sets.rs`). This is rarely a constraint in
//! practice: authored unordered sets model *small independent member sets*;
//! wide or data-driven repetition (loot N items, craft with M ingredients —
//! CDDA-scale) belongs in **recursion over components** (a counter component
//! plus a `fetch_one → fetch_all` cycle), which has no arity limit and plans
//! ~170 ns per item.

use smallvec::SmallVec;

/// How a baked method's subtask members are ordered.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum SubtaskOrder {
    /// A total order: members run in declaration order (the `then` chain).
    /// The planner's fast path — no scheduling machinery at all.
    #[default]
    Total,
    /// A partially-ordered set: members run in any topological order of the
    /// constraint DAG, each exactly once.
    Partial {
        /// Per-member predecessor bitmask: bit `q` set in `preds[p]` means
        /// member `q` must complete before member `p` starts.
        preds: SmallVec<[u64; 4]>,
        /// Number of topological orders (capped at [`LINEARIZATION_CAP`]; a
        /// value equal to the cap means "at least the cap"). Single-order sets
        /// whose order is the declaration order are normalized to
        /// [`SubtaskOrder::Total`]; a single order that *differs* from the
        /// declaration order stays partial (the planner pushes `first`, with no
        /// retries).
        orders: u32,
        /// The first topological order (member positions in execution
        /// order) — what the planner pushes at commitment. Cached at bake so
        /// the hot path never enumerates.
        first: SmallVec<[u8; 4]>,
    },
}

impl SubtaskOrder {
    /// Whether this method's members are a partially-ordered set.
    pub fn is_partial(&self) -> bool {
        matches!(self, SubtaskOrder::Partial { .. })
    }
}

/// Maximum number of linearizations the planner explores per partially-
/// ordered method. Unconstrained sets of n members admit n! orders; the cap
/// bounds the search's worst case while keeping the declaration order (and
/// the 63 next-best deterministic orders) reachable.
pub const LINEARIZATION_CAP: usize = 64;

/// Visit every topological order of the constraint DAG `preds` (bit `q` in
/// `preds[p]` = q before p), in deterministic enumeration order: at each
/// step the ready member with the **lowest position** is chosen first.
/// The visitor returns `false` to stop the enumeration.
///
/// Zero visits means the constraints are cyclic (no complete order exists).
fn each_topo_order(preds: &[u64], mut visit: impl FnMut(&[u8]) -> bool) {
    let n = preds.len();
    debug_assert!(n <= 64, "partial-order methods are limited to 64 members");
    let mut used = 0u64;
    let mut order: SmallVec<[u8; 8]> = SmallVec::new();
    // Per-depth resume cursor: the next member position to consider at that
    // depth (so backtracking resumes rather than restarts the scan).
    let mut next = vec![0u8; n + 1];
    let mut depth = 0usize;
    loop {
        let mut p = next[depth];
        let mut chosen = None;
        while (p as usize) < n {
            let bit = 1u64 << p;
            if used & bit == 0 && preds[p as usize] & !used == 0 {
                chosen = Some(p);
                break;
            }
            p += 1;
        }
        match chosen {
            Some(p) => {
                used |= 1u64 << p;
                order.push(p);
                next[depth] = p + 1;
                depth += 1;
                next[depth] = 0;
                if depth == n {
                    if !visit(&order) {
                        return;
                    }
                    depth -= 1;
                    let p = order.pop().expect("non-empty at full depth");
                    used &= !(1u64 << p);
                }
            }
            None => {
                if depth == 0 {
                    return;
                }
                depth -= 1;
                let p = order.pop().expect("non-empty below depth 0");
                used &= !(1u64 << p);
            }
        }
    }
}

/// The `k`-th topological order of `preds` (0-indexed, member positions in
/// execution order), or `None` when `k` exceeds the (uncapped) order count.
/// Order 0 is the declaration order whenever the declaration order is
/// topological.
pub fn linearize(preds: &[u64], k: usize) -> Option<SmallVec<[u8; 8]>> {
    let mut result = None;
    let mut seen = 0usize;
    each_topo_order(preds, |order| {
        if seen == k {
            result = Some(SmallVec::from_slice(order));
            false
        } else {
            seen += 1;
            true
        }
    });
    result
}

/// The number of topological orders of `preds`, capped at `cap` (a return
/// value equal to `cap` means "at least `cap`"). Zero means the constraints
/// are cyclic.
pub fn topo_order_count(preds: &[u64], cap: usize) -> usize {
    let mut count = 0usize;
    each_topo_order(preds, |_| {
        count += 1;
        count < cap
    });
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn preds_of(edges: &[(u8, u8)], n: usize) -> SmallVec<[u64; 4]> {
        let mut preds = smallvec![0u64; n];
        for &(a, b) in edges {
            preds[b as usize] |= 1 << a;
        }
        preds
    }

    #[test]
    fn order_zero_is_the_declaration_order_when_it_is_topological() {
        // Edges 0→2, 1→2: declaration order [0, 1, 2] is topological.
        let preds = preds_of(&[(0, 2), (1, 2)], 3);
        assert_eq!(
            linearize(&preds, 0).map(|o| o.to_vec()),
            Some(vec![0, 1, 2])
        );
        // The only other order puts 1 before 0.
        assert_eq!(
            linearize(&preds, 1).map(|o| o.to_vec()),
            Some(vec![1, 0, 2])
        );
        assert_eq!(linearize(&preds, 2), None, "exactly two orders exist");
    }

    #[test]
    fn unconstrained_sets_enumerate_every_permutation() {
        let preds = preds_of(&[], 3);
        assert_eq!(topo_order_count(&preds, 1000), 6);
        let mut seen = std::collections::HashSet::new();
        for k in 0..6 {
            let order = linearize(&preds, k).expect("order exists");
            assert_eq!(order.len(), 3);
            seen.insert(order.to_vec());
        }
        assert_eq!(seen.len(), 6, "all orders distinct");
        assert_eq!(linearize(&preds, 6), None);
    }

    #[test]
    fn counts_are_capped() {
        let preds = preds_of(&[], 4);
        assert_eq!(topo_order_count(&preds, 1000), 24);
        assert_eq!(topo_order_count(&preds, 5), 5, "cap reached");
        // 5 unconstrained members admit 120 orders — above the cap.
        let five = preds_of(&[], 5);
        assert_eq!(
            topo_order_count(&five, LINEARIZATION_CAP),
            LINEARIZATION_CAP,
            "capped at the linearization limit"
        );
    }

    #[test]
    fn cyclic_constraints_have_zero_orders() {
        // 0→1 and 1→0: no complete order exists.
        let preds = preds_of(&[(0, 1), (1, 0)], 2);
        assert_eq!(topo_order_count(&preds, 10), 0);
        assert_eq!(linearize(&preds, 0), None);
        // A self-edge is a cycle too.
        let self_loop = preds_of(&[(0, 0)], 1);
        assert_eq!(topo_order_count(&self_loop, 10), 0);
    }

    #[test]
    fn enumeration_is_deterministic_and_lexicographic() {
        let preds = preds_of(&[], 3);
        let orders: Vec<Vec<u8>> = (0..6)
            .map(|k| linearize(&preds, k).expect("order").to_vec())
            .collect();
        let mut sorted = orders.clone();
        sorted.sort();
        assert_eq!(orders, sorted, "ascending-position choice order");
        assert_eq!(orders[0], vec![0, 1, 2]);
    }
}
