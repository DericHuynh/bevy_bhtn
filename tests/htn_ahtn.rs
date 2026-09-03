//! Exhaustive pins for adversarial HTN planning (`src/ahtn.rs`, Ontañón &
//! Buro IJCAI 2015, Algorithm 2 + alpha-beta): minimax semantics against a
//! best-defense opponent, exact interleaved execution ordering, stuck-player
//! semantics, the paper's wait-method completeness pattern, partial-order
//! scheduling, budget/depth bounds, and API validation.

use bevy_bhtn::ahtn::{Ahtn, AhtnOutcome};
use bevy_bhtn::state::PlanState;
use bevy_bhtn::tasks::{TaskBuilder, TaskFn};
use bevy_bhtn::{HtnDomain, HtnError, HtnResult};
use bevy_ecs::prelude::Component;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Gold(pub i32);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Walled(pub bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Bribed(pub bool);
#[derive(Component, Clone, Default, Debug, PartialEq)]
struct Won(pub bool);
#[derive(Component, Clone, Default, Debug)]
struct Seq(Vec<u32>);

/// Always-applicable no-op (a precondition-only primitive).
fn wait(task: &mut TaskBuilder) {
    task.precondition(|g: &Gold| g.0 >= 0);
}

/// The invasion game: max wants gold; the direct way (`ram`) needs the wall
/// down, the costly way (`bribe`) ignores it. Min can build the wall.
fn ram(task: &mut TaskBuilder) {
    task.precondition(|w: &Walled| !w.0)
        .effect(|g: &mut Gold| g.0 = 5);
}
fn bribe(task: &mut TaskBuilder) {
    task.effect(|g: &mut Gold, b: &mut Bribed| {
        g.0 = 5;
        b.0 = true;
    });
}
fn invade(task: &mut TaskBuilder) {
    task.branch().then(wait).then(ram);
    task.branch().then(wait).then(bribe);
}
fn build_wall(task: &mut TaskBuilder) {
    task.effect(|w: &mut Walled| w.0 = true);
}
fn defend(task: &mut TaskBuilder) {
    task.branch().then(build_wall);
}
/// A passive opponent: an empty terminal method.
fn idle(task: &mut TaskBuilder) {
    task.branch();
}

/// Max's payoff: gold minus the bribe's cost.
fn invasion_eval(gold: usize, bribed: usize) -> impl Fn(&PlanState) -> f32 {
    move |s: &PlanState| {
        let g = s.get::<Gold>(gold).0 as f32;
        let b = if s.get::<Bribed>(bribed).0 { 10.0 } else { 0.0 };
        g - b
    }
}

/// `Ahtn::search` with the players' root fn-item types inferred from the fn
/// values (fn-item types cannot be named directly in turbofish).
fn search<MaxRoot: TaskFn, MinRoot: TaskFn>(
    domain: &HtnDomain,
    budget: Option<usize>,
    _max_root: MaxRoot,
    _min_root: MinRoot,
    state: &PlanState,
    eval: impl Fn(&PlanState) -> f32,
    depth: usize,
) -> HtnResult<Option<AhtnOutcome>> {
    let ahtn = Ahtn::new(domain);
    let ahtn = match budget {
        Some(b) => ahtn.with_decomposition_budget(b),
        None => ahtn,
    };
    ahtn.search(_max_root, _min_root, state, eval, depth)
}

// ---------------------------------------------------------------------------
// Minimax semantics
// ---------------------------------------------------------------------------

/// Against a passive opponent, max picks the direct branch: `wait, ram`,
/// value 5 (the bribe's -5 is dominated).
#[test]
fn ahtn_solves_when_opponent_passive() {
    let domain = HtnDomain::from_root(invade).root(idle).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let gold = domain.components.get::<Gold>().unwrap();
    let bribed = domain.components.get::<Bribed>().unwrap();

    let outcome = search(
        &domain,
        None,
        invade,
        idle,
        &state,
        invasion_eval(gold, bribed),
        10,
    )
    .expect("search ok")
    .expect("a plan exists");
    assert_eq!(outcome.value, 5.0);
    assert_eq!(outcome.plan.len(), 2);
    assert_eq!(domain.tasks[outcome.plan[0] as usize].name(), "wait");
    assert_eq!(domain.tasks[outcome.plan[1] as usize].name(), "ram");
}

/// THE adversarial pin: the same max domain, a wall-building opponent — max
/// now picks the bribe (value −5) because the ram branch dies against the
/// wall. Minimax changes the decision; a single-agent planner cannot.
#[test]
fn ahtn_picks_the_branch_that_survives_best_defense() {
    let domain = HtnDomain::from_root(invade).root(defend).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let gold = domain.components.get::<Gold>().unwrap();
    let bribed = domain.components.get::<Bribed>().unwrap();

    let outcome = search(
        &domain,
        None,
        invade,
        defend,
        &state,
        invasion_eval(gold, bribed),
        10,
    )
    .expect("search ok")
    .expect("the bribe branch survives");
    // The ram branch is stuck (the wall is up when it fires); the bribe
    // branch pays 10 for the 5 gold.
    assert_eq!(outcome.value, -5.0);
    assert_eq!(domain.tasks[outcome.plan[1] as usize].name(), "bribe");
}

/// Exact interleaving: max and min primitives alternate in issue order, and
/// the final state reflects every executed action (depth counts both
/// players' actions).
#[test]
fn ahtn_interleaves_execution_in_exact_order() {
    fn inc_a(task: &mut TaskBuilder) {
        task.effect(|t: &mut TicksA| t.0 += 1);
    }
    fn inc_b(task: &mut TaskBuilder) {
        task.effect(|t: &mut TicksB| t.0 += 1);
    }
    fn max_root(task: &mut TaskBuilder) {
        task.branch().then(inc_a).then(inc_a).then(inc_a);
    }
    fn min_root(task: &mut TaskBuilder) {
        task.branch().then(inc_b);
    }

    #[derive(Component, Clone, Default, Debug, PartialEq)]
    #[allow(dead_code)]
    struct TicksA(pub i32);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    #[allow(dead_code)]
    struct TicksB(pub i32);

    let domain = HtnDomain::from_root(max_root)
        .root(min_root)
        .build()
        .unwrap();
    let state = PlanState::build(&domain.components).finish();
    let a = domain.components.get::<TicksA>().unwrap();
    let b = domain.components.get::<TicksB>().unwrap();

    let outcome = search(
        &domain,
        None,
        max_root,
        min_root,
        &state,
        |s| (s.get::<TicksA>(a).0 * 10 + s.get::<TicksB>(b).0) as f32,
        10,
    )
    .expect("search ok")
    .expect("max completes");
    // Depth 10 covers everything: A ticks 3, B ticks 1.
    assert_eq!(outcome.value, 31.0);
    // Max's plan is exactly its three primitives (min's are not in it).
    assert_eq!(outcome.plan.len(), 3);
}

/// A stuck opponent loses the branch: min whose only method is impossible
/// hands max the win (+∞) — max's plan is whatever it executed before min
/// folded.
#[test]
fn ahtn_stuck_opponent_loses_the_branch() {
    fn impossible(task: &mut TaskBuilder) {
        task.precondition(|g: &Gold| g.0 > 1000);
    }
    fn dead_defense(task: &mut TaskBuilder) {
        task.branch().then(impossible);
    }
    let domain = HtnDomain::from_root(invade)
        .root(dead_defense)
        .build()
        .unwrap();
    let state = PlanState::build(&domain.components).finish();
    let gold = domain.components.get::<Gold>().unwrap();
    let bribed = domain.components.get::<Bribed>().unwrap();

    let outcome = search(
        &domain,
        None,
        invade,
        dead_defense,
        &state,
        invasion_eval(gold, bribed),
        10,
    )
    .expect("search ok")
    .expect("max wins");
    assert_eq!(outcome.value, f32::INFINITY);
    // Max's first action executed; then min folded and the game ended.
    assert_eq!(outcome.plan.len(), 1);
    assert_eq!(domain.tasks[outcome.plan[0] as usize].name(), "wait");
}

/// A max branch whose planned primitive fails against the real state is
/// inconsistent: that branch scores −∞. If every branch dies, `search`
/// returns `Ok(None)` — no viable plan.
#[test]
fn ahtn_no_viable_plan_returns_none() {
    fn doomed(task: &mut TaskBuilder) {
        task.branch().then(wait).then(ram); // min always walls first
    }
    let domain = HtnDomain::from_root(doomed).root(defend).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let gold = domain.components.get::<Gold>().unwrap();

    // This domain never registers `Bribed` (no closure touches it), so the
    // eval reads only gold.
    let outcome = search(
        &domain,
        None,
        doomed,
        defend,
        &state,
        |s| s.get::<Gold>(gold).0 as f32,
        10,
    );
    assert!(matches!(outcome, Ok(None)), "every branch is stuck");
}

// ---------------------------------------------------------------------------
// The paper's completeness pattern
// ---------------------------------------------------------------------------

/// The wait-method fallback (the paper's recommended completeness pattern):
/// max's real method is inapplicable until min opens the gate; the fallback
/// method (wait + recurse) keeps max alive across the rounds until it is.
/// Without the fallback, max is stuck immediately.
#[test]
fn ahtn_wait_method_fallback_survives_until_the_gate_opens() {
    fn cross(task: &mut TaskBuilder) {
        task.precondition(|w: &Walled| w.0) // the "gate" is the wall itself
            .effect(|w: &mut Won| w.0 = true);
    }
    fn go(task: &mut TaskBuilder) {
        // Main method: cross when possible.
        task.branch().then(cross);
        // Fallback (the paper's pattern): wait, then try again.
        task.branch().then(wait).then(go);
    }
    fn open_gate(task: &mut TaskBuilder) {
        task.effect(|w: &mut Walled| w.0 = true);
    }
    fn gatekeeper(task: &mut TaskBuilder) {
        task.branch().then(open_gate);
    }

    let domain = HtnDomain::from_root(go).root(gatekeeper).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let won = domain.components.get::<Won>().unwrap();

    let outcome = search(
        &domain,
        None,
        go,
        gatekeeper,
        &state,
        |s| {
            if s.get::<Won>(won).0 {
                1.0
            } else {
                0.0
            }
        },
        10,
    )
    .expect("search ok")
    .expect("the fallback keeps max alive");
    assert_eq!(outcome.value, 1.0);
    let names: Vec<&str> = outcome
        .plan
        .iter()
        .map(|&i| domain.tasks[i as usize].name())
        .collect();
    assert_eq!(names, ["wait", "cross"], "wait once, then cross");
}

// ---------------------------------------------------------------------------
// Bounds & termination
// ---------------------------------------------------------------------------

/// Depth caps the lookahead: with depth 2, only max's first primitive and
/// min's reply execute before evaluation.
#[test]
fn ahtn_depth_caps_the_lookahead() {
    fn inc_a(task: &mut TaskBuilder) {
        task.effect(|t: &mut TicksA| t.0 += 1);
    }
    fn max_root(task: &mut TaskBuilder) {
        task.branch().then(inc_a).then(inc_a).then(inc_a);
    }
    fn idle_min(task: &mut TaskBuilder) {
        task.branch();
    }

    #[derive(Component, Clone, Default, Debug, PartialEq)]
    struct TicksA(pub i32);

    let domain = HtnDomain::from_root(max_root)
        .root(idle_min)
        .build()
        .unwrap();
    let state = PlanState::build(&domain.components).finish();
    let a = domain.components.get::<TicksA>().unwrap();

    let outcome = search(
        &domain,
        None,
        max_root,
        idle_min,
        &state,
        |s| s.get::<TicksA>(a).0 as f32,
        2,
    )
    .expect("search ok")
    .expect("a plan exists");
    // Two actions issued (both max's — min is idle), then evaluation.
    assert_eq!(outcome.plan.len(), 2);
    assert_eq!(outcome.value, 2.0);
}

/// The decomposition budget terminates pure self-recursion: max spiraling
/// alone exhausts it and loses the branch (`Ok(None)`); a spiraling *min*
/// loses its branch to max (+∞).
#[test]
fn ahtn_decomposition_budget_terminates_recursion() {
    fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral);
    }
    fn win(task: &mut TaskBuilder) {
        task.effect(|w: &mut Won| w.0 = true);
    }
    fn simple_max(task: &mut TaskBuilder) {
        task.branch().then(win);
    }

    let domain = HtnDomain::from_root(spiral)
        .root(simple_max)
        .build()
        .unwrap();
    let state = PlanState::build(&domain.components).finish();

    // Max spirals: budget exhausted → stuck → no viable plan. (A small
    // explicit budget: the recursion depth is bounded by the budget, and
    // 1000 nested `node` frames overflow a test thread's 2 MB stack.)
    let outcome =
        search(&domain, Some(50), spiral, simple_max, &state, |_| 0.0, 100).expect("search ok");
    assert!(outcome.is_none());

    // Min spirals: max wins the branch.
    let outcome = search(&domain, Some(50), simple_max, spiral, &state, |_| 0.0, 100)
        .expect("search ok")
        .expect("min folds");
    assert_eq!(outcome.value, f32::INFINITY);
    assert_eq!(domain.tasks[outcome.plan[0] as usize].name(), "win");
}

/// A tight decomposition budget cuts off legitimate deep decompositions —
/// the budget is shared across both players and the whole search.
#[test]
fn ahtn_decomposition_budget_is_shared_and_enforceable() {
    let domain = HtnDomain::from_root(invade).root(defend).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let gold = domain.components.get::<Gold>().unwrap();
    let bribed = domain.components.get::<Bribed>().unwrap();

    // Budget 1: the first method application (max's invade) consumes it;
    // min's defend then cannot decompose → min stuck → +∞ for max.
    let outcome = search(
        &domain,
        Some(1),
        invade,
        defend,
        &state,
        invasion_eval(gold, bribed),
        10,
    )
    .expect("search ok")
    .expect("min cannot decompose");
    assert_eq!(outcome.value, f32::INFINITY);
}

// ---------------------------------------------------------------------------
// Ordering & API
// ---------------------------------------------------------------------------

/// Partially-ordered methods schedule their baked **first** topological
/// order — not declaration order: `b` declared first, `before(a, b)` forces
/// a→b, so the executed order is a then b.
#[test]
fn ahtn_schedules_the_first_topological_order() {
    fn push_one(task: &mut TaskBuilder) {
        task.effect(|s: &mut Seq| s.0.push(1));
    }
    fn push_two(task: &mut TaskBuilder) {
        task.effect(|s: &mut Seq| s.0.push(2));
    }
    fn ordered(task: &mut TaskBuilder) {
        let mut m = task.branch();
        let b = m.subtask(push_two);
        let a = m.subtask(push_one);
        m.before(a, b);
    }
    fn max_root(task: &mut TaskBuilder) {
        task.branch().then(ordered);
    }
    fn idle_min(task: &mut TaskBuilder) {
        task.branch();
    }

    let domain = HtnDomain::from_root(max_root)
        .root(idle_min)
        .build()
        .unwrap();
    let state = PlanState::build(&domain.components).finish();
    let seq_slot = domain.components.get::<Seq>().unwrap();

    let outcome = search(&domain, None, max_root, idle_min, &state, |_| 0.0, 10)
        .expect("search ok")
        .expect("a plan exists");
    assert_eq!(outcome.plan.len(), 2);
    // Execution order (a then b) is observable through the pushed sequence.
    let executed = {
        let mut s = state.clone();
        for &i in &outcome.plan {
            if let bevy_bhtn::Task::Primitive(p) = &domain.tasks[i as usize] {
                for e in &p.effects {
                    e.apply(&mut s);
                }
            }
        }
        s.get::<Seq>(seq_slot).0.clone()
    };
    assert_eq!(
        executed,
        [1, 2],
        "a runs before b despite declaration order"
    );
}

/// Unknown roots error; primitive roots error (both players must
/// decompose). The unknown root is a real fn item that was simply never
/// registered in the domain.
#[test]
fn ahtn_rejects_bad_roots() {
    fn never_registered(_task: &mut TaskBuilder) {}

    let domain = HtnDomain::from_root(invade).root(defend).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let gold = domain.components.get::<Gold>().unwrap();
    let bribed = domain.components.get::<Bribed>().unwrap();
    let eval = invasion_eval(gold, bribed);

    let err = search(&domain, None, never_registered, defend, &state, &eval, 10).unwrap_err();
    assert!(matches!(err, HtnError::UnregisteredTask { .. }));

    // `ram` is a primitive, not a compound root.
    let err = search(&domain, None, ram, defend, &state, &eval, 10).unwrap_err();
    assert!(err.to_string().contains("must be a compound task"));
}

/// Zero depth evaluates the initial state immediately (empty plan).
#[test]
fn ahtn_zero_depth_evaluates_immediately() {
    let domain = HtnDomain::from_root(invade).root(idle).build().unwrap();
    let state = PlanState::build(&domain.components).finish();
    let gold = domain.components.get::<Gold>().unwrap();
    let bribed = domain.components.get::<Bribed>().unwrap();

    let outcome = search(
        &domain,
        None,
        invade,
        idle,
        &state,
        invasion_eval(gold, bribed),
        0,
    )
    .expect("search ok")
    .expect("finite eval");
    assert_eq!(outcome.value, 0.0);
    assert!(outcome.plan.is_empty());
}

// ---------------------------------------------------------------------------
// The dungeon: deep dynamic scenario with multiple reactive adversaries
// ---------------------------------------------------------------------------

/// A dungeon raid against a **coordinated garrison of three monsters**, each
/// reacting to a different signature max leaves:
///
/// - the **gatekeeper** pickproofs the gate when it has been picked (killing
///   the stealth route at the gate);
/// - the **dragon keeper** feeds the dragon when any intrusion is detected
///   (killing the stealth route at the hall — a fed dragon hears everything);
/// - the **warden** alarms the hall when the gate has been smashed (killing
///   the force route).
///
/// Max's three routes: **stealth** (pick → sneak, 7 primitives), **force**
/// (smash → charge, 7 primitives), and **sewers** (bypass everything, 4
/// primitives, worth less). The winning plan is a function of which
/// monsters react — the tests pin every combination.
mod dungeon {
    use super::*;

    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct GatePicked(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct GateSmashed(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct GateReinforced(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct GatePickproofed(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct HasPotion(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct PotionDrunk(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct DragonFed(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct DragonSlain(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct HoardTaken(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct SewersUsed(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct HallAlarmed(pub bool);
    #[derive(Component, Clone, Default, Debug, PartialEq)]
    pub struct PotionPoisoned(pub bool);

    // -- max's primitives (13 unique actions) ----------------------------
    pub fn pick_lock(task: &mut TaskBuilder) {
        task.precondition(
            |p: &GatePicked,
             s: &GateSmashed,
             r: &GateReinforced,
             f: &GatePickproofed,
             dead: &HoardTaken| { !p.0 && !s.0 && !r.0 && !f.0 && !dead.0 },
        )
        .effect(|p: &mut GatePicked| p.0 = true);
    }
    pub fn smash_gate(task: &mut TaskBuilder) {
        task.precondition(
            |p: &GatePicked,
             s: &GateSmashed,
             r: &GateReinforced,
             f: &GatePickproofed,
             dead: &HoardTaken| { !p.0 && !s.0 && !r.0 && !f.0 && !dead.0 },
        )
        .effect(|s: &mut GateSmashed| s.0 = true);
    }
    pub fn sneak_gate(task: &mut TaskBuilder) {
        task.precondition(
            |p: &GatePicked, r: &GateReinforced, f: &GatePickproofed, dead: &HoardTaken| {
                p.0 && !r.0 && !f.0 && !dead.0
            },
        );
    }
    pub fn charge_gate(task: &mut TaskBuilder) {
        task.precondition(|s: &GateSmashed, r: &GateReinforced, dead: &HoardTaken| {
            s.0 && !r.0 && !dead.0
        });
    }
    pub fn grab_potion(task: &mut TaskBuilder) {
        task.precondition(
            |p: &GatePicked, s: &GateSmashed, r: &GateReinforced, dead: &HoardTaken| {
                (p.0 || s.0) && !r.0 && !dead.0
            },
        )
        .effect(|p: &mut HasPotion| p.0 = true);
    }
    pub fn drink_potion(task: &mut TaskBuilder) {
        task.precondition(|p: &HasPotion, x: &PotionPoisoned, dead: &HoardTaken| {
            p.0 && !x.0 && !dead.0
        })
        .effect(|p: &mut HasPotion, d: &mut PotionDrunk| {
            p.0 = false;
            d.0 = true;
        });
    }
    pub fn sneak_hall(task: &mut TaskBuilder) {
        task.precondition(|f: &DragonFed, d: &PotionDrunk, dead: &HoardTaken| {
            (!f.0 || d.0) && !dead.0
        });
    }
    pub fn charge_hall(task: &mut TaskBuilder) {
        task.precondition(|a: &HallAlarmed, d: &PotionDrunk, dead: &HoardTaken| {
            !a.0 && d.0 && !dead.0
        });
    }
    pub fn strike_dragon(task: &mut TaskBuilder) {
        task.precondition(|dead: &HoardTaken| !dead.0)
            .effect(|d: &mut DragonSlain| d.0 = true);
    }
    pub fn take_hoard(task: &mut TaskBuilder) {
        task.precondition(|d: &DragonSlain, s: &SewersUsed, dead: &HoardTaken| {
            (d.0 || s.0) && !dead.0
        })
        .effect(|h: &mut HoardTaken| h.0 = true);
    }
    pub fn enter_sewers(task: &mut TaskBuilder) {
        task.precondition(|dead: &HoardTaken| !dead.0);
    }
    pub fn wade_tunnel(task: &mut TaskBuilder) {
        task.precondition(|dead: &HoardTaken| !dead.0);
    }
    pub fn climb_out(task: &mut TaskBuilder) {
        task.precondition(|dead: &HoardTaken| !dead.0)
            .effect(|s: &mut SewersUsed| s.0 = true);
    }

    // -- max's three routes ----------------------------------------------
    /// Stealth: pick the lock, sneak the gate, potion up, sneak the hall,
    /// strike, hoard — 7 primitives.
    pub fn stealth(task: &mut TaskBuilder) {
        task.branch()
            .then(pick_lock)
            .then(sneak_gate)
            .then(grab_potion)
            .then(drink_potion)
            .then(sneak_hall)
            .then(strike_dragon)
            .then(take_hoard);
    }
    /// Force: smash the gate, charge the gate, potion up, charge the hall,
    /// strike, hoard — 7 primitives.
    pub fn force(task: &mut TaskBuilder) {
        task.branch()
            .then(smash_gate)
            .then(charge_gate)
            .then(grab_potion)
            .then(drink_potion)
            .then(charge_hall)
            .then(strike_dragon)
            .then(take_hoard);
    }
    /// Sewers: bypass everything — 4 primitives, worth less (the eval
    /// discounts a sewer entrance).
    pub fn sewers(task: &mut TaskBuilder) {
        task.branch()
            .then(enter_sewers)
            .then(wade_tunnel)
            .then(climb_out)
            .then(take_hoard);
    }
    /// The campaign: try stealth, then force, then the sewers.
    pub fn raid(task: &mut TaskBuilder) {
        task.branch().then(stealth);
        task.branch().then(force);
        task.branch().then(sewers);
    }

    // -- the garrison (three reactive monsters) ---------------------------
    /// The gatekeeper pickproofs a picked gate; otherwise holds.
    pub fn gatekeeper_turn(task: &mut TaskBuilder) {
        task.branch().then(pickproof);
        task.branch().then(hold);
    }
    pub fn pickproof(task: &mut TaskBuilder) {
        task.precondition(|p: &GatePicked| p.0)
            .effect(|f: &mut GatePickproofed| f.0 = true);
    }
    /// The dragon keeper poisons the potion stash at the first sign of
    /// intrusion — both gate routes then die at `drink_potion`, leaving
    /// only the sewers. (Otherwise holds.)
    pub fn dragon_keeper_turn(task: &mut TaskBuilder) {
        task.branch().then(poison);
        task.branch().then(hold);
    }
    pub fn poison(task: &mut TaskBuilder) {
        task.precondition(|p: &GatePicked, s: &GateSmashed| p.0 || s.0)
            .effect(|x: &mut PotionPoisoned| x.0 = true);
    }
    /// The warden alarms the hall when the gate has been smashed.
    pub fn warden_turn(task: &mut TaskBuilder) {
        task.branch().then(alarm);
        task.branch().then(hold);
    }
    pub fn alarm(task: &mut TaskBuilder) {
        task.precondition(|s: &GateSmashed| s.0)
            .effect(|a: &mut HallAlarmed| a.0 = true);
    }
    /// Hold position: always applicable (the paper's wait action — min must
    /// never be spuriously stuck, e.g. after max has already taken the
    /// hoard).
    pub fn hold(_task: &mut TaskBuilder) {}
}

/// The dungeon scenario helper: builds the domain, state, and eval.
fn dungeon_setup(
    min_root: impl bevy_bhtn::tasks::TaskFn,
) -> (HtnDomain, PlanState, impl Fn(&PlanState) -> f32) {
    let domain = HtnDomain::from_root(dungeon::raid)
        .root(min_root)
        .build()
        .unwrap();
    let state = PlanState::build(&domain.components).finish();
    let hoard = domain.components.get::<dungeon::HoardTaken>().unwrap();
    let sewers = domain.components.get::<dungeon::SewersUsed>().unwrap();
    let eval = move |s: &PlanState| {
        if s.get::<dungeon::HoardTaken>(hoard).0 {
            if s.get::<dungeon::SewersUsed>(sewers).0 {
                50.0
            } else {
                100.0
            }
        } else {
            0.0
        }
    };
    (domain, state, eval)
}

/// Against a fully passive garrison, max takes the stealth route (the first
/// route tried; equal value with force keeps the first) — 7 primitives,
/// pinned by name in order.
#[test]
fn dungeon_passive_garrison_stealth_wins() {
    fn passive(task: &mut TaskBuilder) {
        task.branch()
            .then(dungeon::hold)
            .then(dungeon::hold)
            .then(dungeon::hold);
    }
    let (domain, state, eval) = dungeon_setup(passive);
    let outcome = search(&domain, None, dungeon::raid, passive, &state, eval, 40)
        .expect("search ok")
        .expect("the stealth route works");
    assert_eq!(outcome.value, 100.0);
    let names: Vec<&str> = outcome
        .plan
        .iter()
        .map(|&i| domain.tasks[i as usize].name())
        .collect();
    assert_eq!(
        names,
        [
            "pick_lock",
            "sneak_gate",
            "grab_potion",
            "drink_potion",
            "sneak_hall",
            "strike_dragon",
            "take_hoard"
        ]
    );
}

/// The gatekeeper reacts to a picked gate: the stealth route dies at the
/// gate, and max falls through to the force route — 7 different primitives,
/// pinned by name in order.
#[test]
fn dungeon_gatekeeper_forces_the_smash_route() {
    fn gatekeeper_only(task: &mut TaskBuilder) {
        task.branch().then(dungeon::gatekeeper_turn);
        task.branch().then(dungeon::hold).then(dungeon::hold);
    }
    let (domain, state, eval) = dungeon_setup(gatekeeper_only);
    let outcome = search(
        &domain,
        None,
        dungeon::raid,
        gatekeeper_only,
        &state,
        eval,
        40,
    )
    .expect("search ok")
    .expect("the force route works");
    assert_eq!(outcome.value, 100.0);
    let names: Vec<&str> = outcome
        .plan
        .iter()
        .map(|&i| domain.tasks[i as usize].name())
        .collect();
    assert_eq!(
        names,
        [
            "smash_gate",
            "charge_gate",
            "grab_potion",
            "drink_potion",
            "charge_hall",
            "strike_dragon",
            "take_hoard"
        ]
    );
}

/// The dragon keeper feeds the dragon: the stealth route dies at the hall
/// The dragon keeper poisons the potion stash: both gate routes die at
/// `drink_potion` (the poison is in the stash, not the gate), leaving only
/// the sewers — worth less (50).
#[test]
fn dungeon_dragon_keeper_forces_the_sewers() {
    fn dragon_keeper_only(task: &mut TaskBuilder) {
        task.branch()
            .then(dungeon::hold)
            .then(dungeon::dragon_keeper_turn)
            .then(dungeon::hold);
    }
    let (domain, state, eval) = dungeon_setup(dragon_keeper_only);
    let outcome = search(
        &domain,
        None,
        dungeon::raid,
        dragon_keeper_only,
        &state,
        eval,
        40,
    )
    .expect("search ok")
    .expect("the sewers remain");
    assert_eq!(outcome.value, 50.0);
    let names: Vec<&str> = outcome
        .plan
        .iter()
        .map(|&i| domain.tasks[i as usize].name())
        .collect();
    assert_eq!(
        names,
        ["enter_sewers", "wade_tunnel", "climb_out", "take_hoard"]
    );
}

/// The warden alarms the hall on a smashed gate: the force route dies, and
/// the stealth route (quiet gate, quiet hall) wins.
#[test]
fn dungeon_warden_forces_the_stealth_route() {
    fn warden_only(task: &mut TaskBuilder) {
        task.branch()
            .then(dungeon::hold)
            .then(dungeon::hold)
            .then(dungeon::warden_turn);
    }
    let (domain, state, eval) = dungeon_setup(warden_only);
    let outcome = search(&domain, None, dungeon::raid, warden_only, &state, eval, 40)
        .expect("search ok")
        .expect("the stealth route works");
    assert_eq!(outcome.value, 100.0);
    let names: Vec<&str> = outcome
        .plan
        .iter()
        .map(|&i| domain.tasks[i as usize].name())
        .collect();
    assert_eq!(
        names,
        [
            "pick_lock",
            "sneak_gate",
            "grab_potion",
            "drink_potion",
            "sneak_hall",
            "strike_dragon",
            "take_hoard"
        ]
    );
}

/// The full garrison reacts: pickproof kills stealth, the alarm kills force
/// — only the sewers remain, worth less (50). Max takes them.
#[test]
fn dungeon_full_garrison_forces_the_sewers() {
    fn full_garrison(task: &mut TaskBuilder) {
        task.branch()
            .then(dungeon::gatekeeper_turn)
            .then(dungeon::dragon_keeper_turn)
            .then(dungeon::warden_turn);
    }
    let (domain, state, eval) = dungeon_setup(full_garrison);
    let outcome = search(
        &domain,
        None,
        dungeon::raid,
        full_garrison,
        &state,
        eval,
        40,
    )
    .expect("search ok")
    .expect("the sewers remain");
    assert_eq!(outcome.value, 50.0);
    let names: Vec<&str> = outcome
        .plan
        .iter()
        .map(|&i| domain.tasks[i as usize].name())
        .collect();
    assert_eq!(
        names,
        ["enter_sewers", "wade_tunnel", "climb_out", "take_hoard"]
    );
}

/// The garrison's composition is the difficulty knob: every subset of
/// reacting monsters yields a different (or equal-but-different) plan, and
/// the value never exceeds the passive case. Each garrison is a named
/// wrapper fn — fn items, never fn pointers (a pointer coerces the four
/// distinct identities into one type, the same trap as `any_order` arrays).
#[test]
fn dungeon_garrison_matrix_is_monotone() {
    fn garrison_passive(task: &mut TaskBuilder) {
        task.branch().then(dungeon::hold);
        task.branch().then(dungeon::hold);
        task.branch().then(dungeon::hold);
    }
    fn garrison_pickproof(task: &mut TaskBuilder) {
        task.branch().then(dungeon::gatekeeper_turn);
        task.branch().then(dungeon::dragon_keeper_turn);
        task.branch().then(dungeon::hold);
    }
    fn garrison_alarm(task: &mut TaskBuilder) {
        task.branch().then(dungeon::hold);
        task.branch().then(dungeon::dragon_keeper_turn);
        task.branch().then(dungeon::warden_turn);
    }
    fn garrison_full(task: &mut TaskBuilder) {
        task.branch().then(dungeon::gatekeeper_turn);
        task.branch().then(dungeon::dragon_keeper_turn);
        task.branch().then(dungeon::warden_turn);
    }

    let (domain, state, eval) = dungeon_setup(garrison_passive);
    let value = search(
        &domain,
        None,
        dungeon::raid,
        garrison_passive,
        &state,
        eval,
        40,
    )
    .expect("search ok")
    .expect("a plan exists")
    .value;
    assert_eq!(
        value, 100.0,
        "passive: stealth wins (no intrusion, no poison)"
    );

    let (domain, state, eval) = dungeon_setup(garrison_pickproof);
    let value = search(
        &domain,
        None,
        dungeon::raid,
        garrison_pickproof,
        &state,
        eval,
        40,
    )
    .expect("search ok")
    .expect("a plan exists")
    .value;
    assert_eq!(
        value, 50.0,
        "gatekeeper + reactive keeper: stealth dies at the gate, force dies at the poisoned drink — sewers"
    );

    let (domain, state, eval) = dungeon_setup(garrison_alarm);
    let value = search(
        &domain,
        None,
        dungeon::raid,
        garrison_alarm,
        &state,
        eval,
        40,
    )
    .expect("search ok")
    .expect("a plan exists")
    .value;
    assert_eq!(
        value, 50.0,
        "warden + reactive keeper: both gate routes die — sewers"
    );

    let (domain, state, eval) = dungeon_setup(garrison_full);
    let value = search(
        &domain,
        None,
        dungeon::raid,
        garrison_full,
        &state,
        eval,
        40,
    )
    .expect("search ok")
    .expect("a plan exists")
    .value;
    assert_eq!(value, 50.0, "full garrison: only the sewers remain");
}
