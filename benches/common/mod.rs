//! Shared bench/test scaffolding for `bevy_bhtn`.
//!
//! Single source of truth for everything the benchmarks plan: the **function-
//! defined fixture domains** (miner, outpost, gate chain, doomed recursion),
//! the per-agent component types, the initial-state constructors, and the
//! plan-**execution** helpers. The integration tests include this file
//! directly (`#[path]`) so the "benchmarks produce correct plans" pins run
//! against *exactly* the code the benches run — same components, same domains,
//! same execution semantics.

// Each bench/test target includes this module and uses only its slice; the
// rest is intentionally shared.
#![allow(dead_code)]

use bevy_bhtn::planner::Plan;
use bevy_bhtn::state::PlanState;
use bevy_bhtn::{HtnDomain, Task, TaskBuilder};
use bevy_ecs::prelude::*;
use ustr::Ustr;

// ---------------------------------------------------------------------------
// Miner domain components
// ---------------------------------------------------------------------------

/// The miner's gold count.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Gold(pub i32);
/// Whether the miner is carrying ore.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct HasOre(pub bool);
/// Whether the miner is carrying metal.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct HasMetal(pub bool);
/// The miner's energy level.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Energy(pub i32);
/// The miner's hunger level.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Hunger(pub i32);
/// The world-map location the miner occupies.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub enum Location {
    #[default]
    Outside,
    House,
    Ore,
    Smelter,
    Mushroom,
    Merchant,
}

/// The bench's deterministic per-entity initial components (`i`-th entity of
/// the spawn batch).
///
/// The residue classes must stay inside the domain's **solvable envelope**:
/// the canonical miner has no wired-in `eat`/`sleep` (they are defined but
/// unreferenced by `earn_gold`), so every earning step's `hunger < 75` makes
/// a seed with `i % 60 ∈ 55..60` genuinely unplannable (`Err(NoPlan)`) — that
/// span is why the modulus is 55 (hunger 20..74), not 60. The full-envelope
/// solvability pin in `tests/htn_bench_plans.rs` enforces this for every
/// residue class of the spawn batch.
pub fn miner_components(i: usize) -> impl Bundle {
    (
        Gold((i % 5) as i32),
        HasOre(i.is_multiple_of(3)),
        HasMetal(i.is_multiple_of(7)),
        Energy(80 - (i % 40) as i32),
        Hunger(20 + (i % 55) as i32),
        Location::Outside,
    )
}

/// The bench's deterministic initial scratchpad for the `i`-th actor.
///
/// Same solvable-envelope contract as [`miner_components`] (hunger stays
/// below the 75 work threshold).
pub fn miner_scratch(domain: &HtnDomain, i: usize) -> PlanState {
    let (gold, has_ore, has_metal, energy, hunger) = (
        (i % 5) as i32,
        i.is_multiple_of(3),
        i.is_multiple_of(7),
        80 - (i % 40) as i32,
        20 + (i % 55) as i32,
    );
    PlanState::build(&domain.components)
        .set(Gold(gold))
        .set(HasOre(has_ore))
        .set(HasMetal(has_metal))
        .set(Energy(energy))
        .set(Hunger(hunger))
        .set(Location::Outside)
        .finish()
}

// ---------------------------------------------------------------------------
// Miner domain — task functions (1:1 port of the former miner fixture)
// ---------------------------------------------------------------------------

pub mod miner_tasks {
    use super::*;

    pub fn earn_gold(task: &mut TaskBuilder) {
        task.branch().precondition(|gold: &Gold| gold.0 >= 3);
        task.branch().then(turn_metal_into_gold).then(earn_gold);
        task.branch().then(turn_ore_into_metal).then(earn_gold);
        task.branch()
            .precondition(|has_ore: &HasOre| !has_ore.0)
            .then(go_to_ore)
            .then(mine_ore)
            .then(earn_gold);
    }

    pub fn turn_ore_into_metal(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|has_ore: &HasOre, loc: &Location| has_ore.0 && *loc != Location::Smelter)
            .then(go_to_smelter)
            .then(turn_ore_into_metal);
        task.branch()
            .precondition(|has_ore: &HasOre, loc: &Location| has_ore.0 && *loc == Location::Smelter)
            .then(smelt_ore)
            .then(go_to_outside);
    }

    pub fn turn_metal_into_gold(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|has_metal: &HasMetal, loc: &Location| {
                has_metal.0 && *loc != Location::Merchant
            })
            .then(go_to_merchant)
            .then(turn_metal_into_gold);
        task.branch()
            .precondition(|has_metal: &HasMetal, loc: &Location| {
                has_metal.0 && *loc == Location::Merchant
            })
            .then(sell_metal)
            .then(go_to_outside);
    }

    pub fn miner_done(task: &mut TaskBuilder) {
        task.precondition(|gold: &Gold| gold.0 >= 3);
    }

    pub fn eat(task: &mut TaskBuilder) {
        task.precondition(|hunger: &Hunger, loc: &Location| {
            hunger.0 > 50 && *loc == Location::Mushroom
        })
        .effect(|hunger: &mut Hunger| hunger.0 -= 25)
        .effect(|loc: &mut Location| *loc = Location::Outside);
    }

    pub fn sleep(task: &mut TaskBuilder) {
        task.precondition(|energy: &Energy, loc: &Location| {
            energy.0 < 50 && *loc == Location::House
        })
        .effect(|energy: &mut Energy| energy.0 += 100);
    }

    pub fn mine_ore(task: &mut TaskBuilder) {
        task.precondition(|energy: &Energy, hunger: &Hunger, loc: &Location| {
            energy.0 > 10 && hunger.0 < 75 && *loc == Location::Ore
        })
        .effect(|has_ore: &mut HasOre| has_ore.0 = true)
        .effect(|loc: &mut Location| *loc = Location::Outside);
    }

    pub fn smelt_ore(task: &mut TaskBuilder) {
        task.precondition(
            |energy: &Energy, hunger: &Hunger, loc: &Location, has_ore: &HasOre| {
                energy.0 > 10 && hunger.0 < 75 && *loc == Location::Smelter && has_ore.0
            },
        )
        .effect(|has_ore: &mut HasOre| has_ore.0 = false)
        .effect(|has_metal: &mut HasMetal| has_metal.0 = true);
    }

    pub fn sell_metal(task: &mut TaskBuilder) {
        task.precondition(
            |energy: &Energy, hunger: &Hunger, loc: &Location, has_metal: &HasMetal| {
                energy.0 > 10 && hunger.0 < 75 && *loc == Location::Merchant && has_metal.0
            },
        )
        .effect(|gold: &mut Gold| gold.0 += 1)
        .effect(|has_metal: &mut HasMetal| has_metal.0 = false);
    }

    pub fn go_to_outside(task: &mut TaskBuilder) {
        task.effect(|loc: &mut Location| *loc = Location::Outside);
    }

    pub fn go_to_house(task: &mut TaskBuilder) {
        task.precondition(|loc: &Location| *loc == Location::Outside)
            .effect(|loc: &mut Location| *loc = Location::House);
    }

    pub fn go_to_mushroom(task: &mut TaskBuilder) {
        task.precondition(|loc: &Location| *loc == Location::Outside)
            .effect(|loc: &mut Location| *loc = Location::Mushroom);
    }

    pub fn go_to_ore(task: &mut TaskBuilder) {
        task.effect(|loc: &mut Location| *loc = Location::Ore);
    }

    pub fn go_to_smelter(task: &mut TaskBuilder) {
        task.effect(|loc: &mut Location| *loc = Location::Smelter);
    }

    pub fn go_to_merchant(task: &mut TaskBuilder) {
        task.effect(|loc: &mut Location| *loc = Location::Merchant);
    }
}

/// The flat miner domain (the canonical `bevy_htn`/`bevy_dogoap` miner
/// example) — a faithful function-defined port of the former text fixture.
pub fn miner_domain() -> HtnDomain {
    use miner_tasks::*;
    HtnDomain::from_root(earn_gold)
        .build()
        .expect("miner domain is well-formed")
}

// ---------------------------------------------------------------------------
// Outpost domain components
// ---------------------------------------------------------------------------

/// The colonist's fuel reserve.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Fuel(pub i32);
/// The colonist's food reserve.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Food(pub i32);
/// The colonist's health.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Health(pub i32);
/// The squad's morale.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Morale(pub i32);
/// The squad's ammunition.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Ammo(pub i32);
/// Whether the perimeter is patrolled.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Perimeter(pub bool);
/// Whether the squad is reinforced.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Reinforced(pub bool);
/// Whether the vehicles are armored.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Armored(pub bool);
/// Whether the caches are secured.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Caches(pub bool);
/// Where the colonist is physically posted.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub enum Zone {
    #[default]
    Outside,
    Posting,
    Rally,
    Armory,
}

/// The colonist's physical location — the same component as [`Zone`] (the
/// outpost task closures read/write it under both names).
pub type Position = Zone;

/// The bench's fresh-actor components (plenty of everything).
pub fn fresh_outpost() -> impl Bundle {
    (
        Fuel(5),
        Food(30),
        Health(80),
        Morale(50),
        Ammo(12),
        Zone::Outside,
    )
}

/// The bench's marginal-fuel components (the "queue anyway" trap is eligible
/// but its `drive` leaf fails — genuine in-branch backtracking).
pub fn marginal_outpost() -> impl Bundle {
    (Fuel(3), Food(30), Health(1), Zone::Outside)
}

/// The bench's high-fuel components (the direct drive branch).
pub fn high_fuel_outpost() -> impl Bundle {
    (Fuel(10), Food(0), Health(90), Zone::Outside)
}

/// The bench's initial scratchpad for an outpost actor.
pub fn outpost_scratch(domain: &HtnDomain, b: impl Bundle) -> PlanState {
    let mut world = World::new();
    let e = world.spawn(b).id();
    PlanState::extract(&world, e, &domain.components)
}

// ---------------------------------------------------------------------------
// Outpost domain — task functions (1:1 port of the former outpost fixture)
// ---------------------------------------------------------------------------

pub mod outpost_tasks {
    use super::*;

    pub fn secure_outpost(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|perimeter: &Perimeter| !perimeter.0)
            .then(perimeter_patrol)
            .then(secure_outpost);
        task.branch()
            .precondition(|reinforced: &Reinforced| !reinforced.0)
            .then(reinforce_squad)
            .then(secure_outpost);
        task.branch()
            .precondition(|armored: &Armored| !armored.0)
            .then(armor_vehicles)
            .then(secure_outpost);
        task.branch()
            .precondition(|caches: &Caches| !caches.0)
            .then(secure_cache)
            .then(secure_outpost);
        task.branch()
            .precondition(|morale: &Morale| morale.0 < 5)
            .then(rest)
            .then(secure_outpost);
        task.branch().precondition(
            |perimeter: &Perimeter,
             reinforced: &Reinforced,
             armored: &Armored,
             caches: &Caches,
             morale: &Morale| {
                perimeter.0 && reinforced.0 && armored.0 && caches.0 && morale.0 >= 5
            },
        );
    }

    #[allow(dead_code)]
    pub fn outpost_done(task: &mut TaskBuilder) {
        task.precondition(|perimeter: &Perimeter| perimeter.0);
    }

    pub fn perimeter_patrol(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|pos: &Position| *pos != Zone::Posting)
            .then(reach_posting)
            .then(perimeter_patrol);
        task.branch()
            .precondition(|pos: &Position| *pos == Zone::Posting)
            .then(watch_post);
    }

    pub fn reach_posting(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|fuel: &Fuel| fuel.0 >= 8)
            .then(drive);
        task.branch()
            .precondition(|fuel: &Fuel| fuel.0 >= 2)
            .then(drive);
        task.branch()
            .precondition(|food: &Food| food.0 >= 20)
            .then(hike);
        task.branch()
            .precondition(|health: &Health| health.0 > 5)
            .then(march);
        task.branch().then(rest).then(march);
    }

    pub fn drive(task: &mut TaskBuilder) {
        task.precondition(|fuel: &Fuel| fuel.0 >= 8)
            .effect(|pos: &mut Position| *pos = Zone::Posting)
            .effect(|fuel: &mut Fuel| fuel.0 -= 1);
    }

    pub fn hike(task: &mut TaskBuilder) {
        task.precondition(|food: &Food| food.0 >= 20)
            .effect(|pos: &mut Position| *pos = Zone::Posting)
            .effect(|food: &mut Food| food.0 -= 2);
    }

    pub fn march(task: &mut TaskBuilder) {
        task.precondition(|health: &Health| health.0 > 5)
            .effect(|pos: &mut Position| *pos = Zone::Posting)
            .effect(|food: &mut Food| food.0 -= 2);
    }

    pub fn watch_post(task: &mut TaskBuilder) {
        task.precondition(|pos: &Position| *pos == Zone::Posting)
            .effect(|perimeter: &mut Perimeter| perimeter.0 = true);
    }

    pub fn reinforce_squad(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|pos: &Position| *pos != Zone::Rally)
            .then(to_rally)
            .then(reinforce_squad);
        task.branch()
            .precondition(|pos: &Position| *pos == Zone::Rally)
            .then(rally);
    }

    pub fn to_rally(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|fuel: &Fuel| fuel.0 >= 5)
            .then(convoy);
        task.branch()
            .precondition(|fuel: &Fuel| fuel.0 >= 2)
            .then(siphon)
            .then(convoy);
        task.branch()
            .precondition(|health: &Health| health.0 > 8)
            .then(hike);
        task.branch()
            .precondition(|ammo: &Ammo| ammo.0 >= 10)
            .then(march);
        task.branch().then(walk);
    }

    pub fn siphon(task: &mut TaskBuilder) {
        task.effect(|fuel: &mut Fuel| fuel.0 += 2);
    }

    pub fn convoy(task: &mut TaskBuilder) {
        task.precondition(|fuel: &Fuel| fuel.0 >= 4)
            .effect(|pos: &mut Position| *pos = Zone::Rally)
            .effect(|fuel: &mut Fuel| fuel.0 -= 1);
    }

    pub fn walk(task: &mut TaskBuilder) {
        task.effect(|pos: &mut Position| *pos = Zone::Rally);
    }

    pub fn rally(task: &mut TaskBuilder) {
        task.precondition(|pos: &Position| *pos == Zone::Rally)
            .effect(|reinforced: &mut Reinforced| reinforced.0 = true);
    }

    pub fn armor_vehicles(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|pos: &Position| *pos == Zone::Armory)
            .then(bolt_armor);
        task.branch()
            .precondition(|pos: &Position| *pos != Zone::Armory)
            .then(to_armory)
            .then(armor_vehicles);
    }

    pub fn to_armory(task: &mut TaskBuilder) {
        task.branch()
            .precondition(|fuel: &Fuel| fuel.0 >= 3)
            .then(supply);
        task.branch()
            .precondition(|ammo: &Ammo| ammo.0 >= 5)
            .then(approach);
        task.branch().then(advance);
    }

    pub fn supply(task: &mut TaskBuilder) {
        task.effect(|pos: &mut Position| *pos = Zone::Armory)
            .effect(|fuel: &mut Fuel| fuel.0 -= 3);
    }

    pub fn approach(task: &mut TaskBuilder) {
        task.effect(|pos: &mut Position| *pos = Zone::Armory);
    }

    pub fn advance(task: &mut TaskBuilder) {
        task.effect(|pos: &mut Position| *pos = Zone::Armory);
    }

    pub fn bolt_armor(task: &mut TaskBuilder) {
        task.precondition(|pos: &Position| *pos == Zone::Armory)
            .effect(|armored: &mut Armored| armored.0 = true);
    }

    pub fn secure_cache(task: &mut TaskBuilder) {
        task.branch().then(clear_cache);
        task.branch()
            .precondition(|morale: &Morale| morale.0 >= 30)
            .then(escort);
        task.branch()
            .precondition(|health: &Health| health.0 < 30)
            .then(rest)
            .then(secure_cache);
    }

    pub fn clear_cache(task: &mut TaskBuilder) {
        task.effect(|caches: &mut Caches| caches.0 = true);
    }

    pub fn escort(task: &mut TaskBuilder) {
        task.effect(|caches: &mut Caches| caches.0 = true)
            .effect(|morale: &mut Morale| morale.0 += 5);
    }

    pub fn rest(task: &mut TaskBuilder) {
        task.effect(|morale: &mut Morale| morale.0 = 100)
            .effect(|health: &mut Health| health.0 = 100);
    }
}

/// The deep outpost domain (depth >= 5, dozens of methods, genuine
/// backtracking) — a faithful function-defined port of the former text
/// fixture.
pub fn outpost_domain() -> HtnDomain {
    use outpost_tasks::*;
    HtnDomain::from_root(secure_outpost)
        .build()
        .expect("outpost domain is well-formed")
}

// ---------------------------------------------------------------------------
// Look-ahead A/B domains
// ---------------------------------------------------------------------------

/// The gate chain's gold reserve (only ever written by `strike`).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct GateGold(pub i32);
/// Noise flag flipped by the gate junk tasks.
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
pub struct Noise(pub bool);

pub mod gate_tasks {
    use super::*;

    pub fn gate_root(task: &mut TaskBuilder) {
        task.branch()
            .then(gate0)
            .then(gate1)
            .then(gate2)
            .then(gate3)
            .then(gate4)
            .then(gate5)
            .then(gate6)
            .then(gate7)
            .then(gate8)
            .then(gate9)
            .then(gate10)
            .then(gate11)
            .then(gate_final);
        task.branch().then(strike).then(gate_final);
    }

    pub fn gate_final(task: &mut TaskBuilder) {
        task.precondition(|gold: &GateGold| gold.0 > 1000);
    }

    pub fn strike(task: &mut TaskBuilder) {
        task.effect(|gold: &mut GateGold| gold.0 = 2000);
    }

    pub fn gate0(task: &mut TaskBuilder) {
        task.branch().then(junk_a0);
        task.branch().then(junk_b0);
    }
    pub fn junk_a0(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b0(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate1(task: &mut TaskBuilder) {
        task.branch().then(junk_a1);
        task.branch().then(junk_b1);
    }
    pub fn junk_a1(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b1(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate2(task: &mut TaskBuilder) {
        task.branch().then(junk_a2);
        task.branch().then(junk_b2);
    }
    pub fn junk_a2(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b2(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate3(task: &mut TaskBuilder) {
        task.branch().then(junk_a3);
        task.branch().then(junk_b3);
    }
    pub fn junk_a3(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b3(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate4(task: &mut TaskBuilder) {
        task.branch().then(junk_a4);
        task.branch().then(junk_b4);
    }
    pub fn junk_a4(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b4(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate5(task: &mut TaskBuilder) {
        task.branch().then(junk_a5);
        task.branch().then(junk_b5);
    }
    pub fn junk_a5(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b5(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate6(task: &mut TaskBuilder) {
        task.branch().then(junk_a6);
        task.branch().then(junk_b6);
    }
    pub fn junk_a6(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b6(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate7(task: &mut TaskBuilder) {
        task.branch().then(junk_a7);
        task.branch().then(junk_b7);
    }
    pub fn junk_a7(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b7(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate8(task: &mut TaskBuilder) {
        task.branch().then(junk_a8);
        task.branch().then(junk_b8);
    }
    pub fn junk_a8(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b8(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate9(task: &mut TaskBuilder) {
        task.branch().then(junk_a9);
        task.branch().then(junk_b9);
    }
    pub fn junk_a9(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b9(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate10(task: &mut TaskBuilder) {
        task.branch().then(junk_a10);
        task.branch().then(junk_b10);
    }
    pub fn junk_a10(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b10(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }

    pub fn gate11(task: &mut TaskBuilder) {
        task.branch().then(junk_a11);
        task.branch().then(junk_b11);
    }
    pub fn junk_a11(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }
    pub fn junk_b11(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = false);
    }
}

/// The look-ahead benchmark's exponential-backtracking domain — a faithful
/// function-defined port of the former text fixture: a doomed method whose
/// chain of 12 binary-choice gates never touches `gold`, followed by a task
/// requiring `gold > 1000`. With the look-ahead sweep the doomed method is
/// refuted in one pass; without it, plain MTR backtracking must enumerate all
/// 2^12 gate combinations before abandoning the method (the planner's sanity
/// limit is raised by the bench so the blowup is visible rather than capped).
/// The second root method succeeds directly.
pub fn gate_domain() -> HtnDomain {
    use gate_tasks::*;
    HtnDomain::from_root(gate_root)
        .build()
        .expect("gate domain is well-formed")
}

pub mod doomed_tasks {
    use super::*;

    pub fn act(task: &mut TaskBuilder) {
        task.branch().then(prime).then(spiral).then(impossible);
        task.branch().then(safe);
    }

    pub fn spiral(task: &mut TaskBuilder) {
        task.branch().then(spiral);
    }

    pub fn prime(task: &mut TaskBuilder) {
        task.effect(|n: &mut Noise| n.0 = true);
    }

    pub fn impossible(task: &mut TaskBuilder) {
        task.precondition(|gold: &GateGold| gold.0 > 100);
    }

    pub fn safe(task: &mut TaskBuilder) {
        task.effect(|gold: &mut GateGold| gold.0 = 1);
    }
}

/// The look-ahead benchmark's doomed-recursion domain — a faithful
/// function-defined port of the former text fixture: the first method
/// recurses forever (`spiral`) before an impossible tail task; the second
/// method plans fine. With the look-ahead the recursion is never entered;
/// without it the planner burns its whole step budget (the default 100, i.e.
/// the realistic setting) and returns a partial plan.
pub fn doomed_recursion_domain() -> HtnDomain {
    use doomed_tasks::*;
    HtnDomain::from_root(act)
        .build()
        .expect("doomed-recursion domain is well-formed")
}

// ---------------------------------------------------------------------------
// Plan execution
// ---------------------------------------------------------------------------

/// Execute a plan: apply each planned primitive task's `effects` to the
/// scratchpad, in order. This is the *execution* semantics the integration
/// tests pin (`effects` only — `expected_effects` are planning-only hopes).
pub fn execute_plan(domain: &HtnDomain, state: &mut PlanState, plan: &Plan) {
    for &step in plan.steps() {
        apply_step_effects(domain, state, step);
    }
}

/// Execute **one step** of a plan (the first compiled step's effects) — the
/// agent-tick semantics the benches' plan → execute → replan cycle uses: each
/// cycle advances the world by one action, so every replan sees real, changed
/// state instead of a goal that was already completed by a full-plan execution.
pub fn execute_plan_step(domain: &HtnDomain, state: &mut PlanState, plan: &Plan) {
    if let Some(&first) = plan.steps().first() {
        apply_step_effects(domain, state, first);
    }
}

fn apply_step_effects(domain: &HtnDomain, state: &mut PlanState, step: u32) {
    if let Some(Task::Primitive(p)) = domain.tasks.get(step as usize) {
        for e in &p.effects {
            e.apply(state);
        }
    }
}

/// Interned task-name handle re-exported for bench call sites.
#[allow(dead_code)]
pub type TaskName = Ustr;
