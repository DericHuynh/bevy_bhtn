//! Planner + AI logic tests using the shared [`HtnTestBed`].
//!
//! Exercises **forward planning** (MTR decomposition + backtracking), **backward
//! / goal-state planning**, and **plan validity** (execution: applying a plan's
//! task effects in order reaches the terminal state).
//!
//! Domains are deliberately *terminating* so assertions are exact and robust —
//! unbounded recursion (e.g. a miner that re-loops until a gold target) is the
//! reference `bevy_htn`'s shape and is covered by the builder-fixture tests,
//! not by exact-plan assertions.

// The 300-link chain fixture below instantiates one `TaskFn` impl per link
// (fn-item task identity), so monomorphization depth scales with the longest
// static task-reference chain.
#![recursion_limit = "1024"]

mod common;
use common::bench_common::{gate_tasks::gate_root, miner_tasks::earn_gold};
use common::HtnTestBed;

use bevy_bhtn::prelude::*;
use bevy_ecs::prelude::Component;
use ustr::Ustr;

/// A task fn's item type cannot be named directly, so the lookup-by-type API
/// is reached through these inference helpers: the fn value pins `F` to the
/// fn item's unique type, resolved through the baked `TypeId` index.
fn plan_of<F: TaskFn>(planner: &mut HtnPlanner<'_>, _f: F, state: &PlanState) -> Plan {
    planner.plan(_f, state)
}

fn bed_backward<F: GoalFn>(bed: &HtnTestBed, _f: F, state: &PlanState) -> HtnResult<Vec<Ustr>> {
    bed.plan_backward(_f, state)
}

// ---------------------------------------------------------------------------
// Travel domain — mirrors the classic bevy_htn `test_travel_htn`.
// Two root methods, exercises backtracking (walk when close, taxi when far).
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Copy, Default, Debug, PartialEq, Eq)]
enum Spot {
    #[default]
    Home,
    #[allow(dead_code)]
    Other,
    Park,
}

#[derive(Component, Clone, Default, Debug)]
struct Cash(pub i32);
#[derive(Component, Clone, Default, Debug)]
struct DistanceToPark(pub i32);
#[derive(Component, Clone, Default, Debug)]
struct Happy(pub bool);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct MyLocation(pub Spot);
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct TaxiLocation(pub Spot);

fn go_to_park(task: &mut TaskBuilder) {
    task.branch().then(walk);
    task.branch().then(call_taxi).then(ride_taxi).then(pay_taxi);
}

fn walk(task: &mut TaskBuilder) {
    task.precondition(|d: &DistanceToPark| d.0 <= 4)
        .precondition(|loc: &MyLocation| loc.0 != Spot::Park)
        .precondition(|h: &Happy| !h.0)
        .effect(|loc: &mut MyLocation| loc.0 = Spot::Park)
        .effect(|h: &mut Happy| h.0 = true);
}

fn call_taxi(task: &mut TaskBuilder) {
    task.precondition(|c: &Cash| c.0 >= 1)
        .effect(|taxi: &mut TaxiLocation, my: &mut MyLocation| taxi.0 = my.0);
}

fn ride_taxi(task: &mut TaskBuilder) {
    task.precondition(|taxi: &TaxiLocation, my: &MyLocation| taxi.0 == my.0)
        .precondition(|c: &Cash| c.0 >= 1)
        .effect(|taxi: &mut TaxiLocation| taxi.0 = Spot::Park)
        .effect(|my: &mut MyLocation| my.0 = Spot::Park)
        .effect(|h: &mut Happy| h.0 = true);
}

fn pay_taxi(task: &mut TaskBuilder) {
    task.precondition(|taxi: &TaxiLocation| taxi.0 == Spot::Park)
        .precondition(|c: &Cash| c.0 >= 1)
        .effect(|c: &mut Cash| c.0 -= 1);
}

fn travel_domain() -> HtnDomain {
    HtnDomain::from_root(go_to_park)
        .build()
        .expect("travel domain is well-formed")
}

/// A travel scratchpad: `cash` and `distance_to_park` set, everything else at
/// its default (home, unhappy, no taxi).
fn travel_state(domain: &HtnDomain, cash: i32, distance_to_park: i32) -> PlanState {
    PlanState::build(&domain.components)
        .set(Cash(cash))
        .set(DistanceToPark(distance_to_park))
        .finish()
}

// ---------------------------------------------------------------------------
// Forward planning (deterministic, terminating)
// ---------------------------------------------------------------------------

#[test]
fn forward_plans_walk_when_close() {
    let bed = HtnTestBed::new(travel_domain());
    let state = travel_state(bed.domain(), 0, 1);
    assert_eq!(bed.plan_forward(&state), vec![Ustr::from("walk")]);
}

#[test]
fn forward_plans_taxi_when_far() {
    let bed = HtnTestBed::new(travel_domain());
    let state = travel_state(bed.domain(), 10, 9);
    // Walk fails (too far) -> backtracks -> taxi succeeds.
    assert_eq!(
        bed.plan_forward(&state),
        vec![
            Ustr::from("call_taxi"),
            Ustr::from("ride_taxi"),
            Ustr::from("pay_taxi")
        ]
    );
}

#[test]
fn forward_plan_is_terminal_and_executes() {
    let bed = HtnTestBed::new(travel_domain());
    let mut state = travel_state(bed.domain(), 10, 9);
    let plan = bed.plan_forward(&state);
    assert_eq!(plan.len(), 3);

    // Execute: apply each planned task's effects in order.
    for name in &plan {
        let Some(Task::Primitive(p)) = bed.domain().get_task(name) else {
            panic!("planned task `{name}` missing");
        };
        for e in p.effects.iter() {
            e.apply(&mut state);
        }
    }

    let my_location = bed.domain().components.get::<MyLocation>().unwrap();
    let happy = bed.domain().components.get::<Happy>().unwrap();
    let cash = bed.domain().components.get::<Cash>().unwrap();
    // Terminal state: at the park, happy, taxi paid for.
    assert_eq!(state.get::<MyLocation>(my_location).0, Spot::Park);
    assert!(state.get::<Happy>(happy).0);
    assert_eq!(state.get::<Cash>(cash).0, 9);
}

// ---------------------------------------------------------------------------
// Forward planning: an already-satisfied goal yields an empty plan.
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug)]
struct Powered(pub bool);

fn switch_on(task: &mut TaskBuilder) {
    task.effect(|p: &mut Powered| p.0 = true);
}

fn ensure_on(task: &mut TaskBuilder) {
    task.branch().precondition(|p: &Powered| p.0);
    task.branch().then(switch_on);
}

fn idempotent_domain() -> HtnDomain {
    HtnDomain::from_root(ensure_on)
        .build()
        .expect("idempotent domain is well-formed")
}

#[test]
fn forward_plan_returns_empty_when_goal_already_met() {
    let bed = HtnTestBed::new(idempotent_domain());
    let on = PlanState::build(&bed.domain().components)
        .set(Powered(true))
        .finish();
    assert!(bed.plan_forward(&on).is_empty());
    let off = PlanState::build(&bed.domain().components)
        .set(Powered(false))
        .finish();
    assert_eq!(bed.plan_forward(&off), vec![Ustr::from("switch_on")]);
}

// ---------------------------------------------------------------------------
// Backward (goal-state) planning
// ---------------------------------------------------------------------------

#[derive(Component, Clone, Default, Debug)]
struct HasOre(pub bool);
#[derive(Component, Clone, Default, Debug)]
struct HasRope(pub bool);

fn mine(task: &mut TaskBuilder) {
    task.precondition(|o: &HasOre| !o.0)
        .effect(|o: &mut HasOre| o.0 = true);
}

fn ore_root(task: &mut TaskBuilder) {
    task.branch().then(mine);
}

fn have_ore(goal: &mut GoalBuilder) {
    goal.effect(|o: &mut HasOre| o.0 = true);
}

fn goal_domain() -> HtnDomain {
    HtnDomain::from_root(ore_root)
        .goal(have_ore)
        .build()
        .expect("goal domain is well-formed")
}

#[test]
fn backward_plan_finds_satisfying_leaf() {
    let bed = HtnTestBed::new(goal_domain());
    let state = PlanState::build(&bed.domain().components).finish();
    let plan = bed_backward(&bed, have_ore, &state).expect("back plan reaches goal");
    assert_eq!(plan, vec![Ustr::from("mine")]);
}

fn mine_bare(task: &mut TaskBuilder) {
    task.effect(|o: &mut HasOre| o.0 = true);
}

fn unreachable_root(task: &mut TaskBuilder) {
    task.branch().then(mine_bare);
}

fn want_more(goal: &mut GoalBuilder) {
    goal.effect(|o: &mut HasOre| o.0 = true)
        .effect(|r: &mut HasRope| r.0 = true);
}

#[test]
fn backward_plan_rejects_unreachable_goal() {
    let domain = HtnDomain::from_root(unreachable_root)
        .goal(want_more)
        .build()
        .expect("unreachable-goal domain is well-formed");
    let bed = HtnTestBed::new(domain);
    let state = PlanState::build(&bed.domain().components).finish();
    // `has_rope` is never produced by any primitive -> goal unreachable.
    assert!(bed_backward(&bed, want_more, &state).is_err());
}

// ---------------------------------------------------------------------------
// Task-function-recorded condition/effect details + evaluation
// ---------------------------------------------------------------------------

#[test]
fn task_functions_record_conditions_and_effects() {
    let bed = HtnTestBed::new(travel_domain());
    let Some(Task::Primitive(walk)) = bed.domain().get_task("walk") else {
        panic!("walk primitive missing");
    };
    // The `walk` task function recorded its three preconditions and two
    // effects, in declaration order.
    assert_eq!(walk.preconditions.len(), 3);
    assert_eq!(walk.effects.len(), 2);

    // Conditions evaluate against the scratchpad: close + not at the park +
    // unhappy is pickable; standing at the park is not.
    let near = travel_state(bed.domain(), 10, 3);
    assert!(walk.preconditions_met(&near));
    let at_park = PlanState::build(&bed.domain().components)
        .set(MyLocation(Spot::Park))
        .finish();
    assert!(!walk.preconditions_met(&at_park));

    // Effects apply to the scratchpad: walking lands at the park, happy.
    let mut state = travel_state(bed.domain(), 10, 3);
    for e in walk.effects.iter() {
        e.apply(&mut state);
    }
    let my_location = bed.domain().components.get::<MyLocation>().unwrap();
    let happy = bed.domain().components.get::<Happy>().unwrap();
    assert_eq!(state.get::<MyLocation>(my_location).0, Spot::Park);
    assert!(state.get::<Happy>(happy).0);
}

#[test]
fn conditions_evaluate_against_state() {
    let bed = HtnTestBed::new(travel_domain());
    let Some(Task::Primitive(walk)) = bed.domain().get_task("walk") else {
        panic!("walk primitive missing");
    };
    // The first recorded precondition is the distance gate (`<= 4`).
    let near = travel_state(bed.domain(), 10, 3);
    let far = travel_state(bed.domain(), 10, 9);
    assert!(walk.preconditions[0].evaluate(&near));
    assert!(!walk.preconditions[0].evaluate(&far));
}

#[test]
fn effects_apply_to_state() {
    let bed = HtnTestBed::new(idempotent_domain());
    let Some(Task::Primitive(switch)) = bed.domain().get_task("switch_on") else {
        panic!("switch_on primitive missing");
    };
    let mut state = PlanState::build(&bed.domain().components).finish();
    for e in switch.effects.iter() {
        e.apply(&mut state);
    }
    let powered = bed.domain().components.get::<Powered>().unwrap();
    assert!(state.get::<Powered>(powered).0);
}

// ---------------------------------------------------------------------------
// Task-index width dispatch:
// planner's `u16` search monomorphization (every fixture domain plans on the
// `u8` path). The chain of 300 compound tasks decomposes past the default
// sanity limit, so the budget is raised; the doomed branch burns its 300
// steps, backtracks fully, and the direct branch plans.
// ---------------------------------------------------------------------------

/// The wide-domain fixture's gold component (module-level: the task functions
/// close over the type).
#[derive(Component, Clone, Default, Debug, PartialEq, Eq)]
struct WideGold(pub i32);

macro_rules! chain_link {
    ($name:ident, $next:ident) => {
        fn $name(task: &mut TaskBuilder) {
            task.branch().then($next);
        }
    };
}

chain_link!(c0, c1);
chain_link!(c1, c2);
chain_link!(c2, c3);
chain_link!(c3, c4);
chain_link!(c4, c5);
chain_link!(c5, c6);
chain_link!(c6, c7);
chain_link!(c7, c8);
chain_link!(c8, c9);
chain_link!(c9, c10);
chain_link!(c10, c11);
chain_link!(c11, c12);
chain_link!(c12, c13);
chain_link!(c13, c14);
chain_link!(c14, c15);
chain_link!(c15, c16);
chain_link!(c16, c17);
chain_link!(c17, c18);
chain_link!(c18, c19);
chain_link!(c19, c20);
chain_link!(c20, c21);
chain_link!(c21, c22);
chain_link!(c22, c23);
chain_link!(c23, c24);
chain_link!(c24, c25);
chain_link!(c25, c26);
chain_link!(c26, c27);
chain_link!(c27, c28);
chain_link!(c28, c29);
chain_link!(c29, c30);
chain_link!(c30, c31);
chain_link!(c31, c32);
chain_link!(c32, c33);
chain_link!(c33, c34);
chain_link!(c34, c35);
chain_link!(c35, c36);
chain_link!(c36, c37);
chain_link!(c37, c38);
chain_link!(c38, c39);
chain_link!(c39, c40);
chain_link!(c40, c41);
chain_link!(c41, c42);
chain_link!(c42, c43);
chain_link!(c43, c44);
chain_link!(c44, c45);
chain_link!(c45, c46);
chain_link!(c46, c47);
chain_link!(c47, c48);
chain_link!(c48, c49);
chain_link!(c49, c50);
chain_link!(c50, c51);
chain_link!(c51, c52);
chain_link!(c52, c53);
chain_link!(c53, c54);
chain_link!(c54, c55);
chain_link!(c55, c56);
chain_link!(c56, c57);
chain_link!(c57, c58);
chain_link!(c58, c59);
chain_link!(c59, c60);
chain_link!(c60, c61);
chain_link!(c61, c62);
chain_link!(c62, c63);
chain_link!(c63, c64);
chain_link!(c64, c65);
chain_link!(c65, c66);
chain_link!(c66, c67);
chain_link!(c67, c68);
chain_link!(c68, c69);
chain_link!(c69, c70);
chain_link!(c70, c71);
chain_link!(c71, c72);
chain_link!(c72, c73);
chain_link!(c73, c74);
chain_link!(c74, c75);
chain_link!(c75, c76);
chain_link!(c76, c77);
chain_link!(c77, c78);
chain_link!(c78, c79);
chain_link!(c79, c80);
chain_link!(c80, c81);
chain_link!(c81, c82);
chain_link!(c82, c83);
chain_link!(c83, c84);
chain_link!(c84, c85);
chain_link!(c85, c86);
chain_link!(c86, c87);
chain_link!(c87, c88);
chain_link!(c88, c89);
chain_link!(c89, c90);
chain_link!(c90, c91);
chain_link!(c91, c92);
chain_link!(c92, c93);
chain_link!(c93, c94);
chain_link!(c94, c95);
chain_link!(c95, c96);
chain_link!(c96, c97);
chain_link!(c97, c98);
chain_link!(c98, c99);
chain_link!(c99, c100);
chain_link!(c100, c101);
chain_link!(c101, c102);
chain_link!(c102, c103);
chain_link!(c103, c104);
chain_link!(c104, c105);
chain_link!(c105, c106);
chain_link!(c106, c107);
chain_link!(c107, c108);
chain_link!(c108, c109);
chain_link!(c109, c110);
chain_link!(c110, c111);
chain_link!(c111, c112);
chain_link!(c112, c113);
chain_link!(c113, c114);
chain_link!(c114, c115);
chain_link!(c115, c116);
chain_link!(c116, c117);
chain_link!(c117, c118);
chain_link!(c118, c119);
chain_link!(c119, c120);
chain_link!(c120, c121);
chain_link!(c121, c122);
chain_link!(c122, c123);
chain_link!(c123, c124);
chain_link!(c124, c125);
chain_link!(c125, c126);
chain_link!(c126, c127);
chain_link!(c127, c128);
chain_link!(c128, c129);
chain_link!(c129, c130);
chain_link!(c130, c131);
chain_link!(c131, c132);
chain_link!(c132, c133);
chain_link!(c133, c134);
chain_link!(c134, c135);
chain_link!(c135, c136);
chain_link!(c136, c137);
chain_link!(c137, c138);
chain_link!(c138, c139);
chain_link!(c139, c140);
chain_link!(c140, c141);
chain_link!(c141, c142);
chain_link!(c142, c143);
chain_link!(c143, c144);
chain_link!(c144, c145);
chain_link!(c145, c146);
chain_link!(c146, c147);
chain_link!(c147, c148);
chain_link!(c148, c149);
chain_link!(c149, c150);
chain_link!(c150, c151);
chain_link!(c151, c152);
chain_link!(c152, c153);
chain_link!(c153, c154);
chain_link!(c154, c155);
chain_link!(c155, c156);
chain_link!(c156, c157);
chain_link!(c157, c158);
chain_link!(c158, c159);
chain_link!(c159, c160);
chain_link!(c160, c161);
chain_link!(c161, c162);
chain_link!(c162, c163);
chain_link!(c163, c164);
chain_link!(c164, c165);
chain_link!(c165, c166);
chain_link!(c166, c167);
chain_link!(c167, c168);
chain_link!(c168, c169);
chain_link!(c169, c170);
chain_link!(c170, c171);
chain_link!(c171, c172);
chain_link!(c172, c173);
chain_link!(c173, c174);
chain_link!(c174, c175);
chain_link!(c175, c176);
chain_link!(c176, c177);
chain_link!(c177, c178);
chain_link!(c178, c179);
chain_link!(c179, c180);
chain_link!(c180, c181);
chain_link!(c181, c182);
chain_link!(c182, c183);
chain_link!(c183, c184);
chain_link!(c184, c185);
chain_link!(c185, c186);
chain_link!(c186, c187);
chain_link!(c187, c188);
chain_link!(c188, c189);
chain_link!(c189, c190);
chain_link!(c190, c191);
chain_link!(c191, c192);
chain_link!(c192, c193);
chain_link!(c193, c194);
chain_link!(c194, c195);
chain_link!(c195, c196);
chain_link!(c196, c197);
chain_link!(c197, c198);
chain_link!(c198, c199);
chain_link!(c199, c200);
chain_link!(c200, c201);
chain_link!(c201, c202);
chain_link!(c202, c203);
chain_link!(c203, c204);
chain_link!(c204, c205);
chain_link!(c205, c206);
chain_link!(c206, c207);
chain_link!(c207, c208);
chain_link!(c208, c209);
chain_link!(c209, c210);
chain_link!(c210, c211);
chain_link!(c211, c212);
chain_link!(c212, c213);
chain_link!(c213, c214);
chain_link!(c214, c215);
chain_link!(c215, c216);
chain_link!(c216, c217);
chain_link!(c217, c218);
chain_link!(c218, c219);
chain_link!(c219, c220);
chain_link!(c220, c221);
chain_link!(c221, c222);
chain_link!(c222, c223);
chain_link!(c223, c224);
chain_link!(c224, c225);
chain_link!(c225, c226);
chain_link!(c226, c227);
chain_link!(c227, c228);
chain_link!(c228, c229);
chain_link!(c229, c230);
chain_link!(c230, c231);
chain_link!(c231, c232);
chain_link!(c232, c233);
chain_link!(c233, c234);
chain_link!(c234, c235);
chain_link!(c235, c236);
chain_link!(c236, c237);
chain_link!(c237, c238);
chain_link!(c238, c239);
chain_link!(c239, c240);
chain_link!(c240, c241);
chain_link!(c241, c242);
chain_link!(c242, c243);
chain_link!(c243, c244);
chain_link!(c244, c245);
chain_link!(c245, c246);
chain_link!(c246, c247);
chain_link!(c247, c248);
chain_link!(c248, c249);
chain_link!(c249, c250);
chain_link!(c250, c251);
chain_link!(c251, c252);
chain_link!(c252, c253);
chain_link!(c253, c254);
chain_link!(c254, c255);
chain_link!(c255, c256);
chain_link!(c256, c257);
chain_link!(c257, c258);
chain_link!(c258, c259);
chain_link!(c259, c260);
chain_link!(c260, c261);
chain_link!(c261, c262);
chain_link!(c262, c263);
chain_link!(c263, c264);
chain_link!(c264, c265);
chain_link!(c265, c266);
chain_link!(c266, c267);
chain_link!(c267, c268);
chain_link!(c268, c269);
chain_link!(c269, c270);
chain_link!(c270, c271);
chain_link!(c271, c272);
chain_link!(c272, c273);
chain_link!(c273, c274);
chain_link!(c274, c275);
chain_link!(c275, c276);
chain_link!(c276, c277);
chain_link!(c277, c278);
chain_link!(c278, c279);
chain_link!(c279, c280);
chain_link!(c280, c281);
chain_link!(c281, c282);
chain_link!(c282, c283);
chain_link!(c283, c284);
chain_link!(c284, c285);
chain_link!(c285, c286);
chain_link!(c286, c287);
chain_link!(c287, c288);
chain_link!(c288, c289);
chain_link!(c289, c290);
chain_link!(c290, c291);
chain_link!(c291, c292);
chain_link!(c292, c293);
chain_link!(c293, c294);
chain_link!(c294, c295);
chain_link!(c295, c296);
chain_link!(c296, c297);
chain_link!(c297, c298);
chain_link!(c298, c299);
chain_link!(c299, leaf_check);

fn leaf_check(task: &mut TaskBuilder) {
    task.precondition(|gold: &WideGold| gold.0 > 1000);
}

fn strike(task: &mut TaskBuilder) {
    task.effect(|gold: &mut WideGold| gold.0 = 2000);
}

fn wide_root(task: &mut TaskBuilder) {
    task.branch().then(c0);
    task.branch().then(strike).then(leaf_check);
}

#[test]
fn domains_wider_than_u8_plan_on_the_u16_path() {
    let domain = HtnDomain::from_root(wide_root).build().unwrap();
    assert!(
        domain.tasks.len() > u8::MAX as usize,
        "fixture must exceed the u8 dispatch width (tasks = {})",
        domain.tasks.len()
    );

    let state = PlanState::build(&domain.components)
        .set(WideGold(0))
        .finish();
    let mut planner = HtnPlanner::new(&domain);
    planner.set_sanity_limit(1000);
    let plan = plan_of(&mut planner, wide_root, &state);
    assert!(
        plan.is_complete(),
        "the raised budget fully refutes the doomed branch"
    );
    assert_eq!(plan.task_names(), ["strike", "leaf_check"]);

    // And the look-ahead agrees (same plan, found without the doomed branch's
    // 300-step burn).
    let mut la = HtnPlanner::new(&domain);
    la.set_sanity_limit(1000).set_lookahead(true);
    assert_eq!(
        plan_of(&mut la, wide_root, &state).task_names(),
        ["strike", "leaf_check"]
    );
}

// ---------------------------------------------------------------------------
// Plan status — Complete vs Partial (sanity budget / fail-fast)
// ---------------------------------------------------------------------------

/// `Plan::status` tells a finished decomposition from one the search cut
/// short: terminating domains plan `Complete`; a decomposition that exceeds
/// the sanity budget returns the best `Partial` prefix; a search that
/// exhausts every method is `Complete` (and empty).
#[test]
fn plan_status_reports_complete_vs_partial() {
    // Terminating domain: the plan is final.
    let miner = common::miner_domain();
    let state = PlanState::build(&miner.components).finish();
    let mut planner = HtnPlanner::new(&miner);
    assert!(plan_of(&mut planner, earn_gold, &state).is_complete());

    // Budget-truncated: with the look-ahead off, the gate domain's doomed
    // method must enumerate 2^12 gate combinations — the default sanity
    // limit (100) cuts the search short and returns the prefix found so far.
    let gate = common::gate_domain();
    let state = PlanState::build(&gate.components).finish();
    let mut planner = HtnPlanner::new(&gate);
    planner.set_lookahead(false);
    let partial = plan_of(&mut planner, gate_root, &state);
    assert!(partial.is_partial());
    assert!(
        !partial.is_empty(),
        "a partial plan is the prefix found so far"
    );

    // Same domain, look-ahead on (the default): the doomed method is refuted
    // in one sweep pass and the direct method plans — final, no budget raise
    // needed (full enumeration would cost ~2^12 gate combinations).
    let mut planner = HtnPlanner::new(&gate);
    let done = plan_of(&mut planner, gate_root, &state);
    assert!(done.is_complete());
    assert_eq!(done.task_names(), ["strike", "gate_final"]);

    // Exhausted search: no method can ever apply — the empty result is
    // final, not truncated.
    #[derive(Component, Clone, Default, Debug)]
    struct Wall(bool);
    fn impossible(task: &mut TaskBuilder) {
        task.branch().then(no_way);
    }
    fn no_way(task: &mut TaskBuilder) {
        task.precondition(|w: &Wall| w.0);
    }
    let dead = HtnDomain::from_root(impossible).build().unwrap();
    let state = PlanState::build(&dead.components).finish();
    let mut planner = HtnPlanner::new(&dead);
    let plan = plan_of(&mut planner, impossible, &state);
    assert!(plan.is_complete());
    assert!(plan.is_empty());
}
