//! Parser conformance tests: the `.htn` files shipped with, and identical to,
//! the reference `bevy_htn` examples must parse with our grammar. This pins the
//! `.htn` file API so the planner stays a drop-in for those domains.

use cdda_htn::parse_htn;

fn load(name: &str) -> String {
    std::fs::read_to_string(format!("{}/htn/{name}", env!("CARGO_MANIFEST_DIR")))
        .expect("read htn fixture")
}

#[test]
fn parses_dude_example() {
    let src = load("dude.htn");
    let domain = parse_htn(&src).expect("dude.htn must parse");
    // Root compound + IsScared + 9 primitives.
    assert_eq!(domain.tasks.len(), 11);
    assert_eq!(domain.schema(), "0.1.0");
}

#[test]
fn parses_miner_example() {
    let src = load("miner.htn");
    let domain = parse_htn(&src).expect("miner.htn must parse");
    // EarnGold + TurnOreIntoMetal + TurnMetalIntoGold + 11 primitives.
    assert_eq!(domain.tasks.len(), 14);
}

#[test]
fn parses_troll_example() {
    let src = load("troll.htn");
    let domain = parse_htn(&src).expect("troll.htn must parse");
    // BeTrunkThumper + AttackEnemy + 11 primitives.
    assert_eq!(domain.tasks.len(), 13);
}

#[test]
fn parses_deep_outpost_example() {
    let src = load("outpost.htn");
    let domain = parse_htn(&src).expect("outpost.htn must parse");
    // 8 compounds + 15 primitives = 23 tasks.
    assert_eq!(domain.tasks.len(), 23);
    // The root compound is the forward-planning entry point.
    assert_eq!(domain.root_task().map(|t| t.name()), Some("SecureOutpost"));
}
