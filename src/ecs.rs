//! Idiomatic Bevy ECS integration: agents and the AI driver system.
//!
//! The planner core stays headless (it plans against a [`PlanState`]
//! scratchpad), while this module wires it into a `bevy_ecs` [`World`]:
//!
//! - [`HtnAgent`] — the per-entity AI component: the current [`Plan`] and its
//!   cursor. The agent's *state* is just the entity's own components — there
//!   is no monolithic state struct to mirror.
//! - [`HtnConfig`] — the domain + planner tuning, as a [`Resource`].
//! - [`htn_ai_system`] — an exclusive system (`fn(&mut World)`) that gives
//!   every [`HtnAgent`] one turn per run: plan if planless (extracting a
//!   scratchpad from the entity's components), then execute one planned
//!   primitive — re-checking its preconditions against the real world,
//!   dispatching its action commands, committing its effects to the real
//!   components, and advancing the cursor (or dropping the plan to trigger a
//!   replan).
//!
//! # Working with the real world
//!
//! The planner simulates on a [`PlanState`] scratchpad; the world only ever
//! changes through the driver and the game's own systems. The seam has four
//! moving parts:
//!
//! 1. **Scratchpad extraction** — at plan time (and again at validation and
//!    post-action time, refreshed in place) the driver copies every registered
//!    component off the agent entity ([`PlanState::extract`] /
//!    [`PlanState::refresh`]; missing components materialize as `Default`).
//!    Preconditions and effect closures read and mutate this snapshot — never
//!    the `World` directly.
//! 2. **Effect commit** — when a step executes, the driver applies the
//!    primitive's effects to the scratchpad and writes **only the slots the
//!    effects declare as writes** (`&mut T` parameters; `&T` read parameters
//!    are never committed) back onto the real entity via
//!    [`PlanState::write_back_with`]. An action's commands are dispatched and
//!    flushed first, then effects commit on top of the post-action state.
//! 3. **Intent markers** — an action should not *do* the real-world work; it
//!    inserts a marker component (`cmds.insert(PickupRequest)`), and a
//!    dedicated game system realizes the intent (resolving targets, mutating
//!    the relationship graph) and removes the marker. This keeps world logic
//!    in ordinary systems the planner never sees.
//! 4. **Projection components** — the planner can only simulate components.
//!    World truth that lives in relationships or raw geometry (what is in
//!    which pocket, what lies on the ground) is derived into planner-facing
//!    summary components by sync systems each tick (`PocketContents` from the
//!    `InPocket`/`Pockets` relationship graph, `GroundKnowledge` from ground
//!    entities). The graph is the truth; the projection is what the planner
//!    simulates.
//!
//! Together with the driver's **same-tick replan** this models dynamic
//! worlds: the planner *simulates* the outcome as an effect (e.g.
//! `Arrived(true)`), the movement system recomputes the truth every tick, and
//! the driver's per-step re-validation walks the current plan checking
//! preconditions against reality. A failed check drops the plan and **replans
//! from reality in the same tick** — an enemy walking into view (a threat
//! projection flipping a step's safety precondition) changes behavior the
//! very tick it happens, because the fresh plan is selected against the world
//! as it is *now*. See `tests/htn_cdda_world.rs` for the grounded scenario:
//! an enemy appears mid-journey, the survivor flees the same tick, and
//! resumes the craft when the danger passes.
//!
//! # Paused plans and plan lifecycle events
//!
//! Two driver-level extensions to the plan lifecycle:
//!
//! - **Pause/resume (PausePlan)** — a
//!   [`pause_plan`](crate::tasks::MethodBuilder::pause_plan) marker truncates
//!   a plan at an authoring boundary. The driver executes the compiled prefix,
//!   keeps the exhausted paused plan ([`PlanStatus::Paused`]), and on the next
//!   run **resumes decomposition from the pause**
//!   ([`HtnPlanner::resume`](crate::planner::HtnPlanner::resume)) — against
//!   the world as it is then — instead of replanning from the root. One step
//!   per tick is preserved: a resume tick plans the next leg and executes its
//!   first step, exactly like a planless tick. Drift during a paused plan
//!   still replans from the root as always: the fresh plan re-derives the
//!   executed prefix (its already-reflected steps elide as quiet) and
//!   re-truncates at the markers.
//! - **Lifecycle events** — when [`HtnConfig::plan_events`] is set, the driver
//!   writes [`PlanEvent`]s to `Messages<PlanEvent>`: plans installed and
//!   replaced (with the old plan — a drift repair or a pause-resume
//!   continuation; Fluid HTN's `OnNewPlan` / `OnReplacePlan(old, new)`), steps
//!   that failed re-validation (`OnCurrentTaskFailed`), and plans completed.
//!   The same bridge pattern as the decomposition trace, with zero
//!   planner-core involvement.
//!
//! # Hot-reloading domains (Subsecond)
//!
//! With the `hotpatching` feature (which enables `bevy_ecs/hotpatching`, the
//! same [Dioxus Subsecond](https://dioxuslabs.com/learn/0.7/essentials/ui/hotreload/)
//! engine behind Bevy's hot-patching), `HtnConfig` can carry a **domain
//! rebuild closure** via `HtnConfig::with_rebuild` (feature-gated). The
//! baked domain is
//! *data* — compiled precondition/effect closures — so patching a task
//! function's body does not retroactively change it. Instead, the driver
//! watches `bevy_ecs::HotPatchChanges`; when a hot patch lands it
//! **re-records and re-bakes the domain** through the rebuild closure (fresh
//! closures resolve through the patched jump table), swaps it into the
//! config, and drops every agent's plan so the next tick replans against the
//! new behavior:
//!
//! ```ignore
//! # use bevy_bhtn::prelude::*;
//! # use bevy_ecs::prelude::*;
//! # #[derive(Component, Clone, Default, Debug)] struct Ammo(u32);
//! # fn engage(task: &mut TaskBuilder) { task.branch(); }
//! let domain = HtnDomain::from_root(engage).build().unwrap();
//! let config = HtnConfig::with_rebuild(domain, || {
//!     // Re-run the recording. For best results wrap the body in
//!     // `subsecond::call` (add `subsecond = "0.7"` to your deps) so the
//!     // fresh recording resolves task functions through the latest patch:
//!     HtnDomain::from_root(engage).build()
//! });
//! ```
//!
//! Run the game with the Dioxus CLI (`dx serve --hotpatch`) and edit task
//! functions in the binary crate — the new closure bodies land in the baked
//! domain on the next driver tick, with no restart and no lost world state.
//!
//! Limitations (mirroring Subsecond's): only the *tip* crate is tracked (task
//! functions defined in a dependency crate are not observed); a rebuild that
//! fails keeps the old domain; graph-shape changes are fine (plans are
//! dropped wholesale and the scratchpad layout is rebuilt), but task
//! *signature* changes (different component parameters) change the registry —
//! supported, since the scratchpad is rebuilt from the new domain's registry.
//!
//! ```
//! use bevy_bhtn::prelude::*;
//! use bevy_ecs::prelude::*;
//!
//! #[derive(Component, Clone, Default, Debug)]
//! struct Battery(i32);
//!
//! fn charge(task: &mut TaskBuilder) {
//!     task.branch().precondition(|battery: &Battery| battery.0 >= 3);
//!     task.branch()
//!         .precondition(|battery: &Battery| battery.0 < 3)
//!         .then(gather)
//!         .then(charge);
//! }
//! fn gather(task: &mut TaskBuilder) {
//!     task.effect(|battery: &mut Battery| battery.0 += 1);
//! }
//!
//! let domain = HtnDomain::from_root(charge).build().unwrap();
//! let mut world = World::new();
//! world.insert_resource(HtnConfig::new(domain));
//!
//! // Agents are ordinary entities: independent components + the AI marker.
//! let entity = world.spawn((Battery(0), HtnAgent::default())).id();
//!
//! // One AI tick for every agent (add to a schedule as an exclusive system).
//! htn_ai_system(&mut world);
//! assert_eq!(world.get::<Battery>(entity).unwrap().0, 1);
//! ```

use bevy_ecs::entity::Entity;
use bevy_ecs::message::Message;
use bevy_ecs::prelude::{Component, Resource};
use bevy_ecs::world::World;

use crate::domain::HtnDomain;
use crate::domain::Task;
/// Used by the hot-reload rebuild closure; inert without the feature.
#[cfg_attr(not(feature = "hotpatching"), allow(unused_imports))]
use crate::error::HtnResult;
use crate::planner::{HtnPlanner, Plan, PlanStatus, ResumeStep};
use crate::selection::{DecompositionTrace, HtnSearchStrategy, LookaheadMode, SearchOverride};
use crate::state::{ComponentRegistry, PlanState};

/// The per-entity AI component: current plan and cursor.
///
/// The agent's plan state is the entity's own component set — the driver
/// extracts a [`PlanState`] scratchpad from it at plan time and commits
/// effects back to the real components at execution time.
#[derive(Component, Debug, Default, Clone)]
pub struct HtnAgent {
    /// The current forward plan, if one has been (re)planned. Read via
    /// [`Self::plan`]; mutated only by the driver.
    plan: Option<Plan>,
    /// Index into `plan`'s task list of the next step to execute. Read via
    /// [`Self::cursor`]; mutated only by the driver.
    cursor: usize,
    /// LOD bookkeeping for [`PlanEvery`]: how many more driver runs this
    /// agent must sit planless before it may plan again. Purely internal
    /// driver state — not part of the agent's game-facing data.
    plan_deferral: u32,
}

impl HtnAgent {
    /// The agent's current plan, if one has been (re)planned.
    pub fn plan(&self) -> Option<&Plan> {
        self.plan.as_ref()
    }

    /// The index of the next step to execute within the current plan
    /// (advances as the driver executes; resets when the plan is dropped).
    pub fn cursor(&self) -> usize {
        self.cursor
    }
}

/// Level-of-detail throttle: an agent carrying this component plans **at most
/// once every `n` runs** of [`htn_ai_system`] while it sits planless — the
/// driver's answer to "far away NPCs should think less often". Execution of an
/// already-planned plan is never delayed; only (re)planning is paced. Agents
/// with an active plan or without this component are unaffected (`n <= 1`
/// disables the throttle).
///
/// A planning attempt (successful or not) resets the deferral to `n - 1`, so
/// an agent whose domain genuinely has no plan retries at the same reduced
/// cadence instead of burning a replan every tick.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct PlanEvery(pub u32);

/// Per-agent planning-root override: the domain task index this agent's
/// behavior plans from, instead of the domain's main root. The seam for
/// CDDA-style NPC archetypes — survivor, animal, faction soldier — as
/// separate compound roots registered via
/// [`DomainBuilder::add_root`](crate::domain::DomainBuilder::add_root), one
/// agent per archetype.
///
/// Agents without this component plan from
/// [`HtnConfig::domain`](HtnConfig)'s root. An out-of-bounds index is an
/// [`HtnError::UnregisteredTask`](crate::error::HtnError::UnregisteredTask)
/// planning error — treated exactly like an unsolvable plan (the agent goes
/// planless and idles), never a wedge.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentRoot(pub u32);

/// Plan lifecycle events, emitted by the driver into
/// `Messages<PlanEvent>` when [`HtnConfig::plan_events`] is set — the same
/// bridge pattern as the decomposition trace, with **zero planner-core
/// involvement**: the planner knows nothing about these; the driver reports
/// what it did with the plans it planned and executed.
///
/// Mirrors Fluid HTN's plan callbacks (`OnNewPlan`, `OnReplacePlan(old, new)`,
/// `OnCurrentTaskFailed`): hooks for animation, telemetry, and debug
/// overlays. Every event carries the agent [`Entity`].
// `PlanCompleted`/`StepFailed` are deliberately tiny next to `PlanReplaced`'s
// full plan payloads (the Fluid HTN callback contract — games want the plans,
// not summaries); the bridge is opt-in and event volume is one to a few per
// agent per tick, so the size skew is the design.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Message)]
pub enum PlanEvent {
    /// A plan was installed on an agent. `old` is the plan it replaced:
    /// `None` when the agent was planless (Fluid HTN's `OnNewPlan`), `Some`
    /// when an existing plan was swapped out — a same-tick drift-repair
    /// replan, or a pause-resume continuation (the paused plan is `old`).
    PlanReplaced {
        /// The agent the plan was installed on.
        entity: Entity,
        /// The plan that was replaced, if any.
        old: Option<Plan>,
        /// The freshly installed plan.
        new: Plan,
    },
    /// The next step's preconditions failed re-validation against the live
    /// world (drift): the plan was dropped (Fluid HTN's
    /// `OnCurrentTaskFailed`). The same-tick repair replan that follows on
    /// the driver's second pass is reported as its own `PlanReplaced`.
    StepFailed {
        /// The agent whose step failed.
        entity: Entity,
        /// The failing step's task index into
        /// [`HtnDomain::tasks`](crate::domain::HtnDomain).
        task: u32,
        /// The failing task's display name.
        task_name: &'static str,
    },
    /// The plan's steps all executed and the decomposition ran to completion
    /// (`Complete`). Pause-truncated plans never report this — they resume
    /// (reported as `PlanReplaced`) — and budget-truncated (`Partial`) plans
    /// just end without an event.
    /// A step executed: its action commands were dispatched (if any) and its
    /// effects committed to the real components. The per-action hook — Fluid
    /// HTN's `OnApplyEffect` at driver granularity: one event per executed
    /// step, not per effect — for animation ticks and telemetry.
    StepExecuted {
        /// The agent that executed the step.
        entity: Entity,
        /// The executed step's task index into
        /// [`HtnDomain::tasks`](crate::domain::HtnDomain).
        task: u32,
        /// The executed task's display name.
        task_name: &'static str,
    },
    /// The plan's steps all executed and the decomposition ran to completion
    /// (`Complete`). Pause-truncated plans never report this — they resume
    /// (reported as `PlanReplaced`) — and budget-truncated (`Partial`) plans
    /// just end without an event.
    PlanCompleted {
        /// The agent whose plan completed.
        entity: Entity,
    },
}

/// The planner's world-side configuration: domain and planner tuning.
#[derive(Resource)]
pub struct HtnConfig {
    /// The HTN domain agents plan against.
    pub domain: HtnDomain,
    /// The search strategy agents use unless overridden by a
    /// [`SearchOverride`] component (default [`DepthFirst`]).
    ///
    /// [`DepthFirst`]: HtnSearchStrategy::DepthFirst
    pub strategy: HtnSearchStrategy,
    /// The forward planner's look-ahead gating mode (default
    /// [`LookaheadMode::Always`]).
    pub lookahead: LookaheadMode,
    /// The forward planner's decomposition-step budget (default `100`).
    pub sanity_limit: usize,
    /// Whether the driver forwards the planner's [`DecompositionTrace`]
    /// events to `Messages<DecompositionTrace>` after each plan/resume
    /// (default `false`). The trace half of the diagnostics bridge; the
    /// driver-side lifecycle counterpart is [`Self::plan_events`].
    pub trace_events: bool,
    /// Whether the driver emits plan lifecycle [`PlanEvent`]s to
    /// `Messages<PlanEvent>` (default `false`): plans installed/replaced,
    /// steps failed against live reality, and plans completed.
    pub plan_events: bool,
    /// Domain rebuild closure for Subsecond hot-reloading (see the module
    /// docs). Set via [`HtnConfig::with_rebuild`]; the driver re-records and
    /// re-bakes the domain through it whenever a hot patch lands.
    #[cfg(feature = "hotpatching")]
    pub rebuild: Option<Box<dyn Fn() -> HtnResult<HtnDomain> + Send + Sync>>,
}

impl HtnConfig {
    /// Config with default planner tuning.
    pub fn new(domain: HtnDomain) -> Self {
        Self {
            domain,
            strategy: HtnSearchStrategy::default(),
            lookahead: LookaheadMode::default(),
            sanity_limit: crate::planner::DEFAULT_SANITY_LIMIT,
            trace_events: false,
            plan_events: false,
            #[cfg(feature = "hotpatching")]
            rebuild: None,
        }
    }

    /// Config with a **domain rebuild closure** for Subsecond hot-reloading
    /// (requires the `hotpatching` feature). Whenever a hot patch lands, the
    /// driver calls `rebuild` to re-record and re-bake the domain, swaps it
    /// in, and drops every agent's plan so agents replan against the new
    /// behavior. A rebuild that returns `Err` keeps the current domain.
    #[cfg(feature = "hotpatching")]
    pub fn with_rebuild(
        domain: HtnDomain,
        rebuild: impl Fn() -> HtnResult<HtnDomain> + Send + Sync + 'static,
    ) -> Self {
        Self {
            rebuild: Some(Box::new(rebuild)),
            ..Self::new(domain)
        }
    }

    /// Set the global search strategy.
    pub fn with_strategy(mut self, strategy: HtnSearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the decomposition-step budget.
    pub fn with_sanity_limit(mut self, limit: usize) -> Self {
        self.sanity_limit = limit;
        self
    }

    /// Set the look-ahead gating mode.
    pub fn with_lookahead_mode(mut self, mode: LookaheadMode) -> Self {
        self.lookahead = mode;
        self
    }

    /// Forward the planner's [`DecompositionTrace`] events after each
    /// plan/resume.
    pub fn with_trace_events(mut self, enabled: bool) -> Self {
        self.trace_events = enabled;
        self
    }

    /// Emit plan lifecycle [`PlanEvent`]s to `Messages<PlanEvent>`.
    pub fn with_plan_events(mut self, enabled: bool) -> Self {
        self.plan_events = enabled;
        self
    }
}

/// Per-tick reusable driver buffers, parked as a resource between runs so
/// the driver allocates nothing in steady state: the agent-entity list
/// (cleared and refilled each tick) and one [`PlanState`] scratchpad
/// (refreshed in place per phase — plan, validate, post-action re-extract —
/// instead of re-allocating up to three times per agent per tick).
#[derive(Resource, Default)]
struct DriverScratch {
    entities: Vec<Entity>,
    state: Option<PlanState>,
    /// Working copy for quiet-prefix elision (reused across plannings via
    /// `copy_from` — zero steady-state allocation, same discipline as
    /// `state`). Dropped on a hot-reload domain swap (fresh registry).
    elide: Option<PlanState>,
    /// The `HotPatchChanges` tick observed on the previous run (hot-reload
    /// support): a different value means a patch landed between ticks.
    #[cfg(feature = "hotpatching")]
    hotpatch_tick: Option<u32>,
}

/// Elide the plan's leading run of **quiet** steps and return the index of
/// the first consequential step (== the plan's length when every step is
/// quiet). A step is quiet iff it dispatches **no action** and its real
/// write slots — applied against current reality — already hold the values
/// the effect would produce, i.e. its entire execution footprint is already
/// reflected in the world.
///
/// # Soundness
///
/// A quiet step's execution footprint in the driver is exactly: precondition
/// reads (pure), effect application + commit (verified byte-identical on
/// every committed slot), and an action dispatch (none exists). Executing it
/// or skipping it therefore leaves the world identical, so resuming the
/// fresh plan at the first consequential step is equivalent to executing it
/// from the top — and the planner certified the suffix from exactly the
/// state the skip starts from. Every executed step is still re-validated
/// against live reality before it runs, so later drift is handled exactly
/// like the baseline.
///
/// The per-step no-op check is what makes this safe where structural plan
/// comparison (Fluid HTN's MTR rejection) is not: if a drift **undid** an
/// executed step's effect (a hammer picked up and knocked out of the
/// pocket), re-applying the effect changes its slot, the step is not quiet,
/// and it re-executes — elision can never skip a step the world still
/// needs. Each check runs against a clean copy of reality (the working
/// copy is reset per step), so planning-only effect mutations from earlier
/// checks cannot poison a later one, and each elided step is quiet on its
/// own, independent of the others.
///
/// Deliberately conservative corners: steps with actions are never elided
/// (commands are opaque world effects); steps writing heap-owning
/// components (Vec/HashMap members) never elide — their reallocating writes
/// always move bytes, so they read as "changed"; planning-only
/// (`.expected`) mutations are ignored by the check, matching execution,
/// which never commits them; and a plan whose steps are *all* quiet
/// degrades to planless — executing it would be pure theater.
fn elide_quiet_prefix(
    tasks: &[Task],
    registry: &ComponentRegistry,
    state: &PlanState,
    scratch: &mut Option<PlanState>,
    plan: &Plan,
) -> usize {
    let work = scratch.get_or_insert_with(|| PlanState::build(registry).finish());
    let mut j = 0;
    while let Some(&idx) = plan.steps().get(j) {
        match tasks.get(idx as usize) {
            // A step with no action and no committed writes has an empty
            // footprint by construction (the driver skips apply+commit when
            // the write list is empty).
            Some(Task::Primitive(primitive)) if primitive.action.is_none() => {
                let writes = primitive.write_slot_slice();
                if writes.is_empty() {
                    j += 1;
                    continue;
                }
                // Clean base per step: each elision decision is measured
                // against reality alone, never against earlier checks.
                work.copy_from(state);
                primitive.apply_effects(work);
                let unchanged = work.slots_unchanged(state, writes);
                if !unchanged {
                    break;
                }
                j += 1;
            }
            _ => break,
        }
    }
    j
}

/// One agent's AI turn: **at most two passes**. Pass 0 plans (or resumes a
/// pause-truncated plan) if planless and executes one step. If the step's
/// re-validation fails (the world drifted since planning — an enemy walked
/// into view, the item was taken), the plan is dropped and pass 1 **replans
/// from reality in the same tick**: the fresh plan is selected against the
/// world as it is *now* and executes immediately, so an interrupt changes
/// behavior the very tick it happens instead of a tick later. Pass 1 is the
/// last: if its plan also fails validation (a non-deterministic precondition
/// oscillating between passes, or a system mutating mid-tick), the agent
/// goes planless and retries next tick — bounded work per tick, no spinning.
///
/// Lifecycle events (when `config.plan_events` is set) are pushed onto
/// `events`; the caller flushes them into `Messages<PlanEvent>` once per
/// tick. The config resource is scoped by the caller so domain references
/// stay alive while the world is mutated.
fn agent_turn(
    world: &mut World,
    config: &HtnConfig,
    scratch: &mut DriverScratch,
    entity: Entity,
    events: &mut Vec<PlanEvent>,
) {
    // Lifecycle-bridge bookkeeping: the plan an earlier pass of THIS turn
    // dropped on drift — the `old` side of the `PlanReplaced` event the
    // fresh install reports.
    let mut drifted_off: Option<Plan> = None;
    'turn: for pass in 0..2u32 {
        // 1. Plan if planless — or **resume** an exhausted paused
        // plan: a plan truncated by a pause marker keeps its
        // remaining decomposition work queued, and once its compiled
        // steps have executed the driver resumes decomposition from
        // the pause instead of replanning from the root (the prefix
        // is committed history — it can never be backtracked into).
        // A per-agent [`SearchOverride`] component replaces the
        // global strategy/budget for this entity. The same-tick
        // repair replan (pass 1) bypasses the [`PlanEvery`] throttle
        // — the throttle paces *idle* replanning, and an execution
        // interrupt deserves an immediate reaction; the attempt
        // still resets the deferral, so subsequent planless ticks
        // stay paced.
        let exhausted_pause = world.get::<HtnAgent>(entity).is_some_and(|a| {
            a.plan.as_ref().is_some_and(|p| {
                p.status() == PlanStatus::Paused
                    && a.cursor >= p.len()
                    && p.resume().is_some_and(|r| {
                        r.tasks.iter().any(|s| matches!(s, ResumeStep::Task(_)))
                    })
            })
        });
        if world
            .get::<HtnAgent>(entity)
            .is_some_and(|a| a.plan.is_none())
            || exhausted_pause
        {
            // LOD throttle: [`PlanEvery`] paces (re)planning to once
            // every `n` runs while the agent sits planless. Execution
            // of an existing plan is never delayed.
            let every = world
                .get::<PlanEvery>(entity)
                .map(|p| p.0.max(1))
                .unwrap_or(1);
            if every > 1 && pass == 0 {
                let deferred = world.get_mut::<HtnAgent>(entity).is_some_and(|mut a| {
                    if a.plan_deferral > 0 {
                        a.plan_deferral -= 1;
                        true
                    } else {
                        false
                    }
                });
                if deferred {
                    break 'turn;
                }
            }
            // The per-agent archetype root ([`AgentRoot`]) wins over the
            // domain's main root — the seam for one domain serving many
            // NPC kinds.
            let root = world
                .get::<AgentRoot>(entity)
                .map(|r| r.0 as usize)
                .unwrap_or(config.domain.root);
            let tasks = &config.domain.tasks;
            let registry = &config.domain.components;
            // One scratchpad, refreshed in place: planning, validation,
            // and the post-action re-extract all reuse the same buffer.
            let state = scratch
                .state
                .get_or_insert_with(|| PlanState::build(registry).finish());
            state.refresh(world, entity);
            let (strategy, sanity) = world
                .get::<SearchOverride>(entity)
                .map(|o| {
                    (
                        o.strategy
                            .clone()
                            .unwrap_or_else(|| config.strategy.clone()),
                        o.sanity_limit,
                    )
                })
                .unwrap_or_else(|| (config.strategy.clone(), None));
            let sanity = sanity.unwrap_or(config.sanity_limit);
            let mut planner = HtnPlanner::new(&config.domain);
            planner
                .set_lookahead_mode(config.lookahead)
                .set_sanity_limit(sanity)
                .set_strategy(strategy.clone());
            // The `old` plan for the lifecycle event: a plan dropped
            // on drift earlier this turn (root replan), or the
            // exhausted paused plan being resumed (resume). Drift
            // always replans from the root — the pause context is
            // abandoned and re-derived from reality; resuming keeps
            // the committed chain.
            let dropped = drifted_off.take();
            let resuming = if dropped.is_none() && exhausted_pause {
                world
                    .get_mut::<HtnAgent>(entity)
                    .and_then(|mut a| a.plan.take())
            } else {
                None
            };
            let resume_point = resuming.as_ref().and_then(|p| p.resume());
            let mut trace_buf: Vec<DecompositionTrace> = Vec::new();
            // A planning error (unregistered root, a genuinely
            // unsolvable state, or a resume whose remaining
            // decomposition failed) is treated like an empty plan:
            // the agent stays planless and idles — it never wedges
            // on a bad plan, and the next tick replans from the root.
            let plan = (match resume_point {
                Some(point) => planner.resume(point, state),
                None if config.trace_events => {
                    planner.plan_traced(root, state, &mut trace_buf)
                }
                None => planner.plan(root, state),
            })
            .unwrap_or_default();
            if !trace_buf.is_empty() {
                world
                    .resource_mut::<bevy_ecs::message::Messages<DecompositionTrace>>()
                    .write_batch(trace_buf.drain(..));
            }
            // An empty plan means "nothing to do" — store it as planless
            // so a later world change can trigger a real replan instead
            // of wedging the agent on a zero-length program. Same for a
            // plan whose every step is quiet (see `elide_quiet_prefix`).
            // A paused plan with queued resume work is never degraded:
            // its (possibly empty) prefix is a real leg — executing it
            // ends in a resume, not a replan.
            let quiet =
                elide_quiet_prefix(tasks, registry, state, &mut scratch.elide, &plan);
            let resume_work = plan.resume().is_some_and(|r| {
                r.tasks.iter().any(|s| matches!(s, ResumeStep::Task(_)))
            });
            let (plan, cursor) = if quiet >= plan.len() && !resume_work {
                (None, 0)
            } else {
                (Some(plan), quiet)
            };
            if config.plan_events {
                if let Some(new) = &plan {
                    events.push(PlanEvent::PlanReplaced {
                        entity,
                        old: dropped.or(resuming),
                        new: new.clone(),
                    });
                }
            }
            if let Some(mut agent) = world.get_mut::<HtnAgent>(entity) {
                agent.plan = plan;
                agent.cursor = cursor;
                agent.plan_deferral = every.saturating_sub(1);
            }
        }

        // 2. Resolve the next step from the compiled program: a flat
        // array index into the baked task array — no name lookups.
        let Some(step_idx) = world.get::<HtnAgent>(entity).and_then(|a| {
            let plan = a.plan.as_ref()?;
            plan.step_task(a.cursor)
        }) else {
            break 'turn;
        };
        let Some(Task::Primitive(primitive)) = config.domain.tasks.get(step_idx) else {
            break 'turn;
        };

        // 3. Validate the step against the real world: if the
        // preconditions no longer hold (the world drifted since
        // planning), the plan is dropped and — on the first pass — the
        // agent replans from reality **this tick** (the loop's pass 1).
        // The scratchpad is the shared driver buffer, refreshed in place.
        let registry = &config.domain.components;
        let state = scratch
            .state
            .get_or_insert_with(|| PlanState::build(registry).finish());
        state.refresh(world, entity);
        if !primitive.preconditions_met(state) {
            if config.plan_events {
                events.push(PlanEvent::StepFailed {
                    entity,
                    task: step_idx as u32,
                    task_name: primitive.name,
                });
            }
            if let Some(mut agent) = world.get_mut::<HtnAgent>(entity) {
                drifted_off = agent.plan.take();
                agent.cursor = 0;
            }
            if pass == 0 {
                continue 'turn;
            }
            // Defensive: the pass-1 plan already failed its own first
            // validation. Go planless; the next tick replans fresh.
            break 'turn;
        }

        // 4. Execute: dispatch the action's commands (then flush so the
        // effects observe post-action state), and commit the effects to
        // the real components.
        if let Some(action) = &primitive.action {
            // The command buffers hold borrows of the world: they are
            // released at the end of this block so `flush` can run.
            {
                let mut commands = world.commands();
                let mut entity_commands = commands.entity(entity);
                action(&mut entity_commands);
            }
            world.flush();
            // The action may have mutated planning components: re-extract
            // so effects apply on top of the post-action state.
            state.refresh(world, entity);
        }
        // Commit only the baked write slots (read-only effect parameters
        // are never committed to the real entity).
        let writes = primitive.write_slot_slice();
        if !writes.is_empty() {
            primitive.apply_effects(state);
            state.write_back_with(world, entity, writes);
        }
        // Per-action lifecycle hook: the step ran for real.
        if config.plan_events {
            events.push(PlanEvent::StepExecuted {
                entity,
                task: step_idx as u32,
                task_name: primitive.name,
            });
        }

        // 5. Advance the cursor. A finished `Complete` plan is dropped
        // for replan (with a lifecycle event); a finished **paused**
        // plan is kept — the next run resumes its decomposition from
        // the pause point; a budget-truncated `Partial` plan just
        // ends. One step per tick: the turn ends here even with
        // passes left.
        if let Some(mut agent) = world.get_mut::<HtnAgent>(entity) {
            agent.cursor += 1;
            let done = agent.plan.as_ref().is_some_and(|p| agent.cursor >= p.len());
            if done {
                let status = agent.plan.as_ref().map(|p| p.status());
                if status == Some(PlanStatus::Complete) && config.plan_events {
                    events.push(PlanEvent::PlanCompleted { entity });
                }
                if status != Some(PlanStatus::Paused) {
                    agent.plan = None;
                    agent.cursor = 0;
                }
            }
        }
        break 'turn;
    } // 'turn
}

/// The AI driver: one exclusive-system tick for every [`HtnAgent`].
///
/// Per agent per run:
/// 1. **Plan** — if the agent has no plan, extract a [`PlanState`] scratchpad
///    from the entity's components and forward-plan from [`HtnConfig`]'s
///    domain root.
/// 2. **Validate** — re-check the next primitive's preconditions against the
///    real world. If the world drifted since planning (another system moved
///    the agent, spent the ammo, ...), the plan is dropped immediately so the
///    next tick replans from reality.
/// 3. **Execute one step** — dispatch the primitive's action commands against
///    the agent entity, commit its effects to the real components, flush, and
///    advance the cursor (a finished plan is dropped so the next tick
///    replans).
///
/// Add it to an `App` with `app.add_systems(Update, htn_ai_system)` (exclusive
/// systems take `&mut World` directly).
pub fn htn_ai_system(world: &mut World) {
    // Trace events: ensure the resource exists and double-buffer per tick.
    if world
        .get_resource::<bevy_ecs::message::Messages<DecompositionTrace>>()
        .is_none()
    {
        world.insert_resource(bevy_ecs::message::Messages::<DecompositionTrace>::default());
    }
    world
        .resource_mut::<bevy_ecs::message::Messages<DecompositionTrace>>()
        .update();

    // Plan lifecycle events: same bridge discipline as the trace above.
    if world
        .get_resource::<bevy_ecs::message::Messages<PlanEvent>>()
        .is_none()
    {
        world.insert_resource(bevy_ecs::message::Messages::<PlanEvent>::default());
    }
    world
        .resource_mut::<bevy_ecs::message::Messages<PlanEvent>>()
        .update();

    // Reusable buffers: take them out of the world (their allocations move
    // with them), work, then put them back for the next tick.
    if world.get_resource::<DriverScratch>().is_none() {
        world.insert_resource(DriverScratch::default());
    }
    let mut scratch = world
        .remove_resource::<DriverScratch>()
        .expect("just inserted");
    scratch.entities.clear();

    // Hot-reload pass (Subsecond): if a patch landed since the previous run
    // and the config carries a rebuild closure, re-record + re-bake the
    // domain, swap it in, and drop every agent's plan so the next tick
    // replans against the new behavior. A failing rebuild keeps the old
    // domain (the patch is still recorded as seen, so it is not retried).
    #[cfg(feature = "hotpatching")]
    {
        use bevy_ecs::change_detection::DetectChanges as _;
        let current = world
            .get_resource_ref::<bevy_ecs::HotPatchChanges>()
            .map(|r| r.last_changed().get())
            .unwrap_or(0);
        match scratch.hotpatch_tick {
            // First observation: record the baseline, no rebuild.
            None => scratch.hotpatch_tick = Some(current),
            Some(seen) if seen != current => {
                scratch.hotpatch_tick = Some(current);
                world.resource_scope(|world, mut config: bevy_ecs::prelude::Mut<HtnConfig>| {
                    if let Some(rebuild) = config.rebuild.as_deref() {
                        if let Ok(new_domain) = rebuild() {
                            config.domain = new_domain;
                            // The new domain owns a fresh registry: the
                            // parked scratchpad's layout is stale.
                            scratch.state = None;
                            scratch.elide = None;
                            // Every agent replans against the new domain.
                            let mut agents =
                                world.query_filtered::<Entity, bevy_ecs::prelude::With<HtnAgent>>();
                            let patched: Vec<Entity> = agents.iter(world).collect();
                            for entity in patched {
                                if let Some(mut agent) = world.get_mut::<HtnAgent>(entity) {
                                    agent.plan = None;
                                    agent.cursor = 0;
                                }
                            }
                        }
                    }
                });
            }
            Some(_) => {}
        }
    }

    {
        let mut agents = world.query_filtered::<Entity, bevy_ecs::prelude::With<HtnAgent>>();
        scratch.entities.extend(agents.iter(world));
    }

    // Plan lifecycle events buffered across the agent loop, flushed once at
    // the end of the tick (ownership moves through the buffer — no clones on
    // the `old`/`new` plans themselves).
    let mut events: Vec<PlanEvent> = Vec::new();

    // Take the agent list out of the scratch so each turn can borrow the
    // whole scratch mutably (the allocation is parked back after the loop —
    // zero steady-state allocation).
    let entities = std::mem::take(&mut scratch.entities);

    for &entity in &entities {
        // The config resource is scoped out of the world for the whole tick,
        // so domain references (preconditions, effects, actions) stay alive
        // while the world itself is mutated.
        world.resource_scope(|world, config: bevy_ecs::prelude::Mut<HtnConfig>| {
            agent_turn(world, &config, &mut scratch, entity, &mut events);
        });
    }
    // Park the agent list back on the scratch for the next tick.
    scratch.entities = entities;

    // Flush the lifecycle bridge: events buffered during the agent loop are
    // readable for the rest of this tick.
    if !events.is_empty() {
        world
            .resource_mut::<bevy_ecs::message::Messages<PlanEvent>>()
            .write_batch(events.drain(..));
    }

    // Park the buffers back on the world for the next tick.
    world.insert_resource(scratch);
}
