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
use bevy_ecs::prelude::{Component, Resource};
use bevy_ecs::world::World;

use crate::domain::HtnDomain;
use crate::planner::{HtnPlanner, Plan};
use crate::selection::{DecompositionTrace, HtnSearchStrategy, SearchOverride};
use crate::state::PlanState;
use crate::tasks::Task;

/// The per-entity AI component: current plan and cursor.
///
/// The agent's plan state is the entity's own component set — the driver
/// extracts a [`PlanState`] scratchpad from it at plan time and commits
/// effects back to the real components at execution time.
#[derive(Component, Debug, Default, Clone)]
pub struct HtnAgent {
    /// The current forward plan, if one has been (re)planned.
    pub plan: Option<Plan>,
    /// Index into [`Self::plan`]'s task list of the next step to execute.
    pub cursor: usize,
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
    /// Whether the forward planner's look-ahead sweep runs (default `true`).
    pub lookahead: bool,
    /// The forward planner's decomposition-step budget (default `100`).
    pub sanity_limit: usize,
    /// Whether the driver forwards [`DecompositionTrace`] events to
    /// `Messages<DecompositionTrace>` after each plan (default `false`).
    pub debug_trace: bool,
}

impl HtnConfig {
    /// Config with default planner tuning.
    pub fn new(domain: HtnDomain) -> Self {
        Self {
            domain,
            strategy: HtnSearchStrategy::default(),
            lookahead: true,
            sanity_limit: 100,
            debug_trace: false,
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

    /// Enable or disable the look-ahead sweep.
    pub fn with_lookahead(mut self, enabled: bool) -> Self {
        self.lookahead = enabled;
        self
    }

    /// Forward [`DecompositionTrace`] events after each plan.
    pub fn with_debug_trace(mut self, enabled: bool) -> Self {
        self.debug_trace = enabled;
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

    // Reusable buffers: take them out of the world (their allocations move
    // with them), work, then put them back for the next tick.
    if world.get_resource::<DriverScratch>().is_none() {
        world.insert_resource(DriverScratch::default());
    }
    let mut scratch = world
        .remove_resource::<DriverScratch>()
        .expect("just inserted");
    scratch.entities.clear();
    {
        let mut agents = world.query_filtered::<Entity, bevy_ecs::prelude::With<HtnAgent>>();
        scratch.entities.extend(agents.iter(world));
    }

    for &entity in &scratch.entities {
        // The config resource is scoped out of the world for the whole tick,
        // so domain references (preconditions, effects, actions) stay alive
        // while the world itself is mutated.
        world.resource_scope(|world, config: bevy_ecs::prelude::Mut<HtnConfig>| {
            // 1. Plan if planless. A per-agent [`SearchOverride`] component
            // replaces the global strategy/budget for this entity.
            if world
                .get::<HtnAgent>(entity)
                .is_some_and(|a| a.plan.is_none())
            {
                let root = config.domain.root;
                let registry = &config.domain.components;
                // One scratchpad, refreshed in place: planning, validation,
                // and the post-action re-extract all reuse the same buffer.
                let state = scratch
                    .state
                    .get_or_insert_with(|| PlanState::build(registry).finish());
                state.refresh(world, entity, registry);
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
                    .set_lookahead(config.lookahead)
                    .set_sanity_limit(sanity);
                let mut trace_buf: Vec<DecompositionTrace> = Vec::new();
                let plan = match &strategy {
                    HtnSearchStrategy::DepthFirst => {
                        if config.debug_trace {
                            planner.plan_traced_index(root, state, &mut trace_buf)
                        } else {
                            planner.plan_index(root, state)
                        }
                    }
                    HtnSearchStrategy::DepthFirstFailFast => {
                        planner.set_fail_fast(true);
                        if config.debug_trace {
                            planner.plan_traced_index(root, state, &mut trace_buf)
                        } else {
                            planner.plan_index(root, state)
                        }
                    }
                    HtnSearchStrategy::CostBounded => {
                        planner.set_cost_bounded(true);
                        if config.debug_trace {
                            planner.plan_traced_index(root, state, &mut trace_buf)
                        } else {
                            planner.plan_index(root, state)
                        }
                    }
                    HtnSearchStrategy::Custom(searcher) => {
                        searcher.search(&config.domain, state).unwrap_or_default()
                    }
                };
                if !trace_buf.is_empty() {
                    world
                        .resource_mut::<bevy_ecs::message::Messages<DecompositionTrace>>()
                        .write_batch(trace_buf.drain(..));
                }
                // An empty plan means "nothing to do" — store it as planless
                // so a later world change can trigger a real replan instead
                // of wedging the agent on a zero-length program.
                let plan = if plan.is_empty() { None } else { Some(plan) };
                if let Some(mut agent) = world.get_mut::<HtnAgent>(entity) {
                    agent.plan = plan;
                    agent.cursor = 0;
                }
            }

            // 2. Resolve the next step from the compiled program: a flat
            // array index into the baked task array — no name lookups.
            let Some(step_idx) = world.get::<HtnAgent>(entity).and_then(|a| {
                let plan = a.plan.as_ref()?;
                plan.step_task(a.cursor)
            }) else {
                return;
            };
            let Some(Task::Primitive(primitive)) = config.domain.tasks.get(step_idx) else {
                return;
            };

            // 3. Validate the step against the real world: if the
            // preconditions no longer hold (the world drifted since
            // planning), drop the plan and replan next tick. The scratchpad
            // is the shared driver buffer, refreshed in place.
            let registry = &config.domain.components;
            let state = scratch
                .state
                .get_or_insert_with(|| PlanState::build(registry).finish());
            state.refresh(world, entity, registry);
            if !primitive.preconditions_met(state) {
                if let Some(mut agent) = world.get_mut::<HtnAgent>(entity) {
                    agent.plan = None;
                    agent.cursor = 0;
                }
                return;
            }

            // 4. Execute: dispatch the action's commands (then flush so the
            // effects observe post-action state), and commit the effects to
            // the real components.
            if let Some(action) = &primitive.action {
                let mut commands = world.commands();
                let mut entity_commands = commands.entity(entity);
                action(&mut entity_commands);
                drop(entity_commands);
                drop(commands);
                world.flush();
                // The action may have mutated planning components: re-extract
                // so effects apply on top of the post-action state.
                state.refresh(world, entity, registry);
            }
            let writes: Vec<usize> = primitive.write_slots().collect();
            if !writes.is_empty() {
                primitive.apply_effects(state);
                state.write_back_with(world, entity, registry, &writes);
            }

            // 5. Advance the cursor (a finished plan is dropped for replan).
            if let Some(mut agent) = world.get_mut::<HtnAgent>(entity) {
                agent.cursor += 1;
                let done = agent.plan.as_ref().is_some_and(|p| agent.cursor >= p.len());
                if done {
                    agent.plan = None;
                    agent.cursor = 0;
                }
            }
        });
    }

    // Park the buffers back on the world for the next tick.
    world.insert_resource(scratch);
}
