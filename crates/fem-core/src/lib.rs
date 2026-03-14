//! FEM Core - Portable soft body physics library
//!
//! This library provides platform-independent soft body simulation using
//! XPBD (Extended Position-Based Dynamics) for unconditionally stable physics.
//! It has no dependencies on WebGL, WASM, or any platform-specific APIs.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use fem_core::{PhysicsWorld, BodyConfig, Material};
//! use fem_core::mesh::create_ring_mesh;
//!
//! // Create a physics world
//! let mut world = PhysicsWorld::new();
//! world.set_gravity(-9.8);
//! world.set_ground(Some(-5.0));
//!
//! // Create a mesh and add it as a body
//! let mesh = create_ring_mesh(1.0, 0.5, 16, 4);
//! let player = world.add_body(&mesh, BodyConfig::new()
//!     .with_material(Material::RUBBER)
//!     .at_position(0.0, 2.0));
//!
//! // Simulate
//! world.step(1.0 / 60.0);
//!
//! // Query state
//! let pos = world.get_position(player);
//! ```
//!
//! # Modules
//!
//! - [`world`] - High-level physics world API (recommended)
//! - [`xpbd`] - Low-level XPBD solver and body types
//! - [`mesh`] - Mesh generation utilities
//! - [`math`] - Math primitives (Vec2, Mat2)

pub mod math;
pub mod mesh;
pub mod xpbd;
pub mod trace;
pub mod compute;
pub mod world;

// === High-level API (recommended) ===

pub use world::{
    PhysicsWorld,
    BodyHandle,
    BodyConfig,
    Material,
    CollisionGroups,
};

// === Mesh utilities ===

pub use mesh::Mesh;

// === Low-level API (for advanced usage) ===

pub use math::{Mat2, Vec2};
pub use xpbd::{XPBDSoftBody, CollisionSystem, EdgeConstraint, AreaConstraint};
pub use trace::{SimulationTracer, FrameTrace, TraceStatistics};
