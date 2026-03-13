//! FEM Core - Portable soft body physics library
//!
//! This library provides platform-independent soft body simulation using
//! XPBD (Extended Position-Based Dynamics) for unconditionally stable physics.
//! It has no dependencies on WebGL, WASM, or any platform-specific APIs.

pub mod math;
pub mod mesh;
pub mod xpbd;
pub mod trace;
pub mod compute;

// Re-export commonly used types
pub use math::{Mat2, Vec2};
pub use mesh::Mesh;
pub use xpbd::{XPBDSoftBody, CollisionSystem};
pub use trace::{SimulationTracer, FrameTrace, TraceStatistics};
