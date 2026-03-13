//! FEM Core - Portable Finite Element Method soft body physics library
//!
//! This library provides platform-independent soft body simulation using
//! the Finite Element Method. It has no dependencies on WebGL, WASM, or
//! any platform-specific APIs.
//!
//! Two solvers are available:
//! - `softbody`: Force-based FEM with Neo-Hookean materials
//! - `xpbd`: Position-based dynamics (unconditionally stable)

pub mod math;
pub mod fem;
pub mod mesh;
pub mod softbody;
pub mod xpbd;
pub mod trace;
pub mod compute;

#[cfg(test)]
mod simulation_tests;

// Re-export commonly used types
pub use math::{Mat2, Vec2};
pub use fem::{LameParams, TriangleData, ForceStats, compute_triangle_data};
pub use mesh::Mesh;
pub use softbody::{SoftBody, Material};
pub use xpbd::XPBDSoftBody;
pub use trace::{SimulationTracer, FrameTrace, TraceStatistics};
