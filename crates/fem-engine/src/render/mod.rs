//! Platform-agnostic rendering abstractions
//!
//! This module defines traits and types for rendering that can be implemented
//! by any graphics backend (WebGL, Metal, Vulkan, wgpu, etc).
//!
//! # Design
//!
//! The rendering system uses dependency inversion:
//! - `fem-engine` defines the `Renderer` trait and data types
//! - Platform crates (like `fem-game`) implement the trait for their backend
//! - Game logic works with the abstract `Renderer` trait
//!
//! This allows the same game code to run on web, iOS, Android, and desktop.

mod color;
mod material;
mod mesh;
mod texture;
mod traits;

pub use color::Color;
pub use material::Material;
pub use mesh::{RenderMesh, Vertex};
pub use texture::{Texture, TextureFilter, TextureFormat, TextureId, TextureWrap};
pub use traits::{RenderError, Renderer, RendererExt};

mod sprite;
pub use sprite::{Sprite, SpriteFrame, SpriteSheet};
