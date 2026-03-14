//! FEM Engine - Platform-agnostic game engine
//!
//! This crate provides rendering abstractions and game engine utilities
//! that work with any graphics backend (WebGL, Metal, Vulkan, etc).
//!
//! The key design principle is dependency inversion: this crate defines
//! traits that platform-specific code implements, rather than depending
//! on any specific graphics API.

pub mod render;

// Re-export commonly used types
pub use render::{
    Color,
    Material,
    RenderError,
    RenderMesh,
    Renderer,
    RendererExt,
    Sprite,
    SpriteFrame,
    SpriteSheet,
    Texture,
    TextureFilter,
    TextureFormat,
    TextureId,
    TextureWrap,
    Vertex,
};
