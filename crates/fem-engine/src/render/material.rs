//! Material types for rendering

use super::{Color, TextureId};

/// Rendering material - defines how a surface appears
#[derive(Clone, Debug)]
pub struct Material {
    /// Base color (multiplied with texture if present)
    pub color: Color,
    /// Optional texture
    pub texture: TextureId,
    /// Texture tiling factor (1.0 = no tiling)
    pub texture_scale: (f32, f32),
    /// Texture offset
    pub texture_offset: (f32, f32),
    /// Wireframe color (for debug/outline rendering)
    pub wireframe_color: Option<Color>,
    /// Opacity (0.0 = transparent, 1.0 = opaque)
    pub opacity: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            texture: TextureId::NONE,
            texture_scale: (1.0, 1.0),
            texture_offset: (0.0, 0.0),
            wireframe_color: None,
            opacity: 1.0,
        }
    }
}

impl Material {
    /// Create a solid color material
    pub fn solid(color: Color) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    /// Create a textured material
    pub fn textured(texture: TextureId) -> Self {
        Self {
            texture,
            ..Default::default()
        }
    }

    /// Create a material with color tint and texture
    pub fn tinted(color: Color, texture: TextureId) -> Self {
        Self {
            color,
            texture,
            ..Default::default()
        }
    }

    /// Set texture tiling
    pub fn with_tiling(mut self, scale_u: f32, scale_v: f32) -> Self {
        self.texture_scale = (scale_u, scale_v);
        self
    }

    /// Set texture offset
    pub fn with_offset(mut self, u: f32, v: f32) -> Self {
        self.texture_offset = (u, v);
        self
    }

    /// Enable wireframe overlay
    pub fn with_wireframe(mut self, color: Color) -> Self {
        self.wireframe_color = Some(color);
        self
    }

    /// Set opacity
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}
