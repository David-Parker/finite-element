//! Renderer trait - the core abstraction for platform-specific backends

use super::{Color, Material, RenderMesh, Sprite, Texture, TextureId};

/// Error type for renderer operations
#[derive(Clone, Debug)]
pub enum RenderError {
    /// Texture creation failed
    TextureCreationFailed(String),
    /// Shader compilation failed
    ShaderError(String),
    /// Invalid texture ID
    InvalidTexture(TextureId),
    /// Backend-specific error
    BackendError(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TextureCreationFailed(msg) => write!(f, "Texture creation failed: {}", msg),
            Self::ShaderError(msg) => write!(f, "Shader error: {}", msg),
            Self::InvalidTexture(id) => write!(f, "Invalid texture ID: {:?}", id),
            Self::BackendError(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for RenderError {}

/// Platform-agnostic renderer trait
///
/// Implement this trait for your graphics backend (WebGL, Metal, wgpu, etc).
///
/// # Example Implementation
///
/// ```ignore
/// struct WebGLRenderer { /* ... */ }
///
/// impl Renderer for WebGLRenderer {
///     fn clear(&mut self, color: Color) {
///         self.gl.clear_color(color.r, color.g, color.b, color.a);
///         self.gl.clear(GL::COLOR_BUFFER_BIT);
///     }
///     // ... other methods
/// }
/// ```
pub trait Renderer {
    /// Clear the screen with a color
    fn clear(&mut self, color: Color);

    /// Set the camera/view transformation
    fn set_camera(&mut self, x: f32, y: f32, width: f32, height: f32);

    /// Create a texture and return its ID
    fn create_texture(&mut self, texture: &Texture) -> Result<TextureId, RenderError>;

    /// Delete a texture
    fn delete_texture(&mut self, id: TextureId);

    /// Draw a mesh with a material
    fn draw_mesh(&mut self, mesh: &RenderMesh, material: &Material);

    /// Draw a sprite
    fn draw_sprite(&mut self, sprite: &Sprite);

    /// Draw a line (for debug rendering)
    fn draw_line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, color: Color, width: f32);

    /// Draw multiple meshes efficiently (batch if possible)
    fn draw_meshes(&mut self, meshes: &[(&RenderMesh, &Material)]) {
        // Default implementation: draw one by one
        for (mesh, material) in meshes {
            self.draw_mesh(mesh, material);
        }
    }

    /// Begin a frame (called before any drawing)
    fn begin_frame(&mut self) {}

    /// End a frame (called after all drawing, may trigger buffer swap)
    fn end_frame(&mut self) {}

    /// Get the current viewport size
    fn viewport_size(&self) -> (u32, u32);
}

/// Extension trait for convenient rendering operations
pub trait RendererExt: Renderer {
    /// Draw a solid-color mesh
    fn draw_mesh_solid(&mut self, mesh: &RenderMesh, color: Color) {
        self.draw_mesh(mesh, &Material::solid(color));
    }

    /// Draw a mesh with wireframe overlay
    fn draw_mesh_wireframe(&mut self, mesh: &RenderMesh, fill: Color, wire: Color) {
        self.draw_mesh(mesh, &Material::solid(fill).with_wireframe(wire));
    }

    /// Draw a rectangle
    fn draw_rect(&mut self, x: f32, y: f32, width: f32, height: f32, color: Color) {
        let hw = width * 0.5;
        let hh = height * 0.5;
        let mesh = RenderMesh {
            vertices: vec![
                super::Vertex::with_uv(x - hw, y - hh, 0.0, 0.0),
                super::Vertex::with_uv(x + hw, y - hh, 1.0, 0.0),
                super::Vertex::with_uv(x + hw, y + hh, 1.0, 1.0),
                super::Vertex::with_uv(x - hw, y + hh, 0.0, 1.0),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        };
        self.draw_mesh(&mesh, &Material::solid(color));
    }

    /// Draw a circle (approximated with segments)
    fn draw_circle(&mut self, x: f32, y: f32, radius: f32, color: Color, segments: u32) {
        use std::f32::consts::PI;

        let mut vertices = Vec::with_capacity(segments as usize + 1);
        let mut indices = Vec::with_capacity(segments as usize * 3);

        // Center vertex
        vertices.push(super::Vertex::with_uv(x, y, 0.5, 0.5));

        // Edge vertices
        for i in 0..segments {
            let angle = (i as f32 / segments as f32) * PI * 2.0;
            let px = x + angle.cos() * radius;
            let py = y + angle.sin() * radius;
            let u = 0.5 + angle.cos() * 0.5;
            let v = 0.5 + angle.sin() * 0.5;
            vertices.push(super::Vertex::with_uv(px, py, u, v));

            // Triangle from center to edge
            let next = (i + 1) % segments;
            indices.push(0);
            indices.push(i + 1);
            indices.push(next + 1);
        }

        let mesh = RenderMesh { vertices, indices };
        self.draw_mesh(&mesh, &Material::solid(color));
    }
}

// Blanket implementation
impl<T: Renderer> RendererExt for T {}
