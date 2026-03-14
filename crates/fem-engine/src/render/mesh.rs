//! Render mesh types

use super::Color;

/// A vertex with position, UV, and optional color
#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    /// Position (x, y)
    pub position: [f32; 2],
    /// Texture coordinates (u, v) in 0.0-1.0 range
    pub uv: [f32; 2],
    /// Vertex color (multiplied with material/texture)
    pub color: Color,
}

impl Vertex {
    /// Create a vertex with position only (white color, zero UV)
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            position: [x, y],
            uv: [0.0, 0.0],
            color: Color::WHITE,
        }
    }

    /// Create a vertex with position and UV
    #[inline]
    pub fn with_uv(x: f32, y: f32, u: f32, v: f32) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
            color: Color::WHITE,
        }
    }

    /// Create a vertex with position, UV, and color
    #[inline]
    pub fn with_uv_color(x: f32, y: f32, u: f32, v: f32, color: Color) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
            color,
        }
    }

    /// Set UV coordinates
    #[inline]
    pub fn uv(mut self, u: f32, v: f32) -> Self {
        self.uv = [u, v];
        self
    }

    /// Set vertex color
    #[inline]
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

/// A renderable mesh with vertices and indices
///
/// This is the render-side representation of a mesh. It includes UV coordinates
/// and is designed for efficient GPU upload.
#[derive(Clone, Debug)]
pub struct RenderMesh {
    /// Interleaved vertex data
    pub vertices: Vec<Vertex>,
    /// Triangle indices (3 per triangle)
    pub indices: Vec<u32>,
}

impl RenderMesh {
    /// Create an empty mesh
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Create a mesh with pre-allocated capacity
    pub fn with_capacity(vertex_count: usize, index_count: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(vertex_count),
            indices: Vec::with_capacity(index_count),
        }
    }

    /// Create from raw position data and indices (generates default UVs)
    ///
    /// This is useful for converting from fem-core Mesh format.
    /// Positions are [x0, y0, x1, y1, ...] flat array.
    pub fn from_positions(positions: &[f32], indices: &[u32]) -> Self {
        let vertex_count = positions.len() / 2;
        let vertices = (0..vertex_count)
            .map(|i| Vertex::new(positions[i * 2], positions[i * 2 + 1]))
            .collect();

        Self {
            vertices,
            indices: indices.to_vec(),
        }
    }

    /// Create from positions with UV coordinates
    ///
    /// Both positions and uvs are flat arrays: [x0, y0, x1, y1, ...] and [u0, v0, u1, v1, ...]
    pub fn from_positions_uvs(positions: &[f32], uvs: &[f32], indices: &[u32]) -> Self {
        let vertex_count = positions.len() / 2;
        assert_eq!(uvs.len(), positions.len(), "UV count must match position count");

        let vertices = (0..vertex_count)
            .map(|i| {
                Vertex::with_uv(
                    positions[i * 2],
                    positions[i * 2 + 1],
                    uvs[i * 2],
                    uvs[i * 2 + 1],
                )
            })
            .collect();

        Self {
            vertices,
            indices: indices.to_vec(),
        }
    }

    /// Update vertex positions from a flat array (for animated/physics meshes)
    ///
    /// This preserves UV coordinates while updating positions.
    pub fn update_positions(&mut self, positions: &[f32]) {
        let vertex_count = positions.len() / 2;
        assert_eq!(
            vertex_count,
            self.vertices.len(),
            "Position count must match vertex count"
        );

        for (i, vertex) in self.vertices.iter_mut().enumerate() {
            vertex.position = [positions[i * 2], positions[i * 2 + 1]];
        }
    }

    /// Get flat position array (for compatibility with existing code)
    pub fn positions_flat(&self) -> Vec<f32> {
        let mut positions = Vec::with_capacity(self.vertices.len() * 2);
        for v in &self.vertices {
            positions.push(v.position[0]);
            positions.push(v.position[1]);
        }
        positions
    }

    /// Get flat UV array
    pub fn uvs_flat(&self) -> Vec<f32> {
        let mut uvs = Vec::with_capacity(self.vertices.len() * 2);
        for v in &self.vertices {
            uvs.push(v.uv[0]);
            uvs.push(v.uv[1]);
        }
        uvs
    }

    /// Number of vertices
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangles
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

impl Default for RenderMesh {
    fn default() -> Self {
        Self::new()
    }
}
