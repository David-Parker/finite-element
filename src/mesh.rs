//! Mesh generation utilities

use std::f32::consts::PI;

/// Mesh data
pub struct Mesh {
    pub vertices: Vec<f32>,   // Flat array [x0, y0, x1, y1, ...]
    pub triangles: Vec<u32>,  // Triangle indices [i0, i1, i2, ...]
}

/// Create a ring (annulus) mesh
pub fn create_ring_mesh(
    outer_radius: f32,
    inner_radius: f32,
    segments: u32,
    radial_divisions: u32,
) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    // Create vertex grid (no duplicate vertices at seam)
    for r in 0..=radial_divisions {
        let radius = inner_radius + (outer_radius - inner_radius) * (r as f32 / radial_divisions as f32);
        for i in 0..segments {
            let angle = (i as f32 / segments as f32) * PI * 2.0;
            vertices.push(angle.cos() * radius);
            vertices.push(angle.sin() * radius);
        }
    }

    // Create triangles with proper wrap-around
    let verts_per_ring = segments;
    for r in 0..radial_divisions {
        for i in 0..segments {
            let curr = r * verts_per_ring + i;
            let next = r * verts_per_ring + (i + 1) % segments;
            let curr_outer = (r + 1) * verts_per_ring + i;
            let next_outer = (r + 1) * verts_per_ring + (i + 1) % segments;

            triangles.push(curr);
            triangles.push(curr_outer);
            triangles.push(next);

            triangles.push(next);
            triangles.push(curr_outer);
            triangles.push(next_outer);
        }
    }

    Mesh { vertices, triangles }
}

/// Create wireframe indices from ring mesh
pub fn create_ring_wireframe(segments: u32, radial_divisions: u32) -> Vec<u32> {
    let mut line_indices = Vec::new();
    let verts_per_ring = segments;

    for r in 0..radial_divisions {
        for i in 0..segments {
            let curr = r * verts_per_ring + i;
            let next = r * verts_per_ring + (i + 1) % segments;
            let curr_outer = (r + 1) * verts_per_ring + i;
            let next_outer = (r + 1) * verts_per_ring + (i + 1) % segments;

            // Inner edge
            line_indices.push(curr);
            line_indices.push(next);

            // Radial edge
            line_indices.push(curr);
            line_indices.push(curr_outer);

            // Diagonal edge
            line_indices.push(next);
            line_indices.push(curr_outer);

            // Outer edge (only on last ring)
            if r == radial_divisions - 1 {
                line_indices.push(curr_outer);
                line_indices.push(next_outer);
            }
        }
    }

    line_indices
}

/// Offset all vertices by a fixed amount
pub fn offset_vertices(vertices: &mut [f32], dx: f32, dy: f32) {
    for i in (0..vertices.len()).step_by(2) {
        vertices[i] += dx;
        vertices[i + 1] += dy;
    }
}

/// Create a simple square mesh
pub fn create_square_mesh(size: f32, divisions: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let half_size = size / 2.0;

    for y in 0..=divisions {
        for x in 0..=divisions {
            let px = -half_size + (x as f32 / divisions as f32) * size;
            let py = -half_size + (y as f32 / divisions as f32) * size;
            vertices.push(px);
            vertices.push(py);
        }
    }

    let verts_per_row = divisions + 1;
    for y in 0..divisions {
        for x in 0..divisions {
            let curr = y * verts_per_row + x;
            let right = curr + 1;
            let up = curr + verts_per_row;
            let up_right = up + 1;

            triangles.push(curr);
            triangles.push(right);
            triangles.push(up);

            triangles.push(right);
            triangles.push(up_right);
            triangles.push(up);
        }
    }

    Mesh { vertices, triangles }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_mesh() {
        let mesh = create_ring_mesh(1.0, 0.5, 8, 2);
        assert_eq!(mesh.vertices.len() / 2, 24); // 8 * (2+1) vertices
        assert_eq!(mesh.triangles.len() / 3, 32); // 8 * 2 * 2 triangles
    }

    #[test]
    fn test_square_mesh() {
        let mesh = create_square_mesh(2.0, 4);
        assert_eq!(mesh.vertices.len() / 2, 25); // 5x5 vertices
        assert_eq!(mesh.triangles.len() / 3, 32); // 4x4x2 triangles
    }
}
