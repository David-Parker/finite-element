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

/// Create an ellipse mesh with non-uniform vertex distribution
pub fn create_ellipse_mesh(
    width: f32,
    height: f32,
    segments: u32,
    rings: u32,
) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    // Center vertex
    vertices.push(0.0);
    vertices.push(0.0);

    // Create rings of vertices from center outward
    for r in 1..=rings {
        let t = r as f32 / rings as f32;
        let rx = width * 0.5 * t;
        let ry = height * 0.5 * t;

        for i in 0..segments {
            let angle = (i as f32 / segments as f32) * PI * 2.0;
            vertices.push(angle.cos() * rx);
            vertices.push(angle.sin() * ry);
        }
    }

    // Triangles from center to first ring
    for i in 0..segments {
        let next = (i + 1) % segments;
        triangles.push(0);  // center
        triangles.push(1 + i);
        triangles.push(1 + next);
    }

    // Triangles between rings
    for r in 0..(rings - 1) {
        let ring_start = 1 + r * segments;
        let next_ring_start = 1 + (r + 1) * segments;

        for i in 0..segments {
            let next = (i + 1) % segments;

            triangles.push(ring_start + i);
            triangles.push(next_ring_start + i);
            triangles.push(ring_start + next);

            triangles.push(ring_start + next);
            triangles.push(next_ring_start + i);
            triangles.push(next_ring_start + next);
        }
    }

    Mesh { vertices, triangles }
}

/// Create a star-shaped mesh
pub fn create_star_mesh(
    outer_radius: f32,
    inner_radius: f32,
    points: u32,
    rings: u32,
) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    // Center vertex
    vertices.push(0.0);
    vertices.push(0.0);

    // Create rings with alternating star points
    let total_points = points * 2;  // points + valleys

    for r in 1..=rings {
        let t = r as f32 / rings as f32;

        for i in 0..total_points {
            let angle = (i as f32 / total_points as f32) * PI * 2.0;
            // Alternate between outer and inner radius
            let radius = if i % 2 == 0 {
                outer_radius * t
            } else {
                inner_radius * t
            };
            vertices.push(angle.cos() * radius);
            vertices.push(angle.sin() * radius);
        }
    }

    // Triangles from center to first ring
    for i in 0..total_points {
        let next = (i + 1) % total_points;
        triangles.push(0);
        triangles.push(1 + i);
        triangles.push(1 + next);
    }

    // Triangles between rings
    for r in 0..(rings - 1) {
        let ring_start = 1 + r * total_points;
        let next_ring_start = 1 + (r + 1) * total_points;

        for i in 0..total_points {
            let next = (i + 1) % total_points;

            triangles.push(ring_start + i);
            triangles.push(next_ring_start + i);
            triangles.push(ring_start + next);

            triangles.push(ring_start + next);
            triangles.push(next_ring_start + i);
            triangles.push(next_ring_start + next);
        }
    }

    Mesh { vertices, triangles }
}

/// Create a blob mesh with randomized vertex positions
pub fn create_blob_mesh(
    base_radius: f32,
    variation: f32,
    segments: u32,
    rings: u32,
    seed: u32,
) -> Mesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    // Simple deterministic "random" based on seed
    let pseudo_random = |i: u32, j: u32| -> f32 {
        let x = ((i.wrapping_mul(1103515245).wrapping_add(j.wrapping_mul(12345)).wrapping_add(seed)) % 1000) as f32 / 1000.0;
        x * 2.0 - 1.0  // -1 to 1
    };

    // Center vertex
    vertices.push(0.0);
    vertices.push(0.0);

    // Create rings with randomized radii
    for r in 1..=rings {
        let base_t = r as f32 / rings as f32;

        for i in 0..segments {
            let angle = (i as f32 / segments as f32) * PI * 2.0;
            let random_factor = 1.0 + pseudo_random(r, i) * variation;
            let radius = base_radius * base_t * random_factor;
            vertices.push(angle.cos() * radius);
            vertices.push(angle.sin() * radius);
        }
    }

    // Triangles from center to first ring
    for i in 0..segments {
        let next = (i + 1) % segments;
        triangles.push(0);
        triangles.push(1 + i);
        triangles.push(1 + next);
    }

    // Triangles between rings
    for r in 0..(rings - 1) {
        let ring_start = 1 + r * segments;
        let next_ring_start = 1 + (r + 1) * segments;

        for i in 0..segments {
            let next = (i + 1) % segments;

            triangles.push(ring_start + i);
            triangles.push(next_ring_start + i);
            triangles.push(ring_start + next);

            triangles.push(ring_start + next);
            triangles.push(next_ring_start + i);
            triangles.push(next_ring_start + next);
        }
    }

    Mesh { vertices, triangles }
}

/// Create wireframe for any radial mesh (ellipse, star, blob)
pub fn create_radial_wireframe(segments: u32, rings: u32) -> Vec<u32> {
    let mut line_indices = Vec::new();

    // Lines from center to first ring
    for i in 0..segments {
        line_indices.push(0);
        line_indices.push(1 + i);
    }

    // Lines within and between rings
    for r in 0..rings {
        let ring_start = 1 + r * segments;

        for i in 0..segments {
            let next = (i + 1) % segments;

            // Circumferential line
            line_indices.push(ring_start + i);
            line_indices.push(ring_start + next);

            // Radial line to next ring (if not last ring)
            if r < rings - 1 {
                let next_ring_start = 1 + (r + 1) * segments;
                line_indices.push(ring_start + i);
                line_indices.push(next_ring_start + i);
            }
        }
    }

    line_indices
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

    #[test]
    fn test_ring_wireframe() {
        let wireframe = create_ring_wireframe(8, 2);
        // Each segment has: inner edge, radial edge, diagonal edge
        // Plus outer edges on last ring
        // 8 segments * 2 radial divisions * 3 edges + 8 outer edges = 56 edges
        // Each edge is 2 indices
        assert!(wireframe.len() > 0);
        assert_eq!(wireframe.len() % 2, 0); // Must be pairs
    }

    #[test]
    fn test_offset_vertices() {
        let mut vertices = vec![0.0, 0.0, 1.0, 1.0];
        offset_vertices(&mut vertices, 2.0, 3.0);
        assert_eq!(vertices, vec![2.0, 3.0, 3.0, 4.0]);
    }
}
