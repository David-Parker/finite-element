//! SoftBody - manages a deformable mesh with FEM physics

use crate::math::*;
use crate::fem::*;

/// Material properties
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub young_modulus: f32,   // Stiffness (Pa)
    pub poisson_ratio: f32,   // Incompressibility (0-0.5)
    pub density: f32,         // Mass per unit area (kg/m²)
    pub damping: f32,         // Velocity damping factor
}

impl Material {
    pub const RUBBER: Material = Material {
        young_modulus: 5e5,
        poisson_ratio: 0.45,
        density: 1100.0,
        damping: 0.002,
    };

    pub const WOOD: Material = Material {
        young_modulus: 1e6,
        poisson_ratio: 0.3,
        density: 600.0,
        damping: 0.01,
    };

    pub const METAL: Material = Material {
        young_modulus: 2e7,
        poisson_ratio: 0.3,
        density: 7800.0,
        damping: 0.001,
    };

    pub const JELLO: Material = Material {
        young_modulus: 2e5,
        poisson_ratio: 0.45,
        density: 1000.0,
        damping: 0.003,
    };
}

/// SoftBody manages a deformable mesh with FEM physics
pub struct SoftBody {
    // Vertex data
    pub pos: Vec<f32>,
    pub vel: Vec<f32>,
    pub force: Vec<f32>,
    pub mass: Vec<f32>,

    // Triangle connectivity
    pub triangles: Vec<u32>,

    // Material
    pub material: Material,
    pub mu: f32,
    pub lambda: f32,

    // Per-triangle precomputed data
    pub tri_data: Vec<TriangleData>,

    // Counts
    pub num_verts: usize,
    pub num_tris: usize,
}

impl SoftBody {
    /// Create a new soft body from mesh data
    pub fn new(vertices: &[f32], triangles: &[u32], material: Material) -> Self {
        let num_verts = vertices.len() / 2;
        let num_tris = triangles.len() / 3;

        let lame = compute_lame_parameters(material.young_modulus, material.poisson_ratio);

        let mut body = SoftBody {
            pos: vertices.to_vec(),
            vel: vec![0.0; vertices.len()],
            force: vec![0.0; vertices.len()],
            mass: vec![0.0; num_verts],
            triangles: triangles.to_vec(),
            material,
            mu: lame.mu,
            lambda: lame.lambda,
            tri_data: Vec::with_capacity(num_tris),
            num_verts,
            num_tris,
        };

        body.initialize_rest_state();
        body
    }

    fn initialize_rest_state(&mut self) {
        for t in 0..self.num_tris {
            let i0 = self.triangles[t * 3] as usize;
            let i1 = self.triangles[t * 3 + 1] as usize;
            let i2 = self.triangles[t * 3 + 2] as usize;

            let x0 = self.pos[i0 * 2];
            let y0 = self.pos[i0 * 2 + 1];
            let x1 = self.pos[i1 * 2];
            let y1 = self.pos[i1 * 2 + 1];
            let x2 = self.pos[i2 * 2];
            let y2 = self.pos[i2 * 2 + 1];

            let dm = compute_rest_shape_matrix(x0, y0, x1, y1, x2, y2);
            let dm_inv = mat2_inv(&dm);
            let area = compute_triangle_area(x0, y0, x1, y1, x2, y2);

            self.tri_data.push(TriangleData {
                rest_dm_inv: dm_inv,
                rest_area: area,
            });

            // Distribute triangle mass to vertices (1/3 each)
            let tri_mass = area * self.material.density;
            self.mass[i0] += tri_mass / 3.0;
            self.mass[i1] += tri_mass / 3.0;
            self.mass[i2] += tri_mass / 3.0;
        }
    }

    /// Compute all elastic forces on vertices
    pub fn compute_all_forces(&mut self) -> ForceStats {
        self.force.fill(0.0);

        let mut total_energy = 0.0;
        let mut min_j = f32::INFINITY;
        let mut max_j = f32::NEG_INFINITY;

        for t in 0..self.num_tris {
            let i0 = self.triangles[t * 3] as usize;
            let i1 = self.triangles[t * 3 + 1] as usize;
            let i2 = self.triangles[t * 3 + 2] as usize;

            let x0 = self.pos[i0 * 2];
            let y0 = self.pos[i0 * 2 + 1];
            let x1 = self.pos[i1 * 2];
            let y1 = self.pos[i1 * 2 + 1];
            let x2 = self.pos[i2 * 2];
            let y2 = self.pos[i2 * 2 + 1];

            let result = compute_triangle_forces(
                &self.tri_data[t],
                x0, y0, x1, y1, x2, y2,
                self.mu, self.lambda,
            );

            // Accumulate forces
            self.force[i0 * 2] += result.f0[0];
            self.force[i0 * 2 + 1] += result.f0[1];
            self.force[i1 * 2] += result.f1[0];
            self.force[i1 * 2 + 1] += result.f1[1];
            self.force[i2 * 2] += result.f2[0];
            self.force[i2 * 2 + 1] += result.f2[1];

            total_energy += result.energy * self.tri_data[t].rest_area;
            min_j = min_j.min(result.j);
            max_j = max_j.max(result.j);
        }

        ForceStats { total_energy, min_j, max_j }
    }

    /// Add gravity force to all vertices
    pub fn apply_gravity(&mut self, gravity: f32) {
        for i in 0..self.num_verts {
            self.force[i * 2 + 1] += self.mass[i] * gravity;
        }
    }

    /// Apply velocity damping
    pub fn apply_damping(&mut self) {
        let damping = 1.0 - self.material.damping;
        for i in 0..self.num_verts {
            self.vel[i * 2] *= damping;
            self.vel[i * 2 + 1] *= damping;
        }
    }

    /// Integrate velocities and positions using semi-implicit Euler
    pub fn integrate(&mut self, dt: f32) {
        for i in 0..self.num_verts {
            let m = self.mass[i];
            if m < 1e-10 {
                continue;
            }

            let ax = self.force[i * 2] / m;
            let ay = self.force[i * 2 + 1] / m;

            self.vel[i * 2] += ax * dt;
            self.vel[i * 2 + 1] += ay * dt;

            self.pos[i * 2] += self.vel[i * 2] * dt;
            self.pos[i * 2 + 1] += self.vel[i * 2 + 1] * dt;
        }
    }

    /// Handle collision with ground plane
    pub fn collide_with_ground(&mut self, ground_y: f32) {
        for i in 0..self.num_verts {
            if self.pos[i * 2 + 1] < ground_y {
                self.pos[i * 2 + 1] = ground_y;
                if self.vel[i * 2 + 1] < 0.0 {
                    self.vel[i * 2 + 1] = 0.0;
                }
                self.vel[i * 2] *= 0.98;
            }
        }
    }

    /// Perform one physics substep (no damping - call apply_damping once per frame)
    pub fn substep(&mut self, dt: f32, gravity: f32) -> ForceStats {
        let stats = self.compute_all_forces();
        self.apply_gravity(gravity);
        self.integrate(dt);
        stats
    }

    /// Get total kinetic energy
    pub fn get_kinetic_energy(&self) -> f32 {
        let mut ke = 0.0;
        for i in 0..self.num_verts {
            let vx = self.vel[i * 2];
            let vy = self.vel[i * 2 + 1];
            ke += 0.5 * self.mass[i] * (vx * vx + vy * vy);
        }
        ke
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::create_square_mesh;

    #[test]
    fn test_softbody_creation() {
        let mesh = create_square_mesh(1.0, 2);
        let body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::JELLO);

        assert_eq!(body.num_verts, 9);
        assert_eq!(body.num_tris, 8);
    }

    #[test]
    fn test_no_force_at_rest() {
        let mesh = create_square_mesh(1.0, 2);
        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::JELLO);
        body.compute_all_forces();

        for f in &body.force {
            assert!(f.abs() < 1e-6, "Force should be ~0 at rest");
        }
    }
}
