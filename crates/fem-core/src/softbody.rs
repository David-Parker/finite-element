//! SoftBody - manages a deformable mesh with FEM physics

use crate::math::*;
use crate::fem::*;

/// Interpolate between two matrices: (1-t)*a + t*b
fn mat2_lerp(a: &Mat2, b: &Mat2, t: f32) -> Mat2 {
    [
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
        a[3] * (1.0 - t) + b[3] * t,
    ]
}

/// Material properties
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub young_modulus: f32,   // Stiffness (Pa)
    pub poisson_ratio: f32,   // Incompressibility (0-0.5)
    pub density: f32,         // Mass per unit area (kg/m²)
    pub damping: f32,         // Velocity damping factor
    pub yield_stress: f32,    // Stress threshold for plastic deformation (0 = no plasticity)
    pub plasticity: f32,      // How much deformation becomes permanent (0-1)
}

impl Material {
    // Softest - jiggly, bouncy
    pub const JELLO: Material = Material {
        young_modulus: 2e5,       // Softest
        poisson_ratio: 0.45,      // Nearly incompressible
        density: 1000.0,
        damping: 300.0,           // Lower damping for more jiggle
        yield_stress: 0.0,
        plasticity: 0.0,
    };

    // Soft, elastic, bouncy
    pub const RUBBER: Material = Material {
        young_modulus: 6e5,       // 3x stiffer than jello
        poisson_ratio: 0.45,
        density: 1100.0,
        damping: 500.0,
        yield_stress: 0.0,
        plasticity: 0.0,
    };

    // Medium stiffness, less bounce
    pub const WOOD: Material = Material {
        young_modulus: 1.5e6,     // 2.5x stiffer than rubber
        poisson_ratio: 0.3,
        density: 600.0,
        damping: 1000.0,
        yield_stress: 0.0,
        plasticity: 0.0,
    };

    // Stiff, minimal deformation, can yield permanently
    pub const METAL: Material = Material {
        young_modulus: 5e6,       // 3.3x stiffer than wood
        poisson_ratio: 0.3,
        density: 2000.0,
        damping: 4000.0,
        yield_stress: 3e6,
        plasticity: 0.3,
    };

    // Stiff rubber - bouncy, springy, no permanent deformation
    pub const BOUNCY_RUBBER: Material = Material {
        young_modulus: 2e6,       // Stiff like wood but elastic
        poisson_ratio: 0.48,      // Nearly incompressible (rubber-like)
        density: 1100.0,          // Rubber density
        damping: 150.0,           // Very low damping for max bounce
        yield_stress: 0.0,        // No plastic deformation
        plasticity: 0.0,          // Purely elastic
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
                plastic_def: mat2_identity(),
            });

            // Distribute triangle mass to vertices (1/3 each)
            let tri_mass = area * self.material.density;
            self.mass[i0] += tri_mass / 3.0;
            self.mass[i1] += tri_mass / 3.0;
            self.mass[i2] += tri_mass / 3.0;
        }
    }

    /// Compute timestep-adaptive stiffness multiplier
    /// For explicit integration, stiffness must be reduced when dt is large relative to
    /// the material's natural frequency to maintain stability
    /// Returns (mu_effective, lambda_effective)
    pub fn compute_adaptive_stiffness(&self, dt: f32) -> (f32, f32) {
        // Estimate mesh spacing
        let avg_area = self.tri_data.iter().map(|t| t.rest_area).sum::<f32>() / self.num_tris as f32;
        let h = (avg_area * 2.0).sqrt();

        // Wave speed with full material stiffness
        let c = (self.material.young_modulus / self.material.density).sqrt();

        // CFL number: dimensionless ratio of timestep to stable timestep
        // CFL = c * dt / h; stable when CFL < 1
        let cfl = c * dt / h;

        // CRITICAL: Use extremely conservative threshold for stability at large timesteps
        // The nonlinear feedback in FEM during collision requires much smaller effective CFL
        // Start scaling when CFL > 0.05 and scale aggressively
        let safe_cfl = 0.05;
        let scale = if cfl > safe_cfl {
            // Scale factor = (safe_cfl/CFL)² because stiffness ~ c² ~ E
            // Additional factor of 0.5 for extra safety margin during collisions
            (safe_cfl / cfl).powi(2) * 0.5
        } else {
            1.0
        };

        (self.mu * scale, self.lambda * scale)
    }

    /// Compute all elastic forces on vertices with timestep-adaptive stiffness
    pub fn compute_all_forces_with_dt(&mut self, dt: f32) -> ForceStats {
        self.force.fill(0.0);

        let (mu_eff, lambda_eff) = self.compute_adaptive_stiffness(dt);

        let mut total_energy = 0.0;
        let mut min_j = f32::INFINITY;
        let mut max_j = f32::NEG_INFINITY;

        let yield_stress = self.material.yield_stress;
        let plasticity = self.material.plasticity;

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

            // Check for plastic yielding with original stiffness
            if yield_stress > 0.0 {
                let (stress_mag, f_total) = compute_stress_magnitude(
                    &self.tri_data[t],
                    x0, y0, x1, y1, x2, y2,
                    self.mu, self.lambda,  // Use full stiffness for yield check
                );

                let j_total = mat2_det(&f_total);
                let current_fp_det = mat2_det(&self.tri_data[t].plastic_def);

                if stress_mag > yield_stress
                    && j_total > 0.7
                    && j_total < 1.5
                    && current_fp_det > 0.8
                    && current_fp_det < 1.2
                {
                    let old_fp = self.tri_data[t].plastic_def;
                    let yield_ratio = ((stress_mag - yield_stress) / stress_mag).min(0.5);
                    let blend = plasticity * yield_ratio * 0.02;
                    let new_fp = mat2_lerp(&old_fp, &f_total, blend.min(0.05));

                    let fp_det = mat2_det(&new_fp);
                    if fp_det > 0.85 && fp_det < 1.15 {
                        self.tri_data[t].plastic_def = new_fp;
                    }
                }
            }

            // Use adaptive stiffness for force computation
            let result = compute_triangle_forces(
                &self.tri_data[t],
                x0, y0, x1, y1, x2, y2,
                mu_eff, lambda_eff,
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

    /// Compute all elastic forces on vertices (legacy, uses default stiffness)
    pub fn compute_all_forces(&mut self) -> ForceStats {
        self.force.fill(0.0);

        let mut total_energy = 0.0;
        let mut min_j = f32::INFINITY;
        let mut max_j = f32::NEG_INFINITY;

        let yield_stress = self.material.yield_stress;
        let plasticity = self.material.plasticity;

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

            // Check for plastic yielding
            if yield_stress > 0.0 {
                let (stress_mag, f_total) = compute_stress_magnitude(
                    &self.tri_data[t],
                    x0, y0, x1, y1, x2, y2,
                    self.mu, self.lambda,
                );

                // Only allow plastic deformation if triangle is in good shape
                let j_total = mat2_det(&f_total);
                let current_fp_det = mat2_det(&self.tri_data[t].plastic_def);

                // Very conservative: only yield when triangle is healthy (J between 0.7 and 1.5)
                // and plastic deformation is still close to identity
                if stress_mag > yield_stress
                    && j_total > 0.7
                    && j_total < 1.5
                    && current_fp_det > 0.8
                    && current_fp_det < 1.2
                {
                    // Material has yielded - update plastic deformation very slowly
                    let old_fp = self.tri_data[t].plastic_def;
                    let yield_ratio = ((stress_mag - yield_stress) / stress_mag).min(0.5);
                    let blend = plasticity * yield_ratio * 0.02; // Very slow plastic flow
                    let new_fp = mat2_lerp(&old_fp, &f_total, blend.min(0.05));

                    // Even tighter limits on plastic deformation
                    let fp_det = mat2_det(&new_fp);
                    if fp_det > 0.85 && fp_det < 1.15 {
                        self.tri_data[t].plastic_def = new_fp;
                    }
                }
            }

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

    /// Apply internal damping - damps relative motion between vertices
    /// This doesn't affect center-of-mass velocity, only internal oscillations
    pub fn apply_internal_damping(&mut self) {
        let damping = self.material.damping;

        // For each triangle, damp relative velocities along edges
        for tri in 0..self.num_tris {
            let i0 = self.triangles[tri * 3] as usize;
            let i1 = self.triangles[tri * 3 + 1] as usize;
            let i2 = self.triangles[tri * 3 + 2] as usize;

            // Edge 0-1
            Self::damp_edge(&mut self.force, &self.vel, &self.pos, i0, i1, damping);
            // Edge 1-2
            Self::damp_edge(&mut self.force, &self.vel, &self.pos, i1, i2, damping);
            // Edge 2-0
            Self::damp_edge(&mut self.force, &self.vel, &self.pos, i2, i0, damping);
        }
    }

    /// Damp relative velocity along an edge
    fn damp_edge(force: &mut [f32], vel: &[f32], pos: &[f32], i0: usize, i1: usize, damping: f32) {
        // Edge direction
        let dx = pos[i1 * 2] - pos[i0 * 2];
        let dy = pos[i1 * 2 + 1] - pos[i0 * 2 + 1];
        let len_sq = dx * dx + dy * dy;
        if len_sq < 1e-10 {
            return;
        }
        let len = len_sq.sqrt();
        let nx = dx / len;
        let ny = dy / len;

        // Relative velocity along edge
        let dvx = vel[i1 * 2] - vel[i0 * 2];
        let dvy = vel[i1 * 2 + 1] - vel[i0 * 2 + 1];
        let rel_vel = dvx * nx + dvy * ny;

        // Damping force (equal and opposite)
        let fx = damping * rel_vel * nx;
        let fy = damping * rel_vel * ny;

        force[i0 * 2] += fx;
        force[i0 * 2 + 1] += fy;
        force[i1 * 2] -= fx;
        force[i1 * 2 + 1] -= fy;
    }

    /// Integrate velocities and positions using semi-implicit Euler
    /// Includes safety limits to prevent NaN/inf propagation
    pub fn integrate(&mut self, dt: f32) {
        // Safety limits - these catch extreme cases but shouldn't activate in normal simulation
        const MAX_VELOCITY: f32 = 100.0;  // 100 m/s is very fast for soft bodies
        const MAX_POSITION: f32 = 1000.0; // 1km bounds

        for i in 0..self.num_verts {
            let m = self.mass[i];
            if m < 1e-10 {
                continue;
            }

            let ax = self.force[i * 2] / m;
            let ay = self.force[i * 2 + 1] / m;

            // Skip NaN/inf forces
            if !ax.is_finite() || !ay.is_finite() {
                continue;
            }

            // Semi-implicit Euler: update velocity first, then position
            self.vel[i * 2] += ax * dt;
            self.vel[i * 2 + 1] += ay * dt;

            // Clamp velocity to prevent runaway
            let speed_sq = self.vel[i * 2].powi(2) + self.vel[i * 2 + 1].powi(2);
            if speed_sq > MAX_VELOCITY * MAX_VELOCITY {
                let scale = MAX_VELOCITY / speed_sq.sqrt();
                self.vel[i * 2] *= scale;
                self.vel[i * 2 + 1] *= scale;
            }

            self.pos[i * 2] += self.vel[i * 2] * dt;
            self.pos[i * 2 + 1] += self.vel[i * 2 + 1] * dt;

            // Clamp position to bounds
            self.pos[i * 2] = self.pos[i * 2].clamp(-MAX_POSITION, MAX_POSITION);
            self.pos[i * 2 + 1] = self.pos[i * 2 + 1].clamp(-MAX_POSITION, MAX_POSITION);
        }
    }

    /// Handle collision with ground plane using soft constraint
    /// Uses gradual position correction to avoid mesh distortion
    pub fn collide_with_ground(&mut self, ground_y: f32) {
        const FRICTION: f32 = 0.9;
        const RESTITUTION: f32 = 0.3;  // Bounce coefficient
        // Gradual correction: only fix a fraction of penetration per substep
        // This allows the mesh to deform naturally during impact
        const MAX_CORRECTION_PER_SUBSTEP: f32 = 0.05;  // Max 5cm per substep

        for i in 0..self.num_verts {
            let y = self.pos[i * 2 + 1];
            let vy = self.vel[i * 2 + 1];

            if y < ground_y {
                let penetration = ground_y - y;

                // Gradual position correction (limit max correction per substep)
                let correction = penetration.min(MAX_CORRECTION_PER_SUBSTEP);
                self.pos[i * 2 + 1] += correction;

                // Only modify velocity when vertex is moving into ground
                if vy < 0.0 {
                    // Bounce with restitution, but clamp to prevent excessive bounce
                    let bounce_vel = (-vy * RESTITUTION).min(5.0);
                    self.vel[i * 2 + 1] = bounce_vel;
                }

                // Ground friction
                self.vel[i * 2] *= FRICTION;
            }
        }
    }

    /// Perform one physics substep with timestep-adaptive stiffness
    /// Returns (ForceStats, strain_corrections)
    pub fn substep(&mut self, dt: f32, gravity: f32) -> (ForceStats, u32) {
        self.substep_with_ground(dt, gravity, None)
    }

    /// Perform one physics substep including optional ground collision
    /// Uses velocity-based collision (speculative contacts) for mesh-friendly collision
    pub fn substep_with_ground(&mut self, dt: f32, gravity: f32, ground_y: Option<f32>) -> (ForceStats, u32) {
        // Pre-correction: ensure mesh is valid before computing forces
        self.limit_strain();

        // Sanitize any NaN values
        self.sanitize_state();

        // Use timestep-adaptive stiffness for stability at large dt
        let stats = self.compute_all_forces_with_dt(dt);
        self.apply_gravity(gravity);
        self.apply_internal_damping();

        // SPECULATIVE CONTACT: Adjust velocities BEFORE integration to prevent penetration
        // This is much more mesh-friendly than position correction after penetration
        if let Some(gy) = ground_y {
            self.apply_speculative_ground_collision(dt, gy);
        }

        self.integrate(dt);

        // Post-correction: ensure mesh stays valid
        let mut total_corrections = 0u32;
        for _ in 0..3 {
            total_corrections += self.limit_strain();
        }
        (stats, total_corrections)
    }

    /// Speculative ground collision with strong collision damping
    /// Adjusts velocity before integration to prevent penetration
    /// Applies strong damping during collision to prevent energy buildup
    fn apply_speculative_ground_collision(&mut self, dt: f32, ground_y: f32) {
        const RESTITUTION: f32 = 0.3;
        const FRICTION: f32 = 0.85;
        const SKIN_DEPTH: f32 = 0.03;  // Buffer above ground
        const COLLISION_DAMPING: f32 = 0.95;  // Moderate damping during collision

        // First pass: check if any vertex is in collision zone
        let collision_zone = ground_y + 1.0;  // Within 1m of ground
        let mut in_collision = false;
        for i in 0..self.num_verts {
            if self.pos[i * 2 + 1] < collision_zone {
                in_collision = true;
                break;
            }
        }

        // Apply strong damping to ALL vertices during collision
        // This prevents internal oscillations from building up
        if in_collision {
            for i in 0..self.num_verts {
                self.vel[i * 2] *= COLLISION_DAMPING;
                self.vel[i * 2 + 1] *= COLLISION_DAMPING;
            }
        }

        // Second pass: handle individual vertex collisions
        for i in 0..self.num_verts {
            let y = self.pos[i * 2 + 1];
            let vy = self.vel[i * 2 + 1];

            // Predict where vertex will be after integration
            let predicted_y = y + vy * dt;
            let target_y = ground_y + SKIN_DEPTH;

            if predicted_y < target_y {
                // Calculate velocity that prevents penetration
                let required_vy = (target_y - y) / dt;

                if vy < required_vy {
                    let approach_speed = -vy.min(0.0);
                    let bounce = approach_speed * RESTITUTION;
                    self.vel[i * 2 + 1] = required_vy.max(bounce);
                    self.vel[i * 2] *= FRICTION;
                }
            }

            // Handle vertices currently below ground
            if y < ground_y {
                let push_strength = (ground_y - y).min(0.02);
                self.pos[i * 2 + 1] += push_strength;

                if self.vel[i * 2 + 1] < 0.0 {
                    self.vel[i * 2 + 1] = self.vel[i * 2 + 1].abs() * RESTITUTION;
                }
            }
        }
    }

    /// Clean up any NaN or inf values that may have crept in
    fn sanitize_state(&mut self) {
        for i in 0..self.pos.len() {
            if !self.pos[i].is_finite() {
                // Reset to some reasonable default - this shouldn't happen
                // but prevents NaN propagation
                self.pos[i] = 0.0;
                self.vel[i] = 0.0;
            }
            if !self.vel[i].is_finite() {
                self.vel[i] = 0.0;
            }
        }
    }

    /// Limit strain to prevent triangle inversion
    /// Uses position-based correction for stability
    /// Returns count of triangles that were corrected
    pub fn limit_strain(&mut self) -> u32 {
        // Tighter bounds for stability with large timesteps
        const MIN_J: f32 = 0.7;  // Minimum allowed volume ratio (prevent inversion)
        const MAX_J: f32 = 1.15; // Maximum allowed volume ratio (prevent pancaking)
        let mut corrections = 0u32;

        for t in 0..self.num_tris {
            let i0 = self.triangles[t * 3] as usize;
            let i1 = self.triangles[t * 3 + 1] as usize;
            let i2 = self.triangles[t * 3 + 2] as usize;

            // Current positions
            let x0 = self.pos[i0 * 2];
            let y0 = self.pos[i0 * 2 + 1];
            let x1 = self.pos[i1 * 2];
            let y1 = self.pos[i1 * 2 + 1];
            let x2 = self.pos[i2 * 2];
            let y2 = self.pos[i2 * 2 + 1];

            // Compute current signed area (proportional to J)
            let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
            let rest_area = self.tri_data[t].rest_area;
            let j = area / rest_area;

            // If triangle is too compressed or inverted, push vertices apart
            if j < MIN_J {
                corrections += 1;
                let center_x = (x0 + x1 + x2) / 3.0;
                let center_y = (y0 + y1 + y2) / 3.0;

                // Scale factor to restore minimum area
                let scale = (MIN_J / j.max(0.01)).sqrt().min(1.5);

                // Push vertices away from center
                self.pos[i0 * 2] = center_x + (x0 - center_x) * scale;
                self.pos[i0 * 2 + 1] = center_y + (y0 - center_y) * scale;
                self.pos[i1 * 2] = center_x + (x1 - center_x) * scale;
                self.pos[i1 * 2 + 1] = center_y + (y1 - center_y) * scale;
                self.pos[i2 * 2] = center_x + (x2 - center_x) * scale;
                self.pos[i2 * 2 + 1] = center_y + (y2 - center_y) * scale;

                // Damp velocities to prevent oscillation (gentle to avoid killing freefall)
                self.vel[i0 * 2] *= 0.98;
                self.vel[i0 * 2 + 1] *= 0.98;
                self.vel[i1 * 2] *= 0.98;
                self.vel[i1 * 2 + 1] *= 0.98;
                self.vel[i2 * 2] *= 0.98;
                self.vel[i2 * 2 + 1] *= 0.98;
            }
            // If triangle is too stretched, pull vertices together
            else if j > MAX_J {
                corrections += 1;
                let center_x = (x0 + x1 + x2) / 3.0;
                let center_y = (y0 + y1 + y2) / 3.0;

                let scale = (MAX_J / j).sqrt().max(0.8);

                self.pos[i0 * 2] = center_x + (x0 - center_x) * scale;
                self.pos[i0 * 2 + 1] = center_y + (y0 - center_y) * scale;
                self.pos[i1 * 2] = center_x + (x1 - center_x) * scale;
                self.pos[i1 * 2 + 1] = center_y + (y1 - center_y) * scale;
                self.pos[i2 * 2] = center_x + (x2 - center_x) * scale;
                self.pos[i2 * 2 + 1] = center_y + (y2 - center_y) * scale;

                // Also damp outward velocity to help settle
                self.vel[i0 * 2] *= 0.95;
                self.vel[i0 * 2 + 1] *= 0.95;
                self.vel[i1 * 2] *= 0.95;
                self.vel[i1 * 2 + 1] *= 0.95;
                self.vel[i2 * 2] *= 0.95;
                self.vel[i2 * 2 + 1] *= 0.95;
            }
        }
        corrections
    }

    /// Get lowest Y position of any vertex
    pub fn get_lowest_y(&self) -> f32 {
        let mut lowest = f32::INFINITY;
        for i in 0..self.num_verts {
            lowest = lowest.min(self.pos[i * 2 + 1]);
        }
        lowest
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

    /// Put object to sleep if kinetic energy is below threshold
    /// Returns true if object was put to sleep
    pub fn sleep_if_resting(&mut self, ke_threshold: f32) -> bool {
        let ke = self.get_kinetic_energy();
        if ke < ke_threshold {
            // Zero all velocities
            self.vel.fill(0.0);
            true
        } else {
            false
        }
    }

    /// Get maximum velocity magnitude
    pub fn get_max_velocity(&self) -> f32 {
        let mut max_vel_sq: f32 = 0.0;
        for i in 0..self.num_verts {
            let vx = self.vel[i * 2];
            let vy = self.vel[i * 2 + 1];
            max_vel_sq = max_vel_sq.max(vx * vx + vy * vy);
        }
        max_vel_sq.sqrt()
    }

    /// Get axis-aligned bounding box (min_x, min_y, max_x, max_y)
    pub fn get_aabb(&self) -> (f32, f32, f32, f32) {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for i in 0..self.num_verts {
            let x = self.pos[i * 2];
            let y = self.pos[i * 2 + 1];
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        (min_x, min_y, max_x, max_y)
    }

    /// Check if two AABBs overlap (with margin)
    fn aabb_overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32), margin: f32) -> bool {
        let (a_min_x, a_min_y, a_max_x, a_max_y) = a;
        let (b_min_x, b_min_y, b_max_x, b_max_y) = b;

        a_max_x + margin >= b_min_x && b_max_x + margin >= a_min_x &&
        a_max_y + margin >= b_min_y && b_max_y + margin >= a_min_y
    }

    /// Collide with another soft body using position-based separation with damping
    /// Returns the number of collision responses applied
    pub fn collide_with_body(&mut self, other: &mut SoftBody, min_dist: f32) -> u32 {
        // Broad phase: AABB check
        let self_aabb = self.get_aabb();
        let other_aabb = other.get_aabb();
        if !Self::aabb_overlap(self_aabb, other_aabb, min_dist) {
            return 0;  // Bodies are far apart, skip expensive check
        }

        let mut collisions = 0u32;

        // Safety limits to prevent explosions
        const MAX_POS_CORRECTION: f32 = 0.03;  // Max position change per collision (reduced)
        const MAX_VEL_CORRECTION: f32 = 0.5;   // Max velocity change per collision (reduced)
        const COLLISION_DAMPING: f32 = 0.90;   // Strong damping during collision

        // Accumulate corrections per vertex to avoid compounding
        let mut self_corr_pos: Vec<(f32, f32)> = vec![(0.0, 0.0); self.num_verts];
        let mut self_corr_vel: Vec<(f32, f32)> = vec![(0.0, 0.0); self.num_verts];
        let mut other_corr_pos: Vec<(f32, f32)> = vec![(0.0, 0.0); other.num_verts];
        let mut other_corr_vel: Vec<(f32, f32)> = vec![(0.0, 0.0); other.num_verts];
        let mut self_counts: Vec<u32> = vec![0; self.num_verts];
        let mut other_counts: Vec<u32> = vec![0; other.num_verts];

        for i in 0..self.num_verts {
            let x1 = self.pos[i * 2];
            let y1 = self.pos[i * 2 + 1];

            for j in 0..other.num_verts {
                let x2 = other.pos[j * 2];
                let y2 = other.pos[j * 2 + 1];

                let dx = x2 - x1;
                let dy = y2 - y1;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < min_dist * min_dist && dist_sq > 1e-10 {
                    collisions += 1;
                    let dist = dist_sq.sqrt();
                    let overlap = min_dist - dist;

                    // Normal from self to other
                    let nx = dx / dist;
                    let ny = dy / dist;

                    // Check if vertices are approaching
                    let vx1 = self.vel[i * 2];
                    let vy1 = self.vel[i * 2 + 1];
                    let vx2 = other.vel[j * 2];
                    let vy2 = other.vel[j * 2 + 1];
                    let rel_vel = (vx2 - vx1) * nx + (vy2 - vy1) * ny;

                    // Position correction (clamped)
                    let pos_correction = (overlap * 0.3).min(MAX_POS_CORRECTION);
                    self_corr_pos[i].0 -= nx * pos_correction;
                    self_corr_pos[i].1 -= ny * pos_correction;
                    other_corr_pos[j].0 += nx * pos_correction;
                    other_corr_pos[j].1 += ny * pos_correction;

                    // Velocity correction (only if approaching, clamped)
                    if rel_vel < 0.0 {
                        let vel_correction = (-rel_vel * 0.3).min(MAX_VEL_CORRECTION);
                        self_corr_vel[i].0 -= nx * vel_correction;
                        self_corr_vel[i].1 -= ny * vel_correction;
                        other_corr_vel[j].0 += nx * vel_correction;
                        other_corr_vel[j].1 += ny * vel_correction;
                    }

                    self_counts[i] += 1;
                    other_counts[j] += 1;
                }
            }
        }

        // If any collision detected, apply damping to both bodies
        if collisions > 0 {
            for i in 0..self.num_verts {
                self.vel[i * 2] *= COLLISION_DAMPING;
                self.vel[i * 2 + 1] *= COLLISION_DAMPING;
            }
            for j in 0..other.num_verts {
                other.vel[j * 2] *= COLLISION_DAMPING;
                other.vel[j * 2 + 1] *= COLLISION_DAMPING;
            }
        }

        // Apply averaged corrections with additional clamping
        for i in 0..self.num_verts {
            if self_counts[i] > 0 {
                let n = self_counts[i] as f32;
                let dx = (self_corr_pos[i].0 / n).clamp(-MAX_POS_CORRECTION, MAX_POS_CORRECTION);
                let dy = (self_corr_pos[i].1 / n).clamp(-MAX_POS_CORRECTION, MAX_POS_CORRECTION);
                self.pos[i * 2] += dx;
                self.pos[i * 2 + 1] += dy;

                let dvx = (self_corr_vel[i].0 / n).clamp(-MAX_VEL_CORRECTION, MAX_VEL_CORRECTION);
                let dvy = (self_corr_vel[i].1 / n).clamp(-MAX_VEL_CORRECTION, MAX_VEL_CORRECTION);
                self.vel[i * 2] += dvx;
                self.vel[i * 2 + 1] += dvy;
            }
        }
        for j in 0..other.num_verts {
            if other_counts[j] > 0 {
                let n = other_counts[j] as f32;
                let dx = (other_corr_pos[j].0 / n).clamp(-MAX_POS_CORRECTION, MAX_POS_CORRECTION);
                let dy = (other_corr_pos[j].1 / n).clamp(-MAX_POS_CORRECTION, MAX_POS_CORRECTION);
                other.pos[j * 2] += dx;
                other.pos[j * 2 + 1] += dy;

                let dvx = (other_corr_vel[j].0 / n).clamp(-MAX_VEL_CORRECTION, MAX_VEL_CORRECTION);
                let dvy = (other_corr_vel[j].1 / n).clamp(-MAX_VEL_CORRECTION, MAX_VEL_CORRECTION);
                other.vel[j * 2] += dvx;
                other.vel[j * 2 + 1] += dvy;
            }
        }

        collisions
    }

    /// Get diagnostic information for tracing
    pub fn get_diagnostics(&self) -> (f32, f32, f32, f32, f32, f32) {
        let mut max_vel: f32 = 0.0;
        let mut max_force: f32 = 0.0;

        for i in 0..self.num_verts {
            let vel_sq = self.vel[i * 2] * self.vel[i * 2] + self.vel[i * 2 + 1] * self.vel[i * 2 + 1];
            max_vel = max_vel.max(vel_sq.sqrt());

            let force_sq = self.force[i * 2] * self.force[i * 2] + self.force[i * 2 + 1] * self.force[i * 2 + 1];
            max_force = max_force.max(force_sq.sqrt());
        }

        let mut min_j: f32 = f32::INFINITY;
        let mut max_j: f32 = f32::NEG_INFINITY;
        let mut min_plastic_det: f32 = f32::INFINITY;
        let mut max_plastic_det: f32 = f32::NEG_INFINITY;

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

            // Compute current deformation gradient
            let ds = compute_deformed_shape_matrix(x0, y0, x1, y1, x2, y2);
            let f = compute_deformation_gradient(&ds, &self.tri_data[t].rest_dm_inv);
            let j = mat2_det(&f);

            min_j = min_j.min(j);
            max_j = max_j.max(j);

            let plastic_det = mat2_det(&self.tri_data[t].plastic_def);
            min_plastic_det = min_plastic_det.min(plastic_det);
            max_plastic_det = max_plastic_det.max(plastic_det);
        }

        (min_j, max_j, max_vel, max_force, min_plastic_det, max_plastic_det)
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

    #[test]
    fn test_apply_gravity() {
        let mesh = create_square_mesh(1.0, 2);
        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::JELLO);
        body.force.fill(0.0);
        body.apply_gravity(-9.8);

        // All y-forces should be negative (downward)
        for i in 0..body.num_verts {
            assert!(body.force[i * 2 + 1] < 0.0, "Gravity should pull down");
        }
    }

    #[test]
    fn test_apply_internal_damping() {
        let mesh = create_square_mesh(1.0, 2);
        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::JELLO);

        // Set different velocities on vertices to create relative motion
        body.vel[0] = 10.0;  // vertex 0 moving right
        body.vel[1] = 0.0;
        body.vel[2] = -10.0; // vertex 1 moving left
        body.vel[3] = 0.0;

        // Clear forces and apply damping
        body.force.fill(0.0);
        body.apply_internal_damping();

        // Damping should create forces opposing relative motion
        // Vertex 0 should get negative x force, vertex 1 should get positive x force
        assert!(body.force[0] < 0.0, "Damping should oppose vertex 0's motion");
        assert!(body.force[2] > 0.0, "Damping should oppose vertex 1's motion");
    }

    #[test]
    fn test_kinetic_energy() {
        let mesh = create_square_mesh(1.0, 2);
        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::JELLO);

        // At rest, KE should be 0
        assert!((body.get_kinetic_energy()).abs() < 1e-6, "KE at rest");

        // Add velocity to first vertex
        body.vel[0] = 1.0;
        body.vel[1] = 0.0;

        let ke = body.get_kinetic_energy();
        let expected = 0.5 * body.mass[0] * 1.0; // 0.5 * m * v^2
        assert!((ke - expected).abs() < 1e-6, "KE with velocity");
    }

    #[test]
    fn test_collide_with_ground() {
        let mesh = create_square_mesh(1.0, 2);
        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::JELLO);

        // Move first vertex slightly below ground
        body.pos[1] = -1.05;  // 5cm below ground
        body.vel[1] = -5.0;

        body.collide_with_ground(-1.0);

        // Soft collision should correct the penetration
        assert!(body.pos[1] >= -1.05, "Position should be corrected toward ground");
        // Velocity should be positive (bounced)
        assert!(body.vel[1] > 0.0, "Downward velocity should be reversed");
    }

    #[test]
    fn test_debug_bouncy_rubber_forces() {
        use crate::mesh::create_ring_mesh;

        println!("\n=== BOUNCY_RUBBER Force Debug ===");
        let mesh = create_ring_mesh(1.5, 1.0, 16, 4);
        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::BOUNCY_RUBBER);

        println!("Material: E={}, damping={}, density={}",
            body.material.young_modulus, body.material.damping, body.material.density);
        println!("Lame: mu={:.0}, lambda={:.0}", body.mu, body.lambda);
        println!("Mesh: {} verts, {} tris", body.num_verts, body.num_tris);

        let dt_8 = 1.0 / 60.0 / 8.0;

        // Show adaptive stiffness
        let (mu_eff, lambda_eff) = body.compute_adaptive_stiffness(dt_8 as f32);
        println!("Adaptive stiffness (dt={:.6}): mu={:.0} -> {:.0}, lambda={:.0} -> {:.0}",
            dt_8, body.mu, mu_eff, body.lambda, lambda_eff);

        // Step by step with 8 substeps
        println!("\nStep-by-step (8 substeps):");
        let mut body2 = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::BOUNCY_RUBBER);
        for i in 0..8 {
            let stats = body2.compute_all_forces_with_dt(dt_8 as f32);
            let max_f: f32 = (0..body2.num_verts)
                .map(|j| (body2.force[j*2].powi(2) + body2.force[j*2+1].powi(2)).sqrt())
                .fold(0.0, f32::max);

            body2.apply_gravity(-9.8);
            body2.apply_internal_damping();
            body2.integrate(dt_8 as f32);
            body2.limit_strain();

            let ke = body2.get_kinetic_energy();
            let max_vel = body2.get_max_velocity();
            println!("  Step {}: J=[{:.3},{:.3}], max_f={:.1}, KE={:.1}, vel={:.2}",
                i, stats.min_j, stats.max_j, max_f, ke, max_vel);

            if ke > 1e6 || ke.is_nan() {
                println!("  !!! EXPLOSION DETECTED !!!");
                break;
            }
        }

        // Compare with 64 substeps
        let dt_64 = 1.0 / 60.0 / 64.0;
        let mut body4 = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::BOUNCY_RUBBER);
        for _ in 0..64 {
            body4.substep(dt_64 as f32, -9.8);
        }
        println!("\n64 substeps reference:");
        println!("  KE={:.2}, max_vel={:.2}", body4.get_kinetic_energy(), body4.get_max_velocity());
    }
}
