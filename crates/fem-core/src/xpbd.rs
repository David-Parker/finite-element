//! XPBD (Extended Position-Based Dynamics) solver for soft body simulation
//!
//! XPBD is unconditionally stable unlike force-based FEM with explicit integration.
//! Key insight: position-based constraints solved with compliance give implicit-like stability.
//!
//! References:
//! - "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" (Macklin et al. 2016)
//! - Ten Minute Physics XPBD tutorial

use std::collections::HashMap;

#[cfg(feature = "simd")]
use crate::compute::simd::SimdBackend;
#[cfg(feature = "simd")]
use crate::compute::ComputeBackend;

/// Spatial hash grid for O(1) neighbor queries in collision detection
pub struct SpatialHash {
    cell_size: f32,
    cells: HashMap<(i32, i32), Vec<(usize, usize)>>,  // (body_idx, vert_idx)
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        SpatialHash {
            cell_size,
            cells: HashMap::new(),
        }
    }

    #[inline]
    fn hash(&self, x: f32, y: f32) -> (i32, i32) {
        let cx = (x / self.cell_size).floor() as i32;
        let cy = (y / self.cell_size).floor() as i32;
        (cx, cy)
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    pub fn insert(&mut self, body_idx: usize, vert_idx: usize, x: f32, y: f32) {
        let key = self.hash(x, y);
        self.cells.entry(key).or_insert_with(Vec::new).push((body_idx, vert_idx));
    }

    /// Get all entries in cell and neighboring cells
    pub fn query_neighbors(&self, x: f32, y: f32) -> impl Iterator<Item = &(usize, usize)> {
        let (cx, cy) = self.hash(x, y);
        // Check 3x3 neighborhood
        (-1..=1).flat_map(move |dx| {
            (-1..=1).flat_map(move |dy| {
                self.cells.get(&(cx + dx, cy + dy))
                    .map(|v| v.iter())
                    .into_iter()
                    .flatten()
            })
        })
    }
}

/// Cached edge data for collision detection
#[derive(Clone)]
struct CachedEdge {
    body_idx: usize,
    v0: usize,
    v1: usize,
    mid_x: f32,
    mid_y: f32,
    dx: f32,      // e1.x - e0.x
    dy: f32,      // e1.y - e0.y
    len_sq: f32,  // dx*dx + dy*dy
    w0: f32,      // inv_mass of v0
    w1: f32,      // inv_mass of v1
}

/// Collision system for handling multi-body collisions efficiently
pub struct CollisionSystem {
    edge_hash: SpatialHash,
    min_dist: f32,
    aabbs: Vec<(f32, f32, f32, f32)>,
    overlapping_pairs: Vec<(usize, usize)>,
    cached_edges: Vec<CachedEdge>,  // Precomputed edge data per frame
}

impl CollisionSystem {
    pub fn new(min_dist: f32) -> Self {
        CollisionSystem {
            edge_hash: SpatialHash::new(min_dist * 2.0),
            min_dist,
            aabbs: Vec::with_capacity(32),
            overlapping_pairs: Vec::with_capacity(64),
            cached_edges: Vec::with_capacity(256),
        }
    }

    /// Check if two AABBs overlap (with margin for collision distance)
    #[inline]
    fn aabbs_overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32), margin: f32) -> bool {
        a.2 + margin >= b.0 && b.2 + margin >= a.0 &&  // X overlap
        a.3 + margin >= b.1 && b.3 + margin >= a.1     // Y overlap
    }

    /// Detect and resolve collisions between all bodies using spatial hashing
    /// Uses vertex-edge collision with edge spatial hash for efficiency
    pub fn solve_collisions(&mut self, bodies: &mut [XPBDSoftBody]) -> u32 {
        let num_bodies = bodies.len();

        // Step 1: Compute AABBs for all bodies
        self.aabbs.clear();
        for body in bodies.iter() {
            self.aabbs.push(body.get_aabb());
        }

        // Step 2: Find overlapping body pairs (broad phase)
        self.overlapping_pairs.clear();
        for i in 0..num_bodies {
            for j in (i + 1)..num_bodies {
                if Self::aabbs_overlap(self.aabbs[i], self.aabbs[j], self.min_dist) {
                    self.overlapping_pairs.push((i, j));
                }
            }
        }

        if self.overlapping_pairs.is_empty() {
            return 0;
        }

        // Step 3: Cache edge data and build edge spatial hash
        self.cached_edges.clear();
        self.edge_hash.clear();

        let mut body_needs_collision = vec![false; num_bodies];
        for &(i, j) in &self.overlapping_pairs {
            body_needs_collision[i] = true;
            body_needs_collision[j] = true;
        }

        for (body_idx, body) in bodies.iter().enumerate() {
            if !body_needs_collision[body_idx] { continue; }

            for edge in &body.edge_constraints {
                let w0 = body.inv_mass[edge.v0];
                let w1 = body.inv_mass[edge.v1];

                // Skip edges where both endpoints are fixed
                if w0 == 0.0 && w1 == 0.0 { continue; }

                let e0x = body.pos[edge.v0 * 2];
                let e0y = body.pos[edge.v0 * 2 + 1];
                let e1x = body.pos[edge.v1 * 2];
                let e1y = body.pos[edge.v1 * 2 + 1];

                let dx = e1x - e0x;
                let dy = e1y - e0y;
                let len_sq = dx * dx + dy * dy;

                if len_sq < 1e-10 { continue; }

                let mid_x = (e0x + e1x) * 0.5;
                let mid_y = (e0y + e1y) * 0.5;

                let edge_idx = self.cached_edges.len();
                self.cached_edges.push(CachedEdge {
                    body_idx,
                    v0: edge.v0,
                    v1: edge.v1,
                    mid_x,
                    mid_y,
                    dx,
                    dy,
                    len_sq,
                    w0,
                    w1,
                });

                // Insert edge midpoint into spatial hash
                // Use body_idx in first slot, edge_idx in second
                self.edge_hash.insert(body_idx, edge_idx, mid_x, mid_y);
            }
        }

        // Step 4: For each vertex, query nearby edges and check collisions
        let mut total_collisions = 0u32;
        let min_dist = self.min_dist;
        let min_dist_sq = min_dist * min_dist;

        for body_a_idx in 0..num_bodies {
            if !body_needs_collision[body_a_idx] { continue; }

            for vert_idx in 0..bodies[body_a_idx].num_verts {
                let w_vert = bodies[body_a_idx].inv_mass[vert_idx];
                if w_vert == 0.0 { continue; }

                let vx = bodies[body_a_idx].pos[vert_idx * 2];
                let vy = bodies[body_a_idx].pos[vert_idx * 2 + 1];

                // Query nearby edges (returns body_idx, edge_idx pairs)
                let nearby_edges: Vec<(usize, usize)> = self.edge_hash.query_neighbors(vx, vy)
                    .filter(|&&(b, _)| b != body_a_idx)  // Skip same body
                    .cloned()
                    .collect();

                for (body_b_idx, edge_idx) in nearby_edges {
                    let edge = &self.cached_edges[edge_idx];

                    // Use cached edge data
                    let e0x = bodies[body_b_idx].pos[edge.v0 * 2];
                    let e0y = bodies[body_b_idx].pos[edge.v0 * 2 + 1];

                    // Project vertex onto edge line using cached dx/dy/len_sq
                    let t = ((vx - e0x) * edge.dx + (vy - e0y) * edge.dy) / edge.len_sq;
                    let t = t.clamp(0.0, 1.0);

                    let closest_x = e0x + t * edge.dx;
                    let closest_y = e0y + t * edge.dy;

                    let dx = vx - closest_x;
                    let dy = vy - closest_y;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq < min_dist_sq && dist_sq > 1e-10 {
                        total_collisions += 1;

                        let dist = dist_sq.sqrt();
                        let overlap = min_dist - dist;

                        let nx = dx / dist;
                        let ny = dy / dist;

                        let w_edge = (1.0 - t) * edge.w0 + t * edge.w1;
                        let w_total = w_vert + w_edge;

                        if w_total < 1e-10 { continue; }

                        let vert_corr = overlap * (w_vert / w_total);
                        let edge_corr = overlap * (w_edge / w_total);

                        // Move vertex
                        bodies[body_a_idx].pos[vert_idx * 2] += nx * vert_corr;
                        bodies[body_a_idx].pos[vert_idx * 2 + 1] += ny * vert_corr;

                        // Move edge endpoints
                        let e0_factor = (1.0 - t) * edge.w0 / w_edge.max(1e-10);
                        let e1_factor = t * edge.w1 / w_edge.max(1e-10);

                        bodies[body_b_idx].pos[edge.v0 * 2] -= nx * edge_corr * e0_factor;
                        bodies[body_b_idx].pos[edge.v0 * 2 + 1] -= ny * edge_corr * e0_factor;
                        bodies[body_b_idx].pos[edge.v1 * 2] -= nx * edge_corr * e1_factor;
                        bodies[body_b_idx].pos[edge.v1 * 2 + 1] -= ny * edge_corr * e1_factor;
                    }
                }
            }
        }

        total_collisions
    }
}

/// Edge constraint data
#[derive(Clone, Debug)]
pub struct EdgeConstraint {
    pub v0: usize,           // First vertex index
    pub v1: usize,           // Second vertex index
    pub rest_length: f32,    // Rest length
}

/// Triangle area constraint data
#[derive(Clone, Debug)]
pub struct AreaConstraint {
    pub v0: usize,
    pub v1: usize,
    pub v2: usize,
    pub rest_area: f32,
}

/// XPBD soft body with position-based constraints
pub struct XPBDSoftBody {
    // Vertex data
    pub pos: Vec<f32>,           // Current positions [x0, y0, x1, y1, ...]
    pub prev_pos: Vec<f32>,      // Previous positions (for velocity computation)
    pub vel: Vec<f32>,           // Velocities (used for external forces)
    pub inv_mass: Vec<f32>,      // Inverse masses (0 = fixed)

    // Constraints
    pub edge_constraints: Vec<EdgeConstraint>,
    pub area_constraints: Vec<AreaConstraint>,

    // Material compliance (inverse of stiffness)
    // Lower compliance = stiffer
    pub edge_compliance: f32,    // For distance constraints
    pub area_compliance: f32,    // For area preservation

    // Triangle connectivity (for rendering)
    pub triangles: Vec<u32>,

    // Counts
    pub num_verts: usize,
}

impl XPBDSoftBody {
    /// Create from mesh vertices and triangles
    /// compliance: 0 = infinitely stiff, higher = softer
    pub fn new(
        vertices: &[f32],
        triangles: &[u32],
        density: f32,
        edge_compliance: f32,
        area_compliance: f32,
    ) -> Self {
        let num_verts = vertices.len() / 2;
        let num_tris = triangles.len() / 3;

        // Initialize vertex data
        let pos = vertices.to_vec();
        let prev_pos = vertices.to_vec();
        let vel = vec![0.0; vertices.len()];

        // Compute masses from triangle areas
        let mut mass = vec![0.0f32; num_verts];
        let mut area_constraints = Vec::with_capacity(num_tris);

        for t in 0..num_tris {
            let i0 = triangles[t * 3] as usize;
            let i1 = triangles[t * 3 + 1] as usize;
            let i2 = triangles[t * 3 + 2] as usize;

            let x0 = vertices[i0 * 2];
            let y0 = vertices[i0 * 2 + 1];
            let x1 = vertices[i1 * 2];
            let y1 = vertices[i1 * 2 + 1];
            let x2 = vertices[i2 * 2];
            let y2 = vertices[i2 * 2 + 1];

            // Compute signed area (for winding order)
            let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
            let tri_area = area.abs();

            // Distribute mass to vertices
            let tri_mass = tri_area * density;
            mass[i0] += tri_mass / 3.0;
            mass[i1] += tri_mass / 3.0;
            mass[i2] += tri_mass / 3.0;

            // Area constraint
            area_constraints.push(AreaConstraint {
                v0: i0,
                v1: i1,
                v2: i2,
                rest_area: tri_area,
            });
        }

        // Compute inverse masses
        let inv_mass: Vec<f32> = mass.iter().map(|&m| {
            if m > 1e-10 { 1.0 / m } else { 0.0 }
        }).collect();

        // Build edge constraints from triangles (avoiding duplicates)
        let mut edge_set = std::collections::HashSet::new();
        let mut edge_constraints = Vec::new();

        for t in 0..num_tris {
            let i0 = triangles[t * 3] as usize;
            let i1 = triangles[t * 3 + 1] as usize;
            let i2 = triangles[t * 3 + 2] as usize;

            // Add edges (using sorted indices for uniqueness)
            for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                let key = if a < b { (a, b) } else { (b, a) };
                if edge_set.insert(key) {
                    let x0 = vertices[a * 2];
                    let y0 = vertices[a * 2 + 1];
                    let x1 = vertices[b * 2];
                    let y1 = vertices[b * 2 + 1];
                    let dx = x1 - x0;
                    let dy = y1 - y0;
                    let rest_length = (dx * dx + dy * dy).sqrt();

                    edge_constraints.push(EdgeConstraint {
                        v0: a,
                        v1: b,
                        rest_length,
                    });
                }
            }
        }

        XPBDSoftBody {
            pos,
            prev_pos,
            vel,
            inv_mass,
            edge_constraints,
            area_constraints,
            edge_compliance,
            area_compliance,
            triangles: triangles.to_vec(),
            num_verts,
        }
    }

    /// Create from existing FEM material parameters
    /// Note: XPBD compliance is different from FEM stiffness
    /// We scale to give stable behavior with 8 substeps at 60Hz
    pub fn from_material(
        vertices: &[f32],
        triangles: &[u32],
        young_modulus: f32,
        density: f32,
    ) -> Self {
        // For XPBD, compliance = 1/stiffness
        // Lower compliance = stiffer material
        // Scale appropriately for simulation timestep
        let base_compliance = 1.0 / young_modulus;

        // These multipliers are tuned for 8 substeps at 60Hz
        let edge_compliance = base_compliance * 10.0;
        let area_compliance = base_compliance * 100.0;

        Self::new(vertices, triangles, density, edge_compliance, area_compliance)
    }

    /// Pre-solve: apply external forces and predict positions
    #[cfg(feature = "simd")]
    pub fn pre_solve(&mut self, dt: f32, gravity: f32) {
        SimdBackend::integrate_gravity(
            &mut self.pos,
            &mut self.vel,
            &mut self.prev_pos,
            gravity,
            dt,
            &self.inv_mass,
        );
    }

    /// Pre-solve: apply external forces and predict positions (scalar fallback)
    #[cfg(not(feature = "simd"))]
    pub fn pre_solve(&mut self, dt: f32, gravity: f32) {
        for i in 0..self.num_verts {
            if self.inv_mass[i] == 0.0 {
                continue; // Fixed vertex
            }

            // Store previous position
            self.prev_pos[i * 2] = self.pos[i * 2];
            self.prev_pos[i * 2 + 1] = self.pos[i * 2 + 1];

            // Apply gravity to velocity
            self.vel[i * 2 + 1] += gravity * dt;

            // Predict position
            self.pos[i * 2] += self.vel[i * 2] * dt;
            self.pos[i * 2 + 1] += self.vel[i * 2 + 1] * dt;
        }
    }

    /// Solve distance (edge length) constraint using XPBD
    /// Returns constraint violation before solve
    fn solve_edge_constraint(&mut self, edge: &EdgeConstraint, alpha: f32) -> f32 {
        let i0 = edge.v0;
        let i1 = edge.v1;

        let w0 = self.inv_mass[i0];
        let w1 = self.inv_mass[i1];
        let w_sum = w0 + w1;

        if w_sum < 1e-10 {
            return 0.0; // Both vertices fixed
        }

        // Current edge vector
        let dx = self.pos[i1 * 2] - self.pos[i0 * 2];
        let dy = self.pos[i1 * 2 + 1] - self.pos[i0 * 2 + 1];
        let len = (dx * dx + dy * dy).sqrt();

        if len < 1e-10 {
            return 0.0; // Degenerate edge
        }

        // Constraint: C = len - rest_length
        let c = len - edge.rest_length;

        // Gradient magnitude: |∇C| = 1 for distance constraint
        // XPBD: λ = -C / (w_sum + α/dt²)
        // where α is compliance (1/stiffness)
        let lambda = -c / (w_sum + alpha);

        // Position corrections
        let nx = dx / len;
        let ny = dy / len;

        let corr0 = -lambda * w0;
        let corr1 = lambda * w1;

        self.pos[i0 * 2] += corr0 * nx;
        self.pos[i0 * 2 + 1] += corr0 * ny;
        self.pos[i1 * 2] += corr1 * nx;
        self.pos[i1 * 2 + 1] += corr1 * ny;

        c.abs()
    }

    /// Solve area constraint using XPBD
    /// Preserves triangle area (2D volume)
    fn solve_area_constraint(&mut self, area: &AreaConstraint, alpha: f32) -> f32 {
        let i0 = area.v0;
        let i1 = area.v1;
        let i2 = area.v2;

        let w0 = self.inv_mass[i0];
        let w1 = self.inv_mass[i1];
        let w2 = self.inv_mass[i2];

        // Get positions
        let x0 = self.pos[i0 * 2];
        let y0 = self.pos[i0 * 2 + 1];
        let x1 = self.pos[i1 * 2];
        let y1 = self.pos[i1 * 2 + 1];
        let x2 = self.pos[i2 * 2];
        let y2 = self.pos[i2 * 2 + 1];

        // Current signed area (2 * area = cross product)
        let current_area_2x = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        let current_area = current_area_2x * 0.5;

        // Constraint: C = current_area - rest_area
        let c = current_area - area.rest_area;

        // Gradients of area w.r.t. vertex positions
        // ∇_p0 A = 0.5 * (p1 - p2)^perp = 0.5 * (y1 - y2, x2 - x1)
        // ∇_p1 A = 0.5 * (p2 - p0)^perp = 0.5 * (y2 - y0, x0 - x2)
        // ∇_p2 A = 0.5 * (p0 - p1)^perp = 0.5 * (y0 - y1, x1 - x0)
        let grad0_x = 0.5 * (y1 - y2);
        let grad0_y = 0.5 * (x2 - x1);
        let grad1_x = 0.5 * (y2 - y0);
        let grad1_y = 0.5 * (x0 - x2);
        let grad2_x = 0.5 * (y0 - y1);
        let grad2_y = 0.5 * (x1 - x0);

        // Sum of weighted squared gradient magnitudes
        let grad0_sq = grad0_x * grad0_x + grad0_y * grad0_y;
        let grad1_sq = grad1_x * grad1_x + grad1_y * grad1_y;
        let grad2_sq = grad2_x * grad2_x + grad2_y * grad2_y;

        let w_grad_sum = w0 * grad0_sq + w1 * grad1_sq + w2 * grad2_sq;

        if w_grad_sum < 1e-10 {
            return c.abs();
        }

        // XPBD lambda
        let lambda = -c / (w_grad_sum + alpha);

        // Apply corrections
        self.pos[i0 * 2] += lambda * w0 * grad0_x;
        self.pos[i0 * 2 + 1] += lambda * w0 * grad0_y;
        self.pos[i1 * 2] += lambda * w1 * grad1_x;
        self.pos[i1 * 2 + 1] += lambda * w1 * grad1_y;
        self.pos[i2 * 2] += lambda * w2 * grad2_x;
        self.pos[i2 * 2 + 1] += lambda * w2 * grad2_y;

        c.abs()
    }

    /// Solve ground collision as a simple position constraint
    /// Just projects vertices out of ground - bounce happens from internal elasticity
    pub fn solve_ground_collision(&mut self, ground_y: f32, _dt: f32) {
        const FRICTION: f32 = 0.7;
        const RESTITUTION: f32 = 0.3;

        for i in 0..self.num_verts {
            if self.inv_mass[i] == 0.0 {
                continue;
            }

            let y = self.pos[i * 2 + 1];

            if y < ground_y {
                let prev_x = self.prev_pos[i * 2];
                let prev_y = self.prev_pos[i * 2 + 1];

                // How far did we move this substep?
                let dy = y - prev_y;

                // Project out of ground
                self.pos[i * 2 + 1] = ground_y;

                // If we were moving downward, reflect some velocity
                if dy < 0.0 {
                    // Position-based reflection: mirror the penetration
                    let penetration = ground_y - y;
                    self.pos[i * 2 + 1] = ground_y + penetration * RESTITUTION;
                }

                // Friction: reduce horizontal movement
                let dx = self.pos[i * 2] - prev_x;
                self.pos[i * 2] = prev_x + dx * FRICTION;
            }
        }
    }

    /// Solve all constraints for one iteration
    /// Returns max constraint violation
    pub fn solve_constraints(&mut self, dt: f32) -> f32 {
        let dt_sq = dt * dt;
        let mut max_violation: f32 = 0.0;

        // Edge compliance scaled by dt²
        let edge_alpha = self.edge_compliance / dt_sq;

        // Solve edge (distance) constraints
        for edge in self.edge_constraints.clone() {
            let violation = self.solve_edge_constraint(&edge, edge_alpha);
            max_violation = max_violation.max(violation);
        }

        // Area compliance scaled by dt²
        let area_alpha = self.area_compliance / dt_sq;

        // Solve area constraints
        for area in self.area_constraints.clone() {
            let violation = self.solve_area_constraint(&area, area_alpha);
            max_violation = max_violation.max(violation);
        }

        max_violation
    }

    /// Post-solve: compute velocities from position change
    #[cfg(feature = "simd")]
    pub fn post_solve(&mut self, dt: f32) {
        SimdBackend::derive_velocities(&self.pos, &self.prev_pos, &mut self.vel, dt);
    }

    /// Post-solve: compute velocities from position change (scalar fallback)
    #[cfg(not(feature = "simd"))]
    pub fn post_solve(&mut self, dt: f32) {
        let inv_dt = 1.0 / dt;

        for i in 0..self.num_verts {
            self.vel[i * 2] = (self.pos[i * 2] - self.prev_pos[i * 2]) * inv_dt;
            self.vel[i * 2 + 1] = (self.pos[i * 2 + 1] - self.prev_pos[i * 2 + 1]) * inv_dt;
        }
    }

    /// Apply velocity damping
    pub fn apply_damping(&mut self, damping: f32) {
        let factor = 1.0 - damping;
        for i in 0..self.vel.len() {
            self.vel[i] *= factor;
        }
    }

    /// Pre-solve and constraint solving (call collide_with_body after this, then finalize_substep)
    pub fn substep_pre(&mut self, dt: f32, gravity: f32, ground_y: Option<f32>) {
        self.pre_solve(dt, gravity);

        // Solve constraints multiple times for stiff behavior
        // More iterations = better shape preservation
        for _ in 0..5 {
            self.solve_constraints(dt);
        }

        // Ground collision
        if let Some(gy) = ground_y {
            self.solve_ground_collision(gy, dt);
        }
    }

    /// Finalize substep: compute velocities from position change
    pub fn substep_post(&mut self, dt: f32) {
        self.post_solve(dt);
    }

    /// Complete substep: pre-solve, solve constraints, post-solve (no inter-body collision)
    pub fn substep(&mut self, dt: f32, gravity: f32, ground_y: Option<f32>) {
        self.substep_pre(dt, gravity, ground_y);
        self.substep_post(dt);
    }

    /// Get kinetic energy
    pub fn get_kinetic_energy(&self) -> f32 {
        let mut ke = 0.0;
        for i in 0..self.num_verts {
            if self.inv_mass[i] > 0.0 {
                let m = 1.0 / self.inv_mass[i];
                let vx = self.vel[i * 2];
                let vy = self.vel[i * 2 + 1];
                ke += 0.5 * m * (vx * vx + vy * vy);
            }
        }
        ke
    }

    /// Get lowest Y position
    pub fn get_lowest_y(&self) -> f32 {
        let mut lowest = f32::INFINITY;
        for i in 0..self.num_verts {
            lowest = lowest.min(self.pos[i * 2 + 1]);
        }
        lowest
    }

    /// Get AABB
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

    /// Get center of mass (average position)
    pub fn get_center(&self) -> (f32, f32) {
        let mut cx = 0.0;
        let mut cy = 0.0;
        for i in 0..self.num_verts {
            cx += self.pos[i * 2];
            cy += self.pos[i * 2 + 1];
        }
        let n = self.num_verts as f32;
        (cx / n, cy / n)
    }

    /// Collide with another XPBD body - position-based separation
    /// Call this BEFORE post_solve so velocities are derived correctly
    pub fn collide_with_body(&mut self, other: &mut XPBDSoftBody, min_dist: f32) -> u32 {
        // Broad phase AABB check
        let self_aabb = self.get_aabb();
        let other_aabb = other.get_aabb();

        if self_aabb.2 + min_dist < other_aabb.0 || other_aabb.2 + min_dist < self_aabb.0 ||
           self_aabb.3 + min_dist < other_aabb.1 || other_aabb.3 + min_dist < self_aabb.1 {
            return 0;
        }

        let mut collisions = 0u32;

        for i in 0..self.num_verts {
            let w1 = self.inv_mass[i];
            if w1 == 0.0 { continue; }

            for j in 0..other.num_verts {
                let w2 = other.inv_mass[j];
                if w2 == 0.0 { continue; }

                let x1 = self.pos[i * 2];
                let y1 = self.pos[i * 2 + 1];
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

                    // Push vertices apart proportional to inverse mass
                    let w_sum = w1 + w2;
                    let corr1 = overlap * (w1 / w_sum);
                    let corr2 = overlap * (w2 / w_sum);

                    // Apply position corrections only
                    self.pos[i * 2] -= nx * corr1;
                    self.pos[i * 2 + 1] -= ny * corr1;
                    other.pos[j * 2] += nx * corr2;
                    other.pos[j * 2 + 1] += ny * corr2;
                }
            }
        }

        collisions
    }

    /// Sleep if kinetic energy is below threshold
    pub fn sleep_if_resting(&mut self, ke_threshold: f32) -> bool {
        let ke = self.get_kinetic_energy();
        if ke < ke_threshold {
            self.vel.fill(0.0);
            true
        } else {
            false
        }
    }

    /// Get max velocity
    pub fn get_max_velocity(&self) -> f32 {
        let mut max_vel_sq: f32 = 0.0;
        for i in 0..self.num_verts {
            let vx = self.vel[i * 2];
            let vy = self.vel[i * 2 + 1];
            max_vel_sq = max_vel_sq.max(vx * vx + vy * vy);
        }
        max_vel_sq.sqrt()
    }

    /// Get aspect ratio (width / height) - for detecting pancaking
    pub fn get_aspect_ratio(&self) -> f32 {
        let (min_x, min_y, max_x, max_y) = self.get_aabb();
        let width = max_x - min_x;
        let height = max_y - min_y;
        if height < 1e-6 { return f32::INFINITY; }
        width / height
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{create_square_mesh, create_ring_mesh};

    #[test]
    fn test_xpbd_creation() {
        let mesh = create_square_mesh(1.0, 2);
        let body = XPBDSoftBody::new(&mesh.vertices, &mesh.triangles, 1000.0, 1e-6, 1e-5);

        assert_eq!(body.num_verts, 9);
        assert!(!body.edge_constraints.is_empty());
        assert!(!body.area_constraints.is_empty());
    }

    #[test]
    fn test_xpbd_freefall() {
        let mesh = create_square_mesh(1.0, 2);
        let mut body = XPBDSoftBody::new(&mesh.vertices, &mesh.triangles, 1000.0, 1e-6, 1e-5);

        let initial_lowest = body.get_lowest_y();

        // Run 8 substeps for one frame
        let dt = 1.0 / 60.0 / 8.0;
        for _ in 0..8 {
            body.substep(dt, -9.8, None);
        }

        let final_lowest = body.get_lowest_y();

        // Should have fallen
        assert!(final_lowest < initial_lowest, "Body should fall under gravity");
    }

    #[test]
    fn test_xpbd_ground_collision() {
        let mesh = create_square_mesh(1.0, 2);
        let mut body = XPBDSoftBody::new(&mesh.vertices, &mesh.triangles, 1000.0, 1e-4, 1e-3);

        // Position body ABOVE ground, let it fall naturally
        let ground_y = -2.0;
        for i in 0..body.num_verts {
            body.pos[i * 2 + 1] += 3.0;  // 3m above ground
            body.prev_pos[i * 2 + 1] += 3.0;
        }

        let dt = 1.0 / 60.0 / 8.0;

        // Run many frames (10 seconds)
        for frame in 0..600 {
            for _ in 0..8 {
                body.substep(dt, -9.8, Some(ground_y));
            }

            // Check for explosion
            let ke = body.get_kinetic_energy();
            assert!(ke.is_finite() && ke < 1e5, "Frame {}: KE exploded: {}", frame, ke);
        }

        // Should be resting on ground
        let lowest = body.get_lowest_y();
        assert!(lowest >= ground_y - 0.2, "Should be above ground, got {}", lowest);
    }

    #[test]
    fn test_xpbd_stability_8_substeps() {
        // This is the critical test: stable with EXACTLY 8 substeps
        let mesh = create_ring_mesh(1.5, 1.0, 16, 4);

        // Use compliance values that give bouncy behavior without explosion
        let mut body = XPBDSoftBody::new(
            &mesh.vertices,
            &mesh.triangles,
            1100.0,  // density
            1e-4,    // edge compliance (medium stiff - like "bouncy" in material_range)
            1e-3,    // area compliance
        );

        // Offset up
        for i in 0..body.num_verts {
            body.pos[i * 2 + 1] += 6.0;
            body.prev_pos[i * 2 + 1] += 6.0;
        }

        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;  // EXACTLY 8 substeps

        let mut max_ke: f32 = 0.0;

        // Run 10 seconds (600 frames)
        for frame in 0..600 {
            for _ in 0..8 {
                body.substep(dt, -9.8, Some(ground_y));
                body.apply_damping(0.005);  // Light damping per substep
            }

            let ke = body.get_kinetic_energy();
            max_ke = max_ke.max(ke);

            // Check for explosion
            assert!(
                ke.is_finite() && ke < 1e5,
                "Frame {}: KE exploded: {}", frame, ke
            );

            // Check velocities are reasonable
            let max_vel = body.get_max_velocity();
            assert!(
                max_vel < 50.0,
                "Frame {}: velocity exploded: {}", frame, max_vel
            );
        }

        println!("XPBD 8-substep test passed. Max KE: {:.2}", max_ke);
    }

    #[test]
    fn test_xpbd_two_body_collision() {
        let mesh = create_ring_mesh(1.5, 1.0, 16, 4);

        // Use zero edge compliance (rigid edges) like actual simulation
        let mut body1 = XPBDSoftBody::new(
            &mesh.vertices, &mesh.triangles, 1100.0, 0.0, 1e-6
        );
        let mut body2 = XPBDSoftBody::new(
            &mesh.vertices, &mesh.triangles, 1100.0, 0.0, 1e-6
        );

        // Position body1 above body2
        for i in 0..body1.num_verts {
            body1.pos[i * 2 + 1] += 10.0;
            body1.prev_pos[i * 2 + 1] += 10.0;
        }
        for i in 0..body2.num_verts {
            body2.pos[i * 2 + 1] += 5.0;
            body2.prev_pos[i * 2 + 1] += 5.0;
        }

        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;
        let collision_dist = 0.02;

        // Run 10 seconds
        for frame in 0..600 {
            for _ in 0..8 {
                // Correct order: pre-solve → collisions → post-solve
                body1.substep_pre(dt, -9.8, Some(ground_y));
                body2.substep_pre(dt, -9.8, Some(ground_y));
                body1.collide_with_body(&mut body2, collision_dist);
                body1.substep_post(dt);
                body2.substep_post(dt);
                body1.apply_damping(0.01);
                body2.apply_damping(0.01);
            }

            let ke1 = body1.get_kinetic_energy();
            let ke2 = body2.get_kinetic_energy();

            assert!(
                ke1.is_finite() && ke1 < 1e5 && ke2.is_finite() && ke2 < 1e5,
                "Frame {}: explosion - KE1={}, KE2={}", frame, ke1, ke2
            );
        }

        println!("XPBD two-body collision test passed");
    }

    #[test]
    fn test_xpbd_five_body_collision() {
        // The critical multi-body test
        let mesh = create_ring_mesh(1.5, 1.0, 16, 4);

        let mut bodies: Vec<XPBDSoftBody> = Vec::new();
        let offsets = [
            (0.0, 22.0),
            (-0.5, 18.0),
            (0.5, 14.0),
            (-0.3, 10.0),
            (0.3, 6.0),
        ];

        for (x_off, y_off) in offsets {
            // Use zero edge compliance (rigid edges) like actual simulation
            let mut body = XPBDSoftBody::new(
                &mesh.vertices, &mesh.triangles, 1100.0, 0.0, 1e-6
            );
            for i in 0..body.num_verts {
                body.pos[i * 2] += x_off;
                body.pos[i * 2 + 1] += y_off;
                body.prev_pos[i * 2] += x_off;
                body.prev_pos[i * 2 + 1] += y_off;
            }
            bodies.push(body);
        }

        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;
        let collision_dist = 0.02;

        // Run 10 seconds (600 frames)
        for frame in 0..600 {
            for _ in 0..8 {
                // Correct order: pre-solve → collisions → post-solve
                for body in &mut bodies {
                    body.substep_pre(dt, -9.8, Some(ground_y));
                }

                // Inter-body collisions
                for i in 0..bodies.len() {
                    for j in (i + 1)..bodies.len() {
                        let (left, right) = bodies.split_at_mut(j);
                        left[i].collide_with_body(&mut right[0], collision_dist);
                    }
                }

                for body in &mut bodies {
                    body.substep_post(dt);
                    body.apply_damping(0.01);
                }
            }

            // Check all bodies
            for (idx, body) in bodies.iter().enumerate() {
                let ke = body.get_kinetic_energy();
                assert!(
                    ke.is_finite() && ke < 1e5,
                    "Frame {}, body {}: KE exploded: {}", frame, idx, ke
                );
            }
        }

        println!("XPBD 5-body collision test passed!");
    }

    /// Test different material stiffnesses
    #[test]
    fn test_xpbd_material_range() {
        let mesh = create_ring_mesh(1.5, 1.0, 16, 4);
        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;

        // Test range of compliance values (soft to stiff)
        let compliances = [
            (5e-4, 5e-3, "jello"),      // Very soft
            (2e-4, 2e-3, "rubber"),     // Soft
            (1e-4, 1e-3, "bouncy"),     // Medium
            (5e-5, 5e-4, "wood"),       // Stiff
            (1e-5, 1e-4, "metal"),      // Very stiff
        ];

        for (edge_c, area_c, name) in compliances {
            let mut body = XPBDSoftBody::new(
                &mesh.vertices, &mesh.triangles, 1000.0, edge_c, area_c
            );

            // Offset up
            for i in 0..body.num_verts {
                body.pos[i * 2 + 1] += 6.0;
                body.prev_pos[i * 2 + 1] += 6.0;
            }

            let mut max_ke: f32 = 0.0;

            // Run 10 seconds
            for frame in 0..600 {
                for _ in 0..8 {
                    body.substep(dt, -9.8, Some(ground_y));
                    body.apply_damping(0.01);
                }

                let ke = body.get_kinetic_energy();
                max_ke = max_ke.max(ke);

                assert!(
                    ke.is_finite() && ke < 1e5,
                    "Material '{}' frame {}: KE exploded: {}", name, frame, ke
                );
            }

            println!("Material '{}' passed. Max KE: {:.2}", name, max_ke);
        }
    }

    /// Stress test: 5 bodies with different stiffnesses
    #[test]
    fn test_xpbd_mixed_stiffness_pile() {
        let mesh = create_ring_mesh(1.5, 1.0, 16, 4);

        // Create bodies with different stiffnesses
        let configs = [
            (5e-4, 5e-3, 1000.0),  // Very soft (top)
            (2e-4, 2e-3, 1100.0),  // Soft
            (1e-4, 1e-3, 1100.0),  // Medium
            (5e-5, 5e-4, 600.0),   // Stiff
            (1e-5, 1e-4, 2000.0),  // Very stiff (bottom)
        ];

        let offsets = [
            (0.0, 22.0),
            (-0.3, 18.0),
            (0.3, 14.0),
            (-0.2, 10.0),
            (0.2, 6.0),
        ];

        let mut bodies: Vec<XPBDSoftBody> = Vec::new();
        for (i, ((edge_c, area_c, density), (x_off, y_off))) in
            configs.iter().zip(offsets.iter()).enumerate()
        {
            let mut body = XPBDSoftBody::new(
                &mesh.vertices, &mesh.triangles, *density, *edge_c, *area_c
            );
            for j in 0..body.num_verts {
                body.pos[j * 2] += x_off;
                body.pos[j * 2 + 1] += y_off;
                body.prev_pos[j * 2] += x_off;
                body.prev_pos[j * 2 + 1] += y_off;
            }
            bodies.push(body);
        }

        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;
        let collision_dist = 0.02;

        // Run 10 seconds (600 frames)
        for frame in 0..600 {
            for _ in 0..8 {
                for body in &mut bodies {
                    body.substep(dt, -9.8, Some(ground_y));
                    body.apply_damping(0.01);
                }

                for i in 0..bodies.len() {
                    for j in (i + 1)..bodies.len() {
                        let (left, right) = bodies.split_at_mut(j);
                        left[i].collide_with_body(&mut right[0], collision_dist);
                    }
                }
            }

            for (idx, body) in bodies.iter().enumerate() {
                let ke = body.get_kinetic_energy();
                assert!(
                    ke.is_finite() && ke < 1e5,
                    "Frame {}, body {}: KE exploded: {}", frame, idx, ke
                );
            }
        }

        println!("XPBD mixed stiffness pile test passed!");
    }

    /// Test that shape is preserved (no pancaking)
    #[test]
    fn test_xpbd_shape_preservation() {
        let mesh = create_ring_mesh(1.5, 1.0, 16, 4);

        // Use zero edge compliance for perfectly rigid edges
        let mut body = XPBDSoftBody::new(
            &mesh.vertices, &mesh.triangles, 1100.0,
            0.0,   // Perfectly rigid edges - no stretching allowed
            1e-6,  // Very stiff area
        );

        // Offset up
        for i in 0..body.num_verts {
            body.pos[i * 2 + 1] += 6.0;
            body.prev_pos[i * 2 + 1] += 6.0;
        }

        let initial_aspect = body.get_aspect_ratio();
        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;

        // Run 5 seconds (300 frames)
        for frame in 0..300 {
            for _ in 0..8 {
                body.substep(dt, -9.8, Some(ground_y));
            }

            let aspect = body.get_aspect_ratio();

            // Shape should not pancake: aspect ratio should not exceed 3x initial
            assert!(
                aspect < initial_aspect * 3.0,
                "Frame {}: shape pancaked! Initial aspect: {:.2}, current: {:.2}",
                frame, initial_aspect, aspect
            );
        }

        let final_aspect = body.get_aspect_ratio();
        println!("Shape preservation test passed. Initial: {:.2}, Final: {:.2}", initial_aspect, final_aspect);
    }

    /// Test ellipse mesh simulation stability
    #[test]
    fn test_xpbd_ellipse_mesh() {
        use crate::mesh::create_ellipse_mesh;

        let mesh = create_ellipse_mesh(2.5, 1.8, 16, 4);
        let mut body = XPBDSoftBody::new(
            &mesh.vertices, &mesh.triangles, 1100.0, 0.0, 1e-6
        );

        // Offset up
        for i in 0..body.num_verts {
            body.pos[i * 2 + 1] += 8.0;
            body.prev_pos[i * 2 + 1] += 8.0;
        }

        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;

        // Run 5 seconds
        for frame in 0..300 {
            for _ in 0..8 {
                body.substep(dt, -9.8, Some(ground_y));
                body.apply_damping(0.01);
            }

            let ke = body.get_kinetic_energy();
            assert!(
                ke.is_finite() && ke < 1e5,
                "Frame {}: ellipse KE exploded: {}", frame, ke
            );
        }

        println!("XPBD ellipse mesh test passed");
    }

    /// Test star mesh simulation stability
    #[test]
    fn test_xpbd_star_mesh() {
        use crate::mesh::create_star_mesh;

        let mesh = create_star_mesh(1.6, 0.7, 5, 4);
        let mut body = XPBDSoftBody::new(
            &mesh.vertices, &mesh.triangles, 1100.0, 0.0, 1e-6
        );

        // Offset up
        for i in 0..body.num_verts {
            body.pos[i * 2 + 1] += 8.0;
            body.prev_pos[i * 2 + 1] += 8.0;
        }

        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;

        // Run 5 seconds
        for frame in 0..300 {
            for _ in 0..8 {
                body.substep(dt, -9.8, Some(ground_y));
                body.apply_damping(0.01);
            }

            let ke = body.get_kinetic_energy();
            assert!(
                ke.is_finite() && ke < 1e5,
                "Frame {}: star KE exploded: {}", frame, ke
            );
        }

        println!("XPBD star mesh test passed");
    }

    /// Test blob mesh simulation stability
    #[test]
    fn test_xpbd_blob_mesh() {
        use crate::mesh::create_blob_mesh;

        let mesh = create_blob_mesh(1.4, 0.25, 16, 4, 42);
        let mut body = XPBDSoftBody::new(
            &mesh.vertices, &mesh.triangles, 1100.0, 0.0, 1e-6
        );

        // Offset up
        for i in 0..body.num_verts {
            body.pos[i * 2 + 1] += 8.0;
            body.prev_pos[i * 2 + 1] += 8.0;
        }

        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;

        // Run 5 seconds
        for frame in 0..300 {
            for _ in 0..8 {
                body.substep(dt, -9.8, Some(ground_y));
                body.apply_damping(0.01);
            }

            let ke = body.get_kinetic_energy();
            assert!(
                ke.is_finite() && ke < 1e5,
                "Frame {}: blob KE exploded: {}", frame, ke
            );
        }

        println!("XPBD blob mesh test passed");
    }

    /// Test mixed shape collisions
    #[test]
    fn test_xpbd_mixed_shape_collision() {
        use crate::mesh::{create_ellipse_mesh, create_star_mesh, create_blob_mesh};

        // Create different shapes
        let ring_mesh = create_ring_mesh(1.5, 1.0, 16, 4);
        let ellipse_mesh = create_ellipse_mesh(2.0, 1.5, 16, 4);
        let star_mesh = create_star_mesh(1.4, 0.6, 5, 4);
        let blob_mesh = create_blob_mesh(1.3, 0.2, 16, 4, 123);

        let meshes = [&ring_mesh, &ellipse_mesh, &star_mesh, &blob_mesh];
        let offsets = [(0.0, 18.0), (-0.3, 14.0), (0.3, 10.0), (0.0, 6.0)];

        let mut bodies: Vec<XPBDSoftBody> = Vec::new();
        for (mesh, (x_off, y_off)) in meshes.iter().zip(offsets.iter()) {
            let mut body = XPBDSoftBody::new(
                &mesh.vertices, &mesh.triangles, 1100.0, 0.0, 1e-6
            );
            for i in 0..body.num_verts {
                body.pos[i * 2] += x_off;
                body.pos[i * 2 + 1] += y_off;
                body.prev_pos[i * 2] += x_off;
                body.prev_pos[i * 2 + 1] += y_off;
            }
            bodies.push(body);
        }

        let ground_y = -8.0;
        let dt = 1.0 / 60.0 / 8.0;
        let collision_dist = 0.02;

        // Run 10 seconds
        for frame in 0..600 {
            for _ in 0..8 {
                // Correct order: pre-solve → collisions → post-solve
                for body in &mut bodies {
                    body.substep_pre(dt, -9.8, Some(ground_y));
                }

                // Inter-body collisions
                for i in 0..bodies.len() {
                    for j in (i + 1)..bodies.len() {
                        let (left, right) = bodies.split_at_mut(j);
                        left[i].collide_with_body(&mut right[0], collision_dist);
                    }
                }

                for body in &mut bodies {
                    body.substep_post(dt);
                    body.apply_damping(0.01);
                }
            }

            for (idx, body) in bodies.iter().enumerate() {
                let ke = body.get_kinetic_energy();
                assert!(
                    ke.is_finite() && ke < 1e5,
                    "Frame {}, body {}: KE exploded: {}", frame, idx, ke
                );
            }
        }

        println!("XPBD mixed shape collision test passed!");
    }
}
