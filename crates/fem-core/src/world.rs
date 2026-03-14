//! Physics World - High-level API for soft body simulation
//!
//! Provides a clean interface for managing physics bodies, collision groups,
//! and simulation stepping.

use crate::mesh::Mesh;
use crate::xpbd::{XPBDSoftBody, CollisionSystem};

/// Unique identifier for a physics body
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyHandle(pub(crate) usize);

impl BodyHandle {
    /// Get the raw index (for advanced usage)
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Collision group flags - bodies only collide if their groups overlap
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionGroups {
    /// Groups this body belongs to
    pub membership: u32,
    /// Groups this body can collide with
    pub filter: u32,
}

impl Default for CollisionGroups {
    fn default() -> Self {
        Self {
            membership: 0xFFFF_FFFF,
            filter: 0xFFFF_FFFF,
        }
    }
}

impl CollisionGroups {
    /// No collisions with any body
    pub const NONE: Self = Self { membership: 0, filter: 0 };

    /// Collide with everything (default)
    pub const ALL: Self = Self { membership: 0xFFFF_FFFF, filter: 0xFFFF_FFFF };

    /// Create collision groups with specific membership and filter
    pub fn new(membership: u32, filter: u32) -> Self {
        Self { membership, filter }
    }

    /// Check if two collision groups can interact
    pub fn can_collide(&self, other: &Self) -> bool {
        (self.membership & other.filter) != 0 && (other.membership & self.filter) != 0
    }
}

/// Material properties for creating bodies
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Density in kg/m² (affects mass)
    pub density: f32,
    /// Edge compliance (0 = rigid edges, higher = softer)
    pub edge_compliance: f32,
    /// Area compliance (0 = incompressible, higher = compressible)
    pub area_compliance: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self::RUBBER
    }
}

impl Material {
    /// Soft, jiggly material
    pub const JELLO: Self = Self {
        density: 1000.0,
        edge_compliance: 0.0,
        area_compliance: 1e-6,
    };

    /// Bouncy rubber
    pub const RUBBER: Self = Self {
        density: 1100.0,
        edge_compliance: 0.0,
        area_compliance: 1e-7,
    };

    /// Stiff material
    pub const WOOD: Self = Self {
        density: 600.0,
        edge_compliance: 0.0,
        area_compliance: 1e-8,
    };

    /// Nearly rigid material
    pub const METAL: Self = Self {
        density: 2000.0,
        edge_compliance: 0.0,
        area_compliance: 0.0,
    };

    /// Create custom material
    pub fn new(density: f32, edge_compliance: f32, area_compliance: f32) -> Self {
        Self { density, edge_compliance, area_compliance }
    }
}

/// Body configuration options
#[derive(Clone, Debug)]
pub struct BodyConfig {
    /// Material properties
    pub material: Material,
    /// Collision groups
    pub collision_groups: CollisionGroups,
    /// If true, body is treated as rigid (infinite stiffness)
    pub is_rigid: bool,
    /// Initial position offset
    pub position: (f32, f32),
    /// Initial velocity
    pub velocity: (f32, f32),
}

impl Default for BodyConfig {
    fn default() -> Self {
        Self {
            material: Material::default(),
            collision_groups: CollisionGroups::default(),
            is_rigid: false,
            position: (0.0, 0.0),
            velocity: (0.0, 0.0),
        }
    }
}

impl BodyConfig {
    /// Create a new body config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the material
    pub fn with_material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    /// Set collision groups
    pub fn with_collision_groups(mut self, groups: CollisionGroups) -> Self {
        self.collision_groups = groups;
        self
    }

    /// Disable collisions for this body
    pub fn without_collisions(mut self) -> Self {
        self.collision_groups = CollisionGroups::NONE;
        self
    }

    /// Make this body rigid (no deformation)
    pub fn as_rigid(mut self) -> Self {
        self.is_rigid = true;
        self
    }

    /// Set initial position
    pub fn at_position(mut self, x: f32, y: f32) -> Self {
        self.position = (x, y);
        self
    }

    /// Set initial velocity
    pub fn with_velocity(mut self, vx: f32, vy: f32) -> Self {
        self.velocity = (vx, vy);
        self
    }
}

/// Internal body data stored alongside XPBDSoftBody
struct BodyData {
    collision_groups: CollisionGroups,
    is_rigid: bool,
    /// Original edge rest lengths (for compression effects)
    original_edge_lengths: Vec<f32>,
    /// Original area rest values (for compression effects)
    original_areas: Vec<f32>,
}

/// Physics world containing all bodies and simulation state
pub struct PhysicsWorld {
    bodies: Vec<XPBDSoftBody>,
    body_data: Vec<BodyData>,
    triangles: Vec<Vec<u32>>,
    collision_system: CollisionSystem,

    // Simulation parameters
    gravity: f32,
    ground_y: Option<f32>,
    ground_friction: f32,
    ground_restitution: f32,
    substeps: u32,

    // Tracking
    next_handle: usize,
    handle_to_index: Vec<Option<usize>>,  // Maps handle -> current index
    index_to_handle: Vec<BodyHandle>,      // Maps index -> handle
}

impl PhysicsWorld {
    /// Create a new empty physics world
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            body_data: Vec::new(),
            triangles: Vec::new(),
            collision_system: CollisionSystem::new(0.15),
            gravity: -9.8,
            ground_y: None,
            ground_friction: 0.8,
            ground_restitution: 0.3,
            substeps: 4,
            next_handle: 0,
            handle_to_index: Vec::new(),
            index_to_handle: Vec::new(),
        }
    }

    /// Set gravity (default: -9.8)
    pub fn set_gravity(&mut self, gravity: f32) {
        self.gravity = gravity;
    }

    /// Get current gravity
    pub fn gravity(&self) -> f32 {
        self.gravity
    }

    /// Set ground plane Y coordinate (None to disable)
    pub fn set_ground(&mut self, y: Option<f32>) {
        self.ground_y = y;
    }

    /// Get ground Y coordinate
    pub fn ground(&self) -> Option<f32> {
        self.ground_y
    }

    /// Set ground friction coefficient (default: 0.8)
    /// 0.0 = frictionless ice, 1.0 = very sticky
    pub fn set_ground_friction(&mut self, friction: f32) {
        self.ground_friction = friction.clamp(0.0, 1.0);
    }

    /// Get ground friction coefficient
    pub fn ground_friction(&self) -> f32 {
        self.ground_friction
    }

    /// Set ground restitution/bounciness (default: 0.3)
    /// 0.0 = no bounce, 1.0 = perfect bounce
    pub fn set_ground_restitution(&mut self, restitution: f32) {
        self.ground_restitution = restitution.clamp(0.0, 1.0);
    }

    /// Get ground restitution
    pub fn ground_restitution(&self) -> f32 {
        self.ground_restitution
    }

    /// Set number of substeps per step (default: 4)
    pub fn set_substeps(&mut self, substeps: u32) {
        self.substeps = substeps.max(1);
    }

    /// Add a body from a mesh with configuration
    pub fn add_body(&mut self, mesh: &Mesh, config: BodyConfig) -> BodyHandle {
        let mut vertices = mesh.vertices.clone();

        // Apply position offset
        let (dx, dy) = config.position;
        for i in 0..vertices.len() / 2 {
            vertices[i * 2] += dx;
            vertices[i * 2 + 1] += dy;
        }

        // Create the physics body
        let material = if config.is_rigid {
            Material::METAL  // Use very stiff material for rigid
        } else {
            config.material
        };

        let mut body = XPBDSoftBody::new(
            &vertices,
            &mesh.triangles,
            material.density,
            material.edge_compliance,
            material.area_compliance,
        );

        // Set initial velocity
        let (vx, vy) = config.velocity;
        for i in 0..body.num_verts {
            body.vel[i * 2] = vx;
            body.vel[i * 2 + 1] = vy;
        }

        // Initialize prev_pos for correct first-frame velocity
        body.prev_pos = body.pos.clone();

        // Store original rest lengths/areas for later manipulation
        let original_edge_lengths: Vec<f32> = body.edge_constraints
            .iter()
            .map(|c| c.rest_length)
            .collect();
        let original_areas: Vec<f32> = body.area_constraints
            .iter()
            .map(|c| c.rest_area)
            .collect();

        let body_data = BodyData {
            collision_groups: config.collision_groups,
            is_rigid: config.is_rigid,
            original_edge_lengths,
            original_areas,
        };

        // Allocate handle
        let handle = BodyHandle(self.next_handle);
        self.next_handle += 1;

        let index = self.bodies.len();

        // Extend handle_to_index if needed
        while self.handle_to_index.len() <= handle.0 {
            self.handle_to_index.push(None);
        }
        self.handle_to_index[handle.0] = Some(index);
        self.index_to_handle.push(handle);

        self.bodies.push(body);
        self.body_data.push(body_data);
        self.triangles.push(mesh.triangles.clone());

        handle
    }

    /// Add a body with default configuration
    pub fn add_body_simple(&mut self, mesh: &Mesh, x: f32, y: f32) -> BodyHandle {
        self.add_body(mesh, BodyConfig::new().at_position(x, y))
    }

    /// Remove a body from the world
    pub fn remove_body(&mut self, handle: BodyHandle) -> bool {
        let Some(index) = self.handle_to_index.get(handle.0).copied().flatten() else {
            return false;
        };

        // Remove from vectors
        self.bodies.remove(index);
        self.body_data.remove(index);
        self.triangles.remove(index);
        self.index_to_handle.remove(index);

        // Invalidate the handle
        self.handle_to_index[handle.0] = None;

        // Update indices for all handles after the removed one
        for i in index..self.index_to_handle.len() {
            let h = self.index_to_handle[i];
            self.handle_to_index[h.0] = Some(i);
        }

        true
    }

    /// Check if a body handle is valid
    pub fn contains(&self, handle: BodyHandle) -> bool {
        self.handle_to_index.get(handle.0).copied().flatten().is_some()
    }

    /// Get the number of bodies in the world
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Get a body by handle (immutable)
    pub fn get_body(&self, handle: BodyHandle) -> Option<&XPBDSoftBody> {
        let index = self.handle_to_index.get(handle.0)?.as_ref()?;
        self.bodies.get(*index)
    }

    /// Get a body by handle (mutable)
    pub fn get_body_mut(&mut self, handle: BodyHandle) -> Option<&mut XPBDSoftBody> {
        let index = self.handle_to_index.get(handle.0)?.as_ref()?;
        self.bodies.get_mut(*index)
    }

    /// Get triangles for a body (for rendering)
    pub fn get_triangles(&self, handle: BodyHandle) -> Option<&[u32]> {
        let index = self.handle_to_index.get(handle.0)?.as_ref()?;
        self.triangles.get(*index).map(|v| v.as_slice())
    }

    /// Get collision groups for a body
    pub fn get_collision_groups(&self, handle: BodyHandle) -> Option<CollisionGroups> {
        let index = self.handle_to_index.get(handle.0)?.as_ref()?;
        self.body_data.get(*index).map(|d| d.collision_groups)
    }

    /// Set collision groups for a body
    pub fn set_collision_groups(&mut self, handle: BodyHandle, groups: CollisionGroups) -> bool {
        let Some(index) = self.handle_to_index.get(handle.0).copied().flatten() else {
            return false;
        };
        if let Some(data) = self.body_data.get_mut(index) {
            data.collision_groups = groups;
            true
        } else {
            false
        }
    }

    /// Iterate over all body handles
    pub fn handles(&self) -> impl Iterator<Item = BodyHandle> + '_ {
        self.index_to_handle.iter().copied()
    }

    /// Iterate over all bodies with their handles
    pub fn iter(&self) -> impl Iterator<Item = (BodyHandle, &XPBDSoftBody)> {
        self.index_to_handle.iter().copied()
            .zip(self.bodies.iter())
    }

    /// Iterate over all bodies mutably with their handles
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (BodyHandle, &mut XPBDSoftBody)> {
        self.index_to_handle.iter().copied()
            .zip(self.bodies.iter_mut())
    }

    // === Force/Impulse Application ===

    /// Apply a force to all vertices of a body (in Newtons)
    /// Force is applied over time, so it accumulates velocity
    pub fn apply_force(&mut self, handle: BodyHandle, fx: f32, fy: f32) {
        let Some(body) = self.get_body_mut(handle) else { return };

        for i in 0..body.num_verts {
            if body.inv_mass[i] > 0.0 {
                // F = ma, so a = F * inv_mass
                // This will be integrated in the next step
                body.vel[i * 2] += fx * body.inv_mass[i];
                body.vel[i * 2 + 1] += fy * body.inv_mass[i];
            }
        }
    }

    /// Apply an impulse to all vertices of a body (instantaneous velocity change)
    pub fn apply_impulse(&mut self, handle: BodyHandle, vx: f32, vy: f32) {
        let Some(body) = self.get_body_mut(handle) else { return };

        for i in 0..body.num_verts {
            if body.inv_mass[i] > 0.0 {
                body.vel[i * 2] += vx;
                body.vel[i * 2 + 1] += vy;
            }
        }
    }

    /// Apply a force at the body's center of mass
    pub fn apply_central_force(&mut self, handle: BodyHandle, fx: f32, fy: f32) {
        // For soft bodies, this is the same as apply_force
        self.apply_force(handle, fx, fy);
    }

    /// Apply an impulse at the body's center of mass
    pub fn apply_central_impulse(&mut self, handle: BodyHandle, vx: f32, vy: f32) {
        self.apply_impulse(handle, vx, vy);
    }

    /// Apply acceleration to all vertices (like gravity, doesn't depend on mass)
    pub fn apply_acceleration(&mut self, handle: BodyHandle, ax: f32, ay: f32, dt: f32) {
        let Some(body) = self.get_body_mut(handle) else { return };

        for i in 0..body.num_verts {
            if body.inv_mass[i] > 0.0 {
                body.vel[i * 2] += ax * dt;
                body.vel[i * 2 + 1] += ay * dt;
            }
        }
    }

    /// Set velocity of all vertices
    pub fn set_velocity(&mut self, handle: BodyHandle, vx: f32, vy: f32) {
        let Some(body) = self.get_body_mut(handle) else { return };

        for i in 0..body.num_verts {
            body.vel[i * 2] = vx;
            body.vel[i * 2 + 1] = vy;
        }
    }

    /// Get average velocity of body
    pub fn get_velocity(&self, handle: BodyHandle) -> Option<(f32, f32)> {
        let body = self.get_body(handle)?;
        let mut vx = 0.0;
        let mut vy = 0.0;
        let mut count = 0;

        for i in 0..body.num_verts {
            if body.inv_mass[i] > 0.0 {
                vx += body.vel[i * 2];
                vy += body.vel[i * 2 + 1];
                count += 1;
            }
        }

        if count > 0 {
            Some((vx / count as f32, vy / count as f32))
        } else {
            Some((0.0, 0.0))
        }
    }

    // === Position/Transform ===

    /// Get the center of mass position
    pub fn get_position(&self, handle: BodyHandle) -> Option<(f32, f32)> {
        self.get_body(handle).map(|b| b.get_center())
    }

    /// Translate all vertices by an offset
    pub fn translate(&mut self, handle: BodyHandle, dx: f32, dy: f32) {
        let Some(body) = self.get_body_mut(handle) else { return };

        for i in 0..body.num_verts {
            body.pos[i * 2] += dx;
            body.pos[i * 2 + 1] += dy;
            body.prev_pos[i * 2] += dx;
            body.prev_pos[i * 2 + 1] += dy;
        }
    }

    /// Set the center position (translates whole body)
    pub fn set_position(&mut self, handle: BodyHandle, x: f32, y: f32) {
        let Some(current) = self.get_position(handle) else { return };
        let dx = x - current.0;
        let dy = y - current.1;
        self.translate(handle, dx, dy);
    }

    // === Deformation Control ===

    /// Compress the body vertically (ratio 0.0-1.0, where 1.0 = no compression)
    /// Useful for squash-and-stretch effects
    pub fn set_vertical_compression(&mut self, handle: BodyHandle, ratio: f32) {
        let Some(index) = self.handle_to_index.get(handle.0).copied().flatten() else {
            return;
        };

        let body = &mut self.bodies[index];
        let data = &self.body_data[index];

        for (i, constraint) in body.edge_constraints.iter_mut().enumerate() {
            let v0 = constraint.v0;
            let v1 = constraint.v1;

            let y0 = body.pos[v0 * 2 + 1];
            let y1 = body.pos[v1 * 2 + 1];
            let x0 = body.pos[v0 * 2];
            let x1 = body.pos[v1 * 2];

            let dy = (y1 - y0).abs();
            let dx = (x1 - x0).abs();

            // Compress edges that are more vertical than horizontal
            if dy > dx * 0.5 {
                constraint.rest_length = data.original_edge_lengths[i] * ratio;
            }
        }
    }

    /// Reset all rest lengths to original values
    pub fn reset_rest_lengths(&mut self, handle: BodyHandle) {
        let Some(index) = self.handle_to_index.get(handle.0).copied().flatten() else {
            return;
        };

        let body = &mut self.bodies[index];
        let data = &self.body_data[index];

        for (i, constraint) in body.edge_constraints.iter_mut().enumerate() {
            constraint.rest_length = data.original_edge_lengths[i];
        }

        for (i, constraint) in body.area_constraints.iter_mut().enumerate() {
            constraint.rest_area = data.original_areas[i];
        }
    }

    // === Queries ===

    /// Get the lowest Y coordinate of a body (for ground detection)
    pub fn get_lowest_y(&self, handle: BodyHandle) -> Option<f32> {
        self.get_body(handle).map(|b| b.get_lowest_y())
    }

    /// Check if body is near ground (within threshold)
    pub fn is_grounded(&self, handle: BodyHandle, threshold: f32) -> bool {
        let Some(ground_y) = self.ground_y else { return false };
        let Some(lowest) = self.get_lowest_y(handle) else { return false };
        lowest < ground_y + threshold
    }

    /// Get kinetic energy of a body
    pub fn get_kinetic_energy(&self, handle: BodyHandle) -> Option<f32> {
        self.get_body(handle).map(|b| b.get_kinetic_energy())
    }

    /// Get total kinetic energy of all bodies
    pub fn total_kinetic_energy(&self) -> f32 {
        self.bodies.iter().map(|b| b.get_kinetic_energy()).sum()
    }

    /// Get AABB (min_x, min_y, max_x, max_y) for a body
    pub fn get_aabb(&self, handle: BodyHandle) -> Option<(f32, f32, f32, f32)> {
        self.get_body(handle).map(|b| b.get_aabb())
    }

    // === Simulation ===

    /// Step the simulation forward by dt seconds
    pub fn step(&mut self, dt: f32) {
        if self.bodies.is_empty() {
            return;
        }

        let substep_dt = dt / self.substeps as f32;

        // Prepare collision system
        self.collision_system.prepare(&self.bodies);

        for _ in 0..self.substeps {
            // Pre-solve all bodies with ground friction
            for body in &mut self.bodies {
                body.substep_pre_with_friction(
                    substep_dt,
                    self.gravity,
                    self.ground_y,
                    self.ground_friction,
                    self.ground_restitution,
                );
            }

            // Resolve collisions (respecting collision groups)
            self.resolve_collisions_with_groups();

            // Post-solve all bodies
            for body in &mut self.bodies {
                body.substep_post(substep_dt);
            }
        }
    }

    /// Internal: resolve collisions respecting collision groups
    fn resolve_collisions_with_groups(&mut self) {
        // For now, use the standard collision system
        // TODO: Filter by collision groups
        self.collision_system.resolve_collisions(&mut self.bodies);
    }

    /// Try to put body to sleep if resting
    pub fn sleep_if_resting(&mut self, handle: BodyHandle, threshold: f32) -> bool {
        let Some(body) = self.get_body_mut(handle) else { return false };
        body.sleep_if_resting(threshold)
    }

    // === Rendering Helpers ===

    /// Get all body positions and triangles for rendering
    pub fn get_render_data(&self) -> Vec<(&[f32], &[u32])> {
        self.bodies.iter()
            .zip(self.triangles.iter())
            .map(|(body, tris)| (body.pos.as_slice(), tris.as_slice()))
            .collect()
    }

    /// Get render data for a specific body
    pub fn get_body_render_data(&self, handle: BodyHandle) -> Option<(&[f32], &[u32])> {
        let index = self.handle_to_index.get(handle.0)?.as_ref()?;
        let body = self.bodies.get(*index)?;
        let tris = self.triangles.get(*index)?;
        Some((body.pos.as_slice(), tris.as_slice()))
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::create_ring_mesh;

    #[test]
    fn test_add_remove_body() {
        let mut world = PhysicsWorld::new();
        let mesh = create_ring_mesh(1.0, 0.5, 8, 2);

        let h1 = world.add_body_simple(&mesh, 0.0, 0.0);
        let h2 = world.add_body_simple(&mesh, 2.0, 0.0);

        assert_eq!(world.body_count(), 2);
        assert!(world.contains(h1));
        assert!(world.contains(h2));

        world.remove_body(h1);

        assert_eq!(world.body_count(), 1);
        assert!(!world.contains(h1));
        assert!(world.contains(h2));
    }

    #[test]
    fn test_collision_groups() {
        let group_a = CollisionGroups::new(0b0001, 0b0010);
        let group_b = CollisionGroups::new(0b0010, 0b0001);
        let group_c = CollisionGroups::new(0b0100, 0b0100);

        assert!(group_a.can_collide(&group_b));
        assert!(group_b.can_collide(&group_a));
        assert!(!group_a.can_collide(&group_c));
        assert!(!group_c.can_collide(&group_a));
    }

    #[test]
    fn test_apply_impulse() {
        let mut world = PhysicsWorld::new();
        let mesh = create_ring_mesh(1.0, 0.5, 8, 2);
        let handle = world.add_body_simple(&mesh, 0.0, 0.0);

        world.apply_impulse(handle, 5.0, 10.0);

        let vel = world.get_velocity(handle).unwrap();
        assert!((vel.0 - 5.0).abs() < 0.01);
        assert!((vel.1 - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_position() {
        let mut world = PhysicsWorld::new();
        let mesh = create_ring_mesh(1.0, 0.5, 8, 2);
        let handle = world.add_body(&mesh, BodyConfig::new().at_position(3.0, 4.0));

        let pos = world.get_position(handle).unwrap();
        assert!((pos.0 - 3.0).abs() < 0.01);
        assert!((pos.1 - 4.0).abs() < 0.01);

        world.set_position(handle, 10.0, 20.0);
        let pos = world.get_position(handle).unwrap();
        assert!((pos.0 - 10.0).abs() < 0.01);
        assert!((pos.1 - 20.0).abs() < 0.01);
    }
}
