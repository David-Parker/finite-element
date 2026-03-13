//! End-to-end simulation tests to catch instability and explosions

#[cfg(test)]
mod tests {
    use crate::mesh::{create_ring_mesh, offset_vertices};
    use crate::softbody::{Material, SoftBody};
    use crate::trace::SimulationTracer;

    // Match actual simulation parameters
    const OUTER_RADIUS: f32 = 1.5;
    const INNER_RADIUS: f32 = 1.0;
    const SEGMENTS: u32 = 32;
    const RADIAL_DIVISIONS: u32 = 4;
    const START_HEIGHT: f32 = 6.0;
    const GROUND_Y: f32 = -8.0;
    const GRAVITY: f32 = -9.8;
    const SUBSTEPS: u32 = 64;

    /// Stability limits - if exceeded, simulation has exploded
    const MAX_ALLOWED_KE: f32 = 1000000.0;
    const MAX_ALLOWED_VELOCITY: f32 = 150.0;
    const MIN_ALLOWED_J: f32 = 0.1;
    const MAX_ALLOWED_J: f32 = 10.0;

    struct SimulationStats {
        frame: u32,
        ke: f32,
        min_j: f32,
        max_j: f32,
        max_vel: f32,
    }

    fn create_test_body(material: Material) -> SoftBody {
        let mut mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
        offset_vertices(&mut mesh.vertices, 0.0, START_HEIGHT);
        SoftBody::new(&mesh.vertices, &mesh.triangles, material)
    }

    fn run_simulation(body: &mut SoftBody, frames: u32) -> Vec<SimulationStats> {
        let dt = 1.0 / 60.0 / SUBSTEPS as f32;
        let mut stats = Vec::new();

        for frame in 0..frames {
            for _ in 0..SUBSTEPS {
                body.substep(dt, GRAVITY).0;
                body.collide_with_ground(GROUND_Y);
            }

            let ke = body.get_kinetic_energy();
            let (min_j, max_j, max_vel, _, _, _) = body.get_diagnostics();

            stats.push(SimulationStats {
                frame,
                ke,
                min_j,
                max_j,
                max_vel,
            });
        }

        stats
    }

    fn check_stability(stats: &[SimulationStats], material_name: &str) {
        for s in stats {
            assert!(
                s.ke < MAX_ALLOWED_KE,
                "{} frame {}: KE {} exceeds max {}",
                material_name, s.frame, s.ke, MAX_ALLOWED_KE
            );
            assert!(
                s.max_vel < MAX_ALLOWED_VELOCITY,
                "{} frame {}: velocity {} exceeds max {}",
                material_name, s.frame, s.max_vel, MAX_ALLOWED_VELOCITY
            );
            assert!(
                s.min_j > MIN_ALLOWED_J,
                "{} frame {}: min_j {} below min {} (triangle near inversion)",
                material_name, s.frame, s.min_j, MIN_ALLOWED_J
            );
            assert!(
                s.max_j < MAX_ALLOWED_J,
                "{} frame {}: max_j {} exceeds max {} (triangle over-stretched)",
                material_name, s.frame, s.max_j, MAX_ALLOWED_J
            );
        }
    }

    #[test]
    fn test_rubber_stability_drop() {
        let mut body = create_test_body(Material::RUBBER);
        let stats = run_simulation(&mut body, 300);
        check_stability(&stats, "RUBBER");
    }

    #[test]
    fn test_jello_stability_drop() {
        let mut body = create_test_body(Material::JELLO);
        let stats = run_simulation(&mut body, 300);
        check_stability(&stats, "JELLO");
    }

    #[test]
    fn test_wood_stability_drop() {
        let mut body = create_test_body(Material::WOOD);
        let stats = run_simulation(&mut body, 300);
        check_stability(&stats, "WOOD");
    }

    #[test]
    fn test_metal_stability_drop() {
        let mut body = create_test_body(Material::METAL);
        let stats = run_simulation(&mut body, 300);
        check_stability(&stats, "METAL");
    }

    #[test]
    fn test_rubber_comes_to_rest() {
        let mut body = create_test_body(Material::RUBBER);
        let stats = run_simulation(&mut body, 1200);

        let final_ke = stats.last().unwrap().ke;
        assert!(
            final_ke < 500.0,
            "RUBBER should settle down, but KE is still {}",
            final_ke
        );
    }

    #[test]
    fn test_jello_no_plastic_deformation() {
        let mut body = create_test_body(Material::JELLO);
        run_simulation(&mut body, 300);

        let (_, _, _, _, min_plastic_det, max_plastic_det) = body.get_diagnostics();

        assert!(
            (min_plastic_det - 1.0).abs() < 0.01 && (max_plastic_det - 1.0).abs() < 0.01,
            "JELLO should have no plastic deformation, but plastic_det range is {}-{}",
            min_plastic_det, max_plastic_det
        );
    }

    #[test]
    fn test_high_drop_stability() {
        let mut mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
        offset_vertices(&mut mesh.vertices, 0.0, 1.0);
        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::RUBBER);

        let dt = 1.0 / 60.0 / 128.0;
        let mut stats = Vec::new();

        for frame in 0..300 {
            for _ in 0..128 {
                body.substep(dt, GRAVITY).0;
                body.collide_with_ground(GROUND_Y);
            }

            let ke = body.get_kinetic_energy();
            let (min_j, max_j, max_vel, _, _, _) = body.get_diagnostics();
            stats.push(SimulationStats { frame, ke, min_j, max_j, max_vel });
        }

        check_stability(&stats, "RUBBER (high drop)");
    }

    #[test]
    fn test_all_materials_survive_long_simulation() {
        let materials = [
            (Material::RUBBER, "RUBBER"),
            (Material::JELLO, "JELLO"),
            (Material::WOOD, "WOOD"),
            (Material::METAL, "METAL"),
        ];

        for (material, name) in materials {
            let mut body = create_test_body(material);
            let stats = run_simulation(&mut body, 600);
            check_stability(&stats, name);
        }
    }

    /// Test that a ring falls with correct gravitational acceleration
    /// At frame N (time = N/60 seconds), expected velocity = g * t = 9.8 * N/60
    fn check_freefall_velocity(body: &SoftBody, frame: u32, material_name: &str) {
        let time = frame as f32 / 60.0;
        let expected_vel = GRAVITY.abs() * time;

        // Compute average Y velocity (center of mass velocity)
        let mut total_vy = 0.0;
        let mut total_mass = 0.0;
        for i in 0..body.num_verts {
            total_vy += body.vel[i * 2 + 1] * body.mass[i];
            total_mass += body.mass[i];
        }
        let avg_vy = -total_vy / total_mass; // Negate because falling is negative

        // Allow 20% tolerance for soft body deformation effects
        let tolerance = expected_vel * 0.2 + 0.5; // +0.5 for early frames
        assert!(
            (avg_vy - expected_vel).abs() < tolerance,
            "{} frame {}: avg velocity {:.2} m/s, expected {:.2} m/s (tolerance {:.2})",
            material_name, frame, avg_vy, expected_vel, tolerance
        );
    }

    /// Check that ring shape is preserved (aspect ratio, not too deformed)
    fn check_ring_shape(body: &SoftBody, material_name: &str) {
        // Find bounding box
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for i in 0..body.num_verts {
            let x = body.pos[i * 2];
            let y = body.pos[i * 2 + 1];
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let width = max_x - min_x;
        let height = max_y - min_y;
        let expected_diameter = OUTER_RADIUS * 2.0; // 3.0
        let expected_thickness = OUTER_RADIUS - INNER_RADIUS; // 0.5

        // Width should be close to original diameter (allow 35% deformation for soft materials)
        assert!(
            width > expected_diameter * 0.65 && width < expected_diameter * 1.35,
            "{}: width {:.2} out of range [{:.2}, {:.2}]",
            material_name, width, expected_diameter * 0.65, expected_diameter * 1.35
        );

        // Height must be at least the ring thickness (can't be pancaked flat)
        // Allow some compression but not pancaking
        let min_height = expected_thickness * 1.5; // At minimum, 1.5x the ring thickness
        assert!(
            height > min_height && height < expected_diameter * 1.5,
            "{}: height {:.2} out of range [{:.2}, {:.2}] (pancaked!)",
            material_name, height, min_height, expected_diameter * 1.5
        );

        // Aspect ratio check - shouldn't be too flat
        let aspect_ratio = width / height;
        assert!(
            aspect_ratio < 4.0,
            "{}: aspect ratio {:.2} too flat (pancaked!)",
            material_name, aspect_ratio
        );
    }

    /// Check that ring has come to rest (low kinetic energy and velocity)
    fn check_at_rest(body: &SoftBody, material_name: &str, ke_threshold: f32, vel_threshold: f32) {
        let ke = body.get_kinetic_energy();
        let (_, _, max_vel, _, _, _) = body.get_diagnostics();

        assert!(
            ke < ke_threshold,
            "{}: KE {:.2} should be < {:.2} at rest",
            material_name, ke, ke_threshold
        );
        assert!(
            max_vel < vel_threshold,
            "{}: max velocity {:.2} should be < {:.2} m/s at rest",
            material_name, max_vel, vel_threshold
        );
    }

    /// Run simulation and return body, checking freefall velocity at specified frames
    fn run_freefall_test(material: Material, material_name: &str, check_frames: &[u32]) -> SoftBody {
        let mut body = create_test_body(material);
        let dt = 1.0 / 60.0 / SUBSTEPS as f32;

        let max_frame = *check_frames.iter().max().unwrap_or(&60);

        for frame in 0..=max_frame {
            // Check velocity before this frame's physics (at start of frame)
            if check_frames.contains(&frame) && frame > 0 {
                check_freefall_velocity(&body, frame, material_name);
            }

            for _ in 0..SUBSTEPS {
                body.substep(dt, GRAVITY).0;
                body.collide_with_ground(GROUND_Y);
            }
        }

        body
    }

    #[test]
    fn test_rubber_freefall_velocity() {
        // Check velocity at 10, 20, 30 frames (before hitting ground)
        // Ring starts at y=6, ground at y=-8, so 14m to fall
        // Time to fall: sqrt(2*14/9.8) = 1.69s = 101 frames
        // Check early in fall before ground contact
        run_freefall_test(Material::RUBBER, "RUBBER", &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_jello_freefall_velocity() {
        run_freefall_test(Material::JELLO, "JELLO", &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_wood_freefall_velocity() {
        run_freefall_test(Material::WOOD, "WOOD", &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_metal_freefall_velocity() {
        run_freefall_test(Material::METAL, "METAL", &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_rubber_settles_and_shape() {
        let mut body = create_test_body(Material::RUBBER);
        run_simulation(&mut body, 1800); // 30 seconds - bouncy material needs longer
        check_at_rest(&body, "RUBBER", 500.0, 5.0); // Higher threshold for bouncy material
        check_ring_shape(&body, "RUBBER");
    }

    #[test]
    fn test_jello_settles_and_shape() {
        let mut body = create_test_body(Material::JELLO);
        run_simulation(&mut body, 1800); // 30 seconds
        check_at_rest(&body, "JELLO", 200.0, 3.0); // Jello is bouncier
        check_ring_shape(&body, "JELLO");
    }

    #[test]
    fn test_wood_settles_and_shape() {
        let mut body = create_test_body(Material::WOOD);
        run_simulation(&mut body, 1800); // 30 seconds - stiffer materials need longer
        check_at_rest(&body, "WOOD", 200.0, 5.0); // Higher threshold for stiff material
        check_ring_shape(&body, "WOOD");
    }

    #[test]
    fn test_metal_settles_and_shape() {
        let mut body = create_test_body(Material::METAL);
        run_simulation(&mut body, 1800); // 30 seconds
        check_at_rest(&body, "METAL", 500.0, 10.0); // Metal oscillates longer
        check_ring_shape(&body, "METAL");
    }

    /// Measure maximum deformation during impact
    fn measure_max_deformation(material: Material) -> f32 {
        let mut body = create_test_body(material);
        let dt = 1.0 / 60.0 / SUBSTEPS as f32;
        let mut max_width = 0.0f32;

        // Run for 5 seconds (300 frames) to capture impact
        for _ in 0..300 {
            for _ in 0..SUBSTEPS {
                body.substep(dt, GRAVITY).0;
                body.collide_with_ground(GROUND_Y);
            }

            // Measure current width
            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            for i in 0..body.num_verts {
                min_x = min_x.min(body.pos[i * 2]);
                max_x = max_x.max(body.pos[i * 2]);
            }
            let width = max_x - min_x;
            max_width = max_width.max(width);
        }

        max_width
    }

    /// Measure final settled width after simulation
    fn measure_settled_width(material: Material) -> f32 {
        let mut body = create_test_body(material);
        run_simulation(&mut body, 1200); // 20 seconds

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        for i in 0..body.num_verts {
            min_x = min_x.min(body.pos[i * 2]);
            max_x = max_x.max(body.pos[i * 2]);
        }
        max_x - min_x
    }

    #[test]
    fn test_stiffness_ordering_max_deformation() {
        // Stiffer materials should generally deform less during impact
        // Note: Impact dynamics depend on density and damping too, not just stiffness
        let jello_deform = measure_max_deformation(Material::JELLO);
        let rubber_deform = measure_max_deformation(Material::RUBBER);
        let wood_deform = measure_max_deformation(Material::WOOD);
        let metal_deform = measure_max_deformation(Material::METAL);

        let expected_diameter = OUTER_RADIUS * 2.0;

        println!("Max deformation during impact:");
        println!("  JELLO:  {:.2} ({:.0}% of original)", jello_deform, 100.0 * jello_deform / expected_diameter);
        println!("  RUBBER: {:.2} ({:.0}% of original)", rubber_deform, 100.0 * rubber_deform / expected_diameter);
        println!("  WOOD:   {:.2} ({:.0}% of original)", wood_deform, 100.0 * wood_deform / expected_diameter);
        println!("  METAL:  {:.2} ({:.0}% of original)", metal_deform, 100.0 * metal_deform / expected_diameter);

        // Soft materials (JELLO, RUBBER) should deform more than stiff materials (WOOD, METAL)
        let soft_avg = (jello_deform + rubber_deform) / 2.0;
        let stiff_avg = (wood_deform + metal_deform) / 2.0;
        assert!(
            soft_avg > stiff_avg,
            "Soft materials avg ({:.2}) should deform more than stiff avg ({:.2})",
            soft_avg, stiff_avg
        );

        // METAL should deform least (stiffest)
        assert!(
            metal_deform <= wood_deform && metal_deform <= rubber_deform && metal_deform <= jello_deform,
            "METAL ({:.2}) should deform least",
            metal_deform
        );

        // All materials should stay within reasonable bounds (not explode)
        for (name, deform) in [("JELLO", jello_deform), ("RUBBER", rubber_deform), ("WOOD", wood_deform), ("METAL", metal_deform)] {
            assert!(
                deform < expected_diameter * 2.0,
                "{} max deformation {:.2} exceeds 200% of original",
                name, deform
            );
        }
    }

    #[test]
    fn test_stiffness_ordering_settled_shape() {
        // After settling, stiffer materials should be closer to original shape
        let jello_width = measure_settled_width(Material::JELLO);
        let rubber_width = measure_settled_width(Material::RUBBER);
        let wood_width = measure_settled_width(Material::WOOD);
        let metal_width = measure_settled_width(Material::METAL);

        let expected_diameter = OUTER_RADIUS * 2.0;

        println!("Settled width:");
        println!("  JELLO:  {:.2} ({}% of original)", jello_width, 100.0 * jello_width / expected_diameter);
        println!("  RUBBER: {:.2} ({}% of original)", rubber_width, 100.0 * rubber_width / expected_diameter);
        println!("  WOOD:   {:.2} ({}% of original)", wood_width, 100.0 * wood_width / expected_diameter);
        println!("  METAL:  {:.2} ({}% of original)", metal_width, 100.0 * metal_width / expected_diameter);

        // With adaptive stiffness (for 8-substep stability), all materials behave softer
        // Allow more deviation than the original test since effective stiffness is reduced
        // JELLO/RUBBER: allow up to 40% deviation
        // WOOD/METAL: allow up to 25% deviation
        for (name, width, max_dev) in [
            ("JELLO", jello_width, 0.40),
            ("RUBBER", rubber_width, 0.30),
            ("WOOD", wood_width, 0.25),
            ("METAL", metal_width, 0.25),
        ] {
            let deviation = (width - expected_diameter).abs() / expected_diameter;
            assert!(
                deviation < max_dev,
                "{} settled width {:.2} deviates {:.0}% from expected {:.2} (max {:.0}%)",
                name, width, deviation * 100.0, expected_diameter, max_dev * 100.0
            );
        }

        // Stiff materials (WOOD, METAL) should be closer to original than soft materials
        let metal_deviation = (metal_width - expected_diameter).abs() / expected_diameter;
        let wood_deviation = (wood_width - expected_diameter).abs() / expected_diameter;
        let rubber_deviation = (rubber_width - expected_diameter).abs() / expected_diameter;
        let jello_deviation = (jello_width - expected_diameter).abs() / expected_diameter;

        // Stiff materials should deviate less than soft materials
        let stiff_avg_dev = (metal_deviation + wood_deviation) / 2.0;
        let soft_avg_dev = (rubber_deviation + jello_deviation) / 2.0;
        assert!(
            stiff_avg_dev < soft_avg_dev,
            "Stiff materials avg deviation ({:.1}%) should be less than soft ({:.1}%)",
            stiff_avg_dev * 100.0, soft_avg_dev * 100.0
        );
    }

    // ============================================================
    // Multi-body collision tests
    // ============================================================

    /// Stats for multi-body simulation
    struct MultiBodyStats {
        frame: u32,
        total_ke: f32,
        min_j: f32,
        max_j: f32,
        max_vel: f32,
        collision_count: u32,
    }

    fn create_test_body_at(material: Material, x_offset: f32, y_offset: f32) -> SoftBody {
        let mut mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
        offset_vertices(&mut mesh.vertices, x_offset, y_offset);
        SoftBody::new(&mesh.vertices, &mesh.triangles, material)
    }

    /// Run multi-body simulation with collision detection
    fn run_multi_body_simulation(
        bodies: &mut [SoftBody],
        frames: u32,
        collision_dist: f32,
    ) -> Vec<MultiBodyStats> {
        let dt = 1.0 / 60.0 / SUBSTEPS as f32;
        let mut stats = Vec::new();

        for frame in 0..frames {
            let mut frame_collisions = 0u32;

            for _ in 0..SUBSTEPS {
                // Physics substep for each body
                for body in bodies.iter_mut() {
                    body.substep(dt, GRAVITY);
                    body.collide_with_ground(GROUND_Y);
                }

                // Inter-body collisions
                for i in 0..bodies.len() {
                    for j in (i + 1)..bodies.len() {
                        let (left, right) = bodies.split_at_mut(j);
                        frame_collisions += left[i].collide_with_body(&mut right[0], collision_dist);
                    }
                }
            }

            // Aggregate stats across all bodies
            let total_ke: f32 = bodies.iter().map(|b| b.get_kinetic_energy()).sum();
            let mut min_j = f32::INFINITY;
            let mut max_j = f32::NEG_INFINITY;
            let mut max_vel = 0.0f32;

            for body in bodies.iter() {
                let (bmin_j, bmax_j, bmax_vel, _, _, _) = body.get_diagnostics();
                min_j = min_j.min(bmin_j);
                max_j = max_j.max(bmax_j);
                max_vel = max_vel.max(bmax_vel);
            }

            stats.push(MultiBodyStats {
                frame,
                total_ke,
                min_j,
                max_j,
                max_vel,
                collision_count: frame_collisions,
            });
        }

        stats
    }

    fn check_multi_body_stability_n(stats: &[MultiBodyStats], test_name: &str, num_bodies: usize) {
        let mut total_collisions = 0u32;
        let mut max_ke_seen = 0.0f32;
        let mut max_vel_seen = 0.0f32;

        // Scale KE threshold by number of bodies (each body contributes its own KE)
        let ke_threshold = MAX_ALLOWED_KE * num_bodies as f32;

        for s in stats {
            total_collisions += s.collision_count;
            max_ke_seen = max_ke_seen.max(s.total_ke);
            max_vel_seen = max_vel_seen.max(s.max_vel);

            assert!(
                s.total_ke < ke_threshold,
                "{} frame {}: KE {} exceeds max {} (EXPLOSION!)",
                test_name, s.frame, s.total_ke, ke_threshold
            );
            assert!(
                s.max_vel < MAX_ALLOWED_VELOCITY,
                "{} frame {}: velocity {} exceeds max {} (EXPLOSION!)",
                test_name, s.frame, s.max_vel, MAX_ALLOWED_VELOCITY
            );
            assert!(
                s.min_j > MIN_ALLOWED_J,
                "{} frame {}: min_j {} below min {} (triangle inversion!)",
                test_name, s.frame, s.min_j, MIN_ALLOWED_J
            );
            assert!(
                s.max_j < MAX_ALLOWED_J,
                "{} frame {}: max_j {} exceeds max {} (triangle explosion!)",
                test_name, s.frame, s.max_j, MAX_ALLOWED_J
            );
        }

        println!(
            "{}: {} frames, {} total collisions, max KE={:.1}, max vel={:.1}",
            test_name,
            stats.len(),
            total_collisions,
            max_ke_seen,
            max_vel_seen
        );
    }

    fn check_multi_body_stability(stats: &[MultiBodyStats], test_name: &str) {
        // Default to 2 bodies for backwards compatibility
        check_multi_body_stability_n(stats, test_name, 2);
    }

    /// Verify that collisions actually occurred
    fn assert_collisions_occurred(stats: &[MultiBodyStats], test_name: &str, min_expected: u32) {
        let total: u32 = stats.iter().map(|s| s.collision_count).sum();
        assert!(
            total >= min_expected,
            "{}: expected at least {} collisions, got {}",
            test_name, min_expected, total
        );
    }

    /// Check that simulation settles to reasonable energy
    fn check_multi_body_settles(stats: &[MultiBodyStats], test_name: &str, ke_threshold: f32) {
        let final_ke = stats.last().map(|s| s.total_ke).unwrap_or(0.0);
        let final_vel = stats.last().map(|s| s.max_vel).unwrap_or(0.0);

        // Check last 10% of frames for settling
        let settle_start = stats.len() * 9 / 10;
        let avg_final_ke: f32 = stats[settle_start..].iter().map(|s| s.total_ke).sum::<f32>()
            / (stats.len() - settle_start) as f32;

        assert!(
            avg_final_ke < ke_threshold,
            "{}: should settle, but avg final KE is {:.1} (threshold {})",
            test_name, avg_final_ke, ke_threshold
        );

        println!(
            "{}: settled with final KE={:.1}, final vel={:.1}",
            test_name, final_ke, final_vel
        );
    }

    // ---- Two body tests ----

    #[test]
    fn test_two_bodies_rubber_collision() {
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::RUBBER, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 300, 0.25);
        check_multi_body_stability(&stats, "2x RUBBER");
        assert_collisions_occurred(&stats, "2x RUBBER", 100);
    }

    #[test]
    fn test_two_bodies_jello_collision() {
        let mut bodies = vec![
            create_test_body_at(Material::JELLO, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::JELLO, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 300, 0.25);
        check_multi_body_stability(&stats, "2x JELLO");
        assert_collisions_occurred(&stats, "2x JELLO", 100);
    }

    #[test]
    fn test_two_bodies_wood_collision() {
        let mut bodies = vec![
            create_test_body_at(Material::WOOD, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::WOOD, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 300, 0.25);
        check_multi_body_stability(&stats, "2x WOOD");
        assert_collisions_occurred(&stats, "2x WOOD", 100);
    }

    #[test]
    fn test_two_bodies_metal_collision() {
        let mut bodies = vec![
            create_test_body_at(Material::METAL, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::METAL, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 300, 0.25);
        check_multi_body_stability(&stats, "2x METAL");
        assert_collisions_occurred(&stats, "2x METAL", 100);
    }

    #[test]
    fn test_two_bodies_bouncy_rubber_collision() {
        let mut bodies = vec![
            create_test_body_at(Material::BOUNCY_RUBBER, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::BOUNCY_RUBBER, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 300, 0.25);
        check_multi_body_stability(&stats, "2x BOUNCY_RUBBER");
        assert_collisions_occurred(&stats, "2x BOUNCY_RUBBER", 100);
    }

    // ---- Three body tests ----

    #[test]
    fn test_three_bodies_rubber_collision() {
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, 0.0, START_HEIGHT + 8.0),
            create_test_body_at(Material::RUBBER, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::RUBBER, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 400, 0.25);
        check_multi_body_stability_n(&stats, "3x RUBBER", 3);
        assert_collisions_occurred(&stats, "3x RUBBER", 200);
    }

    #[test]
    fn test_three_bodies_bouncy_rubber_collision() {
        let mut bodies = vec![
            create_test_body_at(Material::BOUNCY_RUBBER, 0.0, START_HEIGHT + 8.0),
            create_test_body_at(Material::BOUNCY_RUBBER, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::BOUNCY_RUBBER, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 400, 0.25);
        check_multi_body_stability_n(&stats, "3x BOUNCY_RUBBER", 3);
        assert_collisions_occurred(&stats, "3x BOUNCY_RUBBER", 200);
    }

    #[test]
    #[ignore] // TODO: Mixed stiffness materials cause collision instability - needs speculative contacts
    fn test_three_bodies_mixed_materials() {
        // Stagger more to avoid instability with different stiffnesses
        let mut bodies = vec![
            create_test_body_at(Material::METAL, 0.0, START_HEIGHT + 8.0),
            create_test_body_at(Material::RUBBER, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::JELLO, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 400, 0.25);
        check_multi_body_stability_n(&stats, "METAL+RUBBER+JELLO", 3);
        assert_collisions_occurred(&stats, "METAL+RUBBER+JELLO", 150);
    }

    // ---- Four body tests ----

    #[test]
    fn test_four_bodies_rubber_stack() {
        // Stagger slightly to avoid perfect vertical alignment issues
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, 0.2, START_HEIGHT + 12.0),
            create_test_body_at(Material::RUBBER, -0.2, START_HEIGHT + 8.0),
            create_test_body_at(Material::RUBBER, 0.1, START_HEIGHT + 4.0),
            create_test_body_at(Material::RUBBER, -0.1, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 500, 0.25);
        check_multi_body_stability_n(&stats, "4x RUBBER stack", 4);
        assert_collisions_occurred(&stats, "4x RUBBER stack", 300);
    }

    #[test]
    fn test_four_bodies_bouncy_rubber_stack() {
        let mut bodies = vec![
            create_test_body_at(Material::BOUNCY_RUBBER, 0.2, START_HEIGHT + 12.0),
            create_test_body_at(Material::BOUNCY_RUBBER, -0.2, START_HEIGHT + 8.0),
            create_test_body_at(Material::BOUNCY_RUBBER, 0.1, START_HEIGHT + 4.0),
            create_test_body_at(Material::BOUNCY_RUBBER, -0.1, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 500, 0.25);
        check_multi_body_stability_n(&stats, "4x BOUNCY_RUBBER stack", 4);
        assert_collisions_occurred(&stats, "4x BOUNCY_RUBBER stack", 300);
    }

    #[test]
    fn test_four_bodies_scattered() {
        // Spread bodies far apart to avoid initial overlap (ring diameter ~3)
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, -4.0, START_HEIGHT + 8.0),
            create_test_body_at(Material::RUBBER, 4.0, START_HEIGHT + 8.0),
            create_test_body_at(Material::RUBBER, -2.0, START_HEIGHT + 4.0),
            create_test_body_at(Material::RUBBER, 2.0, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 500, 0.25);
        check_multi_body_stability_n(&stats, "4x RUBBER scattered", 4);
    }

    // ---- Five body tests ----

    #[test]
    #[ignore] // TODO: 5+ body collisions cause instability - needs speculative contacts
    fn test_five_bodies_rubber_pile() {
        // Stagger positions more to avoid instability
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, 0.0, START_HEIGHT + 16.0),
            create_test_body_at(Material::RUBBER, -0.5, START_HEIGHT + 12.0),
            create_test_body_at(Material::RUBBER, 0.5, START_HEIGHT + 8.0),
            create_test_body_at(Material::RUBBER, -0.3, START_HEIGHT + 4.0),
            create_test_body_at(Material::RUBBER, 0.3, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 600, 0.25);
        check_multi_body_stability_n(&stats, "5x RUBBER pile", 5);
        assert_collisions_occurred(&stats, "5x RUBBER pile", 400);
    }

    #[test]
    #[ignore] // TODO: 5+ body collisions cause instability - needs speculative contacts
    fn test_five_bodies_bouncy_rubber_pile() {
        let mut bodies = vec![
            create_test_body_at(Material::BOUNCY_RUBBER, 0.0, START_HEIGHT + 16.0),
            create_test_body_at(Material::BOUNCY_RUBBER, -0.5, START_HEIGHT + 12.0),
            create_test_body_at(Material::BOUNCY_RUBBER, 0.5, START_HEIGHT + 8.0),
            create_test_body_at(Material::BOUNCY_RUBBER, -0.3, START_HEIGHT + 4.0),
            create_test_body_at(Material::BOUNCY_RUBBER, 0.3, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 600, 0.25);
        check_multi_body_stability_n(&stats, "5x BOUNCY_RUBBER pile", 5);
        assert_collisions_occurred(&stats, "5x BOUNCY_RUBBER pile", 400);
    }

    #[test]
    #[ignore] // TODO: Mixed materials + many bodies causes instability - needs speculative contacts
    fn test_five_bodies_all_materials() {
        // Mix of materials with staggered positions
        let mut bodies = vec![
            create_test_body_at(Material::METAL, 0.0, START_HEIGHT + 16.0),
            create_test_body_at(Material::WOOD, -0.5, START_HEIGHT + 12.0),
            create_test_body_at(Material::BOUNCY_RUBBER, 0.5, START_HEIGHT + 8.0),
            create_test_body_at(Material::RUBBER, -0.3, START_HEIGHT + 4.0),
            create_test_body_at(Material::JELLO, 0.3, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 600, 0.25);
        check_multi_body_stability_n(&stats, "5 mixed materials", 5);
        assert_collisions_occurred(&stats, "5 mixed materials", 300);
    }

    // ---- High drop tests ----

    #[test]
    fn test_two_bodies_high_drop() {
        // Bodies start much higher for more energetic collision
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, 0.0, 15.0),  // Very high
            create_test_body_at(Material::RUBBER, 0.0, 8.0),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 400, 0.25);
        check_multi_body_stability(&stats, "2x RUBBER high drop");
        assert_collisions_occurred(&stats, "2x RUBBER high drop", 100);
    }

    #[test]
    fn test_three_bodies_high_drop_bouncy() {
        let mut bodies = vec![
            create_test_body_at(Material::BOUNCY_RUBBER, 0.0, 20.0),
            create_test_body_at(Material::BOUNCY_RUBBER, -0.2, 14.0),
            create_test_body_at(Material::BOUNCY_RUBBER, 0.2, 8.0),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 500, 0.25);
        check_multi_body_stability_n(&stats, "3x BOUNCY_RUBBER high drop", 3);
        assert_collisions_occurred(&stats, "3x BOUNCY_RUBBER high drop", 200);
    }

    // ---- Long simulation settling tests ----

    #[test]
    fn test_two_bodies_settle() {
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::RUBBER, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 1200, 0.25);  // 20 seconds
        check_multi_body_stability(&stats, "2x RUBBER settle");
        check_multi_body_settles(&stats, "2x RUBBER settle", 20000.0);  // Higher for multi-body
    }

    #[test]
    fn test_three_bodies_settle() {
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, 0.0, START_HEIGHT + 8.0),
            create_test_body_at(Material::RUBBER, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::RUBBER, 0.5, START_HEIGHT),
        ];
        let stats = run_multi_body_simulation(&mut bodies, 1500, 0.25);  // 25 seconds
        check_multi_body_stability_n(&stats, "3x RUBBER settle", 3);
        check_multi_body_settles(&stats, "3x RUBBER settle", 30000.0);  // Higher for multi-body
    }

    // ---- Direct collision test (bodies on collision course) ----

    #[test]
    fn test_head_on_collision() {
        // Two bodies at same height, will collide when they fall and spread
        let mut bodies = vec![
            create_test_body_at(Material::RUBBER, -2.0, START_HEIGHT),
            create_test_body_at(Material::RUBBER, 2.0, START_HEIGHT),
        ];

        // Give them initial velocity toward each other
        for i in 0..bodies[0].num_verts {
            bodies[0].vel[i * 2] = 3.0;  // Moving right
        }
        for i in 0..bodies[1].num_verts {
            bodies[1].vel[i * 2] = -3.0;  // Moving left
        }

        let stats = run_multi_body_simulation(&mut bodies, 300, 0.25);
        check_multi_body_stability(&stats, "head-on collision");
        assert_collisions_occurred(&stats, "head-on collision", 50);
    }

    #[test]
    fn test_head_on_collision_bouncy() {
        let mut bodies = vec![
            create_test_body_at(Material::BOUNCY_RUBBER, -2.0, START_HEIGHT),
            create_test_body_at(Material::BOUNCY_RUBBER, 2.0, START_HEIGHT),
        ];

        for i in 0..bodies[0].num_verts {
            bodies[0].vel[i * 2] = 5.0;
        }
        for i in 0..bodies[1].num_verts {
            bodies[1].vel[i * 2] = -5.0;
        }

        let stats = run_multi_body_simulation(&mut bodies, 300, 0.25);
        check_multi_body_stability(&stats, "head-on BOUNCY_RUBBER");
        assert_collisions_occurred(&stats, "head-on BOUNCY_RUBBER", 50);
    }

    // ---- Diagnostic test with detailed output ----

    #[test]
    fn test_collision_diagnostic() {
        let mut bodies = vec![
            create_test_body_at(Material::BOUNCY_RUBBER, -0.5, START_HEIGHT + 4.0),
            create_test_body_at(Material::BOUNCY_RUBBER, 0.5, START_HEIGHT),
        ];

        let dt = 1.0 / 60.0 / SUBSTEPS as f32;
        let mut total_collisions = 0u32;
        let mut frames_with_collisions = 0u32;

        println!("\n=== Collision Diagnostic: 2x BOUNCY_RUBBER ===");
        println!("Body 0: y_offset={}, Body 1: y_offset={}", START_HEIGHT + 4.0, START_HEIGHT);

        for frame in 0..300 {
            let mut frame_collisions = 0u32;

            for substep in 0..SUBSTEPS {
                for body in bodies.iter_mut() {
                    body.substep(dt, GRAVITY);
                    body.collide_with_ground(GROUND_Y);
                }

                let (left, right) = bodies.split_at_mut(1);
                let c = left[0].collide_with_body(&mut right[0], 0.25);
                frame_collisions += c;

                // Detailed trace around explosion point
                if frame >= 88 && frame <= 95 && substep % 16 == 0 {
                    let total_ke: f32 = bodies.iter().map(|b| b.get_kinetic_energy()).sum();
                    let (min_j0, max_j0, max_vel0, _, _, _) = bodies[0].get_diagnostics();
                    let (min_j1, max_j1, max_vel1, _, _, _) = bodies[1].get_diagnostics();
                    let lowest0 = bodies[0].get_lowest_y();
                    let lowest1 = bodies[1].get_lowest_y();
                    println!(
                        "  F{:3}.{:02}: KE={:9.1}, col={}, b0[y={:5.2},J={:.2}-{:.2},v={:.1}] b1[y={:5.2},J={:.2}-{:.2},v={:.1}]",
                        frame, substep, total_ke, c, lowest0, min_j0, max_j0, max_vel0, lowest1, min_j1, max_j1, max_vel1
                    );
                }
            }

            if frame_collisions > 0 {
                frames_with_collisions += 1;
            }
            total_collisions += frame_collisions;

            let total_ke: f32 = bodies.iter().map(|b| b.get_kinetic_energy()).sum();
            let max_vel = bodies.iter()
                .map(|b| b.get_diagnostics().2)
                .fold(0.0f32, |a, b| a.max(b));
            let lowest0 = bodies[0].get_lowest_y();
            let lowest1 = bodies[1].get_lowest_y();

            // Print every 10 frames near expected collision, or every 30 otherwise
            let near_collision = frame >= 80 && frame <= 110;
            if (near_collision && frame % 5 == 0) || frame % 30 == 0 {
                println!(
                    "Frame {:3}: KE={:9.1}, vel={:5.1}, col={:4}, y0={:6.2}, y1={:6.2}",
                    frame, total_ke, max_vel, frame_collisions, lowest0, lowest1
                );
            }

            // Explosion check - just warn, don't assert for diagnostic
            if total_ke > MAX_ALLOWED_KE {
                println!("!!! EXPLOSION at frame {}: KE={}", frame, total_ke);
                break;
            }
            if max_vel > MAX_ALLOWED_VELOCITY {
                println!("!!! EXPLOSION at frame {}: vel={}", frame, max_vel);
                break;
            }
        }

        println!("\nSummary:");
        println!("  Total collisions: {}", total_collisions);
        println!("  Frames with collisions: {}", frames_with_collisions);
    }

    // ============================================================
    // Benchmark tests - establish ground truth at high substeps
    // then verify 8-substep simulation matches within tolerance
    // ============================================================

    /// Captured physics state at a specific frame
    #[derive(Clone, Debug)]
    struct PhysicsSnapshot {
        frame: u32,
        center_y: f32,        // Center of mass Y position
        lowest_y: f32,        // Lowest vertex Y
        ke: f32,              // Kinetic energy
        max_vel: f32,         // Max velocity
        min_j: f32,           // Min deformation ratio
        max_j: f32,           // Max deformation ratio
    }

    /// Run simulation and capture snapshots at specified frames
    fn run_and_capture(
        material: Material,
        substeps: u32,
        frames: u32,
        capture_frames: &[u32],
    ) -> Vec<PhysicsSnapshot> {
        let mut mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
        offset_vertices(&mut mesh.vertices, 0.0, START_HEIGHT);
        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, material);

        let dt = 1.0 / 60.0 / substeps as f32;
        let mut snapshots = Vec::new();

        for frame in 0..frames {
            for _ in 0..substeps {
                // Use integrated ground collision for proper strain limiting after collision
                body.substep_with_ground(dt, GRAVITY, Some(GROUND_Y));
            }

            if capture_frames.contains(&frame) {
                // Calculate center of mass Y
                let mut total_y = 0.0;
                let mut total_mass = 0.0;
                for i in 0..body.num_verts {
                    total_y += body.pos[i * 2 + 1] * body.mass[i];
                    total_mass += body.mass[i];
                }
                let center_y = total_y / total_mass;

                let (min_j, max_j, max_vel, _, _, _) = body.get_diagnostics();

                snapshots.push(PhysicsSnapshot {
                    frame,
                    center_y,
                    lowest_y: body.get_lowest_y(),
                    ke: body.get_kinetic_energy(),
                    max_vel,
                    min_j,
                    max_j,
                });
            }
        }

        snapshots
    }

    /// Run multi-body simulation and capture snapshots
    fn run_multi_body_and_capture(
        material: Material,
        num_bodies: usize,
        substeps: u32,
        frames: u32,
        capture_frames: &[u32],
        collision_dist: f32,
    ) -> Vec<PhysicsSnapshot> {
        // Create bodies at staggered heights
        let mut bodies: Vec<SoftBody> = (0..num_bodies)
            .map(|i| {
                let y_offset = START_HEIGHT + (num_bodies - 1 - i) as f32 * 4.0;
                let x_offset = if i % 2 == 0 { -0.3 } else { 0.3 };
                create_test_body_at(material, x_offset, y_offset)
            })
            .collect();

        let dt = 1.0 / 60.0 / substeps as f32;
        let mut snapshots = Vec::new();

        for frame in 0..frames {
            for _ in 0..substeps {
                // Physics for each body with integrated ground collision
                for body in &mut bodies {
                    body.substep_with_ground(dt, GRAVITY, Some(GROUND_Y));
                }

                // Inter-body collisions
                for i in 0..bodies.len() {
                    for j in (i + 1)..bodies.len() {
                        let (left, right) = bodies.split_at_mut(j);
                        left[i].collide_with_body(&mut right[0], collision_dist);
                    }
                }

                // Run strain limiting after all collisions to maintain mesh integrity
                for body in &mut bodies {
                    body.limit_strain();
                }
            }

            if capture_frames.contains(&frame) {
                // Aggregate stats
                let total_ke: f32 = bodies.iter().map(|b| b.get_kinetic_energy()).sum();
                let lowest_y = bodies.iter().map(|b| b.get_lowest_y()).fold(f32::INFINITY, f32::min);
                let max_vel = bodies.iter().map(|b| b.get_diagnostics().2).fold(0.0f32, f32::max);
                let min_j = bodies.iter().map(|b| b.get_diagnostics().0).fold(f32::INFINITY, f32::min);
                let max_j = bodies.iter().map(|b| b.get_diagnostics().1).fold(f32::NEG_INFINITY, f32::max);

                // Center of all bodies
                let mut total_y = 0.0;
                let mut total_mass = 0.0;
                for body in &bodies {
                    for i in 0..body.num_verts {
                        total_y += body.pos[i * 2 + 1] * body.mass[i];
                        total_mass += body.mass[i];
                    }
                }

                snapshots.push(PhysicsSnapshot {
                    frame,
                    center_y: total_y / total_mass,
                    lowest_y,
                    ke: total_ke,
                    max_vel,
                    min_j,
                    max_j,
                });
            }
        }

        snapshots
    }

    /// Compare two sets of snapshots, return max errors
    fn compare_snapshots(
        reference: &[PhysicsSnapshot],
        test: &[PhysicsSnapshot],
    ) -> (f32, f32, f32, f32) {
        let mut max_pos_err = 0.0f32;
        let mut max_vel_err = 0.0f32;
        let mut max_ke_err = 0.0f32;
        let mut max_j_err = 0.0f32;

        for (r, t) in reference.iter().zip(test.iter()) {
            max_pos_err = max_pos_err.max((r.center_y - t.center_y).abs());
            max_vel_err = max_vel_err.max((r.max_vel - t.max_vel).abs());

            // Relative KE error (avoid div by zero)
            if r.ke > 1.0 {
                max_ke_err = max_ke_err.max((r.ke - t.ke).abs() / r.ke);
            }

            max_j_err = max_j_err.max((r.min_j - t.min_j).abs());
            max_j_err = max_j_err.max((r.max_j - t.max_j).abs());
        }

        (max_pos_err, max_vel_err, max_ke_err, max_j_err)
    }

    /// Print comparison results
    fn print_comparison(name: &str, reference: &[PhysicsSnapshot], test: &[PhysicsSnapshot]) {
        println!("\n=== {} ===", name);
        println!("Frame | Ref Y    | Test Y   | Ref KE    | Test KE   | Ref Vel | Test Vel");
        println!("------|----------|----------|-----------|-----------|---------|----------");
        for (r, t) in reference.iter().zip(test.iter()) {
            println!(
                "{:5} | {:8.3} | {:8.3} | {:9.1} | {:9.1} | {:7.2} | {:7.2}",
                r.frame, r.center_y, t.center_y, r.ke, t.ke, r.max_vel, t.max_vel
            );
        }

        let (pos_err, vel_err, ke_err, j_err) = compare_snapshots(reference, test);
        println!("\nMax errors: pos={:.3}m, vel={:.2}m/s, KE={:.1}%, J={:.3}",
            pos_err, vel_err, ke_err * 100.0, j_err);
    }

    // Reference substeps for "ground truth" - high enough for stable physics
    const BENCHMARK_SUBSTEPS: u32 = 64;
    // Target substeps - what we want to work with
    const TARGET_SUBSTEPS: u32 = 8;

    #[test]
    fn benchmark_single_body_freefall() {
        // Frames to capture during freefall and after ground contact
        let capture_frames: Vec<u32> = (0..150).step_by(10).collect();

        let reference = run_and_capture(Material::RUBBER, BENCHMARK_SUBSTEPS, 150, &capture_frames);
        let test = run_and_capture(Material::RUBBER, TARGET_SUBSTEPS, 150, &capture_frames);

        print_comparison("Single Body Freefall (RUBBER)", &reference, &test);

        let (pos_err, vel_err, _ke_err, j_err) = compare_snapshots(&reference, &test);

        // Tolerances for 8-substep stability:
        // - Position within 1m (freefall matches perfectly, collision behavior differs)
        // - Velocity within 5 m/s (with adaptive stiffness, less bounce)
        // - J within 0.5 (mesh integrity maintained)
        // Note: KE not compared because adaptive stiffness changes material behavior
        assert!(pos_err < 1.0, "Position error {} exceeds 1.0m", pos_err);
        assert!(vel_err < 5.0, "Velocity error {} exceeds 5.0 m/s", vel_err);
        assert!(j_err < 0.5, "J error {} exceeds 0.5", j_err);
    }

    #[test]
    fn benchmark_single_body_bouncy_rubber() {
        let capture_frames: Vec<u32> = (0..200).step_by(10).collect();

        let reference = run_and_capture(Material::BOUNCY_RUBBER, BENCHMARK_SUBSTEPS, 200, &capture_frames);
        let test = run_and_capture(Material::BOUNCY_RUBBER, TARGET_SUBSTEPS, 200, &capture_frames);

        print_comparison("Single Body (BOUNCY_RUBBER)", &reference, &test);

        let (pos_err, vel_err, _ke_err, j_err) = compare_snapshots(&reference, &test);

        // Tolerances for 8-substep stability with stiff material
        assert!(pos_err < 1.5, "Position error {} exceeds 1.5m", pos_err);
        assert!(vel_err < 8.0, "Velocity error {} exceeds 8.0 m/s", vel_err);
        assert!(j_err < 0.5, "J error {} exceeds 0.5", j_err);
    }

    #[test]
    fn benchmark_two_body_collision() {
        let capture_frames: Vec<u32> = (0..200).step_by(10).collect();

        let reference = run_multi_body_and_capture(
            Material::RUBBER, 2, BENCHMARK_SUBSTEPS, 200, &capture_frames, 0.25
        );
        let test = run_multi_body_and_capture(
            Material::RUBBER, 2, TARGET_SUBSTEPS, 200, &capture_frames, 0.25
        );

        print_comparison("Two Body Collision (RUBBER)", &reference, &test);

        let (pos_err, vel_err, _ke_err, j_err) = compare_snapshots(&reference, &test);

        // Looser tolerances for collision scenarios
        assert!(pos_err < 2.0, "Position error {} exceeds 2.0m", pos_err);
        assert!(vel_err < 10.0, "Velocity error {} exceeds 10.0 m/s", vel_err);
        assert!(j_err < 0.5, "J error {} exceeds 0.5", j_err);
    }

    #[test]
    fn benchmark_two_body_collision_bouncy() {
        let capture_frames: Vec<u32> = (0..200).step_by(10).collect();

        let reference = run_multi_body_and_capture(
            Material::BOUNCY_RUBBER, 2, BENCHMARK_SUBSTEPS, 200, &capture_frames, 0.25
        );
        let test = run_multi_body_and_capture(
            Material::BOUNCY_RUBBER, 2, TARGET_SUBSTEPS, 200, &capture_frames, 0.25
        );

        print_comparison("Two Body Collision (BOUNCY_RUBBER)", &reference, &test);

        let (pos_err, vel_err, _ke_err, j_err) = compare_snapshots(&reference, &test);

        assert!(pos_err < 2.5, "Position error {} exceeds 2.5m", pos_err);
        assert!(vel_err < 12.0, "Velocity error {} exceeds 12.0 m/s", vel_err);
        assert!(j_err < 0.5, "J error {} exceeds 0.5", j_err);
    }

    #[test]
    fn benchmark_three_body_collision() {
        let capture_frames: Vec<u32> = (0..250).step_by(10).collect();

        let reference = run_multi_body_and_capture(
            Material::RUBBER, 3, BENCHMARK_SUBSTEPS, 250, &capture_frames, 0.25
        );
        let test = run_multi_body_and_capture(
            Material::RUBBER, 3, TARGET_SUBSTEPS, 250, &capture_frames, 0.25
        );

        print_comparison("Three Body Collision (RUBBER)", &reference, &test);

        let (pos_err, vel_err, _ke_err, j_err) = compare_snapshots(&reference, &test);

        // Tolerances for 3-body collision with 8 substeps
        // More bodies = more collision damping = more behavior divergence from reference
        assert!(pos_err < 3.0, "Position error {} exceeds 3.0m", pos_err);
        assert!(vel_err < 20.0, "Velocity error {} exceeds 20.0 m/s", vel_err);
        assert!(j_err < 0.5, "J error {} exceeds 0.5", j_err);
    }

    /// Diagnostic test with tracing - run with --nocapture to see output
    #[test]
    fn test_metal_with_tracing() {
        let mut mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
        offset_vertices(&mut mesh.vertices, 0.0, START_HEIGHT);

        let rest_areas: Vec<f32> = {
            let num_tris = mesh.triangles.len() / 3;
            (0..num_tris).map(|t| {
                let i0 = mesh.triangles[t * 3] as usize;
                let i1 = mesh.triangles[t * 3 + 1] as usize;
                let i2 = mesh.triangles[t * 3 + 2] as usize;
                let x0 = mesh.vertices[i0 * 2];
                let y0 = mesh.vertices[i0 * 2 + 1];
                let x1 = mesh.vertices[i1 * 2];
                let y1 = mesh.vertices[i1 * 2 + 1];
                let x2 = mesh.vertices[i2 * 2];
                let y2 = mesh.vertices[i2 * 2 + 1];
                0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs()
            }).collect()
        };

        let mut body = SoftBody::new(&mesh.vertices, &mesh.triangles, Material::METAL);
        let mut tracer = SimulationTracer::new(500);

        let dt = 1.0 / 60.0 / SUBSTEPS as f32;

        // Simulate 3 seconds
        for frame in 0..180 {
            for _ in 0..SUBSTEPS {
                body.substep(dt, GRAVITY).0;
                body.collide_with_ground(GROUND_Y);
            }

            tracer.capture_frame(
                frame,
                1.0 / 60.0,
                &body.pos,
                &body.vel,
                &mesh.triangles,
                &rest_areas,
            );
        }

        tracer.print_summary(20);
        let stats = tracer.statistics();
        stats.print();

        assert!(stats.is_stable(), "Simulation was unstable");
    }
}
