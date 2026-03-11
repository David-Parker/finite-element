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

        // Soft materials can spread more, stiff materials should hold shape better
        // JELLO/RUBBER: allow up to 35% deviation
        // WOOD/METAL: allow up to 15% deviation
        for (name, width, max_dev) in [
            ("JELLO", jello_width, 0.35),
            ("RUBBER", rubber_width, 0.25),
            ("WOOD", wood_width, 0.15),
            ("METAL", metal_width, 0.15),
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
