//! End-to-end simulation tests to catch instability and explosions

#[cfg(test)]
mod tests {
    use crate::mesh::{create_ring_mesh, offset_vertices};
    use crate::softbody::{Material, SoftBody};
    use crate::trace::SimulationTracer;

    // Match actual simulation parameters
    const OUTER_RADIUS: f32 = 1.5;
    const INNER_RADIUS: f32 = 1.0;
    const SEGMENTS: u32 = 24;
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
                body.substep(dt, GRAVITY);
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
                body.substep(dt, GRAVITY);
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
                body.substep(dt, GRAVITY);
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
