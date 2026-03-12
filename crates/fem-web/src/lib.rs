//! FEM Soft Body Simulation - Web/WebAssembly application

mod renderer;
mod trace;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;
use web_sys::{console, WebGlRenderingContext, HtmlCanvasElement, KeyboardEvent};

// Import from the portable core library
use fem_core::mesh::{create_ring_mesh, create_ring_wireframe, offset_vertices};
use fem_core::softbody::{SoftBody, Material};
use fem_core::math;
use fem_core::fem;

use crate::renderer::Renderer;
use crate::trace::Tracer;

// Configuration
const SEGMENTS: u32 = 16;        // Reduced for 128 triangles (16 * 4 * 2 = 128)
const RADIAL_DIVISIONS: u32 = 4;
const OUTER_RADIUS: f32 = 1.5;  // 3m diameter
const INNER_RADIUS: f32 = 1.0;
const START_HEIGHT: f32 = 6.0;  // 6m above origin
const GROUND_Y: f32 = -8.0;     // ground at -8m
const GRAVITY: f32 = -9.8;
const SUBSTEPS: u32 = 64;
const FIXED_DT: f64 = 1.0 / 60.0;  // Physics runs at 60Hz
const MAX_FRAME_TIME: f64 = 0.1;   // Cap to prevent spiral of death

fn create_soft_body(material: Material, x_offset: f32, y_offset: f32) -> SoftBody {
    let mut mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
    offset_vertices(&mut mesh.vertices, x_offset, y_offset);
    SoftBody::new(&mesh.vertices, &mesh.triangles, material)
}

fn material_name(material: Material) -> &'static str {
    // Compare by young_modulus since we can't derive PartialEq easily
    if material.young_modulus == Material::JELLO.young_modulus {
        "JELLO"
    } else if material.young_modulus == Material::RUBBER.young_modulus {
        "RUBBER"
    } else if material.young_modulus == Material::WOOD.young_modulus {
        "WOOD"
    } else if material.young_modulus == Material::METAL.young_modulus {
        "METAL"
    } else if material.young_modulus == Material::BOUNCY_RUBBER.young_modulus {
        "BOUNCY_RUBBER"
    } else {
        "UNKNOWN"
    }
}

/// Simulation state
struct Simulation {
    bodies: Vec<SoftBody>,
    renderer: Renderer,
    #[allow(dead_code)]
    triangles: Vec<u32>,
    #[allow(dead_code)]
    line_indices: Vec<u32>,
    paused: bool,
    frame_count: u32,
    last_update_time: f64,
    accumulator: f64,
    fps: f64,
    tracer: Tracer,
    current_material: Material,
}

// Colors for each body (cyan, orange)
const BODY_COLORS: [(f32, f32, f32); 2] = [
    (0.4, 0.8, 1.0),  // cyan
    (1.0, 0.6, 0.2),  // orange
];

impl Simulation {
    fn new(gl: WebGlRenderingContext) -> Result<Self, JsValue> {
        let current_material = Material::BOUNCY_RUBBER;
        // Create two bodies: one higher and slightly left, one lower and slightly right
        let body1 = create_soft_body(current_material, -0.5, START_HEIGHT + 4.0);
        let body2 = create_soft_body(current_material, 0.5, START_HEIGHT);
        let mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
        let line_indices = create_ring_wireframe(SEGMENTS, RADIAL_DIVISIONS);

        let mut renderer = Renderer::new(gl)?;
        renderer.set_mesh(&mesh.triangles, &line_indices);
        renderer.set_ground(GROUND_Y);

        Ok(Simulation {
            bodies: vec![body1, body2],
            renderer,
            triangles: mesh.triangles,
            line_indices,
            paused: false,
            frame_count: 0,
            last_update_time: 0.0,
            accumulator: 0.0,
            fps: 0.0,
            tracer: Tracer::new(),
            current_material,
        })
    }

    fn reset(&mut self) {
        self.bodies = vec![
            create_soft_body(self.current_material, -0.5, START_HEIGHT + 4.0),
            create_soft_body(self.current_material, 0.5, START_HEIGHT),
        ];
        self.frame_count = 0;
        self.accumulator = 0.0;
    }

    fn set_material(&mut self, material: Material) {
        self.current_material = material;
        self.reset();
        console::log_1(&format!("Material: {}", material_name(material)).into());
    }

    fn get_material_name(&self) -> &'static str {
        material_name(self.current_material)
    }

    fn update(&mut self, delta_time: f64) {
        if self.paused {
            return;
        }

        // Cap delta time to prevent spiral of death after lag spikes
        let delta_time = delta_time.min(MAX_FRAME_TIME);
        self.accumulator += delta_time;

        // Collision distance threshold (based on mesh density)
        let collision_dist = 0.25;

        // Run fixed timestep physics updates
        while self.accumulator >= FIXED_DT {
            let substep_dt = (FIXED_DT / SUBSTEPS as f64) as f32;

            // Debug: log state before substeps
            if self.frame_count < 5 {
                let ke_before: f32 = self.bodies.iter().map(|b| b.get_kinetic_energy()).sum();
                console::log_1(&format!(
                    "Frame {}: KE={:.2}, dt={:.6}",
                    self.frame_count, ke_before, substep_dt
                ).into());
            }

            for _ in 0..SUBSTEPS {
                // Physics substep for each body
                for body in &mut self.bodies {
                    body.substep(substep_dt, GRAVITY);
                    body.collide_with_ground(GROUND_Y);
                }

                // Inter-body collisions (need to handle borrow checker)
                if self.bodies.len() >= 2 {
                    let (first, rest) = self.bodies.split_at_mut(1);
                    first[0].collide_with_body(&mut rest[0], collision_dist);
                }
            }

            // Debug: log state after substeps
            if self.frame_count < 5 {
                let ke_after: f32 = self.bodies.iter().map(|b| b.get_kinetic_energy()).sum();
                console::log_1(&format!(
                    "  After: KE={:.2}",
                    ke_after
                ).into());
            }

            // Only sleep if object is on the ground and has very low energy
            for body in &mut self.bodies {
                let lowest_y = body.get_lowest_y();
                if lowest_y < GROUND_Y + 0.1 {
                    body.sleep_if_resting(1.0);
                }
            }

            self.accumulator -= FIXED_DT;
            self.frame_count += 1;
            self.collect_trace();
        }
    }

    fn render(&self) {
        let positions: Vec<&[f32]> = self.bodies.iter().map(|b| b.pos.as_slice()).collect();
        self.renderer.render_bodies(&positions, &BODY_COLORS);
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        let state = if self.paused { "Paused" } else { "Resumed" };
        console::log_1(&state.into());
    }

    fn get_kinetic_energy(&self) -> f32 {
        self.bodies.iter().map(|b| b.get_kinetic_energy()).sum()
    }

    fn get_max_velocity(&self) -> f32 {
        self.bodies.iter()
            .map(|b| b.get_diagnostics().2)
            .fold(0.0f32, |a, b| a.max(b))
    }

    fn collect_trace(&mut self) {
        // Trace first body only for now
        if let Some(body) = self.bodies.first() {
            let ke = body.get_kinetic_energy();
            let (min_j, max_j, max_vel, max_force, min_plastic_det, max_plastic_det) =
                body.get_diagnostics();
            self.tracer.record(
                self.frame_count,
                ke,
                min_j,
                max_j,
                max_vel,
                max_force,
                min_plastic_det,
                max_plastic_det,
            );
        }
    }

    fn toggle_tracing(&mut self) {
        self.tracer.toggle();
    }
}

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: HtmlCanvasElement = canvas.dyn_into()?;

    let gl: WebGlRenderingContext = canvas
        .get_context("webgl")?
        .unwrap()
        .dyn_into()?;

    let simulation = Rc::new(RefCell::new(Simulation::new(gl)?));

    // Set up keyboard handler
    {
        let sim = simulation.clone();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let key = event.key().to_lowercase();
            match key.as_str() {
                " " => {
                    sim.borrow_mut().toggle_pause();
                    event.prevent_default();
                }
                "r" => {
                    sim.borrow_mut().reset();
                    console::log_1(&"Simulation reset".into());
                }
                "t" => {
                    sim.borrow_mut().toggle_tracing();
                }
                "1" => {
                    sim.borrow_mut().set_material(Material::JELLO);
                }
                "2" => {
                    sim.borrow_mut().set_material(Material::RUBBER);
                }
                "3" => {
                    sim.borrow_mut().set_material(Material::WOOD);
                }
                "4" => {
                    sim.borrow_mut().set_material(Material::METAL);
                }
                "5" => {
                    sim.borrow_mut().set_material(Material::BOUNCY_RUBBER);
                }
                _ => {}
            }
        }) as Box<dyn FnMut(_)>);

        document.add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Set up animation loop
    {
        let sim = simulation.clone();
        let f = Rc::new(RefCell::new(None));
        let g = f.clone();

        let window_clone = window.clone();
        let perf = window.performance().expect("performance should be available");

        *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            let now = perf.now();

            {
                let mut sim = sim.borrow_mut();

                // Calculate delta time and FPS
                let delta_ms = if sim.last_update_time > 0.0 {
                    now - sim.last_update_time
                } else {
                    0.0
                };
                let delta_secs = delta_ms / 1000.0;

                if delta_ms > 0.0 {
                    let instant_fps = 1000.0 / delta_ms;
                    sim.fps = sim.fps * 0.9 + instant_fps * 0.1;
                }
                sim.last_update_time = now;

                sim.update(delta_secs);

                // Update status display
                if let Some(document) = web_sys::window().and_then(|w| w.document()) {
                    if let Some(el) = document.get_element_by_id("material") {
                        el.set_text_content(Some(sim.get_material_name()));
                    }
                    if let Some(el) = document.get_element_by_id("fps") {
                        el.set_text_content(Some(&format!("{:.0}", sim.fps)));
                    }
                    if let Some(el) = document.get_element_by_id("frameCount") {
                        el.set_text_content(Some(&sim.frame_count.to_string()));
                    }
                    if let Some(el) = document.get_element_by_id("kineticEnergy") {
                        el.set_text_content(Some(&format!("{:.4}", sim.get_kinetic_energy())));
                    }
                    if let Some(el) = document.get_element_by_id("maxVelocity") {
                        el.set_text_content(Some(&format!("{:.1}", sim.get_max_velocity())));
                    }
                }

                sim.render();
            }

            request_animation_frame(&window_clone, f.borrow().as_ref().unwrap());
        }) as Box<dyn FnMut()>));

        request_animation_frame(&window, g.borrow().as_ref().unwrap());
    }

    console::log_1(&"FEM Soft Body Simulation (Rust/WebAssembly) started".into());
    console::log_1(&"Controls: Space = Pause, R = Reset, T = Trace".into());
    console::log_1(&"Materials: 1=Jello, 2=Rubber, 3=Wood, 4=Metal, 5=Bouncy Rubber".into());

    Ok(())
}

fn request_animation_frame(window: &web_sys::Window, f: &Closure<dyn FnMut()>) {
    window
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}

/// Run tests (callable from JS console)
#[wasm_bindgen]
pub fn run_tests() -> bool {
    console::log_1(&"Running FEM tests...".into());

    let i = math::mat2_identity();
    assert_eq!(i, [1.0, 0.0, 0.0, 1.0]);
    console::log_1(&"  mat2_identity".into());

    let det = math::mat2_det(&[2.0, 0.0, 0.0, 3.0]);
    assert!((det - 6.0).abs() < 1e-6);
    console::log_1(&"  mat2_det".into());

    let area = fem::compute_triangle_area(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    assert!((area - 0.5).abs() < 1e-6);
    console::log_1(&"  triangle_area".into());

    let f = math::mat2_identity();
    let energy = fem::compute_neo_hookean_energy(&f, 1.0, 1000.0, 2000.0);
    assert!(energy.abs() < 1e-6);
    console::log_1(&"  neo_hookean_energy_at_rest".into());

    let p = fem::compute_neo_hookean_stress(&f, 1.0, 1000.0, 2000.0);
    for val in &p {
        assert!(val.abs() < 1e-6);
    }
    console::log_1(&"  neo_hookean_stress_at_rest".into());

    console::log_1(&"All tests passed!".into());
    true
}
