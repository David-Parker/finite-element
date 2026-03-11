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
const SEGMENTS: u32 = 32;
const RADIAL_DIVISIONS: u32 = 4;
const OUTER_RADIUS: f32 = 1.5;  // 3m diameter
const INNER_RADIUS: f32 = 1.0;
const START_HEIGHT: f32 = 6.0;  // 6m above origin
const GROUND_Y: f32 = -8.0;     // ground at -8m
const GRAVITY: f32 = -9.8;
const SUBSTEPS: u32 = 64;
const FIXED_DT: f64 = 1.0 / 60.0;  // Physics runs at 60Hz
const MAX_FRAME_TIME: f64 = 0.1;   // Cap to prevent spiral of death

fn create_soft_body(material: Material) -> SoftBody {
    let mut mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
    offset_vertices(&mut mesh.vertices, 0.0, START_HEIGHT);
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
    } else {
        "UNKNOWN"
    }
}

/// Simulation state
struct Simulation {
    soft_body: SoftBody,
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

impl Simulation {
    fn new(gl: WebGlRenderingContext) -> Result<Self, JsValue> {
        let current_material = Material::RUBBER;
        let soft_body = create_soft_body(current_material);
        let mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
        let line_indices = create_ring_wireframe(SEGMENTS, RADIAL_DIVISIONS);

        let mut renderer = Renderer::new(gl)?;
        renderer.set_mesh(&mesh.triangles, &line_indices);
        renderer.set_ground(GROUND_Y);

        Ok(Simulation {
            soft_body,
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
        self.soft_body = create_soft_body(self.current_material);
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

        // Run fixed timestep physics updates
        while self.accumulator >= FIXED_DT {
            let substep_dt = (FIXED_DT / SUBSTEPS as f64) as f32;

            // Debug: log state before substeps
            if self.frame_count < 5 {
                let ke_before = self.soft_body.get_kinetic_energy();
                let (_, _, max_vel, _, _, _) = self.soft_body.get_diagnostics();
                console::log_1(&format!(
                    "Frame {}: KE={:.2}, maxVel={:.2}, dt={:.6}",
                    self.frame_count, ke_before, max_vel, substep_dt
                ).into());
            }

            let mut total_corrections = 0u32;
            for _ in 0..SUBSTEPS {
                let (_, corrections) = self.soft_body.substep(substep_dt, GRAVITY);
                total_corrections += corrections;
                self.soft_body.collide_with_ground(GROUND_Y);
            }

            // Debug: log state after substeps
            if self.frame_count < 5 {
                let ke_after = self.soft_body.get_kinetic_energy();
                let (min_j, max_j, max_vel, _, _, _) = self.soft_body.get_diagnostics();
                console::log_1(&format!(
                    "  After: KE={:.2}, maxVel={:.2}, J=[{:.2},{:.2}], strain_corrections={}",
                    ke_after, max_vel, min_j, max_j, total_corrections
                ).into());
            }

            // Only sleep if object is on the ground and has very low energy
            let lowest_y = self.soft_body.get_lowest_y();
            if lowest_y < GROUND_Y + 0.1 {
                self.soft_body.sleep_if_resting(1.0);  // Much lower threshold
            }

            self.accumulator -= FIXED_DT;
            self.frame_count += 1;
            self.collect_trace();
        }
    }

    fn render(&self) {
        self.renderer.render(&self.soft_body.pos);
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        let state = if self.paused { "Paused" } else { "Resumed" };
        console::log_1(&state.into());
    }

    fn get_kinetic_energy(&self) -> f32 {
        self.soft_body.get_kinetic_energy()
    }

    fn get_max_velocity(&self) -> f32 {
        let (_, _, max_vel, _, _, _) = self.soft_body.get_diagnostics();
        max_vel
    }

    fn collect_trace(&mut self) {
        let ke = self.soft_body.get_kinetic_energy();
        let (min_j, max_j, max_vel, max_force, min_plastic_det, max_plastic_det) =
            self.soft_body.get_diagnostics();
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
