//! FEM Soft Body Simulation - Rust/WebAssembly implementation

mod math;
mod fem;
mod mesh;
mod softbody;
mod renderer;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;
use web_sys::{console, WebGlRenderingContext, HtmlCanvasElement, KeyboardEvent};

use crate::mesh::{create_ring_mesh, create_ring_wireframe, offset_vertices};
use crate::softbody::{SoftBody, Material};
use crate::renderer::Renderer;

// Configuration
const SEGMENTS: u32 = 24;
const RADIAL_DIVISIONS: u32 = 4;
const OUTER_RADIUS: f32 = 0.5;
const INNER_RADIUS: f32 = 0.35;
const START_HEIGHT: f32 = 0.5;
const GROUND_Y: f32 = -1.0;
const GRAVITY: f32 = -9.8;
const SUBSTEPS: u32 = 32;

fn create_soft_body() -> SoftBody {
    let mut mesh = create_ring_mesh(OUTER_RADIUS, INNER_RADIUS, SEGMENTS, RADIAL_DIVISIONS);
    offset_vertices(&mut mesh.vertices, 0.0, START_HEIGHT);
    SoftBody::new(&mesh.vertices, &mesh.triangles, Material::RUBBER)
}

/// Simulation state
struct Simulation {
    soft_body: SoftBody,
    renderer: Renderer,
    triangles: Vec<u32>,
    line_indices: Vec<u32>,
    paused: bool,
    frame_count: u32,
    last_update_time: f64,
    last_frame_slot: u64,
    fps: f64,
}

impl Simulation {
    fn new(gl: WebGlRenderingContext) -> Result<Self, JsValue> {
        let soft_body = create_soft_body();
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
            last_frame_slot: 0,
            fps: 0.0,
        })
    }

    fn reset(&mut self) {
        self.soft_body = create_soft_body();
        self.frame_count = 0;
    }

    fn update(&mut self) {
        if self.paused {
            return;
        }

        let dt = 1.0 / 60.0 / SUBSTEPS as f32;

        for _ in 0..SUBSTEPS {
            self.soft_body.substep(dt, GRAVITY);
            self.soft_body.collide_with_ground(GROUND_Y);
        }
        self.soft_body.apply_damping();

        self.frame_count += 1;
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

        const FRAME_TIME: f64 = 1000.0 / 60.0; // 16.67ms

        *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            let now = perf.now();

            {
                let mut sim = sim.borrow_mut();

                // Determine which 60Hz frame slot we're in
                let current_slot = (now / FRAME_TIME) as u64;

                // Update once per frame slot
                if current_slot > sim.last_frame_slot {
                    // Calculate FPS
                    if sim.last_update_time > 0.0 {
                        let delta = now - sim.last_update_time;
                        if delta > 0.0 {
                            let instant_fps = 1000.0 / delta;
                            sim.fps = sim.fps * 0.9 + instant_fps * 0.1;
                        }
                    }
                    sim.last_update_time = now;
                    sim.last_frame_slot = current_slot;

                    sim.update();

                    // Update status display
                    if let Some(document) = web_sys::window().and_then(|w| w.document()) {
                        if let Some(el) = document.get_element_by_id("fps") {
                            el.set_text_content(Some(&format!("{:.0}", sim.fps)));
                        }
                        if let Some(el) = document.get_element_by_id("frameCount") {
                            el.set_text_content(Some(&sim.frame_count.to_string()));
                        }
                        if let Some(el) = document.get_element_by_id("kineticEnergy") {
                            el.set_text_content(Some(&format!("{:.4}", sim.get_kinetic_energy())));
                        }
                    }
                }

                // Always render (for smooth visuals on high-refresh displays)
                sim.render();
            }

            request_animation_frame(&window_clone, f.borrow().as_ref().unwrap());
        }) as Box<dyn FnMut()>));

        request_animation_frame(&window, g.borrow().as_ref().unwrap());
    }

    console::log_1(&"FEM Soft Body Simulation (Rust/WebAssembly) started".into());
    console::log_1(&"Controls: Space = Pause, R = Reset".into());

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

    // Basic math tests
    let i = math::mat2_identity();
    assert_eq!(i, [1.0, 0.0, 0.0, 1.0]);
    console::log_1(&"✓ mat2_identity".into());

    let det = math::mat2_det(&[2.0, 0.0, 0.0, 3.0]);
    assert!((det - 6.0).abs() < 1e-6);
    console::log_1(&"✓ mat2_det".into());

    // FEM tests
    let area = fem::compute_triangle_area(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
    assert!((area - 0.5).abs() < 1e-6);
    console::log_1(&"✓ triangle_area".into());

    let f = math::mat2_identity();
    let energy = fem::compute_neo_hookean_energy(&f, 1.0, 1000.0, 2000.0);
    assert!(energy.abs() < 1e-6);
    console::log_1(&"✓ neo_hookean_energy_at_rest".into());

    let p = fem::compute_neo_hookean_stress(&f, 1.0, 1000.0, 2000.0);
    for val in &p {
        assert!(val.abs() < 1e-6);
    }
    console::log_1(&"✓ neo_hookean_stress_at_rest".into());

    console::log_1(&"All tests passed!".into());
    true
}
