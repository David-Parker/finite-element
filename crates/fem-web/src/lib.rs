//! FEM Soft Body Simulation - Web/WebAssembly application
//!
//! Uses XPBD (Extended Position-Based Dynamics) for unconditionally stable simulation

mod renderer;
mod trace;
mod profiler;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;
use web_sys::{console, WebGlRenderingContext, HtmlCanvasElement, KeyboardEvent, MouseEvent};

// Import from the portable core library
use fem_core::mesh::{create_ring_mesh, offset_vertices};
use fem_core::xpbd::{XPBDSoftBody, CollisionSystem};

use crate::renderer::{Renderer, BodyMesh};
use crate::trace::Tracer;
use crate::profiler::Profiler;

// Configuration
const SEGMENTS: u32 = 16;        // 128 triangles (16 * 4 * 2 = 128)
const RADIAL_DIVISIONS: u32 = 4;
const OUTER_RADIUS: f32 = 1.5;   // 3m diameter
const INNER_RADIUS: f32 = 1.0;
const START_HEIGHT: f32 = 6.0;   // 6m above origin
const GROUND_Y: f32 = -8.0;      // ground at -8m
const GRAVITY: f32 = -9.8;
// 4 substeps for good performance while maintaining stability
const SUBSTEPS: u32 = 4;
const FIXED_DT: f64 = 1.0 / 60.0;  // Physics runs at 60Hz
const MAX_FRAME_TIME: f64 = 0.1;   // Cap to prevent spiral of death

/// Material presets for XPBD
/// (edge_compliance, area_compliance, density, damping)
#[derive(Clone, Copy)]
struct XPBDMaterial {
    edge_compliance: f32,
    area_compliance: f32,
    density: f32,
    name: &'static str,
}

impl XPBDMaterial {
    // Soft, jiggly
    const JELLO: XPBDMaterial = XPBDMaterial {
        edge_compliance: 0.0,
        area_compliance: 1e-6,   // Stiffer area
        density: 1000.0,
        name: "JELLO",
    };

    // Bouncy rubber
    const RUBBER: XPBDMaterial = XPBDMaterial {
        edge_compliance: 0.0,
        area_compliance: 1e-7,
        density: 1100.0,
        name: "RUBBER",
    };

    // Stiff wood
    const WOOD: XPBDMaterial = XPBDMaterial {
        edge_compliance: 0.0,
        area_compliance: 1e-8,
        density: 600.0,
        name: "WOOD",
    };

    // Very stiff metal
    const METAL: XPBDMaterial = XPBDMaterial {
        edge_compliance: 0.0,
        area_compliance: 0.0,    // Perfectly rigid
        density: 2000.0,
        name: "METAL",
    };

    // Bouncy rubber
    const BOUNCY_RUBBER: XPBDMaterial = XPBDMaterial {
        edge_compliance: 0.0,
        area_compliance: 1e-7,
        density: 1100.0,
        name: "BOUNCY_RUBBER",
    };
}

/// Shape types for bodies
#[derive(Clone, Copy)]
enum ShapeType {
    Ring,
    Ellipse,
    Star,
    Blob(u32),  // seed for randomization
}

#[allow(dead_code)]
fn create_xpbd_body(material: XPBDMaterial, x_offset: f32, y_offset: f32) -> XPBDSoftBody {
    create_shaped_body(material, x_offset, y_offset, ShapeType::Ring, 1.0).0
}

fn create_shaped_body(material: XPBDMaterial, x_offset: f32, y_offset: f32, shape: ShapeType, scale: f32) -> (XPBDSoftBody, Vec<u32>) {
    use fem_core::mesh::{create_ellipse_mesh, create_star_mesh, create_blob_mesh};

    // Create mesh based on shape type (base sizes, will be scaled)
    let mut mesh = match shape {
        ShapeType::Ring => create_ring_mesh(OUTER_RADIUS * scale, INNER_RADIUS * scale, SEGMENTS, RADIAL_DIVISIONS),
        ShapeType::Ellipse => create_ellipse_mesh(2.5 * scale, 1.8 * scale, SEGMENTS, RADIAL_DIVISIONS),
        ShapeType::Star => create_star_mesh(1.6 * scale, 0.7 * scale, 5, RADIAL_DIVISIONS),
        ShapeType::Blob(seed) => create_blob_mesh(1.4 * scale, 0.25, SEGMENTS, RADIAL_DIVISIONS, seed),
    };

    offset_vertices(&mut mesh.vertices, x_offset, y_offset);

    let triangles = mesh.triangles.clone();

    let mut body = XPBDSoftBody::new(
        &mesh.vertices,
        &mesh.triangles,
        material.density,
        material.edge_compliance,
        material.area_compliance,
    );

    // Initialize prev_pos to current position (important for first frame velocity calculation)
    body.prev_pos = body.pos.clone();

    (body, triangles)
}

/// Simulation state
struct Simulation {
    bodies: Vec<XPBDSoftBody>,
    body_triangles: Vec<Vec<u32>>,  // Triangle indices per body
    collision_system: CollisionSystem,
    renderer: Renderer,
    profiler: Profiler,
    paused: bool,
    frame_count: u32,
    last_update_time: f64,
    accumulator: f64,
    fps: f64,
    tracer: Tracer,
    current_material: XPBDMaterial,
    // Mouse attractor
    attractor_active: bool,
    attractor_x: f32,
    attractor_y: f32,
}

// Colors for bodies (cycles through these)
const BODY_COLORS: [(f32, f32, f32); 10] = [
    (0.4, 0.8, 1.0),  // cyan
    (1.0, 0.6, 0.2),  // orange
    (0.5, 1.0, 0.5),  // green
    (1.0, 0.4, 0.7),  // pink
    (0.9, 0.9, 0.3),  // yellow
    (0.7, 0.5, 1.0),  // purple
    (1.0, 0.3, 0.3),  // red
    (0.3, 0.7, 0.9),  // light blue
    (0.9, 0.7, 0.5),  // tan
    (0.6, 0.9, 0.7),  // mint
];

impl Simulation {
    fn new(gl: WebGlRenderingContext) -> Result<Self, JsValue> {
        let current_material = XPBDMaterial::METAL;
        // Create bodies with mixed shapes
        let (bodies, body_triangles) = Self::create_bodies(current_material);

        let mut renderer = Renderer::new(gl)?;
        renderer.set_ground(GROUND_Y);

        // Spatial hash collision system
        let collision_system = CollisionSystem::new(0.15);

        // Profiler reports every 60 frames
        let profiler = Profiler::new(60);

        Ok(Simulation {
            bodies,
            body_triangles,
            collision_system,
            renderer,
            profiler,
            paused: false,
            frame_count: 0,
            last_update_time: 0.0,
            accumulator: 0.0,
            fps: 0.0,
            tracer: Tracer::new(),
            current_material,
            attractor_active: false,
            attractor_x: 0.0,
            attractor_y: 0.0,
        })
    }

    fn create_bodies(material: XPBDMaterial) -> (Vec<XPBDSoftBody>, Vec<Vec<u32>>) {
        let mut bodies = Vec::with_capacity(20);
        let mut triangles = Vec::with_capacity(20);

        // Mix of shapes on the ground with varying sizes
        let ground_rest_y = GROUND_Y + OUTER_RADIUS + 0.1;
        let shapes_and_scales = [
            (ShapeType::Ring, 0.8),
            (ShapeType::Star, 1.2),
            (ShapeType::Ellipse, 0.7),
            (ShapeType::Blob(42), 1.0),
            (ShapeType::Ring, 1.3),
        ];
        for (i, &(shape, scale)) in shapes_and_scales.iter().enumerate() {
            let x = -6.0 + (i as f32) * 3.0;
            let (body, tris) = create_shaped_body(material, x, ground_rest_y, shape, scale);
            bodies.push(body);
            triangles.push(tris);
        }

        // Falling bodies with mixed shapes and sizes
        let drop_start = START_HEIGHT + 5.0;
        let vertical_spacing = 4.0;
        let horizontal_spacing = 3.5;

        let falling_shapes = [
            (ShapeType::Blob(1), 0.7), (ShapeType::Star, 1.1), (ShapeType::Ellipse, 0.9), (ShapeType::Ring, 1.3),
            (ShapeType::Star, 0.8), (ShapeType::Blob(2), 1.2), (ShapeType::Ring, 0.6), (ShapeType::Ellipse, 1.0),
            (ShapeType::Ellipse, 1.1), (ShapeType::Ring, 0.9), (ShapeType::Blob(3), 0.8), (ShapeType::Star, 1.3),
            (ShapeType::Ring, 1.0), (ShapeType::Ellipse, 0.7), (ShapeType::Star, 0.9), (ShapeType::Blob(4), 1.1),
        ];

        for (i, &(shape, scale)) in falling_shapes.iter().enumerate() {
            let row = i / 4;
            let col = i % 4;
            let x = -5.25 + (col as f32) * horizontal_spacing;
            let y = drop_start + (row as f32) * vertical_spacing;
            let x_offset = ((row + col) % 3) as f32 * 0.3 - 0.3;
            let (body, tris) = create_shaped_body(material, x + x_offset, y, shape, scale);
            bodies.push(body);
            triangles.push(tris);
        }

        (bodies, triangles)
    }

    fn reset(&mut self) {
        let (bodies, triangles) = Self::create_bodies(self.current_material);
        self.bodies = bodies;
        self.body_triangles = triangles;
        self.frame_count = 0;
        self.accumulator = 0.0;
    }

    fn set_material(&mut self, material: XPBDMaterial) {
        self.current_material = material;
        self.reset();
        console::log_1(&format!("Material: {} (XPBD)", material.name).into());
    }

    fn get_material_name(&self) -> &'static str {
        self.current_material.name
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
                let ke_before: f32 = self.bodies.iter().map(|b| b.get_kinetic_energy()).sum();
                console::log_1(&format!(
                    "Frame {}: KE={:.2}, dt={:.6}",
                    self.frame_count, ke_before, substep_dt
                ).into());
            }

            self.profiler.begin("substeps");

            // Build collision data once per frame (broad phase + edge cache + hash)
            self.profiler.begin("col_prepare");
            self.collision_system.prepare(&self.bodies);
            self.profiler.end("col_prepare");

            for _substep in 0..SUBSTEPS {
                // Apply attractor acceleration if active
                if self.attractor_active {
                    const ATTRACTOR_STRENGTH: f32 = 25.0;
                    const MIN_DIST: f32 = 3.0;  // Don't pull harder when closer than this
                    for body in &mut self.bodies {
                        // Get body center
                        let (cx, cy) = body.get_center();
                        // Direction to attractor
                        let dx = self.attractor_x - cx;
                        let dy = self.attractor_y - cy;
                        let dist = (dx * dx + dy * dy).sqrt().max(MIN_DIST);
                        // Apply velocity change towards attractor (normalized direction)
                        let ax = (dx / dist) * ATTRACTOR_STRENGTH * substep_dt;
                        let ay = (dy / dist) * ATTRACTOR_STRENGTH * substep_dt;
                        for i in 0..body.num_verts {
                            if body.inv_mass[i] > 0.0 {
                                body.vel[i * 2] += ax;
                                body.vel[i * 2 + 1] += ay;
                            }
                        }
                    }
                }

                // Pre-solve and constraints for all bodies
                self.profiler.begin("constraints");
                for body in &mut self.bodies {
                    body.substep_pre(substep_dt, GRAVITY, Some(GROUND_Y));
                }
                self.profiler.end("constraints");

                // Inter-body collisions: reuse prepared data, resolve with fresh positions
                self.profiler.begin("col_resolve");
                self.collision_system.resolve_collisions(&mut self.bodies);
                self.profiler.end("col_resolve");

                // Finalize substep: derive velocities from position change
                self.profiler.begin("post_solve");
                for body in &mut self.bodies {
                    body.substep_post(substep_dt);
                }
                self.profiler.end("post_solve");
            }
            self.profiler.end("substeps");

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

    fn render(&mut self) {
        self.profiler.begin("render");
        let meshes: Vec<BodyMesh> = self.bodies.iter()
            .zip(self.body_triangles.iter())
            .map(|(body, tris)| BodyMesh {
                positions: body.pos.as_slice(),
                triangles: tris.as_slice(),
            })
            .collect();
        self.renderer.render_meshes(&meshes, &BODY_COLORS);
        self.profiler.end("render");

        // Log collision stats periodically
        if self.frame_count % 60 == 0 && self.frame_count > 0 {
            let cs = &self.collision_system;
            console::log_1(&format!(
                "COL STATS: pairs={} edges={} candidates={} collisions={} iters={}",
                cs.stats_overlapping_pairs, cs.stats_cached_edges,
                cs.stats_candidates, cs.stats_collisions_found, cs.stats_iterations_run
            ).into());
        }

        self.profiler.end_frame();
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
            .map(|b| b.get_max_velocity())
            .fold(0.0f32, |a, b| a.max(b))
    }

    fn collect_trace(&mut self) {
        // Trace first body only for now
        if let Some(body) = self.bodies.first() {
            let ke = body.get_kinetic_energy();
            let max_vel = body.get_max_velocity();
            // XPBD doesn't have J (volume ratio) or stress info directly
            // Use placeholder values
            self.tracer.record(
                self.frame_count,
                ke,
                1.0,  // min_j placeholder
                1.0,  // max_j placeholder
                max_vel,
                0.0,  // max_force placeholder
                1.0,  // min_plastic_det placeholder
                1.0,  // max_plastic_det placeholder
            );
        }
    }

    fn toggle_tracing(&mut self) {
        self.tracer.toggle();
    }

    fn toggle_profiler(&mut self) {
        self.profiler.toggle();
    }

    fn set_attractor(&mut self, x: f32, y: f32) {
        self.attractor_x = x;
        self.attractor_y = y;
    }

    fn start_attracting(&mut self) {
        self.attractor_active = true;
    }

    fn stop_attracting(&mut self) {
        self.attractor_active = false;
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
                "p" => {
                    sim.borrow_mut().toggle_profiler();
                }
                "1" => {
                    sim.borrow_mut().set_material(XPBDMaterial::JELLO);
                }
                "2" => {
                    sim.borrow_mut().set_material(XPBDMaterial::RUBBER);
                }
                "3" => {
                    sim.borrow_mut().set_material(XPBDMaterial::WOOD);
                }
                "4" => {
                    sim.borrow_mut().set_material(XPBDMaterial::METAL);
                }
                "5" => {
                    sim.borrow_mut().set_material(XPBDMaterial::BOUNCY_RUBBER);
                }
                _ => {}
            }
        }) as Box<dyn FnMut(_)>);

        document.add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Set up mouse handlers for attractor
    let canvas_width = canvas.width() as f32;
    let canvas_height = canvas.height() as f32;
    let view_size = 10.0f32;  // Must match renderer's view_size

    // Helper to convert screen coords to world coords
    let screen_to_world = move |screen_x: f32, screen_y: f32| -> (f32, f32) {
        // screen (0,0) is top-left, world (0,0) is center
        let world_x = (screen_x / canvas_width - 0.5) * 2.0 * view_size;
        let world_y = -(screen_y / canvas_height - 0.5) * 2.0 * view_size;
        (world_x, world_y)
    };

    // Mouse down - start attracting
    {
        let sim = simulation.clone();
        let convert = screen_to_world;
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            let (wx, wy) = convert(event.offset_x() as f32, event.offset_y() as f32);
            let mut sim = sim.borrow_mut();
            sim.set_attractor(wx, wy);
            sim.start_attracting();
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Mouse move - update attractor position
    {
        let sim = simulation.clone();
        let convert = screen_to_world;
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            let (wx, wy) = convert(event.offset_x() as f32, event.offset_y() as f32);
            sim.borrow_mut().set_attractor(wx, wy);
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Mouse up - stop attracting
    {
        let sim = simulation.clone();
        let closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
            sim.borrow_mut().stop_attracting();
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Mouse leave - stop attracting
    {
        let sim = simulation.clone();
        let closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
            sim.borrow_mut().stop_attracting();
        }) as Box<dyn FnMut(_)>);
        canvas.add_event_listener_with_callback("mouseleave", closure.as_ref().unchecked_ref())?;
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

    console::log_1(&"XPBD Soft Body Simulation (Rust/WebAssembly) started".into());
    console::log_1(&"Controls: Space = Pause, R = Reset, T = Trace".into());
    console::log_1(&"Materials: 1=Jello, 2=Rubber, 3=Wood, 4=Metal, 5=Bouncy Rubber".into());

    Ok(())
}

fn request_animation_frame(window: &web_sys::Window, f: &Closure<dyn FnMut()>) {
    window
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}

