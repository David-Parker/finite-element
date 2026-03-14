//! FEM Ring Game - WebGL platformer with soft body physics
//!
//! Player controls a ring that can move left/right and jump by compressing and releasing.

mod renderer;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;
use web_sys::{console, WebGlRenderingContext, HtmlCanvasElement, KeyboardEvent};

use fem_core::{PhysicsWorld, BodyHandle, BodyConfig, Material};
use fem_core::mesh::create_ring_mesh;

use crate::renderer::Renderer;

// Game configuration
const SEGMENTS: u32 = 16;
const RADIAL_DIVISIONS: u32 = 4;
const PLAYER_OUTER_RADIUS: f32 = 1.2;
const PLAYER_INNER_RADIUS: f32 = 0.8;
const GROUND_Y: f32 = -5.0;
const GRAVITY: f32 = -15.0;
const FIXED_DT: f64 = 1.0 / 60.0;
const MAX_FRAME_TIME: f64 = 0.1;

// Player movement
const MOVE_ACCELERATION: f32 = 35.0;
const MAX_HORIZONTAL_SPEED: f32 = 12.0;

// Jump mechanics
const MAX_JUMP_CHARGE: f32 = 1.0;  // seconds to full charge
const JUMP_CHARGE_RATE: f32 = 2.0; // charge per second
const MAX_COMPRESSION: f32 = 0.4;  // maximum compression ratio (40%)
const JUMP_IMPULSE: f32 = 20.0;    // base jump velocity

// Camera
const VIEW_WIDTH: f32 = 20.0;
const VIEW_HEIGHT: f32 = 15.0;
const CAMERA_SMOOTH: f32 = 0.1;   // camera smoothing factor

// Ring drops
const DROP_INTERVAL: f32 = 3.0;    // seconds between drops
const DROP_OUTER_RADIUS: f32 = 0.8;
const DROP_INNER_RADIUS: f32 = 0.5;
const MAX_DROPPED_RINGS: usize = 10;

/// Input state
#[derive(Default)]
struct InputState {
    left: bool,
    right: bool,
    space: bool,
}

/// Game state
struct Game {
    // Physics
    world: PhysicsWorld,
    player: BodyHandle,
    dropped_rings: Vec<BodyHandle>,

    // Jump mechanics
    jump_charge: f32,
    is_charging: bool,

    // Camera
    camera_x: f32,
    camera_y: f32,

    // Ring drops
    drop_timer: f32,
    rng_state: u32,

    // Rendering
    renderer: Renderer,

    // Timing
    last_update_time: f64,
    accumulator: f64,
    fps: f64,
    frame_count: u32,

    // Input
    input: InputState,
}

impl Game {
    fn new(gl: WebGlRenderingContext) -> Result<Self, JsValue> {
        // Create physics world
        let mut world = PhysicsWorld::new();
        world.set_gravity(GRAVITY);
        world.set_ground(Some(GROUND_Y));
        world.set_substeps(4);

        // Create player ring
        let player_mesh = create_ring_mesh(
            PLAYER_OUTER_RADIUS,
            PLAYER_INNER_RADIUS,
            SEGMENTS,
            RADIAL_DIVISIONS,
        );

        let start_y = GROUND_Y + PLAYER_OUTER_RADIUS + 0.5;
        let player = world.add_body(&player_mesh, BodyConfig::new()
            .with_material(Material::RUBBER)
            .at_position(0.0, start_y));

        let mut renderer = Renderer::new(gl)?;
        renderer.set_ground(GROUND_Y);
        renderer.set_view(VIEW_WIDTH, VIEW_HEIGHT);

        Ok(Game {
            world,
            player,
            dropped_rings: Vec::new(),
            jump_charge: 0.0,
            is_charging: false,
            camera_x: 0.0,
            camera_y: start_y,
            drop_timer: DROP_INTERVAL,
            rng_state: 12345,
            renderer,
            last_update_time: 0.0,
            accumulator: 0.0,
            fps: 0.0,
            frame_count: 0,
            input: InputState::default(),
        })
    }

    fn simple_random(&mut self) -> f32 {
        // Simple xorshift RNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        (self.rng_state as f32) / (u32::MAX as f32)
    }

    fn is_player_grounded(&self) -> bool {
        self.world.is_grounded(self.player, 0.2)
    }

    fn update(&mut self, delta_time: f64) {
        let delta_time = delta_time.min(MAX_FRAME_TIME);
        self.accumulator += delta_time;

        // Handle jump charging
        if self.input.space && self.is_player_grounded() {
            if !self.is_charging {
                self.is_charging = true;
                self.jump_charge = 0.0;
            }
            self.jump_charge = (self.jump_charge + delta_time as f32 * JUMP_CHARGE_RATE).min(MAX_JUMP_CHARGE);

            // Compress the ring vertically
            let compression = 1.0 - (self.jump_charge / MAX_JUMP_CHARGE) * MAX_COMPRESSION;
            self.world.set_vertical_compression(self.player, compression);
        } else if self.is_charging {
            // Release jump
            self.is_charging = false;

            // Restore original shape
            self.world.reset_rest_lengths(self.player);

            // Apply upward impulse based on charge
            let impulse = JUMP_IMPULSE * (self.jump_charge / MAX_JUMP_CHARGE);
            self.world.apply_impulse(self.player, 0.0, impulse);

            self.jump_charge = 0.0;
        }

        // Update ring drop timer
        self.drop_timer -= delta_time as f32;
        if self.drop_timer <= 0.0 {
            self.drop_ring();
            self.drop_timer = DROP_INTERVAL;
        }

        // Fixed timestep physics
        while self.accumulator >= FIXED_DT {
            let dt = FIXED_DT as f32;

            // Apply horizontal movement to player
            if self.input.left || self.input.right {
                let dir = if self.input.right { 1.0 } else { -1.0 };
                self.world.apply_acceleration(self.player, MOVE_ACCELERATION * dir, 0.0, dt);

                // Clamp velocity
                if let Some((vx, vy)) = self.world.get_velocity(self.player) {
                    let clamped_vx = vx.clamp(-MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED);
                    if (clamped_vx - vx).abs() > 0.01 {
                        self.world.set_velocity(self.player, clamped_vx, vy);
                    }
                }
            }

            // Note: Ground friction is now handled by the physics engine
            // The high friction causes the ring to roll naturally

            // Step physics
            self.world.step(dt);

            // Sleep rings that are resting
            for &ring in &self.dropped_rings {
                if self.world.is_grounded(ring, 0.1) {
                    self.world.sleep_if_resting(ring, 1.0);
                }
            }

            self.accumulator -= FIXED_DT;
            self.frame_count += 1;
        }

        // Update camera
        self.update_camera();
    }

    fn drop_ring(&mut self) {
        // Remove oldest if at max
        if self.dropped_rings.len() >= MAX_DROPPED_RINGS {
            let oldest = self.dropped_rings.remove(0);
            self.world.remove_body(oldest);
        }

        let (player_x, _) = self.world.get_position(self.player).unwrap_or((0.0, 0.0));

        // Random X offset near player
        let offset = (self.simple_random() - 0.5) * 10.0;
        let drop_x = player_x + offset;
        let drop_y = self.camera_y + VIEW_HEIGHT / 2.0 + 2.0;

        let mesh = create_ring_mesh(
            DROP_OUTER_RADIUS,
            DROP_INNER_RADIUS,
            SEGMENTS,
            RADIAL_DIVISIONS,
        );

        let handle = self.world.add_body(&mesh, BodyConfig::new()
            .with_material(Material::RUBBER)
            .at_position(drop_x, drop_y));

        self.dropped_rings.push(handle);
    }

    fn update_camera(&mut self) {
        let (player_x, player_y) = self.world.get_position(self.player).unwrap_or((0.0, 0.0));

        // Calculate target camera position
        let target_x = player_x;
        let target_y = player_y.max(GROUND_Y + VIEW_HEIGHT / 2.0 - 2.0);

        // Smooth camera movement
        self.camera_x += (target_x - self.camera_x) * CAMERA_SMOOTH;
        self.camera_y += (target_y - self.camera_y) * CAMERA_SMOOTH;

        self.renderer.set_camera(self.camera_x, self.camera_y);
    }

    fn render(&mut self) {
        // Collect render data
        let render_data = self.world.get_render_data();
        self.renderer.render_bodies(&render_data);
    }

    fn handle_key_down(&mut self, key: &str) {
        match key {
            "a" | "arrowleft" => self.input.left = true,
            "d" | "arrowright" => self.input.right = true,
            " " => self.input.space = true,
            _ => {}
        }
    }

    fn handle_key_up(&mut self, key: &str) {
        match key {
            "a" | "arrowleft" => self.input.left = false,
            "d" | "arrowright" => self.input.right = false,
            " " => self.input.space = false,
            _ => {}
        }
    }

    fn get_charge_percent(&self) -> u32 {
        ((self.jump_charge / MAX_JUMP_CHARGE) * 100.0) as u32
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

    let game = Rc::new(RefCell::new(Game::new(gl)?));

    // Keyboard down handler
    {
        let g = game.clone();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let key = event.key().to_lowercase();
            g.borrow_mut().handle_key_down(&key);

            // Prevent scrolling with arrow keys/space
            if matches!(key.as_str(), " " | "arrowleft" | "arrowright" | "arrowup" | "arrowdown") {
                event.prevent_default();
            }
        }) as Box<dyn FnMut(_)>);
        document.add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Keyboard up handler
    {
        let g = game.clone();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let key = event.key().to_lowercase();
            g.borrow_mut().handle_key_up(&key);
        }) as Box<dyn FnMut(_)>);
        document.add_event_listener_with_callback("keyup", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Animation loop
    {
        let g = game.clone();
        let f = Rc::new(RefCell::new(None));
        let g2 = f.clone();

        let window_clone = window.clone();
        let perf = window.performance().expect("performance should be available");

        *g2.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            let now = perf.now();

            {
                let mut game = g.borrow_mut();

                let delta_ms = if game.last_update_time > 0.0 {
                    now - game.last_update_time
                } else {
                    0.0
                };
                let delta_secs = delta_ms / 1000.0;

                if delta_ms > 0.0 {
                    let instant_fps = 1000.0 / delta_ms;
                    game.fps = game.fps * 0.9 + instant_fps * 0.1;
                }
                game.last_update_time = now;

                game.update(delta_secs);

                // Update UI
                if let Some(document) = web_sys::window().and_then(|w| w.document()) {
                    if let Some(el) = document.get_element_by_id("fps") {
                        el.set_text_content(Some(&format!("{:.0}", game.fps)));
                    }
                    if let Some(el) = document.get_element_by_id("charge") {
                        el.set_text_content(Some(&format!("{}", game.get_charge_percent())));
                    }
                }

                game.render();
            }

            request_animation_frame(&window_clone, f.borrow().as_ref().unwrap());
        }) as Box<dyn FnMut()>));

        request_animation_frame(&window, g2.borrow().as_ref().unwrap());
    }

    console::log_1(&"FEM Ring Game started!".into());
    console::log_1(&"Controls: A/Left = move left, D/Right = move right, Space = charge jump".into());

    Ok(())
}

fn request_animation_frame(window: &web_sys::Window, f: &Closure<dyn FnMut()>) {
    window
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}
