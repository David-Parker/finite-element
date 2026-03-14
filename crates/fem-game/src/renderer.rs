//! WebGL renderer for the ring game

use wasm_bindgen::prelude::*;
use web_sys::{WebGlRenderingContext as GL, WebGlProgram, WebGlBuffer, WebGlUniformLocation};

pub struct Renderer {
    gl: GL,
    program: WebGlProgram,
    vertex_buffer: WebGlBuffer,
    index_buffer: WebGlBuffer,
    ground_buffer: WebGlBuffer,
    u_offset: WebGlUniformLocation,
    u_scale: WebGlUniformLocation,
    u_color: WebGlUniformLocation,
    a_position: u32,
    view_width: f32,
    view_height: f32,
    camera_x: f32,
    camera_y: f32,
    ground_y: f32,
}

impl Renderer {
    pub fn new(gl: GL) -> Result<Self, JsValue> {
        // Create shaders with camera offset
        let vs_source = r#"
            attribute vec2 aPosition;
            uniform vec2 uOffset;
            uniform vec2 uScale;
            void main() {
                vec2 pos = (aPosition - uOffset) * uScale;
                gl_Position = vec4(pos, 0.0, 1.0);
            }
        "#;

        let fs_source = r#"
            precision mediump float;
            uniform vec3 uColor;
            void main() {
                gl_FragColor = vec4(uColor, 1.0);
            }
        "#;

        let vs = compile_shader(&gl, GL::VERTEX_SHADER, vs_source)?;
        let fs = compile_shader(&gl, GL::FRAGMENT_SHADER, fs_source)?;
        let program = link_program(&gl, &vs, &fs)?;

        gl.use_program(Some(&program));

        let a_position = gl.get_attrib_location(&program, "aPosition") as u32;
        let u_offset = gl.get_uniform_location(&program, "uOffset")
            .ok_or("Failed to get uOffset location")?;
        let u_scale = gl.get_uniform_location(&program, "uScale")
            .ok_or("Failed to get uScale location")?;
        let u_color = gl.get_uniform_location(&program, "uColor")
            .ok_or("Failed to get uColor location")?;

        gl.enable_vertex_attrib_array(a_position);

        let vertex_buffer = gl.create_buffer().ok_or("Failed to create vertex buffer")?;
        let index_buffer = gl.create_buffer().ok_or("Failed to create index buffer")?;
        let ground_buffer = gl.create_buffer().ok_or("Failed to create ground buffer")?;

        Ok(Renderer {
            gl,
            program,
            vertex_buffer,
            index_buffer,
            ground_buffer,
            u_offset,
            u_scale,
            u_color,
            a_position,
            view_width: 20.0,
            view_height: 15.0,
            camera_x: 0.0,
            camera_y: 0.0,
            ground_y: -5.0,
        })
    }

    pub fn set_view(&mut self, width: f32, height: f32) {
        self.view_width = width;
        self.view_height = height;
    }

    pub fn set_camera(&mut self, x: f32, y: f32) {
        self.camera_x = x;
        self.camera_y = y;
    }

    pub fn set_ground(&mut self, y: f32) {
        self.ground_y = y;
    }

    fn update_ground_buffer(&self) {
        let gl = &self.gl;
        // Ground line spans far beyond visible area
        let left = self.camera_x - self.view_width * 2.0;
        let right = self.camera_x + self.view_width * 2.0;
        let vertices: [f32; 4] = [left, self.ground_y, right, self.ground_y];

        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.ground_buffer));
        unsafe {
            let array = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }
    }

    pub fn render_bodies(&self, bodies: &[(&[f32], &[u32])]) {
        let gl = &self.gl;

        // Clear
        gl.clear_color(0.05, 0.05, 0.08, 1.0);
        gl.clear(GL::COLOR_BUFFER_BIT);

        // Set uniforms
        let scale_x = 2.0 / self.view_width;
        let scale_y = 2.0 / self.view_height;
        gl.uniform2f(Some(&self.u_offset), self.camera_x, self.camera_y);
        gl.uniform2f(Some(&self.u_scale), scale_x, scale_y);

        // Draw ground
        self.update_ground_buffer();
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.ground_buffer));
        gl.vertex_attrib_pointer_with_i32(self.a_position, 2, GL::FLOAT, false, 0, 0);
        gl.uniform3f(Some(&self.u_color), 0.3, 0.6, 0.3);  // Green ground
        gl.draw_arrays(GL::LINES, 0, 2);

        // Draw all bodies - player (index 0) is cyan, others are orange
        for (i, (positions, triangles)) in bodies.iter().enumerate() {
            let color = if i == 0 {
                (0.3, 0.9, 1.0)  // Player: cyan
            } else {
                (1.0, 0.6, 0.2)  // Dropped rings: orange
            };
            self.draw_body(positions, triangles, color);
        }
    }

    fn draw_body(&self, positions: &[f32], triangles: &[u32], color: (f32, f32, f32)) {
        let gl = &self.gl;
        let (r, g, b) = color;

        // Upload positions
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
        unsafe {
            let array = js_sys::Float32Array::view(positions);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }
        gl.vertex_attrib_pointer_with_i32(self.a_position, 2, GL::FLOAT, false, 0, 0);

        // Upload indices
        gl.bind_buffer(GL::ELEMENT_ARRAY_BUFFER, Some(&self.index_buffer));
        let indices_u16: Vec<u16> = triangles.iter().map(|&x| x as u16).collect();
        unsafe {
            let array = js_sys::Uint16Array::view(&indices_u16);
            gl.buffer_data_with_array_buffer_view(GL::ELEMENT_ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }

        // Draw filled (darker)
        gl.uniform3f(Some(&self.u_color), r * 0.25, g * 0.25, b * 0.25);
        gl.draw_elements_with_i32(GL::TRIANGLES, triangles.len() as i32, GL::UNSIGNED_SHORT, 0);

        // Draw wireframe
        gl.uniform3f(Some(&self.u_color), r, g, b);
        for tri in 0..(triangles.len() / 3) {
            let base = (tri * 3) as i32;
            gl.draw_elements_with_i32(GL::LINE_LOOP, 3, GL::UNSIGNED_SHORT, base * 2);
        }
    }
}

fn compile_shader(gl: &GL, shader_type: u32, source: &str) -> Result<web_sys::WebGlShader, String> {
    let shader = gl.create_shader(shader_type)
        .ok_or_else(|| String::from("Failed to create shader"))?;

    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl.get_shader_parameter(&shader, GL::COMPILE_STATUS).as_bool().unwrap_or(false) {
        Ok(shader)
    } else {
        Err(gl.get_shader_info_log(&shader).unwrap_or_else(|| String::from("Unknown error")))
    }
}

fn link_program(gl: &GL, vs: &web_sys::WebGlShader, fs: &web_sys::WebGlShader) -> Result<WebGlProgram, String> {
    let program = gl.create_program()
        .ok_or_else(|| String::from("Failed to create program"))?;

    gl.attach_shader(&program, vs);
    gl.attach_shader(&program, fs);
    gl.link_program(&program);

    if gl.get_program_parameter(&program, GL::LINK_STATUS).as_bool().unwrap_or(false) {
        Ok(program)
    } else {
        Err(gl.get_program_info_log(&program).unwrap_or_else(|| String::from("Unknown error")))
    }
}
