//! WebGL renderer for soft body simulation

use wasm_bindgen::prelude::*;
use web_sys::{WebGlRenderingContext as GL, WebGlProgram, WebGlBuffer, WebGlUniformLocation};

/// Body mesh data for rendering
pub struct BodyMesh<'a> {
    pub positions: &'a [f32],
    pub triangles: &'a [u32],
}

pub struct Renderer {
    gl: GL,
    program: WebGlProgram,
    vertex_buffer: WebGlBuffer,
    index_buffer: WebGlBuffer,
    ground_buffer: WebGlBuffer,
    u_scale: WebGlUniformLocation,
    u_color: WebGlUniformLocation,
    a_position: u32,
    pub view_size: f32,
    pub scale: f32,
}

impl Renderer {
    pub fn new(gl: GL) -> Result<Self, JsValue> {
        let view_size = 10.0;
        let scale = 1.0 / view_size;

        // Create shaders
        let vs_source = r#"
            attribute vec2 aPosition;
            uniform float uScale;
            void main() {
                gl_Position = vec4(aPosition * uScale, 0.0, 1.0);
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

        // Get locations
        let a_position = gl.get_attrib_location(&program, "aPosition") as u32;
        let u_scale = gl.get_uniform_location(&program, "uScale")
            .ok_or("Failed to get uScale location")?;
        let u_color = gl.get_uniform_location(&program, "uColor")
            .ok_or("Failed to get uColor location")?;

        gl.enable_vertex_attrib_array(a_position);

        // Create buffers
        let vertex_buffer = gl.create_buffer().ok_or("Failed to create vertex buffer")?;
        let index_buffer = gl.create_buffer().ok_or("Failed to create index buffer")?;
        let ground_buffer = gl.create_buffer().ok_or("Failed to create ground buffer")?;

        Ok(Renderer {
            gl,
            program,
            vertex_buffer,
            index_buffer,
            ground_buffer,
            u_scale,
            u_color,
            a_position,
            view_size,
            scale,
        })
    }

    pub fn set_ground(&self, ground_y: f32) {
        let gl = &self.gl;
        let vertices: [f32; 4] = [-self.view_size, ground_y, self.view_size, ground_y];

        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.ground_buffer));
        unsafe {
            let array = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::STATIC_DRAW);
        }
    }

    /// Render bodies with individual mesh data
    pub fn render_meshes(&self, bodies: &[BodyMesh], colors: &[(f32, f32, f32)]) {
        let gl = &self.gl;

        gl.clear_color(0.07, 0.07, 0.07, 1.0);
        gl.clear(GL::COLOR_BUFFER_BIT);
        gl.uniform1f(Some(&self.u_scale), self.scale);

        // Draw ground
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.ground_buffer));
        gl.vertex_attrib_pointer_with_i32(self.a_position, 2, GL::FLOAT, false, 0, 0);
        gl.uniform3f(Some(&self.u_color), 0.5, 0.5, 0.5);
        gl.draw_arrays(GL::LINES, 0, 2);

        // Draw each body
        for (i, body) in bodies.iter().enumerate() {
            let (r, g, b) = colors[i % colors.len()];

            // Upload positions
            gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
            unsafe {
                let array = js_sys::Float32Array::view(body.positions);
                gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
            }
            gl.vertex_attrib_pointer_with_i32(self.a_position, 2, GL::FLOAT, false, 0, 0);

            // Upload indices
            gl.bind_buffer(GL::ELEMENT_ARRAY_BUFFER, Some(&self.index_buffer));
            unsafe {
                let indices_u16 = u32_to_u16(body.triangles);
                let array = js_sys::Uint16Array::view(&indices_u16);
                gl.buffer_data_with_array_buffer_view(GL::ELEMENT_ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
            }

            // Draw filled triangles
            gl.uniform3f(Some(&self.u_color), r * 0.3, g * 0.3, b * 0.3);
            gl.draw_elements_with_i32(GL::TRIANGLES, body.triangles.len() as i32, GL::UNSIGNED_SHORT, 0);

            // Draw wireframe (using triangle edges)
            gl.uniform3f(Some(&self.u_color), r, g, b);
            // Draw as line loop per triangle for wireframe effect
            for tri in 0..(body.triangles.len() / 3) {
                let base = (tri * 3) as i32;
                gl.draw_elements_with_i32(GL::LINE_LOOP, 3, GL::UNSIGNED_SHORT, base * 2);
            }
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

fn u32_to_u16(data: &[u32]) -> Vec<u16> {
    data.iter().map(|&x| x as u16).collect()
}
