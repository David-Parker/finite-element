//! WebGL renderer for the ring game with texture support

use wasm_bindgen::prelude::*;
use web_sys::{
    WebGlRenderingContext as GL, WebGlProgram, WebGlBuffer,
    WebGlTexture, WebGlUniformLocation, HtmlImageElement,
};

/// Render data for a body including positions, UVs, and indices
pub struct BodyRenderData<'a> {
    pub positions: &'a [f32],
    pub uvs: &'a [f32],
    pub indices: &'a [u32],
    pub use_texture: bool,
}

pub struct Renderer {
    gl: GL,

    // Solid color shader
    solid_program: WebGlProgram,
    solid_a_position: u32,
    solid_u_offset: WebGlUniformLocation,
    solid_u_scale: WebGlUniformLocation,
    solid_u_color: WebGlUniformLocation,

    // Textured shader
    textured_program: WebGlProgram,
    textured_a_position: u32,
    textured_a_uv: u32,
    textured_u_offset: WebGlUniformLocation,
    textured_u_scale: WebGlUniformLocation,
    textured_u_sampler: WebGlUniformLocation,

    // Buffers
    vertex_buffer: WebGlBuffer,
    uv_buffer: WebGlBuffer,
    index_buffer: WebGlBuffer,
    ground_buffer: WebGlBuffer,

    // Textures
    player_texture: Option<WebGlTexture>,
    dropped_texture: Option<WebGlTexture>,

    // View state
    view_width: f32,
    view_height: f32,
    camera_x: f32,
    camera_y: f32,
    ground_y: f32,

    // Render mode
    wireframe_mode: bool,
}

impl Renderer {
    pub fn new(gl: GL) -> Result<Self, JsValue> {
        // === Solid color shader ===
        let solid_vs = r#"
            attribute vec2 aPosition;
            uniform vec2 uOffset;
            uniform vec2 uScale;
            void main() {
                vec2 pos = (aPosition - uOffset) * uScale;
                gl_Position = vec4(pos, 0.0, 1.0);
            }
        "#;

        let solid_fs = r#"
            precision mediump float;
            uniform vec3 uColor;
            void main() {
                gl_FragColor = vec4(uColor, 1.0);
            }
        "#;

        let solid_program = create_program(&gl, solid_vs, solid_fs)?;
        gl.use_program(Some(&solid_program));

        let solid_a_position = gl.get_attrib_location(&solid_program, "aPosition") as u32;
        let solid_u_offset = gl.get_uniform_location(&solid_program, "uOffset")
            .ok_or("Failed to get solid uOffset")?;
        let solid_u_scale = gl.get_uniform_location(&solid_program, "uScale")
            .ok_or("Failed to get solid uScale")?;
        let solid_u_color = gl.get_uniform_location(&solid_program, "uColor")
            .ok_or("Failed to get solid uColor")?;

        // === Textured shader ===
        let textured_vs = r#"
            attribute vec2 aPosition;
            attribute vec2 aUV;
            uniform vec2 uOffset;
            uniform vec2 uScale;
            varying vec2 vUV;
            void main() {
                vec2 pos = (aPosition - uOffset) * uScale;
                gl_Position = vec4(pos, 0.0, 1.0);
                vUV = aUV;
            }
        "#;

        let textured_fs = r#"
            precision mediump float;
            uniform sampler2D uSampler;
            varying vec2 vUV;
            void main() {
                gl_FragColor = texture2D(uSampler, vUV);
            }
        "#;

        let textured_program = create_program(&gl, textured_vs, textured_fs)?;
        gl.use_program(Some(&textured_program));

        let textured_a_position = gl.get_attrib_location(&textured_program, "aPosition") as u32;
        let textured_a_uv = gl.get_attrib_location(&textured_program, "aUV") as u32;
        let textured_u_offset = gl.get_uniform_location(&textured_program, "uOffset")
            .ok_or("Failed to get textured uOffset")?;
        let textured_u_scale = gl.get_uniform_location(&textured_program, "uScale")
            .ok_or("Failed to get textured uScale")?;
        let textured_u_sampler = gl.get_uniform_location(&textured_program, "uSampler")
            .ok_or("Failed to get textured uSampler")?;

        // Create buffers
        let vertex_buffer = gl.create_buffer().ok_or("Failed to create vertex buffer")?;
        let uv_buffer = gl.create_buffer().ok_or("Failed to create UV buffer")?;
        let index_buffer = gl.create_buffer().ok_or("Failed to create index buffer")?;
        let ground_buffer = gl.create_buffer().ok_or("Failed to create ground buffer")?;

        // Enable blending for textures with transparency
        gl.enable(GL::BLEND);
        gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);

        Ok(Renderer {
            gl,
            solid_program,
            solid_a_position,
            solid_u_offset,
            solid_u_scale,
            solid_u_color,
            textured_program,
            textured_a_position,
            textured_a_uv,
            textured_u_offset,
            textured_u_scale,
            textured_u_sampler,
            vertex_buffer,
            uv_buffer,
            index_buffer,
            ground_buffer,
            player_texture: None,
            dropped_texture: None,
            view_width: 20.0,
            view_height: 15.0,
            camera_x: 0.0,
            camera_y: 0.0,
            ground_y: -5.0,
            wireframe_mode: false,
        })
    }

    /// Load a texture from an HtmlImageElement
    fn create_texture_from_image(&self, image: &HtmlImageElement) -> Result<WebGlTexture, JsValue> {
        let gl = &self.gl;

        let texture = gl.create_texture().ok_or("Failed to create texture")?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&texture));

        // Upload image to texture
        gl.tex_image_2d_with_u32_and_u32_and_image(
            GL::TEXTURE_2D,
            0,
            GL::RGBA as i32,
            GL::RGBA,
            GL::UNSIGNED_BYTE,
            image,
        )?;

        // Check if texture is power-of-two (required for mipmaps in WebGL 1)
        let width = image.width();
        let height = image.height();
        let is_pot = width.is_power_of_two() && height.is_power_of_two();

        // Set texture parameters
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);

        if is_pot {
            // Generate mipmaps for better quality when scaled down
            gl.generate_mipmap(GL::TEXTURE_2D);
            // Use trilinear filtering (linear mipmap + linear) to reduce aliasing
            gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::LINEAR_MIPMAP_LINEAR as i32);
        } else {
            // NPOT textures can't have mipmaps in WebGL 1, use linear filtering
            gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::LINEAR as i32);
        }
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::LINEAR as i32);

        Ok(texture)
    }

    /// Load player texture from an HtmlImageElement
    pub fn load_player_texture(&mut self, image: &HtmlImageElement) -> Result<(), JsValue> {
        self.player_texture = Some(self.create_texture_from_image(image)?);
        Ok(())
    }

    /// Load dropped ring texture from an HtmlImageElement
    pub fn load_dropped_texture(&mut self, image: &HtmlImageElement) -> Result<(), JsValue> {
        self.dropped_texture = Some(self.create_texture_from_image(image)?);
        Ok(())
    }

    /// Check if player texture is loaded
    pub fn has_player_texture(&self) -> bool {
        self.player_texture.is_some()
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

    /// Toggle between mesh and wireframe rendering modes
    pub fn toggle_wireframe(&mut self) {
        self.wireframe_mode = !self.wireframe_mode;
    }

    /// Check if wireframe mode is enabled
    pub fn is_wireframe(&self) -> bool {
        self.wireframe_mode
    }

    fn draw_ground(&self) {
        let gl = &self.gl;

        // Use solid shader
        gl.use_program(Some(&self.solid_program));

        let scale_x = 2.0 / self.view_width;
        let scale_y = 2.0 / self.view_height;
        gl.uniform2f(Some(&self.solid_u_offset), self.camera_x, self.camera_y);
        gl.uniform2f(Some(&self.solid_u_scale), scale_x, scale_y);

        // Ground line
        let left = self.camera_x - self.view_width * 2.0;
        let right = self.camera_x + self.view_width * 2.0;
        let vertices: [f32; 4] = [left, self.ground_y, right, self.ground_y];

        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.ground_buffer));
        unsafe {
            let array = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }

        gl.enable_vertex_attrib_array(self.solid_a_position);
        gl.vertex_attrib_pointer_with_i32(self.solid_a_position, 2, GL::FLOAT, false, 0, 0);
        gl.uniform3f(Some(&self.solid_u_color), 0.3, 0.6, 0.3);
        gl.draw_arrays(GL::LINES, 0, 2);
    }

    /// Render bodies with the new format that includes UVs
    pub fn render_bodies_with_uvs(&self, bodies: &[BodyRenderData]) {
        let gl = &self.gl;

        // Clear
        gl.clear_color(0.05, 0.05, 0.08, 1.0);
        gl.clear(GL::COLOR_BUFFER_BIT);

        // Draw ground
        self.draw_ground();

        // Draw all bodies
        for (i, body) in bodies.iter().enumerate() {
            let color = if i == 0 {
                (0.3, 0.9, 1.0)  // Player: cyan
            } else {
                (1.0, 0.6, 0.2)  // Dropped rings: orange
            };

            if self.wireframe_mode {
                // Wireframe only mode
                self.draw_wireframe_only(body.positions, body.indices, color);
            } else if i == 0 && body.use_texture && self.player_texture.is_some() {
                // Player with pink donut texture
                self.draw_textured_body_with(body.positions, body.uvs, body.indices, self.player_texture.as_ref());
            } else if i > 0 && body.use_texture && self.dropped_texture.is_some() {
                // Dropped rings with chocolate donut texture
                self.draw_textured_body_with(body.positions, body.uvs, body.indices, self.dropped_texture.as_ref());
            } else {
                // Fallback to solid color with wireframe
                self.draw_solid_body(body.positions, body.indices, color);
            }
        }
    }

    /// Legacy render method for compatibility
    pub fn render_bodies(&self, bodies: &[(&[f32], &[u32])]) {
        let gl = &self.gl;

        // Clear
        gl.clear_color(0.05, 0.05, 0.08, 1.0);
        gl.clear(GL::COLOR_BUFFER_BIT);

        // Draw ground
        self.draw_ground();

        // Draw all bodies with solid colors
        for (i, (positions, triangles)) in bodies.iter().enumerate() {
            let color = if i == 0 {
                (0.3, 0.9, 1.0)
            } else {
                (1.0, 0.6, 0.2)
            };
            self.draw_solid_body(positions, triangles, color);
        }
    }

    fn draw_textured_body_with(&self, positions: &[f32], uvs: &[f32], indices: &[u32], texture: Option<&WebGlTexture>) {
        let gl = &self.gl;

        gl.use_program(Some(&self.textured_program));

        let scale_x = 2.0 / self.view_width;
        let scale_y = 2.0 / self.view_height;
        gl.uniform2f(Some(&self.textured_u_offset), self.camera_x, self.camera_y);
        gl.uniform2f(Some(&self.textured_u_scale), scale_x, scale_y);

        // Bind texture
        gl.active_texture(GL::TEXTURE0);
        gl.bind_texture(GL::TEXTURE_2D, texture);
        gl.uniform1i(Some(&self.textured_u_sampler), 0);

        // Upload positions
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
        unsafe {
            let array = js_sys::Float32Array::view(positions);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }
        gl.enable_vertex_attrib_array(self.textured_a_position);
        gl.vertex_attrib_pointer_with_i32(self.textured_a_position, 2, GL::FLOAT, false, 0, 0);

        // Upload UVs
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.uv_buffer));
        unsafe {
            let array = js_sys::Float32Array::view(uvs);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }
        gl.enable_vertex_attrib_array(self.textured_a_uv);
        gl.vertex_attrib_pointer_with_i32(self.textured_a_uv, 2, GL::FLOAT, false, 0, 0);

        // Upload indices
        gl.bind_buffer(GL::ELEMENT_ARRAY_BUFFER, Some(&self.index_buffer));
        let indices_u16: Vec<u16> = indices.iter().map(|&x| x as u16).collect();
        unsafe {
            let array = js_sys::Uint16Array::view(&indices_u16);
            gl.buffer_data_with_array_buffer_view(GL::ELEMENT_ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }

        // Draw
        gl.draw_elements_with_i32(GL::TRIANGLES, indices.len() as i32, GL::UNSIGNED_SHORT, 0);
    }

    fn draw_wireframe_only(&self, positions: &[f32], indices: &[u32], color: (f32, f32, f32)) {
        let gl = &self.gl;
        let (r, g, b) = color;

        gl.use_program(Some(&self.solid_program));

        let scale_x = 2.0 / self.view_width;
        let scale_y = 2.0 / self.view_height;
        gl.uniform2f(Some(&self.solid_u_offset), self.camera_x, self.camera_y);
        gl.uniform2f(Some(&self.solid_u_scale), scale_x, scale_y);

        // Upload positions
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
        unsafe {
            let array = js_sys::Float32Array::view(positions);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }
        gl.enable_vertex_attrib_array(self.solid_a_position);
        gl.vertex_attrib_pointer_with_i32(self.solid_a_position, 2, GL::FLOAT, false, 0, 0);

        // Upload indices
        gl.bind_buffer(GL::ELEMENT_ARRAY_BUFFER, Some(&self.index_buffer));
        let indices_u16: Vec<u16> = indices.iter().map(|&x| x as u16).collect();
        unsafe {
            let array = js_sys::Uint16Array::view(&indices_u16);
            gl.buffer_data_with_array_buffer_view(GL::ELEMENT_ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }

        // Draw wireframe only (no fill)
        gl.uniform3f(Some(&self.solid_u_color), r, g, b);
        for tri in 0..(indices.len() / 3) {
            let base = (tri * 3) as i32;
            gl.draw_elements_with_i32(GL::LINE_LOOP, 3, GL::UNSIGNED_SHORT, base * 2);
        }
    }

    fn draw_solid_body(&self, positions: &[f32], indices: &[u32], color: (f32, f32, f32)) {
        let gl = &self.gl;
        let (r, g, b) = color;

        gl.use_program(Some(&self.solid_program));

        let scale_x = 2.0 / self.view_width;
        let scale_y = 2.0 / self.view_height;
        gl.uniform2f(Some(&self.solid_u_offset), self.camera_x, self.camera_y);
        gl.uniform2f(Some(&self.solid_u_scale), scale_x, scale_y);

        // Upload positions
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.vertex_buffer));
        unsafe {
            let array = js_sys::Float32Array::view(positions);
            gl.buffer_data_with_array_buffer_view(GL::ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }
        gl.enable_vertex_attrib_array(self.solid_a_position);
        gl.vertex_attrib_pointer_with_i32(self.solid_a_position, 2, GL::FLOAT, false, 0, 0);

        // Upload indices
        gl.bind_buffer(GL::ELEMENT_ARRAY_BUFFER, Some(&self.index_buffer));
        let indices_u16: Vec<u16> = indices.iter().map(|&x| x as u16).collect();
        unsafe {
            let array = js_sys::Uint16Array::view(&indices_u16);
            gl.buffer_data_with_array_buffer_view(GL::ELEMENT_ARRAY_BUFFER, &array, GL::DYNAMIC_DRAW);
        }

        // Draw filled (darker)
        gl.uniform3f(Some(&self.solid_u_color), r * 0.25, g * 0.25, b * 0.25);
        gl.draw_elements_with_i32(GL::TRIANGLES, indices.len() as i32, GL::UNSIGNED_SHORT, 0);

        // Draw wireframe
        gl.uniform3f(Some(&self.solid_u_color), r, g, b);
        for tri in 0..(indices.len() / 3) {
            let base = (tri * 3) as i32;
            gl.draw_elements_with_i32(GL::LINE_LOOP, 3, GL::UNSIGNED_SHORT, base * 2);
        }
    }
}

fn create_program(gl: &GL, vs_source: &str, fs_source: &str) -> Result<WebGlProgram, String> {
    let vs = compile_shader(gl, GL::VERTEX_SHADER, vs_source)?;
    let fs = compile_shader(gl, GL::FRAGMENT_SHADER, fs_source)?;
    link_program(gl, &vs, &fs)
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
