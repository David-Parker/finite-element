//! Sprite types for 2D rendering

use super::{Color, TextureId};

/// A 2D sprite for rendering textured quads
#[derive(Clone, Debug)]
pub struct Sprite {
    /// Texture to render
    pub texture: TextureId,
    /// Position (center)
    pub position: (f32, f32),
    /// Size (width, height)
    pub size: (f32, f32),
    /// Rotation in radians
    pub rotation: f32,
    /// UV region within texture (min_u, min_v, max_u, max_v)
    /// Defaults to (0, 0, 1, 1) for full texture
    pub uv_rect: (f32, f32, f32, f32),
    /// Color tint
    pub color: Color,
    /// Anchor point (0.5, 0.5 = center, 0.0, 0.0 = top-left)
    pub anchor: (f32, f32),
    /// Flip horizontally
    pub flip_x: bool,
    /// Flip vertically
    pub flip_y: bool,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            texture: TextureId::NONE,
            position: (0.0, 0.0),
            size: (1.0, 1.0),
            rotation: 0.0,
            uv_rect: (0.0, 0.0, 1.0, 1.0),
            color: Color::WHITE,
            anchor: (0.5, 0.5),
            flip_x: false,
            flip_y: false,
        }
    }
}

impl Sprite {
    /// Create a sprite with a texture
    pub fn new(texture: TextureId) -> Self {
        Self {
            texture,
            ..Default::default()
        }
    }

    /// Set position
    pub fn at(mut self, x: f32, y: f32) -> Self {
        self.position = (x, y);
        self
    }

    /// Set size
    pub fn sized(mut self, width: f32, height: f32) -> Self {
        self.size = (width, height);
        self
    }

    /// Set uniform scale
    pub fn scaled(mut self, scale: f32) -> Self {
        self.size = (self.size.0 * scale, self.size.1 * scale);
        self
    }

    /// Set rotation in radians
    pub fn rotated(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }

    /// Set UV rectangle for sprite sheet
    pub fn with_uv(mut self, min_u: f32, min_v: f32, max_u: f32, max_v: f32) -> Self {
        self.uv_rect = (min_u, min_v, max_u, max_v);
        self
    }

    /// Set color tint
    pub fn tinted(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set anchor point
    pub fn anchored(mut self, x: f32, y: f32) -> Self {
        self.anchor = (x, y);
        self
    }

    /// Flip horizontally
    pub fn flipped_x(mut self) -> Self {
        self.flip_x = !self.flip_x;
        self
    }

    /// Flip vertically
    pub fn flipped_y(mut self) -> Self {
        self.flip_y = !self.flip_y;
        self
    }
}

/// Frame within a sprite sheet
#[derive(Clone, Copy, Debug)]
pub struct SpriteFrame {
    /// UV rectangle (min_u, min_v, max_u, max_v)
    pub uv_rect: (f32, f32, f32, f32),
    /// Duration in seconds (for animations)
    pub duration: f32,
}

/// Sprite sheet for animations
#[derive(Clone, Debug)]
pub struct SpriteSheet {
    /// Texture containing all frames
    pub texture: TextureId,
    /// Individual frames
    pub frames: Vec<SpriteFrame>,
}

impl SpriteSheet {
    /// Create from a grid of uniform cells
    pub fn from_grid(
        texture: TextureId,
        cols: u32,
        rows: u32,
        frame_duration: f32,
    ) -> Self {
        let mut frames = Vec::with_capacity((cols * rows) as usize);
        let cell_w = 1.0 / cols as f32;
        let cell_h = 1.0 / rows as f32;

        for row in 0..rows {
            for col in 0..cols {
                let min_u = col as f32 * cell_w;
                let min_v = row as f32 * cell_h;
                frames.push(SpriteFrame {
                    uv_rect: (min_u, min_v, min_u + cell_w, min_v + cell_h),
                    duration: frame_duration,
                });
            }
        }

        Self { texture, frames }
    }

    /// Get frame at index (wraps around)
    pub fn frame(&self, index: usize) -> &SpriteFrame {
        &self.frames[index % self.frames.len()]
    }

    /// Get frame for a given time (looping animation)
    pub fn frame_at_time(&self, time: f32) -> &SpriteFrame {
        let total_duration: f32 = self.frames.iter().map(|f| f.duration).sum();
        let mut t = time % total_duration;

        for frame in &self.frames {
            if t < frame.duration {
                return frame;
            }
            t -= frame.duration;
        }

        self.frames.last().unwrap()
    }
}
