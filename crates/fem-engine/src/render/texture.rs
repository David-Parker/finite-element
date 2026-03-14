//! Texture types and traits

/// Opaque handle to a texture resource
///
/// The actual texture data lives in the platform-specific renderer.
/// This is just an ID that the renderer uses to look up the texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

impl TextureId {
    /// A null/invalid texture ID
    pub const NONE: Self = Self(u32::MAX);

    /// Check if this is a valid texture ID
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

/// Texture pixel format
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFormat {
    /// 8-bit grayscale
    R8,
    /// 8-bit red + green
    Rg8,
    /// 24-bit RGB
    Rgb8,
    /// 32-bit RGBA
    Rgba8,
}

impl TextureFormat {
    /// Bytes per pixel for this format
    #[inline]
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::Rg8 => 2,
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }
}

/// Texture filtering mode
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextureFilter {
    /// Nearest-neighbor (pixelated)
    Nearest,
    /// Bilinear (smooth)
    #[default]
    Linear,
}

/// Texture wrapping mode
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextureWrap {
    /// Repeat the texture
    #[default]
    Repeat,
    /// Clamp to edge pixels
    ClampToEdge,
    /// Mirror and repeat
    MirroredRepeat,
}

/// Texture descriptor for creating textures
#[derive(Clone, Debug)]
pub struct Texture {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel format
    pub format: TextureFormat,
    /// Pixel data (row-major, top-to-bottom)
    pub data: Vec<u8>,
    /// Minification filter
    pub min_filter: TextureFilter,
    /// Magnification filter
    pub mag_filter: TextureFilter,
    /// Horizontal wrap mode
    pub wrap_u: TextureWrap,
    /// Vertical wrap mode
    pub wrap_v: TextureWrap,
}

impl Texture {
    /// Create a new texture with default filtering (linear) and wrapping (repeat)
    pub fn new(width: u32, height: u32, format: TextureFormat, data: Vec<u8>) -> Self {
        let expected_size = (width * height) as usize * format.bytes_per_pixel();
        assert_eq!(
            data.len(),
            expected_size,
            "Texture data size mismatch: expected {}, got {}",
            expected_size,
            data.len()
        );

        Self {
            width,
            height,
            format,
            data,
            min_filter: TextureFilter::Linear,
            mag_filter: TextureFilter::Linear,
            wrap_u: TextureWrap::Repeat,
            wrap_v: TextureWrap::Repeat,
        }
    }

    /// Create a 1x1 solid color texture
    pub fn solid_color(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::new(1, 1, TextureFormat::Rgba8, vec![r, g, b, a])
    }

    /// Create a checkerboard pattern texture
    pub fn checkerboard(size: u32, tile_size: u32, color1: [u8; 4], color2: [u8; 4]) -> Self {
        let mut data = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                let tile_x = x / tile_size;
                let tile_y = y / tile_size;
                let color = if (tile_x + tile_y) % 2 == 0 {
                    color1
                } else {
                    color2
                };
                data.extend_from_slice(&color);
            }
        }
        Self::new(size, size, TextureFormat::Rgba8, data)
    }

    /// Set filtering mode
    pub fn with_filter(mut self, filter: TextureFilter) -> Self {
        self.min_filter = filter;
        self.mag_filter = filter;
        self
    }

    /// Set wrap mode
    pub fn with_wrap(mut self, wrap: TextureWrap) -> Self {
        self.wrap_u = wrap;
        self.wrap_v = wrap;
        self
    }
}
