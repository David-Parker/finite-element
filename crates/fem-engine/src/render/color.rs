//! Color types

/// RGBA color with components in 0.0-1.0 range
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self::rgb(1.0, 1.0, 1.0);
    pub const BLACK: Self = Self::rgb(0.0, 0.0, 0.0);
    pub const RED: Self = Self::rgb(1.0, 0.0, 0.0);
    pub const GREEN: Self = Self::rgb(0.0, 1.0, 0.0);
    pub const BLUE: Self = Self::rgb(0.0, 0.0, 1.0);
    pub const CYAN: Self = Self::rgb(0.3, 0.9, 1.0);
    pub const ORANGE: Self = Self::rgb(1.0, 0.6, 0.2);
    pub const TRANSPARENT: Self = Self::rgba(0.0, 0.0, 0.0, 0.0);

    /// Create an RGB color with full opacity
    #[inline]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create an RGBA color
    #[inline]
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create from 0-255 integer components
    #[inline]
    pub fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    /// Create from hex color (e.g., 0xFF5500 for orange)
    #[inline]
    pub fn from_hex(hex: u32) -> Self {
        Self::from_u8(
            ((hex >> 16) & 0xFF) as u8,
            ((hex >> 8) & 0xFF) as u8,
            (hex & 0xFF) as u8,
            255,
        )
    }

    /// Create from hex with alpha (e.g., 0xFF5500FF)
    #[inline]
    pub fn from_hex_alpha(hex: u32) -> Self {
        Self::from_u8(
            ((hex >> 24) & 0xFF) as u8,
            ((hex >> 16) & 0xFF) as u8,
            ((hex >> 8) & 0xFF) as u8,
            (hex & 0xFF) as u8,
        )
    }

    /// Multiply RGB by a factor (for darkening/lightening)
    #[inline]
    pub fn scale(self, factor: f32) -> Self {
        Self {
            r: (self.r * factor).clamp(0.0, 1.0),
            g: (self.g * factor).clamp(0.0, 1.0),
            b: (self.b * factor).clamp(0.0, 1.0),
            a: self.a,
        }
    }

    /// Return as [r, g, b] array
    #[inline]
    pub fn to_rgb_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Return as [r, g, b, a] array
    #[inline]
    pub fn to_rgba_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

impl From<[f32; 3]> for Color {
    fn from([r, g, b]: [f32; 3]) -> Self {
        Self::rgb(r, g, b)
    }
}

impl From<[f32; 4]> for Color {
    fn from([r, g, b, a]: [f32; 4]) -> Self {
        Self::rgba(r, g, b, a)
    }
}
