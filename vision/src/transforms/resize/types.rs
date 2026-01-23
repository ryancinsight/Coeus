/// Interpolation mode for resizing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationMode {
    /// Bilinear interpolation (default for images)
    #[default]
    Bilinear,
    /// Nearest neighbor interpolation
    Nearest,
    /// Bicubic interpolation
    Bicubic,
}
