//! Depth estimator interface plus the pure-Rust pre/post-processing shared by every backend.
//!
//! The neural network itself (Depth Anything V3 Small, ONNX) lives in [`crate::onnx`] and is only
//! compiled for the native CLI (`--features onnx`). Everything here is dependency-free and testable.

/// ImageNet statistics used by the Depth Anything family.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// ViT patch size; inference dimensions must be multiples of it.
pub const PATCH: usize = 14;

/// A monocular depth backend. Returns a `width * height` map normalised to `[0, 1]`
/// where `0.0` is nearest and `1.0` is farthest.
pub trait DepthEstimator {
    fn id(&self) -> &str;
    fn estimate(&mut self, rgb: &[[f32; 3]], width: usize, height: usize) -> Result<Vec<f32>, String>;
}

/// Inference size for an image: aspect preserved, long side ≈ `long_side`, both sides a multiple of [`PATCH`].
pub fn inference_size(width: usize, height: usize, long_side: usize) -> (usize, usize) {
    let long = width.max(height).max(1) as f32;
    let scale = long_side as f32 / long;
    let snap = |v: usize| (((v as f32 * scale) / PATCH as f32).round() as usize).max(1) * PATCH;
    (snap(width), snap(height))
}

/// Bilinear resample of a single-channel plane.
pub fn resize_plane(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || src.len() < sw * sh {
        return vec![0.0; dw * dh];
    }
    if sw == dw && sh == dh {
        return src[..sw * sh].to_vec();
    }
    let mut out = vec![0.0f32; dw * dh];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f32;
            let top = src[y0 * sw + x0] * (1.0 - tx) + src[y0 * sw + x1] * tx;
            let bot = src[y1 * sw + x0] * (1.0 - tx) + src[y1 * sw + x1] * tx;
            out[y * dw + x] = top * (1.0 - ty) + bot * ty;
        }
    }
    out
}

/// Resize RGB to `dw x dh` and return a planar `[3, dh, dw]` tensor, ImageNet-normalised.
pub fn to_nchw_normalized(rgb: &[[f32; 3]], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * dw * dh);
    for c in 0..3 {
        let plane: Vec<f32> = rgb.iter().take(sw * sh).map(|p| p[c].clamp(0.0, 1.0)).collect();
        let resized = resize_plane(&plane, sw, sh, dw, dh);
        out.extend(resized.into_iter().map(|v| (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c]));
    }
    out
}

/// Min–max normalise to `[0, 1]` (non-finite values become 0). A flat map becomes all zeros.
pub fn normalize_depth(plane: &mut [f32]) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for v in plane.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    let span = hi - lo;
    if !span.is_finite() || span < 1e-9 {
        plane.fill(0.0);
        return;
    }
    for v in plane.iter_mut() {
        *v = (*v - lo) / span;
    }
}
