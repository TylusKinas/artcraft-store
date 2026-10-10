//! Depth filtering, edge-preserving bilateral refinement, remapping, and visualization modes.

/// Remap raw depth in [0, 1] through near/far clipping, inversion, and gamma curve.
#[inline]
pub fn remap_depth(raw: f32, near: f32, far: f32, invert: bool, gamma: f32) -> f32 {
    let near_val = (near / 100.0).clamp(0.0, 1.0);
    let far_val = (far / 100.0).clamp(0.0, 1.0);
    let span = (far_val - near_val).abs().max(1e-5);

    let mut val = if far_val >= near_val { (raw - near_val) / span } else { (near_val - raw) / span };
    val = val.clamp(0.0, 1.0);

    if invert {
        val = 1.0 - val;
    }

    if gamma > 0.0 && (gamma - 1.0).abs() > 1e-4 {
        val = val.powf(1.0 / gamma.clamp(0.1, 10.0));
    }

    val.clamp(0.0, 1.0)
}

/// Cross-bilateral filter that refines depth edges using image RGB guidance (see
/// [`refine_depth_guided`]). Aligns depth boundaries with object contours, preventing depth bleed.
pub fn refine_depth_edges(depth: &[f32], rgb: &[[f32; 3]], width: usize, height: usize, radius: usize, out: &mut [f32]) {
    if rgb.len() < width * height {
        return;
    }
    let lum: Vec<f32> = rgb.iter().take(width * height).map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).collect();
    refine_depth_guided(depth, &lum, width, height, radius, out);
}

/// Bilateral filter of `depth` guided by a luminance plane (`guide`; the depth itself for a depth layer).
///
/// Cost is bounded for hosts that meter instructions (EffectCraft gives wasm plug-ins a fuel budget
/// per pixel): weights come from lookup tables instead of `exp`, radius is capped at 3 (49 taps).
pub fn refine_depth_guided(depth: &[f32], guide: &[f32], width: usize, height: usize, radius: usize, out: &mut [f32]) {
    let n = width * height;
    if width == 0 || height == 0 || depth.len() < n || guide.len() < n || out.len() < n {
        return;
    }
    if radius == 0 {
        out[..n].copy_from_slice(&depth[..n]);
        return;
    }

    let r = radius.min(3);
    let side = 2 * r + 1;
    let spatial_sigma = (r as f32) * 0.75;
    let inv_spatial = 1.0 / (2.0 * spatial_sigma * spatial_sigma).max(1e-4);
    let mut spatial = [0.0f32; 49];
    for dy in 0..side {
        for dx in 0..side {
            let (ox, oy) = (dx as f32 - r as f32, dy as f32 - r as f32);
            spatial[dy * side + dx] = (-(ox * ox + oy * oy) * inv_spatial).exp();
        }
    }
    // Range weight by |luminance difference| quantised to 1/128.
    let inv_range = 1.0 / (2.0 * 0.15f32 * 0.15f32);
    let mut range = [0.0f32; 130];
    for (i, w) in range.iter_mut().enumerate() {
        let d = i as f32 / 128.0;
        *w = (-d * d * inv_range).exp();
    }

    for y in 0..height {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(height - 1);
        for x in 0..width {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(width - 1);
            let c_idx = y * width + x;
            let cg = guide[c_idx];
            let mut sw = 0.0f32;
            let mut sd = 0.0f32;
            for ny in y0..=y1 {
                let s_row = (ny + r - y) * side + r;
                let row = ny * width;
                for nx in x0..=x1 {
                    let i = row + nx;
                    let q = (((guide[i] - cg).abs() * 128.0) as usize).min(129);
                    let w = spatial[s_row + nx - x] * range[q];
                    sw += w;
                    sd += w * depth[i];
                }
            }
            out[c_idx] = if sw > 1e-6 { sd / sw } else { depth[c_idx] };
        }
    }
}

/// Maps normalized depth in [0, 1] to a high-contrast heatmap (Turbo colormap approximation).
/// Smooth perceptual gradient from near (warm reds/yellows) to far (cool blues/purples).
#[inline]
pub fn turbo_colormap(t: f32) -> [f32; 3] {
    let x = t.clamp(0.0, 1.0);

    // Polynomial approximation of Google's Turbo colormap (Anton Mikhailov, 2019)
    let r = (0.1357 + x * (4.5974 - x * (42.3277 - x * (130.5887 - x * (150.5667 - x * 58.1375))))).clamp(0.0, 1.0);
    let g = (0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 - x * (4.2109 - x * 2.7747))))).clamp(0.0, 1.0);
    let b = (0.1067 + x * (12.5833 - x * (78.4357 - x * (207.6083 - x * (244.9818 - x * 106.6667))))).clamp(0.0, 1.0);

    [r, g, b]
}

/// Blends depth-based volumetric atmospheric fog over source RGB pixels.
#[inline]
pub fn blend_fog(src_rgb: [f32; 3], depth: f32, density: f32, fog_color: [f32; 3]) -> [f32; 3] {
    let d = (depth * (density / 100.0)).clamp(0.0, 1.0);
    [src_rgb[0] * (1.0 - d) + fog_color[0] * d, src_rgb[1] * (1.0 - d) + fog_color[1] * d, src_rgb[2] * (1.0 - d) + fog_color[2] * d]
}

/// Generates depth slice / isoline highlight.
#[inline]
pub fn depth_slice_highlight(depth: f32, center: f32, width: f32) -> f32 {
    let c = (center / 100.0).clamp(0.0, 1.0);
    let w = (width / 100.0).clamp(0.01, 1.0);
    let dist = (depth - c).abs() / w;
    if dist <= 1.0 { 1.0 - dist } else { 0.0 }
}
