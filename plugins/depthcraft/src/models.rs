//! Depth model registry, license documentation, and weight verification.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceSelection {
    /// Always works, slower.
    Cpu,
    /// CUDA / DirectML / CoreML through ONNX Runtime, much faster. Errors if no GPU provider is present.
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPreset {
    Fast,     // long side 252 px
    Balanced, // long side 364 px
    Ultra,    // long side 518 px (native Depth Anything resolution)
}

impl QualityPreset {
    /// Inference long side in pixels (multiples of the 14 px ViT patch).
    pub fn long_side(&self) -> usize {
        match self {
            QualityPreset::Fast => 252,
            QualityPreset::Balanced => 364,
            QualityPreset::Ultra => 518,
        }
    }
}

/// Metadata describing the supported depth estimation model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub authors: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub homepage: &'static str,
    /// Where the ONNX export is published.
    pub weights_url: &'static str,
    /// Pinned SHA-256 of the ONNX file (`None` = unpinned).
    pub sha256: Option<&'static str>,
}

/// The registry of supported depth estimation models.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
    id: "depth_anything_v3_small",
    name: "Depth Anything V3 Small (8-bit ONNX)",
    description: "Depth Anything V3 Small, 8-bit dynamically quantised ONNX export (en970/onnx-community, 27.6 MB, Pearson r >= 0.99 vs float32). Runs on CPU, or on GPU through ONNX Runtime execution providers.",
    authors: "ByteDance Seed, Depth Anything 3 team (ONNX export: en970)",
    license: "Apache-2.0",
    license_url: "https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE",
    homepage: "https://github.com/ByteDance-Seed/Depth-Anything-3",
    weights_url: "https://huggingface.co/en970/depth-anything-v3-small-onnx",
    sha256: Some("b6eaf064414e1474c88ca58cdafe414771e058996b99ea58e694e256e8738b43"),
    },
];

pub fn find_model(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

/// Verify a weights buffer against `expected` (or the model's pinned SHA-256 when `None`).
/// With nothing to compare against this succeeds; callers should surface the computed hash.
pub fn verify_weights(model: &ModelSpec, bytes: &[u8], expected: Option<&str>) -> Result<(), String> {
    let Some(want) = expected.or(model.sha256) else {
        return Ok(());
    };
    let got = sha256_hex(bytes);
    if !got.eq_ignore_ascii_case(want) {
        return Err(format!("model `{}` checksum mismatch: expected {want}, got {got}", model.id));
    }
    Ok(())
}

/// Simple SHA-256 digest function (FIPS 180-4).
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let bits = (data.len() as u64).wrapping_mul(8);
    let mut tail = data[data.len() - data.len() % 64..].to_vec();
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&bits.to_be_bytes());

    for block in data[..data.len() - data.len() % 64].as_chunks::<64>().0.iter().chain(tail.as_chunks::<64>().0.iter()) {
        let mut w = [0u32; 64];
        for (i, c) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(k[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }

    let mut out = String::with_capacity(64);
    for v in h {
        out.push_str(&format!("{v:08x}"));
    }
    out
}

/// The 8-bit Depth Anything model compiled into `depthcraft-cli` (feature `embed-model`).
pub const EMBEDDED_MODEL_ID: &str = "depth_anything_v3_small";
