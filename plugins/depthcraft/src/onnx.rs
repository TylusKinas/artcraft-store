//! Native Depth Anything V3 inference through ONNX Runtime (CPU, or GPU via CUDA / DirectML / CoreML).
//!
//! ONNX Runtime is loaded dynamically (`ORT_DYLIB_PATH` or `--ort-lib`), so install the build
//! that matches your hardware (CPU, CUDA, DirectML, CoreML) and DepthCraft will use it.

use std::path::Path;

use ort::ep::{self, ExecutionProvider};
use ort::session::Session;
use ort::value::TensorRef;

use crate::estimator::{DepthEstimator, inference_size, normalize_depth, resize_plane, to_nchw_normalized};
use crate::models::{DeviceSelection, QualityPreset};

enum ModelSource<'a> {
    File(&'a Path),
    Memory(&'a [u8]),
}

pub struct OnnxEstimator {
    id: String,
    session: Session,
    input_name: String,
    /// Fixed (H, W) when the exported graph is static, otherwise `None` (dynamic axes).
    fixed_hw: Option<(usize, usize)>,
    /// Model input rank: 4 = `[B,3,H,W]`, 5 = `[B,N,3,H,W]` (DA3 multi-view export).
    rank: usize,
    long_side: usize,
    invert_output: bool,
    pub active_providers: Vec<&'static str>,
}

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

impl OnnxEstimator {
    pub fn load(id: &str, model_path: &Path, device: DeviceSelection, quality: QualityPreset, invert_output: bool, threads: Option<usize>) -> Result<Self, String> {
        Self::build(id, ModelSource::File(model_path), device, quality, invert_output, threads)
    }

    pub fn load_from_memory(id: &str, bytes: &[u8], device: DeviceSelection, quality: QualityPreset, invert_output: bool, threads: Option<usize>) -> Result<Self, String> {
        Self::build(id, ModelSource::Memory(bytes), device, quality, invert_output, threads)
    }

    fn build(
        id: &str,
        source: ModelSource<'_>,
        device: DeviceSelection,
        quality: QualityPreset,
        invert_output: bool,
        threads: Option<usize>,
    ) -> Result<Self, String> {
        let mut builder = Session::builder().map_err(err)?;
        if let Some(n) = threads {
            builder = builder.with_intra_threads(n).map_err(err)?;
        }

        let mut active: Vec<&'static str> = Vec::new();
        if device == DeviceSelection::Gpu {
            let candidates: Vec<(&'static str, ort::ep::ExecutionProviderDispatch, bool)> = vec![
                ("CUDA", ep::CUDA::default().build(), ep::CUDA::default().is_available().unwrap_or(false)),
                ("DirectML", ep::DirectML::default().build(), ep::DirectML::default().is_available().unwrap_or(false)),
                ("CoreML", ep::CoreML::default().build(), ep::CoreML::default().is_available().unwrap_or(false)),
            ];
            let usable: Vec<_> = candidates.into_iter().filter(|c| c.2).collect();
            if usable.is_empty() {
                return Err("no GPU execution provider available in the loaded ONNX Runtime (need a CUDA / DirectML / CoreML build); use --device cpu".to_string());
            }
            active.extend(usable.iter().map(|c| c.0));
            let eps: Vec<_> = usable.into_iter().map(|c| c.1).collect();
            // DirectML requires sequential execution and no memory-pattern optimisation.
            builder = builder.with_memory_pattern(false).map_err(err)?.with_parallel_execution(false).map_err(err)?;
            builder = builder.with_execution_providers(eps).map_err(err)?;
        }
        active.push("CPU");

        let session = match source {
            ModelSource::File(p) => builder.commit_from_file(p),
            ModelSource::Memory(b) => builder.commit_from_memory(b),
        }
        .map_err(err)?;

        let input = session.inputs().first().ok_or("model has no inputs")?;
        let input_name = input.name().to_string();
        let shape = input.dtype().tensor_shape().ok_or("model input is not a tensor")?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        let rank = dims.len();
        if rank != 4 && rank != 5 {
            return Err(format!("unsupported model input rank {rank} (expected 4 or 5)"));
        }
        let (h, w) = (dims[rank - 2], dims[rank - 1]);
        let fixed_hw = if h > 0 && w > 0 { Some((h as usize, w as usize)) } else { None };

        Ok(Self { id: id.to_string(), session, input_name, fixed_hw, rank, long_side: quality.long_side(), invert_output, active_providers: active })
    }
}

impl OnnxEstimator {
    /// Change inference resolution / output polarity between requests (server mode).
    pub fn configure(&mut self, quality: QualityPreset, invert_output: bool) {
        self.long_side = quality.long_side();
        self.invert_output = invert_output;
    }
}

impl DepthEstimator for OnnxEstimator {
    fn id(&self) -> &str {
        &self.id
    }

    fn estimate(&mut self, rgb: &[[f32; 3]], width: usize, height: usize) -> Result<Vec<f32>, String> {
        if width == 0 || height == 0 || rgb.len() < width * height {
            return Err("invalid frame dimensions".to_string());
        }
        let (iw, ih) = self.fixed_hw.map(|(h, w)| (w, h)).unwrap_or_else(|| inference_size(width, height, self.long_side));
        let data = to_nchw_normalized(rgb, width, height, iw, ih);
        let shape: Vec<usize> = if self.rank == 5 { vec![1, 1, 3, ih, iw] } else { vec![1, 3, ih, iw] };
        let input = TensorRef::from_array_view((shape, data.as_slice())).map_err(err)?;

        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => input]).map_err(err)?;
        let (out_shape, out) = outputs[0].try_extract_tensor::<f32>().map_err(err)?;
        let dims: Vec<i64> = out_shape.iter().copied().collect();
        if dims.len() < 2 {
            return Err(format!("unexpected model output shape {dims:?}"));
        }
        let (oh, ow) = (dims[dims.len() - 2] as usize, dims[dims.len() - 1] as usize);
        if out.len() < oh * ow {
            return Err("model output smaller than its declared shape".to_string());
        }

        let mut depth = resize_plane(&out[..oh * ow], ow, oh, width, height);
        normalize_depth(&mut depth);
        if self.invert_output {
            for v in &mut depth {
                *v = 1.0 - *v;
            }
        }
        Ok(depth)
    }
}
