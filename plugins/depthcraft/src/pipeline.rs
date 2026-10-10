//! Preprocessing pipeline with caching, cancellation, progress reporting, and memory bounds.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::estimator::DepthEstimator;
use crate::filter::refine_depth_edges;
use crate::models::sha256_hex;

/// Cancellation token allowing cooperative abortion of multi-frame batch processing.
#[derive(Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Progress info emitted after each processed frame.
#[derive(Clone, Debug)]
pub struct ProgressReport {
    pub current_frame: usize,
    pub total_frames: usize,
    pub percentage: f32,
    pub cache_hit: bool,
    pub memory_used_bytes: usize,
}

pub type ProgressCallback = Box<dyn FnMut(ProgressReport) + Send + Sync>;

/// Blend `depth` towards the previous frame's map. `weight` is 0..100 (0 = off).
pub fn blend_temporal(depth: &mut [f32], prev: &[f32], weight: f32) {
    if weight <= 0.0 || prev.len() != depth.len() {
        return;
    }
    let alpha = 1.0 - (weight.min(100.0) / 100.0 * 0.75);
    for (c, p) in depth.iter_mut().zip(prev) {
        *c = *c * alpha + *p * (1.0 - alpha);
    }
}

/// In-memory LRU cache storing processed depth maps to prevent redundant re-inference.
pub struct DepthCache {
    entries: HashMap<String, Vec<f32>>,
    order: VecDeque<(String, usize)>, // (key, size_in_bytes)
    total_bytes: usize,
    max_memory_bytes: usize,
}

impl DepthCache {
    pub fn new(max_memory_bytes: usize) -> Self {
        Self { entries: HashMap::new(), order: VecDeque::new(), total_bytes: 0, max_memory_bytes }
    }

    pub fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: String, depth: Vec<f32>) {
        let size = depth.len() * std::mem::size_of::<f32>();

        // Evict if over budget
        while self.total_bytes + size > self.max_memory_bytes && !self.order.is_empty() {
            if let Some((old_key, old_size)) = self.order.pop_front() {
                self.entries.remove(&old_key);
                self.total_bytes = self.total_bytes.saturating_sub(old_size);
            }
        }

        self.total_bytes += size;
        self.order.push_back((key.clone(), size));
        self.entries.insert(key, depth);
    }

    pub fn memory_used(&self) -> usize {
        self.total_bytes
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.total_bytes = 0;
    }
}

/// Pipeline runner for single images or video sequences.
pub struct PipelineConfig {
    pub edge_refine_radius: usize,
    pub temporal_smooth_weight: f32,
    pub memory_limit_bytes: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            edge_refine_radius: 2,
            temporal_smooth_weight: 50.0,
            memory_limit_bytes: 256 * 1024 * 1024, // 256 MB default
        }
    }
}

impl PipelineConfig {
    pub fn default_memory_bytes() -> usize {
        256 * 1024 * 1024
    }
}

pub struct DepthPipeline<E: DepthEstimator> {
    estimator: E,
    config: PipelineConfig,
    cache: DepthCache,
}

impl<E: DepthEstimator> DepthPipeline<E> {
    pub fn new(estimator: E, config: PipelineConfig) -> Self {
        let max_mem = config.memory_limit_bytes;
        Self { estimator, config, cache: DepthCache::new(max_mem) }
    }

    pub fn estimator(&self) -> &E {
        &self.estimator
    }

    pub fn estimator_mut(&mut self) -> &mut E {
        &mut self.estimator
    }

    pub fn set_edge_refine(&mut self, radius: usize) {
        self.config.edge_refine_radius = radius;
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Compute frame content hash for cache lookup.
    pub fn compute_frame_key(&self, rgb: &[[f32; 3]], width: usize, height: usize) -> String {
        let mut sample_bytes = Vec::with_capacity(rgb.len() * 3);
        for c in rgb {
            sample_bytes.push((c[0].clamp(0.0, 1.0) * 255.0) as u8);
            sample_bytes.push((c[1].clamp(0.0, 1.0) * 255.0) as u8);
            sample_bytes.push((c[2].clamp(0.0, 1.0) * 255.0) as u8);
        }
        let digest = sha256_hex(&sample_bytes);
        format!("{}_{}x{}_{}_{}", self.estimator.id(), width, height, self.config.edge_refine_radius, digest)
    }

    /// Process a single frame, returning depth map in [0, 1].
    pub fn process_frame(&mut self, rgb: &[[f32; 3]], width: usize, height: usize) -> Result<(Vec<f32>, bool), String> {
        let key = self.compute_frame_key(rgb, width, height);

        if let Some(cached) = self.cache.get(&key) {
            return Ok((cached.clone(), true));
        }

        let raw = self.estimator.estimate(rgb, width, height)?;

        let mut refined = vec![0.0f32; width * height];
        refine_depth_edges(&raw, rgb, width, height, self.config.edge_refine_radius, &mut refined);

        self.cache.insert(key, refined.clone());
        Ok((refined, false))
    }

    /// Process a video frame sequence with progress reporting and cancellation support.
    pub fn process_sequence(
        &mut self,
        frames: &[(&[[f32; 3]], usize, usize)],
        mut callback: Option<ProgressCallback>,
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, String> {
        let total = frames.len();
        if total == 0 {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(total);
        let mut prev_depth: Option<Vec<f32>> = None;

        for (idx, (rgb, w, h)) in frames.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err("processing cancelled by user".to_string());
            }

            let (mut depth, hit) = self.process_frame(rgb, *w, *h)?;

            if let Some(ref prev) = prev_depth {
                blend_temporal(&mut depth, prev, self.config.temporal_smooth_weight);
            }

            prev_depth = Some(depth.clone());
            results.push(depth);

            if let Some(ref mut cb) = callback {
                cb(ProgressReport {
                    current_frame: idx + 1,
                    total_frames: total,
                    percentage: ((idx + 1) as f32 / total as f32) * 100.0,
                    cache_hit: hit,
                    memory_used_bytes: self.cache.memory_used(),
                });
            }
        }

        Ok(results)
    }

    pub fn cache_memory_bytes(&self) -> usize {
        self.cache.memory_used()
    }
}
