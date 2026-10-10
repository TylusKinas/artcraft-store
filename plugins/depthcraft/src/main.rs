//! depthcraft-cli: Depth Anything V3 (ONNX) depth-map generator for EffectCraft.
//!
//! Runs on CPU (slower, works everywhere) or GPU through ONNX Runtime (CUDA / DirectML / CoreML).

use std::path::{Path, PathBuf};
use std::time::Instant;

use depthcraft::models::{DeviceSelection, MODELS, QualityPreset, find_model, sha256_hex, verify_weights};
use depthcraft::onnx::OnnxEstimator;
use depthcraft::pipeline::{DepthPipeline, PipelineConfig, blend_temporal};

#[cfg(feature = "embed-model")]
const EMBEDDED_BYTES: &[u8] = include_bytes!("../models/depth_anything_v3_small_q8.onnx");
#[cfg(feature = "embed-model")]
const EMBEDDED: bool = true;
#[cfg(not(feature = "embed-model"))]
const EMBEDDED: bool = false;

const USAGE: &str = "DepthCraft CLI - Depth Anything V3 depth maps for EffectCraft

Usage:
  depthcraft-cli models
  depthcraft-cli depth [--model <model.onnx>] --input <image|folder> --output <folder> [options]
  depthcraft-cli serve [--port 47831] [options]   keep the model loaded for the DepthCraft.jsx panel

Options:
  --model <file>       ONNX model (optional in builds with the embedded 8-bit model)
  --device cpu|gpu     cpu (default) works everywhere; gpu needs a CUDA/DirectML/CoreML ONNX Runtime
  --quality fast|balanced|ultra   inference resolution (default balanced; ignored for fixed-size models)
  --refine <0-20>      edge-preserving bilateral radius (default 2)
  --temporal <0-100>   flicker smoothing across a frame folder (default 0)
  --invert-model       flip the model output if it returns disparity (near = bright)
  --sha256 <hex>       verify the model file against this SHA-256
  --threads <n>        CPU intra-op threads
  --ort-lib <path>     ONNX Runtime library (default: $ORT_DYLIB_PATH, else onnxruntime.dll next to this exe)

Input folders are processed in file-name order as a frame sequence. Output: 16-bit grayscale PNG
named <name>_depth.png (black = near, white = far).";

struct Opts {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    device: DeviceSelection,
    quality: QualityPreset,
    refine: usize,
    temporal: f32,
    invert_model: bool,
    sha256: Option<String>,
    threads: Option<usize>,
    ort_lib: Option<PathBuf>,
    port: u16,
}

fn parse(args: &[String], need_io: bool) -> Result<Opts, String> {
    let mut o = Opts {
        model: PathBuf::new(),
        input: PathBuf::new(),
        output: PathBuf::new(),
        device: DeviceSelection::Cpu,
        quality: QualityPreset::Balanced,
        refine: 2,
        temporal: 0.0,
        invert_model: false,
        sha256: None,
        threads: None,
        ort_lib: None,
        port: 47831,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--model" => o.model = val("--model")?.into(),
            "--input" => o.input = val("--input")?.into(),
            "--output" => o.output = val("--output")?.into(),
            "--device" => {
                o.device = match val("--device")?.as_str() {
                    "cpu" => DeviceSelection::Cpu,
                    "gpu" => DeviceSelection::Gpu,
                    v => return Err(format!("unknown device `{v}` (cpu|gpu)")),
                }
            }
            "--quality" => {
                o.quality = match val("--quality")?.as_str() {
                    "fast" => QualityPreset::Fast,
                    "balanced" => QualityPreset::Balanced,
                    "ultra" => QualityPreset::Ultra,
                    v => return Err(format!("unknown quality `{v}` (fast|balanced|ultra)")),
                }
            }
            "--refine" => o.refine = val("--refine")?.parse::<usize>().map_err(|e| e.to_string())?.min(20),
            "--temporal" => o.temporal = val("--temporal")?.parse::<f32>().map_err(|e| e.to_string())?.clamp(0.0, 100.0),
            "--invert-model" => o.invert_model = true,
            "--sha256" => o.sha256 = Some(val("--sha256")?),
            "--threads" => o.threads = Some(val("--threads")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?),
            "--ort-lib" => o.ort_lib = Some(val("--ort-lib")?.into()),
            "--port" => o.port = val("--port")?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
            other => return Err(format!("unknown option `{other}`")),
        }
    }
    if o.model.as_os_str().is_empty() && !EMBEDDED {
        return Err("--model is required (this build has no embedded model)".to_string());
    }
    if need_io && (o.input.as_os_str().is_empty() || o.output.as_os_str().is_empty()) {
        return Err("--input and --output are required".to_string());
    }
    Ok(o)
}

fn list_models() {
    for m in MODELS {
        println!("ID:          {}", m.id);
        println!("Name:        {}", m.name);
        println!("License:     {} ({})", m.license, m.license_url);
        println!("Homepage:    {}", m.homepage);
        println!("Weights:     {}", m.weights_url);
        println!("SHA-256:     {}", m.sha256.unwrap_or("not pinned (use --sha256 to verify)"));
        println!("Description: {}\n", m.description);
    }
}

fn collect_inputs(input: &Path) -> Result<Vec<PathBuf>, String> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(input)
        .map_err(|e| format!("{}: {e}", input.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg")))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no png/jpg images in {}", input.display()));
    }
    Ok(files)
}

fn find_ort() -> Option<PathBuf> {
    let names = ["onnxruntime.dll", "libonnxruntime.so", "libonnxruntime.dylib"];
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        dirs.push(dir);
    }
    dirs.push(PathBuf::from("."));
    dirs.iter().flat_map(|d| names.iter().map(move |n| d.join(n))).find(|p| p.is_file())
}

fn init_ort(o: &Opts) -> Result<(), String> {
    // ONNX Runtime library: --ort-lib, else $ORT_DYLIB_PATH (read by `ort`), else next to the executable / in the current folder.
    let lib = o.ort_lib.clone().or_else(|| if std::env::var_os("ORT_DYLIB_PATH").is_some() { None } else { find_ort() });
    if let Some(lib) = &lib {
        ort::init_from(lib).map_err(|e| format!("cannot load ONNX Runtime {}: {e}", lib.display()))?.commit();
    }
    Ok(())
}

fn build_pipeline(o: &Opts) -> Result<DepthPipeline<OnnxEstimator>, String> {
    let spec = find_model("depth_anything_v3_small").ok_or("model registry is empty")?;
    let embedded = o.model.as_os_str().is_empty();
    let estimator = if embedded {
        #[cfg(feature = "embed-model")]
        {
            verify_weights(spec, EMBEDDED_BYTES, None)?;
            OnnxEstimator::load_from_memory(spec.id, EMBEDDED_BYTES, o.device, o.quality, o.invert_model, o.threads)?
        }
        #[cfg(not(feature = "embed-model"))]
        unreachable!("parse() requires --model without the embedded model")
    } else {
        let bytes = std::fs::read(&o.model).map_err(|e| format!("{}: {e}", o.model.display()))?;
        verify_weights(spec, &bytes, o.sha256.as_deref())?;
        if o.sha256.is_none() && spec.sha256.is_none() {
            eprintln!("note: model checksum not pinned; SHA-256 of {} is {}", o.model.display(), sha256_hex(&bytes));
        }
        drop(bytes);
        OnnxEstimator::load(spec.id, &o.model, o.device, o.quality, o.invert_model, o.threads)?
    };
    eprintln!("{}: running on {}", spec.name, estimator.active_providers.join(" > "));
    Ok(DepthPipeline::new(estimator, PipelineConfig { edge_refine_radius: o.refine, temporal_smooth_weight: 0.0, ..PipelineConfig::default() }))
}

/// Depth for every file in order (frame sequence); writes `<name>_depth.png` into `output`.
/// `progress(done, total, written_path)` runs after each frame.
fn process_files(
    pipeline: &mut DepthPipeline<OnnxEstimator>,
    files: &[PathBuf],
    output: &Path,
    temporal: f32,
    progress: &mut dyn FnMut(usize, usize, &Path, f64),
) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(output).map_err(|e| format!("{}: {e}", output.display()))?;
    let total = files.len();
    let mut prev: Option<Vec<f32>> = None;
    let mut written = Vec::with_capacity(total);
    for (n, path) in files.iter().enumerate() {
        let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb: Vec<[f32; 3]> = img.pixels().map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]).collect();

        let t = Instant::now();
        let (mut depth, _) = pipeline.process_frame(&rgb, w, h)?;
        if let Some(p) = &prev {
            blend_temporal(&mut depth, p, temporal);
        }

        let samples: Vec<u16> = depth.iter().map(|d| (d.clamp(0.0, 1.0) * 65535.0).round() as u16).collect();
        let out = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(w as u32, h as u32, samples).ok_or("depth buffer size mismatch")?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
        let dest = output.join(format!("{stem}_depth.png"));
        out.save(&dest).map_err(|e| format!("{}: {e}", dest.display()))?;
        progress(n + 1, total, &dest, t.elapsed().as_secs_f64() * 1000.0);
        written.push(dest);
        prev = Some(depth);
    }
    Ok(written)
}

fn run(o: Opts) -> Result<(), String> {
    init_ort(&o)?;
    let mut pipeline = build_pipeline(&o)?;
    let files = collect_inputs(&o.input)?;
    let started = Instant::now();
    let total = files.len();
    process_files(&mut pipeline, &files, &o.output, o.temporal, &mut |n, total, dest, ms| {
        eprintln!("[{n}/{total}] -> {} ({ms:.0} ms)", dest.display());
    })?;
    let secs = started.elapsed().as_secs_f64();
    eprintln!("done: {total} frame(s) in {secs:.2} s ({:.2} fps)", total as f64 / secs.max(1e-9));
    Ok(())
}

/// `serve`: keeps the model loaded and answers JSON-line requests on 127.0.0.1 (used by DepthCraft.jsx,
/// because EffectCraft scripts can open TCP sockets but cannot run programs).
///   -> {"cmd":"ping"}                                   <- {"type":"pong","providers":"CUDA > CPU"}
///   -> {"cmd":"depth","input":"C:\\a.png"|folder,"output":"C:\\out","quality":"balanced","refine":0,"temporal":0,"invert":false}
///   <- {"type":"progress","done":1,"total":3,"file":"..."} ... {"type":"done","files":[...]}  |  {"type":"error","message":"..."}
fn serve(o: Opts) -> Result<(), String> {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    init_ort(&o)?;
    let mut pipeline = build_pipeline(&o)?;
    let providers = pipeline.estimator().active_providers.join(" > ");
    let listener = TcpListener::bind(("127.0.0.1", o.port)).map_err(|e| format!("cannot listen on 127.0.0.1:{}: {e}", o.port))?;
    eprintln!("DepthCraft server ready on 127.0.0.1:{} ({providers}). Close this window to stop.", o.port);
    let mut state = (o.quality, o.invert_model);

    for conn in listener.incoming() {
        let Ok(mut stream) = conn else { continue };
        let mut line = String::new();
        if BufReader::new(&stream).read_line(&mut line).is_err() {
            continue;
        }
        let mut send = |v: serde_json::Value| {
            let _ = stream.write_all(format!("{v}\n").as_bytes());
            let _ = stream.flush();
        };
        let Ok(req) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
            send(serde_json::json!({"type": "error", "message": "expected one JSON request per line"}));
            continue;
        };
        match req["cmd"].as_str() {
            Some("ping") => send(serde_json::json!({"type": "pong", "providers": providers})),
            Some("depth") => {
                let input = PathBuf::from(req["input"].as_str().unwrap_or(""));
                let output = PathBuf::from(req["output"].as_str().unwrap_or(""));
                let quality = match req["quality"].as_str() {
                    Some("fast") => QualityPreset::Fast,
                    Some("ultra") => QualityPreset::Ultra,
                    _ => QualityPreset::Balanced,
                };
                let invert = req["invert"].as_bool().unwrap_or(false);
                if state != (quality, invert) {
                    pipeline.estimator_mut().configure(quality, invert);
                    pipeline.clear_cache();
                    state = (quality, invert);
                }
                pipeline.set_edge_refine(req["refine"].as_u64().unwrap_or(0).min(3) as usize);
                let temporal = req["temporal"].as_f64().unwrap_or(0.0).clamp(0.0, 100.0) as f32;
                let result = collect_inputs(&input).and_then(|files| {
                    process_files(&mut pipeline, &files, &output, temporal, &mut |n, total, dest, _| {
                        send(serde_json::json!({"type": "progress", "done": n, "total": total, "file": dest.to_string_lossy()}));
                    })
                });
                match result {
                    Ok(files) => send(serde_json::json!({"type": "done", "files": files.iter().map(|f| f.to_string_lossy()).collect::<Vec<_>>()})),
                    Err(e) => send(serde_json::json!({"type": "error", "message": e})),
                }
            }
            _ => send(serde_json::json!({"type": "error", "message": "unknown cmd"})),
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("models") | Some("--models") => {
            list_models();
            Ok(())
        }
        Some("depth") => parse(&args[1..], true).and_then(run),
        Some("serve") => parse(&args[1..], false).and_then(serve),
        _ => {
            println!("{USAGE}");
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
