#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["onnxruntime", "numpy", "pillow"]
# ///
"""DepthCraft depth server: AI depth maps (Depth Anything V3 Small) on Windows, macOS and Linux.

One file, no build step. Start it and leave it running; the DepthCraft panel in EffectCraft sends it an
image path and imports the depth map it writes (EffectCraft scripts can talk to a local port but cannot
start programs, and a neural network is too heavy for an EffectCraft wasm plug-in).

    pip install onnxruntime numpy pillow          # once  (GPU: onnxruntime-directml / onnxruntime-gpu instead)
    python depthcraft_server.py                    # serve on 127.0.0.1:47831 (this computer only)
    # or, with uv, no install at all:  uv run depthcraft_server.py

The 29 MB model is downloaded once to a cache folder and verified (SHA-256). Command line use without the panel:

    python depthcraft_server.py depth photo.jpg --output out        (a folder of frames works too)

Options: --device auto|cpu|gpu, --quality fast|balanced|ultra, --port N, --model file.onnx, --invert, --temporal 0-100

Protocol (one JSON object per line over TCP):
  {"cmd":"ping"}                                                -> {"type":"pong","providers":"CPUExecutionProvider"}
  {"cmd":"depth","input":path|folder,"output":folder,"quality":"balanced","invert":false,"temporal":0}
      -> {"type":"progress","done":1,"total":3,"file":...} ... {"type":"done","files":[...]}  |  {"type":"error","message":...}
Original work by EffectCraft contributors under MIT OR Apache-2.0. Model: Depth Anything V3 Small (Apache-2.0).
"""
import argparse
import hashlib
import json
import os
import socketserver
import sys
import time
import urllib.request

try:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
except ImportError as e:  # pragma: no cover - friendly message instead of a traceback
    sys.exit(f"Missing package ({e.name}). Install once with:\n    {sys.executable} -m pip install onnxruntime numpy pillow")

MODEL_NAME = "depth_anything_v3_small_q8.onnx"
MODEL_SHA256 = "b6eaf064414e1474c88ca58cdafe414771e058996b99ea58e694e256e8738b43"
MODEL_URLS = [
    "https://raw.githubusercontent.com/TylusKinas/artcraft-store/main/plugins/depthcraft/models/" + MODEL_NAME,
    "https://raw.githubusercontent.com/TylusKinas/artcraft-store/claude/epic-mccarthy-por9ji/plugins/depthcraft/models/" + MODEL_NAME,
    "https://huggingface.co/en970/depth-anything-v3-small-onnx/resolve/main/model_quantized.onnx",
    "https://huggingface.co/en970/depth-anything-v3-small-onnx/resolve/main/onnx/model_quantized.onnx",
]
LONG_SIDE = {"fast": 252, "balanced": 364, "ultra": 518}
PATCH = 14
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir():
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "DepthCraft")


def ensure_model(path=None):
    """Return a verified model path, downloading it on first use."""
    if path:
        return path
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", MODEL_NAME)  # source checkout
    if os.path.isfile(local) and sha256(local) == MODEL_SHA256:
        return local
    dest = os.path.join(cache_dir(), MODEL_NAME)
    if os.path.isfile(dest) and sha256(dest) == MODEL_SHA256:
        return dest
    os.makedirs(cache_dir(), exist_ok=True)
    errors = []
    for url in MODEL_URLS:
        tmp = dest + ".part"
        try:
            print(f"Downloading the depth model (29 MB) from {url} ...", flush=True)
            urllib.request.urlretrieve(url, tmp)
            if sha256(tmp) != MODEL_SHA256:
                errors.append(f"{url}: checksum mismatch")
                continue
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # network / 404: try the next mirror
            errors.append(f"{url}: {e}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    sys.exit("Could not download the model:\n  " + "\n  ".join(errors) + f"\nDownload {MODEL_NAME} manually and pass --model <file>.")


class Depth:
    """Depth Anything V3 through ONNX Runtime (CPU, or GPU when a GPU provider is installed)."""

    def __init__(self, model_path, device="auto"):
        available = ort.get_available_providers()
        gpu = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider") if p in available]
        if device == "gpu" and not gpu:
            sys.exit("--device gpu needs a GPU build of ONNX Runtime (pip install onnxruntime-directml on Windows, "
                     "onnxruntime-gpu for NVIDIA); available: " + ", ".join(available))
        providers = (gpu if device in ("auto", "gpu") else []) + ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        if "DmlExecutionProvider" in providers:  # DirectML requirements
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.rank = len(inp.shape)
        h, w = inp.shape[-2], inp.shape[-1]
        self.fixed = (h, w) if isinstance(h, int) and isinstance(w, int) else None
        self.providers = self.session.get_providers()

    def estimate(self, rgb, quality="balanced", invert=False):
        """rgb: PIL image. Returns float32 HxW in [0, 1], 0 = near, 1 = far."""
        w, h = rgb.size
        if self.fixed:
            ih, iw = self.fixed
        else:
            scale = LONG_SIDE.get(quality, 364) / max(w, h)
            iw, ih = (max(1, round(w * scale / PATCH)) * PATCH, max(1, round(h * scale / PATCH)) * PATCH)
        small = np.asarray(rgb.resize((iw, ih), Image.BILINEAR), dtype=np.float32) / 255.0
        x = ((small.transpose(2, 0, 1) - MEAN) / STD).astype(np.float32)
        x = x[None, None] if self.rank == 5 else x[None]
        out = self.session.run(None, {self.input_name: x})[0]
        d = np.asarray(out, dtype=np.float32).reshape(-1, out.shape[-2], out.shape[-1])[0]
        d = np.asarray(Image.fromarray(d, mode="F").resize((w, h), Image.BILINEAR), dtype=np.float32)
        d = np.nan_to_num(d)
        lo, hi = float(d.min()), float(d.max())
        d = (d - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(d)
        return 1.0 - d if invert else d


def collect_inputs(path):
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path}: No such file or directory")
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise FileNotFoundError(f"no png/jpg images in {path}")
    return files


def process(depth, input_path, output_dir, quality="balanced", invert=False, temporal=0.0, progress=None):
    """Depth for a file or a frame folder; writes <name>_depth.png (16-bit grayscale)."""
    files = collect_inputs(input_path)
    os.makedirs(output_dir, exist_ok=True)
    alpha = 1.0 - (min(max(float(temporal), 0.0), 100.0) / 100.0 * 0.75)
    prev, written = None, []
    for n, f in enumerate(files, 1):
        d = depth.estimate(Image.open(f).convert("RGB"), quality, invert)
        if prev is not None and prev.shape == d.shape and alpha < 1.0:
            d = d * alpha + prev * (1.0 - alpha)
        prev = d
        dest = os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + "_depth.png")
        Image.fromarray(np.round(np.clip(d, 0, 1) * 65535).astype("<u2")).save(dest)
        written.append(dest)
        if progress:
            progress(n, len(files), dest)
    return written


def make_handler(depth):
    class Handler(socketserver.StreamRequestHandler):
        def send(self, obj):
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass

        def handle(self):
            line = self.rfile.readline(1 << 20).decode("utf-8", "replace").strip()
            if not line.startswith("{"):  # refuse anything that is not one of our JSON requests
                return self.send({"type": "error", "message": "expected one JSON request per line"})
            try:
                req = json.loads(line)
            except ValueError:
                return self.send({"type": "error", "message": "invalid JSON"})
            cmd = req.get("cmd")
            if cmd == "ping":
                return self.send({"type": "pong", "providers": " > ".join(depth.providers)})
            if cmd != "depth":
                return self.send({"type": "error", "message": "unknown cmd"})
            try:
                files = process(
                    depth, str(req.get("input", "")), str(req.get("output", "")), str(req.get("quality", "balanced")),
                    bool(req.get("invert", False)), float(req.get("temporal", 0) or 0),
                    lambda n, total, dest: self.send({"type": "progress", "done": n, "total": total, "file": dest}),
                )
                self.send({"type": "done", "files": files})
            except Exception as e:  # report to the panel instead of dropping the connection
                self.send({"type": "error", "message": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description="DepthCraft depth server / command line (Depth Anything V3 Small).")
    ap.add_argument("command", nargs="?", choices=["serve", "depth"], default="serve")
    ap.add_argument("input", nargs="?", help="depth: image or folder of frames")
    ap.add_argument("--output", default="depthcraft_out", help="depth: output folder")
    ap.add_argument("--port", type=int, default=47831)
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--quality", choices=list(LONG_SIDE), default="balanced")
    ap.add_argument("--model", help="ONNX model file (default: download once and cache)")
    ap.add_argument("--invert", action="store_true", help="flip near/far if a model returns disparity")
    ap.add_argument("--temporal", type=float, default=0, help="flicker smoothing for frame folders, 0-100")
    a = ap.parse_args()

    depth = Depth(ensure_model(a.model), a.device)
    print("DepthCraft: ONNX Runtime providers:", " > ".join(depth.providers), flush=True)

    if a.command == "depth":
        if not a.input:
            ap.error("depth needs an input image or folder")
        t = time.time()
        files = process(depth, a.input, a.output, a.quality, a.invert, a.temporal,
                        lambda n, total, dest: print(f"[{n}/{total}] -> {dest}", flush=True))
        print(f"done: {len(files)} frame(s) in {time.time() - t:.2f} s")
        return

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", a.port), make_handler(depth)) as srv:
        print(f"DepthCraft server ready on 127.0.0.1:{a.port} (this computer only). Press Ctrl+C to stop.", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
