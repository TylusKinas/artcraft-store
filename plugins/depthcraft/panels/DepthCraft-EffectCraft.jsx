// DepthCraft: AI depth maps (Depth Anything V3) for EffectCraft, as a ScriptUI panel.
//
// An EffectCraft wasm plug-in cannot run a neural network (the host meters wasm with a small fuel
// budget) and scripts cannot launch programs, but scripts can open TCP sockets. This panel talks to
// the local `depthcraft-cli serve` program, which keeps the model loaded and writes the depth map.
//
//   1. Preferences > Scripting & Expressions > Allow Scripts to Write Files and Access Network: ON
//   2. Install this single file with File > Scripts > Install ScriptUI Panel..., open it from the Window menu.
//   3. Press "Start server": the panel writes the server (embedded below) and a launcher into a folder and opens
//      the launcher. A console window installs what is missing (Python packages, a 29 MB model, once) and keeps
//      the server running; leave it open. Windows starts it for you; on macOS/Linux the panel shows the one
//      command to paste into a terminal. Press "Check server" until it says OK.
//   4. Select a footage layer and press Render Depth. The depth map is imported and added as a layer
//      (black = near, white = far). Apply the DepthCraft effect (depthcraft.wasm, folder EffectCraft\) to it for heatmap,
//      fog, slices, cutouts.
//
// Original work by EffectCraft contributors under MIT OR Apache-2.0.

// The depth server (panels/depthcraft_server.py), embedded so this one file is all a user installs.
// Regenerate with: python3 scripts/embed_server.py
/*PY_BEGIN*/
var SERVER_PY = [
  "#!/usr/bin/env python3",
  "# /// script",
  "# requires-python = \">=3.9\"",
  "# dependencies = [\"onnxruntime\", \"numpy\", \"pillow\"]",
  "# ///",
  "\"\"\"DepthCraft depth server: AI depth maps (Depth Anything V3 Small) on Windows, macOS and Linux.",
  "",
  "One file, no build step. Start it and leave it running; the DepthCraft panel in EffectCraft sends it an",
  "image path and imports the depth map it writes (EffectCraft scripts can talk to a local port but cannot",
  "start programs, and a neural network is too heavy for an EffectCraft wasm plug-in).",
  "",
  "    pip install onnxruntime numpy pillow          # once  (GPU: onnxruntime-directml / onnxruntime-gpu instead)",
  "    python depthcraft_server.py                    # serve on 127.0.0.1:47831 (this computer only)",
  "    # or, with uv, no install at all:  uv run depthcraft_server.py",
  "",
  "The 29 MB model is downloaded once to a cache folder and verified (SHA-256). Command line use without the panel:",
  "",
  "    python depthcraft_server.py depth photo.jpg --output out        (a folder of frames works too)",
  "",
  "Options: --device auto|cpu|gpu, --quality fast|balanced|ultra, --port N, --model file.onnx, --invert, --temporal 0-100",
  "",
  "Protocol (one JSON object per line over TCP):",
  "  {\"cmd\":\"ping\"}                                                -> {\"type\":\"pong\",\"providers\":\"CPUExecutionProvider\"}",
  "  {\"cmd\":\"depth\",\"input\":path|folder,\"output\":folder,\"quality\":\"balanced\",\"invert\":false,\"temporal\":0}",
  "      -> {\"type\":\"progress\",\"done\":1,\"total\":3,\"file\":...} ... {\"type\":\"done\",\"files\":[...]}  |  {\"type\":\"error\",\"message\":...}",
  "Original work by EffectCraft contributors under MIT OR Apache-2.0. Model: Depth Anything V3 Small (Apache-2.0).",
  "\"\"\"",
  "import argparse",
  "import hashlib",
  "import json",
  "import os",
  "import socketserver",
  "import sys",
  "import time",
  "import urllib.request",
  "",
  "try:",
  "    import numpy as np",
  "    import onnxruntime as ort",
  "    from PIL import Image",
  "except ImportError as e:  # pragma: no cover - friendly message instead of a traceback",
  "    sys.exit(f\"Missing package ({e.name}). Install once with:\\n    {sys.executable} -m pip install onnxruntime numpy pillow\")",
  "",
  "MODEL_NAME = \"depth_anything_v3_small_q8.onnx\"",
  "MODEL_SHA256 = \"b6eaf064414e1474c88ca58cdafe414771e058996b99ea58e694e256e8738b43\"",
  "MODEL_URLS = [",
  "    \"https://raw.githubusercontent.com/TylusKinas/artcraft-store/main/plugins/depthcraft/models/\" + MODEL_NAME,",
  "    \"https://raw.githubusercontent.com/TylusKinas/artcraft-store/claude/epic-mccarthy-por9ji/plugins/depthcraft/models/\" + MODEL_NAME,",
  "    \"https://huggingface.co/en970/depth-anything-v3-small-onnx/resolve/main/model_quantized.onnx\",",
  "    \"https://huggingface.co/en970/depth-anything-v3-small-onnx/resolve/main/onnx/model_quantized.onnx\",",
  "]",
  "LONG_SIDE = {\"fast\": 252, \"balanced\": 364, \"ultra\": 518}",
  "PATCH = 14",
  "MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)",
  "STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)",
  "IMAGE_EXTS = (\".png\", \".jpg\", \".jpeg\")",
  "",
  "",
  "def sha256(path):",
  "    h = hashlib.sha256()",
  "    with open(path, \"rb\") as f:",
  "        for chunk in iter(lambda: f.read(1 << 20), b\"\"):",
  "            h.update(chunk)",
  "    return h.hexdigest()",
  "",
  "",
  "def cache_dir():",
  "    if os.name == \"nt\":",
  "        base = os.environ.get(\"LOCALAPPDATA\") or os.path.expanduser(\"~\")",
  "    elif sys.platform == \"darwin\":",
  "        base = os.path.expanduser(\"~/Library/Caches\")",
  "    else:",
  "        base = os.environ.get(\"XDG_CACHE_HOME\") or os.path.expanduser(\"~/.cache\")",
  "    return os.path.join(base, \"DepthCraft\")",
  "",
  "",
  "def ensure_model(path=None):",
  "    \"\"\"Return a verified model path, downloading it on first use.\"\"\"",
  "    if path:",
  "        return path",
  "    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), \"..\", \"models\", MODEL_NAME)  # source checkout",
  "    if os.path.isfile(local) and sha256(local) == MODEL_SHA256:",
  "        return local",
  "    dest = os.path.join(cache_dir(), MODEL_NAME)",
  "    if os.path.isfile(dest) and sha256(dest) == MODEL_SHA256:",
  "        return dest",
  "    os.makedirs(cache_dir(), exist_ok=True)",
  "    errors = []",
  "    for url in MODEL_URLS:",
  "        tmp = dest + \".part\"",
  "        try:",
  "            print(f\"Downloading the depth model (29 MB) from {url} ...\", flush=True)",
  "            urllib.request.urlretrieve(url, tmp)",
  "            if sha256(tmp) != MODEL_SHA256:",
  "                errors.append(f\"{url}: checksum mismatch\")",
  "                continue",
  "            os.replace(tmp, dest)",
  "            return dest",
  "        except Exception as e:  # network / 404: try the next mirror",
  "            errors.append(f\"{url}: {e}\")",
  "        finally:",
  "            if os.path.exists(tmp):",
  "                os.remove(tmp)",
  "    sys.exit(\"Could not download the model:\\n  \" + \"\\n  \".join(errors) + f\"\\nDownload {MODEL_NAME} manually and pass --model <file>.\")",
  "",
  "",
  "class Depth:",
  "    \"\"\"Depth Anything V3 through ONNX Runtime (CPU, or GPU when a GPU provider is installed).\"\"\"",
  "",
  "    def __init__(self, model_path, device=\"auto\"):",
  "        available = ort.get_available_providers()",
  "        gpu = [p for p in (\"CUDAExecutionProvider\", \"DmlExecutionProvider\", \"CoreMLExecutionProvider\") if p in available]",
  "        if device == \"gpu\" and not gpu:",
  "            sys.exit(\"--device gpu needs a GPU build of ONNX Runtime (pip install onnxruntime-directml on Windows, \"",
  "                     \"onnxruntime-gpu for NVIDIA); available: \" + \", \".join(available))",
  "        providers = (gpu if device in (\"auto\", \"gpu\") else []) + [\"CPUExecutionProvider\"]",
  "        opts = ort.SessionOptions()",
  "        if \"DmlExecutionProvider\" in providers:  # DirectML requirements",
  "            opts.enable_mem_pattern = False",
  "            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL",
  "        self.session = ort.InferenceSession(model_path, opts, providers=providers)",
  "        inp = self.session.get_inputs()[0]",
  "        self.input_name = inp.name",
  "        self.rank = len(inp.shape)",
  "        h, w = inp.shape[-2], inp.shape[-1]",
  "        self.fixed = (h, w) if isinstance(h, int) and isinstance(w, int) else None",
  "        self.providers = self.session.get_providers()",
  "",
  "    def estimate(self, rgb, quality=\"balanced\", invert=False):",
  "        \"\"\"rgb: PIL image. Returns float32 HxW in [0, 1], 0 = near, 1 = far.\"\"\"",
  "        w, h = rgb.size",
  "        if self.fixed:",
  "            ih, iw = self.fixed",
  "        else:",
  "            scale = LONG_SIDE.get(quality, 364) / max(w, h)",
  "            iw, ih = (max(1, round(w * scale / PATCH)) * PATCH, max(1, round(h * scale / PATCH)) * PATCH)",
  "        small = np.asarray(rgb.resize((iw, ih), Image.BILINEAR), dtype=np.float32) / 255.0",
  "        x = ((small.transpose(2, 0, 1) - MEAN) / STD).astype(np.float32)",
  "        x = x[None, None] if self.rank == 5 else x[None]",
  "        out = self.session.run(None, {self.input_name: x})[0]",
  "        d = np.asarray(out, dtype=np.float32).reshape(-1, out.shape[-2], out.shape[-1])[0]",
  "        d = np.asarray(Image.fromarray(d, mode=\"F\").resize((w, h), Image.BILINEAR), dtype=np.float32)",
  "        d = np.nan_to_num(d)",
  "        lo, hi = float(d.min()), float(d.max())",
  "        d = (d - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(d)",
  "        return 1.0 - d if invert else d",
  "",
  "",
  "def collect_inputs(path):",
  "    if os.path.isfile(path):",
  "        return [path]",
  "    if not os.path.isdir(path):",
  "        raise FileNotFoundError(f\"{path}: No such file or directory\")",
  "    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))",
  "    if not files:",
  "        raise FileNotFoundError(f\"no png/jpg images in {path}\")",
  "    return files",
  "",
  "",
  "def process(depth, input_path, output_dir, quality=\"balanced\", invert=False, temporal=0.0, progress=None):",
  "    \"\"\"Depth for a file or a frame folder; writes <name>_depth.png (16-bit grayscale).\"\"\"",
  "    files = collect_inputs(input_path)",
  "    os.makedirs(output_dir, exist_ok=True)",
  "    alpha = 1.0 - (min(max(float(temporal), 0.0), 100.0) / 100.0 * 0.75)",
  "    prev, written = None, []",
  "    for n, f in enumerate(files, 1):",
  "        d = depth.estimate(Image.open(f).convert(\"RGB\"), quality, invert)",
  "        if prev is not None and prev.shape == d.shape and alpha < 1.0:",
  "            d = d * alpha + prev * (1.0 - alpha)",
  "        prev = d",
  "        dest = os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + \"_depth.png\")",
  "        Image.fromarray(np.round(np.clip(d, 0, 1) * 65535).astype(\"<u2\")).save(dest)",
  "        written.append(dest)",
  "        if progress:",
  "            progress(n, len(files), dest)",
  "    return written",
  "",
  "",
  "def make_handler(depth):",
  "    class Handler(socketserver.StreamRequestHandler):",
  "        def send(self, obj):",
  "            try:",
  "                self.wfile.write((json.dumps(obj) + \"\\n\").encode(\"utf-8\"))",
  "                self.wfile.flush()",
  "            except OSError:",
  "                pass",
  "",
  "        def handle(self):",
  "            line = self.rfile.readline(1 << 20).decode(\"utf-8\", \"replace\").strip()",
  "            if not line.startswith(\"{\"):  # refuse anything that is not one of our JSON requests",
  "                return self.send({\"type\": \"error\", \"message\": \"expected one JSON request per line\"})",
  "            try:",
  "                req = json.loads(line)",
  "            except ValueError:",
  "                return self.send({\"type\": \"error\", \"message\": \"invalid JSON\"})",
  "            cmd = req.get(\"cmd\")",
  "            if cmd == \"ping\":",
  "                return self.send({\"type\": \"pong\", \"providers\": \" > \".join(depth.providers)})",
  "            if cmd != \"depth\":",
  "                return self.send({\"type\": \"error\", \"message\": \"unknown cmd\"})",
  "            try:",
  "                files = process(",
  "                    depth, str(req.get(\"input\", \"\")), str(req.get(\"output\", \"\")), str(req.get(\"quality\", \"balanced\")),",
  "                    bool(req.get(\"invert\", False)), float(req.get(\"temporal\", 0) or 0),",
  "                    lambda n, total, dest: self.send({\"type\": \"progress\", \"done\": n, \"total\": total, \"file\": dest}),",
  "                )",
  "                self.send({\"type\": \"done\", \"files\": files})",
  "            except Exception as e:  # report to the panel instead of dropping the connection",
  "                self.send({\"type\": \"error\", \"message\": str(e)})",
  "",
  "    return Handler",
  "",
  "",
  "def main():",
  "    ap = argparse.ArgumentParser(description=\"DepthCraft depth server / command line (Depth Anything V3 Small).\")",
  "    ap.add_argument(\"command\", nargs=\"?\", choices=[\"serve\", \"depth\"], default=\"serve\")",
  "    ap.add_argument(\"input\", nargs=\"?\", help=\"depth: image or folder of frames\")",
  "    ap.add_argument(\"--output\", default=\"depthcraft_out\", help=\"depth: output folder\")",
  "    ap.add_argument(\"--port\", type=int, default=47831)",
  "    ap.add_argument(\"--device\", choices=[\"auto\", \"cpu\", \"gpu\"], default=\"auto\")",
  "    ap.add_argument(\"--quality\", choices=list(LONG_SIDE), default=\"balanced\")",
  "    ap.add_argument(\"--model\", help=\"ONNX model file (default: download once and cache)\")",
  "    ap.add_argument(\"--invert\", action=\"store_true\", help=\"flip near/far if a model returns disparity\")",
  "    ap.add_argument(\"--temporal\", type=float, default=0, help=\"flicker smoothing for frame folders, 0-100\")",
  "    a = ap.parse_args()",
  "",
  "    depth = Depth(ensure_model(a.model), a.device)",
  "    print(\"DepthCraft: ONNX Runtime providers:\", \" > \".join(depth.providers), flush=True)",
  "",
  "    if a.command == \"depth\":",
  "        if not a.input:",
  "            ap.error(\"depth needs an input image or folder\")",
  "        t = time.time()",
  "        files = process(depth, a.input, a.output, a.quality, a.invert, a.temporal,",
  "                        lambda n, total, dest: print(f\"[{n}/{total}] -> {dest}\", flush=True))",
  "        print(f\"done: {len(files)} frame(s) in {time.time() - t:.2f} s\")",
  "        return",
  "",
  "    socketserver.TCPServer.allow_reuse_address = True",
  "    with socketserver.TCPServer((\"127.0.0.1\", a.port), make_handler(depth)) as srv:",
  "        print(f\"DepthCraft server ready on 127.0.0.1:{a.port} (this computer only). Press Ctrl+C to stop.\", flush=True)",
  "        try:",
  "            srv.serve_forever()",
  "        except KeyboardInterrupt:",
  "            print(\"\\nstopped\")",
  "",
  "",
  "if __name__ == \"__main__\":",
  "    main()"
].join("\n") + "\n";
/*PY_END*/

(function depthCraftPanel(thisObj) {
  var SECTION = "DepthCraft";
  var HOST = "127.0.0.1:47831";

  function setting(key, fallback) {
    try {
      return app.settings.haveSetting(SECTION, key) ? app.settings.getSetting(SECTION, key) : fallback;
    } catch (e) {
      return fallback;
    }
  }
  function save(key, value) {
    try {
      app.settings.saveSetting(SECTION, key, String(value));
    } catch (e) {}
  }

  var ui = thisObj instanceof Panel ? thisObj : new Window("palette", "DepthCraft", undefined, { resizeable: true });
  ui.orientation = "column";
  ui.alignChildren = ["fill", "top"];
  ui.spacing = 6;

  ui.add("statictext", undefined, "AI depth map (Depth Anything V3)");

  var inRow = ui.add("group");
  inRow.add("statictext", undefined, "Input:");
  var input = inRow.add("edittext", undefined, setting("input", ""), { name: "input" });
  input.characters = 30;
  var fromLayer = inRow.add("button", undefined, "From layer", { name: "fromLayer" });

  var outRow = ui.add("group");
  outRow.add("statictext", undefined, "Output folder:");
  var output = outRow.add("edittext", undefined, setting("output", ""), { name: "output" });
  output.characters = 26;

  var optRow = ui.add("group");
  optRow.add("statictext", undefined, "Quality:");
  var quality = optRow.add("dropdownlist", undefined, ["Fast", "Balanced", "Ultra"], { name: "quality" });
  quality.selection = Number(setting("quality", "1"));
  var invert = optRow.add("checkbox", undefined, "Invert", { name: "invert" });
  invert.value = setting("invert", "false") === "true";
  var addLayer = optRow.add("checkbox", undefined, "Add as layer", { name: "addLayer" });
  addLayer.value = true;

  var tRow = ui.add("group");
  tRow.add("statictext", undefined, "Flicker smoothing (folders):");
  var temporal = tRow.add("slider", undefined, Number(setting("temporal", "0")), 0, 100);
  temporal.preferredSize.width = 100;

  var srvRow = ui.add("group");
  srvRow.add("statictext", undefined, "Server folder:");
  var srvDir = srvRow.add("edittext", undefined, setting("serverDir", ""), { name: "serverDir" });
  srvDir.characters = 26;
  var useGpu = srvRow.add("checkbox", undefined, "GPU", { name: "gpu" });
  useGpu.value = setting("gpu", "false") === "true";

  var btnRow = ui.add("group");
  var start = btnRow.add("button", undefined, "Start server", { name: "start" });
  var check = btnRow.add("button", undefined, "Check server", { name: "check" });
  var render = btnRow.add("button", undefined, "Render Depth", { name: "render" });

  var bar = ui.add("progressbar", undefined, 0, 100);
  bar.preferredSize.width = 260;
  var status = ui.add("statictext", undefined, "Press Start server, wait for its window to say ready, then Check server.", { name: "status" });
  status.characters = 40;

  function say(text) {
    status.text = text;
  }

  // Path of the footage file behind the selected layer, or "".
  function layerFile() {
    var c = app.project.activeItem;
    if (!(c instanceof CompItem) || c.selectedLayers.length === 0) return "";
    var src = c.selectedLayers[0].source;
    try {
      if (src && src.mainSource && src.mainSource.file) return src.mainSource.file.fsName;
      if (src && src.file) return src.file.fsName;
    } catch (e) {}
    return "";
  }

  function folderOf(path) {
    var i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
    return i > 0 ? path.substring(0, i) : path;
  }

  function open() {
    var s = new Socket();
    s.encoding = "UTF-8";
    s.timeout = 600;
    if (!s.open(HOST, "UTF-8")) {
      say("Server not running. Start it: python depthcraft_server.py (and allow script network access).");
      return null;
    }
    return s;
  }

  function request(req, onLine) {
    var s = open();
    if (!s) return null;
    s.writeln(JSON.stringify(req));
    var last = null;
    for (;;) {
      var line = s.readln();
      if (!line) break;
      last = JSON.parse(line);
      if (onLine) onLine(last);
      if (last.type === "done" || last.type === "error" || last.type === "pong") break;
    }
    s.close();
    return last;
  }

  function isWindowsPath(p) {
    return /^[A-Za-z]:[\\\/]/.test(p) || p.indexOf("\\") >= 0;
  }

  // The folder the server files go to: what the user typed, else next to the saved project, else C:\DepthCraft.
  function serverFolder() {
    if (srvDir.text) return srvDir.text;
    var proj = app.project.file;
    if (proj) {
      var base = folderOf(proj.fsName);
      return base + (isWindowsPath(base) ? "\\DepthCraft" : "/DepthCraft");
    }
    return "C:\\DepthCraft";
  }

  function writeText(path, text) {
    var f = new File(path);
    f.encoding = "UTF-8";
    if (!f.open("w")) throw new Error("cannot write " + path);
    f.write(text);
    f.close();
  }

  function windowsLauncher(gpuOn) {
    var pkg = gpuOn ? "onnxruntime-directml" : "onnxruntime";
    return [
      "@echo off",
      "title DepthCraft depth server",
      "cd /d \"%~dp0\"",
      "rem DepthCraft depth server: keeps the AI model loaded and waits on 127.0.0.1:47831 (this computer only).",
      "rem Leave this window open while you use the DepthCraft panel in EffectCraft; close it to stop.",
      "where py >nul 2>nul && (set PY=py -3) || (set PY=python)",
      "%PY% --version >nul 2>nul || (echo Python 3 was not found. Install it from https://www.python.org/downloads/ ^(tick Add python.exe to PATH^) and press Start server again. & pause & exit /b 1)",
      "%PY% -c \"import onnxruntime, numpy, PIL\" >nul 2>nul || (echo Installing the Python packages ^(once^)... & %PY% -m pip install --user " + pkg + " numpy pillow)",
      "%PY% depthcraft_server.py" + (gpuOn ? " --device gpu" : ""),
      "echo.",
      "echo The server stopped. Read any error above.",
      "pause",
    ].join("\r\n") + "\r\n";
  }

  start.onClick = function () {
    try {
      var dir = serverFolder();
      var win = isWindowsPath(dir);
      var folder = new Folder(dir);
      if (!folder.exists && !folder.create()) {
        say("Cannot create " + dir + ". Type another folder in Server folder.");
        return;
      }
      save("serverDir", dir);
      save("gpu", useGpu.value);
      var sep = win ? "\\" : "/";
      writeText(dir + sep + "depthcraft_server.py", SERVER_PY);
      if (!win) {
        say("Run in a terminal: python3 \"" + dir + "/depthcraft_server.py\"  (first: pip install onnxruntime numpy pillow)");
        srvDir.text = dir;
        return;
      }
      var launcher = dir + sep + "Start-DepthCraft-Server.bat";
      writeText(launcher, windowsLauncher(useGpu.value));
      srvDir.text = dir;
      var r = new File(launcher).execute();
      say(r === false ? "Could not open " + launcher + ". Double-click it yourself, then Check server." : "Starting... wait for the server window to say ready (first run installs packages and downloads the model), then Check server.");
    } catch (e) {
      say("Error: " + e);
    }
  };

  fromLayer.onClick = function () {
    var f = layerFile();
    if (!f) {
      say("Select a footage layer first.");
      return;
    }
    input.text = f;
    if (!output.text) output.text = folderOf(f) + "\\depthcraft";
    say("Input: " + f);
  };

  check.onClick = function () {
    var r = request({ cmd: "ping" });
    if (r && r.type === "pong") say("Server OK (" + r.providers + ").");
    else if (r) say("Unexpected reply: " + r.type);
  };

  render.onClick = function () {
    try {
      renderDepth();
    } catch (e) {
      say("Error: " + e);
    }
  };

  function renderDepth() {
    var inPath = input.text;
    if (!inPath) {
      inPath = layerFile();
      input.text = inPath;
    }
    if (!inPath) {
      say("Choose an input: select a footage layer (From layer) or type a file/folder path.");
      return;
    }
    if (!output.text) output.text = folderOf(inPath) + "\\depthcraft";
    var qualities = ["fast", "balanced", "ultra"];
    save("input", inPath);
    save("output", output.text);
    save("quality", quality.selection ? quality.selection.index : 1);
    save("invert", invert.value);
    save("temporal", Math.round(temporal.value));

    bar.value = 0;
    say("Rendering depth...");
    var req = {
      cmd: "depth",
      input: inPath,
      output: output.text,
      quality: qualities[quality.selection ? quality.selection.index : 1],
      invert: invert.value,
      refine: 0,
      temporal: Math.round(temporal.value),
    };
    var r = request(req, function (m) {
      if (m.type === "progress") {
        bar.value = Math.round((100 * m.done) / m.total);
        say("Depth " + m.done + "/" + m.total);
      }
    });
    if (!r) return;
    if (r.type === "error") {
      say("Error: " + r.message);
      return;
    }
    if (r.type !== "done") {
      say("Server closed the connection.");
      return;
    }

    bar.value = 100;
    app.beginUndoGroup("DepthCraft");
    var comp = app.project.activeItem;
    var src = comp instanceof CompItem && comp.selectedLayers.length ? comp.selectedLayers[0] : null;
    var imported = 0;
    for (var i = 0; i < r.files.length; i++) {
      var item = app.project.importFile(new ImportOptions(new File(r.files[i])));
      imported++;
      if (addLayer.value && comp instanceof CompItem && i === 0) {
        var layer = comp.layers.add(item);
        layer.name = "Depth - " + (src ? src.name : item.name);
        try {
          if (src) {
            layer.startTime = src.startTime;
            layer.moveBefore(src);
          }
        } catch (e) {}
      }
    }
    app.endUndoGroup();
    say("Done: " + imported + " depth map(s) imported" + (addLayer.value ? " (first added as a layer)." : "."));
  }

  if (ui instanceof Window) {
    ui.center();
    ui.show();
  } else {
    ui.layout.layout(true);
  }
})(this);
