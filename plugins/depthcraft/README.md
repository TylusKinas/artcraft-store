# DepthCraft

AI depth maps for **EffectCraft** and **Adobe After Effects**, powered by **Depth Anything V3 Small** (CPU, or GPU through DirectML).

## Quick start: one file

1. EffectCraft: Preferences ▸ Scripting & Expressions ▸ tick *Allow Scripts to Write Files and Access Network* (and leave *Warn User When Executing Files* on: it asks once before opening the launcher).
2. File ▸ Scripts ▸ Install ScriptUI Panel… ▸ [`DepthCraft-EffectCraft.jsx`](panels/DepthCraft-EffectCraft.jsx), then open **DepthCraft** from the Window menu.
3. Press **Start server**. The panel writes the server (it is embedded in the file) and a launcher into the *Server folder*, and on Windows opens the launcher: a console window installs what is missing (Python packages and the 29 MB model, once; needs [Python 3](https://www.python.org/downloads/)) and keeps the server running. Leave it open. On macOS/Linux the panel shows the one command to paste into a terminal instead (EffectCraft can only "open" a file, which would show a `.py` in an editor there).
4. Press **Check server** until it says OK, select a footage layer ▸ **From layer** ▸ **Render Depth**. The depth map is imported and added as a layer.

*GPU* checkbox: installs `onnxruntime-directml` and starts the server with `--device gpu` (Windows, any DirectX 12 GPU). Optional: [`depthcraft.wasm`](depthcraft.wasm) (Effect ▸ Load Effect Plug-in…) for heatmap / fog / slice / cutout on the depth layer. No Python? **Windows users can download [`release/DepthCraft-Windows.zip`](release/DepthCraft-Windows.zip)** (exe with everything built in, panels, `README.txt`). The same server also runs by hand: `python depthcraft_server.py` (or `uv run depthcraft_server.py`), and as a command line: `python depthcraft_server.py depth photo.jpg --output out`.

## How it fits together

| Piece | Where | What it does |
|---|---|---|
| `depthcraft_server.py` | `panels/` | **Cross-platform server in one Python file** (ONNX Runtime; CPU or GPU). Also usable as a command line: `python depthcraft_server.py depth photo.jpg --output out`. |
| `depthcraft-cli.exe` | zip | The same engine as a native Windows program. Model built in, ONNX Runtime + DirectML next to it. `depth` renders files/folders, `serve` keeps the model loaded for the EffectCraft panel. |
| `Start-DepthCraft-Server.bat` | `panels/` | Starts `depthcraft-cli serve` (explained inside the file). Needed only for EffectCraft. |
| `DepthCraft-EffectCraft.jsx` | `panels/` | **The one file to install** in EffectCraft: Start server / Render Depth buttons, progress bar; embeds `depthcraft_server.py` (`python3 scripts/embed_server.py` refreshes it) and talks to it over `127.0.0.1:47831`. |
| `DepthCraft-AfterEffects.jsx` | `panels/` | Adobe After Effects panel; runs the CLI directly (experimental, untested in Adobe's app). |
| `depthcraft.wasm` (37 KB) | root | EffectCraft effect (Effect ▸ Stylize ▸ DepthCraft): near/far remap, gamma, invert, **Grayscale / Heatmap / Fog Layer / Depth Slice / Depth Cutout** on a depth-map layer. |

**Why not one wasm file?** EffectCraft runs wasm plug-ins in the `wasmi` interpreter with a fuel budget of 50M + 20k instructions per pixel (`crates/plugin/src/wasm.rs` in the EffectCraft repo). I measured it with a replica of the host: MiDaS Small compiled into the wasm exhausted the budget even at 1080p, after which the host leaves the frame unchanged without an error. A neural network cannot run inside an EffectCraft wasm plug-in, and scripts cannot launch programs (`system.callSystem` is blocked), so EffectCraft gets a TCP-socket panel plus a local server. The effect itself stays tiny (6–28 % of the budget at 1080p, verified).

Verified: the panel's socket flow (check server, render request, progress, depth PNG written) runs in EffectCraft's real script engine against a live server; the CLI/server runs on Linux with ONNX Runtime 1.24 and 1.29 (~0.2–0.3 s per 320×240 image on 4 CPU cores). Not verified here: the Windows exe and DirectML on real hardware, Adobe After Effects, and the final import step inside the full EffectCraft app.

GPU note: the model is 8-bit quantised (CPU-friendly). DirectML runs unsupported integer ops on the CPU, so the GPU gain can be small; a float32/fp16 Depth Anything V3 ONNX would benefit more.

## Command line
```bash
depthcraft-cli depth --input photo.jpg --output out [--device gpu] [--quality fast|balanced|ultra] [--invert-model]
depthcraft-cli depth --input frames --output out --temporal 40      # frame folder, flicker smoothing
depthcraft-cli serve [--port 47831] [--device gpu]
```
Output: 16-bit grayscale `<name>_depth.png` (black = near, white = far). ONNX Runtime is found via `--ort-lib`, `$ORT_DYLIB_PATH`, or `onnxruntime.dll` next to the exe.

## Build
```bash
cargo test                                                      # effect post-processing, pure Rust
cargo build --release --target wasm32-unknown-unknown           # depthcraft.wasm
cargo build --release --features embed-model                    # native CLI (Linux/macOS), model built in
# Windows package from Linux:
pip install ziglang cargo-zigbuild && rustup target add x86_64-pc-windows-gnu
cargo zigbuild --release --features embed-model --target x86_64-pc-windows-gnu --bin depthcraft-cli
python3 scripts/package_windows.py                              # -> release/DepthCraft-Windows.zip
```
`models/` holds the 8-bit Depth Anything V3 Small ONNX (the build embeds it) and its model card. `ec_alloc` pre-grows wasm memory so `ec_render` never calls `memory.grow` (that would detach the host's pixel buffer); the module has no host imports.
