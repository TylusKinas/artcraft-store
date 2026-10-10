#!/usr/bin/env python3
"""Build release/DepthCraft-Windows.zip: the single download for regular users.

Needs the Windows exe built first (see README "Build") and internet for the ONNX Runtime / DirectML NuGet packages.

    cargo zigbuild --release --features embed-model --target x86_64-pc-windows-gnu --bin depthcraft-cli
    cargo build --release --target wasm32-unknown-unknown
    python3 scripts/package_windows.py
"""
import pathlib
import urllib.request
import zipfile

root = pathlib.Path(__file__).resolve().parent.parent
NUGET = "https://api.nuget.org/v3-flatcontainer"
ORT_DML = f"{NUGET}/microsoft.ml.onnxruntime.directml/1.24.4/microsoft.ml.onnxruntime.directml.1.24.4.nupkg"
DML = f"{NUGET}/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg"

README = r"""DepthCraft - AI depth maps for EffectCraft and Adobe After Effects
===================================================================

WHAT IS IN THIS FOLDER
  depthcraft-cli.exe, *.dll       The depth engine (Depth Anything V3 Small, built in). Runs on your CPU, or on a
                                  DirectX 12 GPU with --device gpu. Keep the .dll files next to the .exe.
  Start-DepthCraft-Server.bat     EffectCraft only: starts the engine and keeps it ready (see below).
  depthcraft_server.py            The same server as one Python file (Windows/macOS/Linux): pip install onnxruntime
                                  numpy pillow, then python depthcraft_server.py. Use it instead of the .exe/.bat if you like.
  EffectCraft\DepthCraft-EffectCraft.jsx   EffectCraft panel: "Start server" and "Render Depth" buttons. If you have Python, this
                                           one file is enough: the panel starts the server for you (it is embedded).
  EffectCraft\depthcraft.wasm              EffectCraft effect: colorize / fog / slice / cutout a depth map.
  AfterEffects\DepthCraft-AfterEffects.jsx  Adobe After Effects panel (no server needed).

WHY A SERVER?  (EffectCraft)
  EffectCraft scripts can talk to a port on your own computer but cannot start programs, and a neural network is far
  too heavy for an EffectCraft wasm plug-in. So Start-DepthCraft-Server.bat runs the engine once, with the model loaded;
  the panel sends it an image path and imports the depth map it writes. Nothing leaves your computer (127.0.0.1 only).

ADOBE AFTER EFFECTS (experimental, not yet tested in Adobe's app)
  1. Copy AfterEffects\DepthCraft-AfterEffects.jsx into  ...\Adobe After Effects <version>\Support Files\Scripts\ScriptUI Panels
     and restart After Effects. Window > DepthCraft-AfterEffects.jsx.
  2. Browse... to depthcraft-cli.exe once, select a footage layer, Render Depth.

EFFECTCRAFT, STEP BY STEP
  1. Double-click Start-DepthCraft-Server.bat and leave that window open.
  2. EffectCraft: Preferences > Scripting & Expressions > tick "Allow Scripts to Write Files and Access Network".
  3. File > Scripts > Install ScriptUI Panel... > EffectCraft\DepthCraft-EffectCraft.jsx. Open it from the Window menu.
  4. Select a footage layer, press "From layer", then "Render Depth". The depth map (black = near, white = far) is imported
     and added above the layer.
  5. Optional: Effect > Load Effect Plug-in... > EffectCraft\depthcraft.wasm, then Effect > Stylize > DepthCraft on the
     depth layer for heatmap, fog layer, slices and cutouts.

WITHOUT ANY PANEL (command line)
  depthcraft-cli.exe depth --input C:\photo.jpg --output C:\out            (add --device gpu for the GPU)
  depthcraft-cli.exe depth --input C:\frames --output C:\out --temporal 40  (folder = frame sequence, flicker smoothing)

LICENSES  Depth Anything V3 Apache-2.0; ONNX Runtime MIT; DirectML see DirectML-LICENSE.txt.
"""


def fetch(url, name):
    dest = root / "target" / "pkg" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    return zipfile.ZipFile(dest)


exe = root / "target/x86_64-pc-windows-gnu/release/depthcraft-cli.exe"
wasm = root / "target/wasm32-unknown-unknown/release/depthcraft.wasm"
ort, dml = fetch(ORT_DML, "ortdml.nupkg"), fetch(DML, "dml.nupkg")

out = root / "release" / "DepthCraft-Windows.zip"
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    top = "DepthCraft/"
    z.writestr(top + "README.txt", README.replace("\n", "\r\n"))
    z.write(exe, top + "depthcraft-cli.exe")
    z.writestr(top + "onnxruntime.dll", ort.read("runtimes/win-x64/native/onnxruntime.dll"))
    z.writestr(top + "onnxruntime_providers_shared.dll", ort.read("runtimes/win-x64/native/onnxruntime_providers_shared.dll"))
    z.writestr(top + "DirectML.dll", dml.read("bin/x64-win/DirectML.dll"))
    z.writestr(top + "ONNXRuntime-LICENSE.txt", ort.read("LICENSE"))
    z.writestr(top + "DirectML-LICENSE.txt", dml.read("LICENSE.txt"))
    z.write(root / "panels/Start-DepthCraft-Server.bat", top + "Start-DepthCraft-Server.bat")
    z.write(root / "panels/depthcraft_server.py", top + "depthcraft_server.py")
    z.write(root / "panels/DepthCraft-EffectCraft.jsx", top + "EffectCraft/DepthCraft-EffectCraft.jsx")
    z.write(wasm, top + "EffectCraft/depthcraft.wasm")
    z.write(root / "panels/DepthCraft-AfterEffects.jsx", top + "AfterEffects/DepthCraft-AfterEffects.jsx")
print("wrote", out, out.stat().st_size // 1024 // 1024, "MB")
