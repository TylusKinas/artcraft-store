// DepthCraft: AI depth maps (Depth Anything V3) for EffectCraft, as a ScriptUI panel.
//
// An EffectCraft wasm plug-in cannot run a neural network (the host meters wasm with a small fuel
// budget) and scripts cannot launch programs, but scripts can open TCP sockets. This panel talks to
// the local `depthcraft-cli serve` program, which keeps the model loaded and writes the depth map.
//
//   1. Start the server once:  run Start-DepthCraft-Server.bat (or: depthcraft-cli.exe serve)
//   2. Preferences > Scripting & Expressions > Allow Scripts to Write Files and Access Network: ON
//   3. Install DepthCraft-EffectCraft.jsx with File > Scripts > Install ScriptUI Panel..., open it from the Window menu.
//   4. Select a footage layer and press Render Depth. The depth map is imported and added as a layer
//      (black = near, white = far). Apply the DepthCraft effect (depthcraft.wasm, folder EffectCraft\) to it for heatmap,
//      fog, slices, cutouts.
//
// Original work by EffectCraft contributors under MIT OR Apache-2.0.

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

  var btnRow = ui.add("group");
  var check = btnRow.add("button", undefined, "Check server", { name: "check" });
  var render = btnRow.add("button", undefined, "Render Depth", { name: "render" });

  var bar = ui.add("progressbar", undefined, 0, 100);
  bar.preferredSize.width = 260;
  var status = ui.add("statictext", undefined, "Start the server (Start-DepthCraft-Server.bat), then Render Depth.", { name: "status" });
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
      say("Server not running. Start Start-DepthCraft-Server.bat (and allow script network access).");
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
