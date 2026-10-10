// DepthCraft for Adobe After Effects: AI depth map (Depth Anything V3) from the selected layer.
//
// After Effects scripts can run programs, so this panel calls depthcraft-cli.exe directly (no server).
//   1. Unzip DepthCraft somewhere permanent, e.g. C:\DepthCraft  (keep the .dll files next to the .exe).
//   2. Install: copy this file to  ...\Adobe After Effects <version>\Support Files\Scripts\ScriptUI Panels\
//      and restart After Effects, then open it from the Window menu (or run it with File > Script > Run Script File).
//   3. Click "Browse" once and choose depthcraft-cli.exe. Enable Preferences > Scripting & Expressions >
//      "Allow Scripts to Write Files and Access Network" if After Effects asks.
//   4. Select a footage layer and press "Render Depth". The depth map (black = near, white = far) is imported and
//      added above the layer. A folder of frames can be typed into Input; it is processed frame by frame.
//
// Experimental: written for After Effects' documented scripting API, not yet tested in Adobe's application.
// Original work by EffectCraft contributors under MIT OR Apache-2.0.

(function depthCraftAE(thisObj) {
  var SECTION = "DepthCraft";

  function setting(key, fallback) {
    return app.settings.haveSetting(SECTION, key) ? app.settings.getSetting(SECTION, key) : fallback;
  }
  function save(key, value) {
    app.settings.saveSetting(SECTION, key, String(value));
  }

  var ui = thisObj instanceof Panel ? thisObj : new Window("palette", "DepthCraft", undefined, { resizeable: true });
  ui.orientation = "column";
  ui.alignChildren = ["fill", "top"];

  var exeRow = ui.add("group");
  exeRow.add("statictext", undefined, "depthcraft-cli.exe:");
  var exe = exeRow.add("edittext", undefined, setting("exe", "C:\\DepthCraft\\depthcraft-cli.exe"));
  exe.characters = 26;
  var browse = exeRow.add("button", undefined, "Browse");

  var inRow = ui.add("group");
  inRow.add("statictext", undefined, "Input (file or folder):");
  var input = inRow.add("edittext", undefined, "");
  input.characters = 24;
  var fromLayer = inRow.add("button", undefined, "From layer");

  var optRow = ui.add("group");
  optRow.add("statictext", undefined, "Quality:");
  var quality = optRow.add("dropdownlist", undefined, ["Fast", "Balanced", "Ultra"]);
  quality.selection = Number(setting("quality", "1"));
  var gpu = optRow.add("checkbox", undefined, "GPU");
  gpu.value = setting("gpu", "false") === "true";
  var invert = optRow.add("checkbox", undefined, "Invert");

  var render = ui.add("button", undefined, "Render Depth");
  var bar = ui.add("progressbar", undefined, 0, 100);
  bar.preferredSize.width = 260;
  var status = ui.add("statictext", undefined, "Select a footage layer, then Render Depth.");
  status.characters = 40;

  function say(t) {
    status.text = t;
    try {
      ui.update();
    } catch (e) {}
  }

  function layerFile() {
    var c = app.project.activeItem;
    if (!(c instanceof CompItem) || c.selectedLayers.length === 0) return null;
    var src = c.selectedLayers[0].source;
    return src && src.mainSource && src.mainSource.file ? src.mainSource.file : null;
  }

  browse.onClick = function () {
    var f = File.openDialog("Choose depthcraft-cli.exe", "Program:*.exe");
    if (f) exe.text = f.fsName;
  };
  fromLayer.onClick = function () {
    var f = layerFile();
    if (f) input.text = f.fsName;
    else say("Select a footage layer first.");
  };

  // Runs the CLI for one file or folder; returns its output text.
  function runCli(inPath, outPath) {
    var q = ["fast", "balanced", "ultra"][quality.selection ? quality.selection.index : 1];
    var args = ' depth --input "' + inPath + '" --output "' + outPath + '" --quality ' + q + (gpu.value ? " --device gpu" : "") + (invert.value ? " --invert-model" : "");
    // cmd /c with an extra pair of quotes keeps paths with spaces intact.
    return system.callSystem('cmd.exe /c ""' + exe.text + '"' + args + " 2>&1\"");
  }

  render.onClick = function () {
    try {
      var exeFile = new File(exe.text);
      if (!exeFile.exists) {
        say("depthcraft-cli.exe not found. Click Browse.");
        return;
      }
      var inFile = input.text ? (new Folder(input.text).exists ? new Folder(input.text) : new File(input.text)) : layerFile();
      if (!inFile || !inFile.exists) {
        say("Choose an input: select a footage layer or type a file/folder path.");
        return;
      }
      save("exe", exe.text);
      save("quality", quality.selection ? quality.selection.index : 1);
      save("gpu", gpu.value);

      var outDir = new Folder((inFile instanceof Folder ? inFile.fsName : inFile.parent.fsName) + "/depthcraft");
      if (!outDir.exists) outDir.create();

      var files = inFile instanceof Folder ? inFile.getFiles(/\.(png|jpe?g)$/i) : [inFile];
      files.sort();
      if (files.length === 0) {
        say("No png/jpg images in that folder.");
        return;
      }

      var comp = app.project.activeItem;
      var src = comp instanceof CompItem && comp.selectedLayers.length ? comp.selectedLayers[0] : null;
      app.beginUndoGroup("DepthCraft");
      for (var i = 0; i < files.length; i++) {
        bar.value = Math.round((100 * i) / files.length);
        say("Depth " + (i + 1) + "/" + files.length + " ...");
        var log = runCli(files[i].fsName, outDir.fsName);
        var name = files[i].name.replace(/\.[^.]*$/, "") + "_depth.png";
        var result = new File(outDir.fsName + "/" + name);
        if (!result.exists) {
          app.endUndoGroup();
          say("Failed: " + String(log).replace(/\s+/g, " ").substring(0, 160));
          return;
        }
        var item = app.project.importFile(new ImportOptions(result));
        if (i === 0 && comp instanceof CompItem) {
          var layer = comp.layers.add(item);
          layer.name = "Depth - " + (src ? src.name : item.name);
          if (src) {
            layer.startTime = src.startTime;
            layer.moveBefore(src);
          }
        }
      }
      app.endUndoGroup();
      bar.value = 100;
      say("Done: " + files.length + " depth map(s) imported (first added as a layer).");
    } catch (e) {
      say("Error: " + e);
    }
  };

  if (ui instanceof Window) {
    ui.center();
    ui.show();
  } else {
    ui.layout.layout(true);
  }
})(this);
