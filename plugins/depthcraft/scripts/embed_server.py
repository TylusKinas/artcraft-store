#!/usr/bin/env python3
"""Embed panels/depthcraft_server.py into panels/DepthCraft-EffectCraft.jsx (between the PY_BEGIN/PY_END markers).

The panel is the single file a user installs; it writes the embedded server to disk when "Start server" is pressed.
Run after editing depthcraft_server.py:  python3 scripts/embed_server.py
"""
import json
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent.parent / "panels"
py = (root / "depthcraft_server.py").read_text(encoding="utf-8")
lines = ",\n".join("  " + json.dumps(line) for line in py.splitlines())
block = "/*PY_BEGIN*/\nvar SERVER_PY = [\n" + lines + '\n].join("\\n") + "\\n";\n/*PY_END*/'
jsx = root / "DepthCraft-EffectCraft.jsx"
text = jsx.read_text(encoding="utf-8")
new, n = re.subn(r"/\*PY_BEGIN\*/.*?/\*PY_END\*/", lambda _: block, text, flags=re.S)
assert n == 1, "markers not found"
jsx.write_text(new, encoding="utf-8")
print("embedded", len(py.splitlines()), "lines into", jsx.name)
