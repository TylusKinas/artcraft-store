@echo off
rem ============================================================================
rem  DepthCraft depth server
rem
rem  What it does: starts depthcraft-cli.exe in "serve" mode. The AI model is loaded
rem  once and then waits on 127.0.0.1:47831 (this computer only, nothing leaves it).
rem  The DepthCraft panel in EffectCraft sends it an image path, this window makes the
rem  depth map and writes it next to your footage, and the panel imports it.
rem
rem  Why it is needed: EffectCraft scripts can talk to a local port but cannot start
rem  programs, and a neural network is too heavy for an EffectCraft wasm plug-in.
rem
rem  How to use: double-click this file, leave the window open while you work in
rem  EffectCraft, close it when you are done. Add --device gpu below for DirectX 12 GPUs.
rem  (Adobe After Effects users do not need this: use AfterEffects\DepthCraft-AE.jsx.)
rem ============================================================================
cd /d "%~dp0"
echo DepthCraft depth server - keep this window open while using the DepthCraft panel.
echo.
depthcraft-cli.exe serve %*
echo.
echo The server stopped. If there is an error above, check that onnxruntime.dll and
echo DirectML.dll are in this folder.
pause
