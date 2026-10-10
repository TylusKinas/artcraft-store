# DepthCraft Attribution & Licensing

DepthCraft is an original EffectCraft plugin developed under the project license (MIT OR Apache-2.0).

## Open Model References & Specifications

1. **Depth Anything 3 (Small)**
   - **Authors**: ByteDance Seed, Depth Anything 3 team
   - **Homepage**: https://github.com/ByteDance-Seed/Depth-Anything-3
   - **License**: Apache-2.0 for the Small model (verify on the model card before redistributing weights)
   - **ONNX export**: https://huggingface.co/en970/depth-anything-v3-small-onnx
   - Used by `depthcraft-cli` only; its weights are in `models/` of this repository and embedded in the CLI.

2. **ONNX Runtime** and **DirectML** (Microsoft, MIT; DirectML under its own license, included in the Windows package) and the `ort` Rust bindings (MIT OR Apache-2.0), loaded dynamically at runtime.

3. **Turbo Colormap**
   - **Author**: Anton Mikhailov (Google LLC, 2019)
   - **License**: Apache-2.0
   - **Reference**: https://gist.github.com/mikhailov-work/ee72ba419ac4274247068a81a13600f6

## Original Implementation
All plugin code, bilateral cross-edge refinement, pre/post-processing, adaptive temporal smoothing filters, and pipeline caching are original works licensed under **MIT OR Apache-2.0**.
No proprietary code, binaries, or assets from third-party plugins were accessed, decompiled, or copied.
