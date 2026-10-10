---
license: apache-2.0
base_model: onnx-community/depth-anything-v3-small
library_name: transformers.js
pipeline_tag: depth-estimation
tags:
  - onnx
  - depth-estimation
---

# Depth Anything V3 Small — 8-bit ONNX

An 8-bit dynamically quantised export of
[onnx-community/depth-anything-v3-small](https://huggingface.co/onnx-community/depth-anything-v3-small),
produced for [depth-realtime](https://github.com/en970/depth-realtime), a
browser-based real-time depth viewer.

The upstream repository ships float32 only, at 105 MB, which is more than a web
page should ask a visitor to download before it can show anything. This export
is 28.9 MB.

## Fidelity

Measured against the float32 original on three scenes, at 322 px input:

| Scene | Pearson r | float32 time | 8-bit time |
| --- | --- | --- | --- |
| Portrait, deep background | 0.9977 | 218 ms | 101 ms |
| Indoor, mid range | 0.9902 | 171 ms | 113 ms |
| Indoor, close range | 0.9984 | 127 ms | 90 ms |

Distinct depth levels surviving in the far third of the frame, after the
percentile normalisation and 8-bit transport a real viewer applies: 179 → 177,
84 → 91, 50 → 53. The quantisation does not cost background structure.

## Licence

Apache-2.0, inherited from the base model.
