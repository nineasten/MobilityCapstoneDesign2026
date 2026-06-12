# Jetson AGX Orin low-level optimization playbook

This directory holds Orin-specific optimization tools for the HLoc AR
navigation pipeline. The pipeline itself (`hloc_persistent_localization_node.py`)
is unchanged and continues to work as-is on the laptop (RTX3060). Everything
here is additive prep work / checklists for when the pipeline is deployed to
the Orin.

Current laptop baseline (RTX3060 Laptop, 80W cap, FP16 backbone + FP16-autocast
matcher): ~0.276s/frame avg (~3.6Hz), 94-96/121 frames accepted, avg 90-94
inliers, well above the 0.5Hz target.

## 1. Power mode (`setup_power_mode.sh`)

Run once after boot on the Orin:

```bash
./setup_power_mode.sh MAXN   # or 30W / 15W
```

This calls `nvpmodel -m <id>` + `jetson_clocks` to lock GPU/CPU/EMC clocks to
the max for the chosen mode, so latency is reproducible. **Verify the mode
IDs first** with `sudo nvpmodel -q --verbose` -- they differ across JetPack
versions and the script's IDs are a starting guess, not guaranteed for this
board's JetPack version.

Given our 0.5Hz target has 2x-6x margin even in the worst-case estimate, start
at **30W** rather than MAXN -- it leaves thermal headroom for sustained
operation and is closer to where the laptop's 80W-capped RTX3060 sits
relative to its max.

## 2. ONNX export (`export_onnx_models.py`)

Run on the laptop (already done, outputs in `model_cache/onnx/`, gitignored):

```bash
TORCH_HOME=$HOME/capstonews/model_cache/torch \
  python3 export_onnx_models.py --max-keypoints 512 --resize 800x500
```

Produces:
- `netvlad.onnx` (569MB, full NetVLAD incl. VGG16 backbone + whitening)
- `superpoint_dense.onnx` (5MB, SuperPoint backbone+heads up to the NMS'd
  dense score map and normalized dense descriptor map)

Both were verified against PyTorch (max abs diff ~1e-6) at 800x500 and
640x400, confirming the dynamic height/width axes work correctly.

SuperPoint's keypoint extraction (`torch.nonzero`, border removal, top-k,
descriptor sampling via `grid_sample`) has data-dependent output shapes and
is **not** exported -- it stays as the existing lightweight PyTorch/numpy
post-processing on dense outputs, unchanged.

**Copy these two `.onnx` files to the Orin** (they are architecture-portable;
only the compiled TensorRT *engine* is not).

## 3. Building TensorRT engines (run ON the Orin)

TensorRT engines are tied to the exact GPU (compute capability + TensorRT
version) they're built on -- an engine built on the RTX3060 (sm_86) will not
load on Orin (sm_87). Engines must be built on-device:

```bash
# FP16 engine, fixed batch=1, input size matching the live pipeline
# (camera is 960x600, resize_max=800 -> resized image is ~800x500)
trtexec --onnx=netvlad.onnx \
  --saveEngine=netvlad_fp16.engine \
  --fp16 \
  --shapes=image:1x3x500x800

trtexec --onnx=superpoint_dense.onnx \
  --saveEngine=superpoint_dense_fp16.engine \
  --fp16 \
  --shapes=image:1x1x500x800
```

Then benchmark in isolation before touching the pipeline:

```bash
trtexec --loadEngine=netvlad_fp16.engine --shapes=image:1x3x500x800
trtexec --loadEngine=superpoint_dense_fp16.engine --shapes=image:1x1x500x800
```

Compare the reported "GPU Compute" latency against the PyTorch `netvlad_s`
and `superpoint_s` numbers from `results.jsonl` (currently ~0.064s and
~0.020s respectively on RTX3060) to see the actual speedup on Orin before
deciding whether integration is worth the added complexity.

### Integration (only if the benchmark shows a meaningful win)

This would require a new `TRTSuperPoint`/`TRTNetVLAD` wrapper in
`hloc_persistent_localization_node.py` that:
1. Runs the TensorRT engine via `pycuda`/`torch2trt`/`onnxruntime-gpu` with
   the TensorRT execution provider, instead of `self.netvlad`/`self.superpoint`.
2. Feeds the dense outputs into the existing (unchanged) NMS/keypoint
   extraction + `write_feature` code path.

Not implemented yet -- this is a checklist for after the Orin benchmark above
justifies the engineering cost.

## 4. DLA offload

Orin has 2x DLA (Deep Learning Accelerator) cores, usable via TensorRT
(`--useDLACore=0/1` in `trtexec`, with `--allowGPUFallback` for unsupported
layers). Candidates and caveats:

- **NetVLAD's VGG16 backbone**: mostly standard conv/relu/maxpool -- DLA's
  best case. Worth trying `trtexec --onnx=netvlad.onnx --useDLACore=0
  --allowGPUFallback --fp16` and comparing latency to the GPU-only engine.
- **SuperPoint dense backbone**: similar conv-heavy structure, also a
  candidate.
- DLA only supports a fixed op subset and FP16/INT8 (no FP32). Layers it
  can't run fall back to GPU automatically with `--allowGPUFallback`, but
  check the fallback log -- if most layers fall back, DLA isn't helping.
- Running NetVLAD on DLA while SuperPoint+matcher run on GPU could let both
  execute **concurrently** (DLA and GPU are separate engines), which is a
  real latency win beyond raw per-op speed -- worth testing once the TensorRT
  engines above exist.

## 5. nvjpeg hardware JPEG encoding

Relevant only if the live debug image logging (`live_query.jpg` write in
`hloc_persistent_localization_node.py`, controlled by `enable_result_log`)
or any image transport step becomes a bottleneck on Orin. The Jetson
multimedia API / `nvjpegEncoder` (via `jetson-utils` or GStreamer's
`nvjpegenc`) offloads JPEG encode to a dedicated hardware block, freeing the
GPU/CPU.

Given the current pipeline already disables most debug image I/O for
performance (see prior optimization pass), this is **low priority** --
revisit only if profiling on the actual Orin shows JPEG encode/IO as a
measurable fraction of frame time.
