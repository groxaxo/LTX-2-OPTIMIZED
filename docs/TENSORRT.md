# TensorRT acceleration

This fork includes an opt-in Torch-TensorRT backend for the two compact, tensor-heavy parts of the LTX pipeline:

- the spatial latent upsampler used between the distilled pipeline's two stages;
- the video VAE encoder and decoder.

The 22B transformer remains on the repository's existing PyTorch/FP8 path. On an RTX 3090, compiling a second
BF16 copy of that model would exceed the 24 GiB memory budget. Torch-TensorRT 2.8 also predates the later native
TensorRT attention lowering, so this integration does not decompose the transformer's large attention operation.

## Runtime target

The dependency versions match the repository's verified runtime:

- PyTorch 2.8;
- CUDA 12.8;
- Torch-TensorRT 2.8;
- TensorRT 10.12;
- Ubuntu x86-64;
- RTX 3090 / compute capability 8.6.

The backend uses hybrid execution. Supported graph regions run in TensorRT, while unsupported regions remain in
PyTorch. Engine compilation is lazy on the first invocation of each tensor signature.

## Installation

From the repository virtual environment:

```bash
python -m pip install -r requirements-tensorrt.txt
```

Verify the matched stack:

```bash
python - <<'PY'
import tensorrt
import torch
import torch_tensorrt

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Torch-TensorRT:", torch_tensorrt.__version__)
print("TensorRT:", tensorrt.__version__)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

## Web UI v4

Launch the UI through the TensorRT wrapper:

```bash
python run_tensorrt.py web-v4
```

In **Advanced**, enable **Multi-GPU (3× 3090)**. The UI launches generation in a child process, which inherits the
TensorRT environment from the wrapper.

The default three-GPU allocation remains:

- GPU 0: FP8 transformer;
- GPU 1: video VAE and spatial upsampler, with TensorRT regions;
- GPU 2: Gemma and audio models.

## Distilled CLI

Pass pipeline arguments after `--`:

```bash
python run_tensorrt.py distilled -- \
  --distilled-checkpoint-path ./models/ltx-2.3-22b-dev.safetensors \
  --gemma-root ./models/gemma3 \
  --spatial-upsampler-path ./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --quantization fp8-cast \
  --multi-gpu \
  --prompt "A cinematic aerial shot over an alpine lake" \
  --output-path output-tensorrt.mp4 \
  --width 1280 --height 720 --num-frames 121
```

Other launch targets are available through `python run_tensorrt.py --help`.

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `LTX_TENSORRT` | `0` | Enable the backend. The launcher sets it to `1`. |
| `LTX_TENSORRT_COMPONENTS` | `upsampler,vae` | Components to compile. |
| `LTX_TENSORRT_CACHE_DIR` | `~/.cache/ltx/tensorrt` | Timing and engine cache root. |
| `LTX_TENSORRT_CACHE_GB` | `64` | Maximum persistent engine-cache size. |
| `LTX_TENSORRT_WORKSPACE_GB` | `4` | Per-engine builder workspace ceiling. |
| `LTX_TENSORRT_MIN_BLOCK_SIZE` | `5` | Minimum contiguous TensorRT region size. |
| `LTX_TENSORRT_OPT_LEVEL` | `3` | TensorRT builder optimization level, from 0 to 5. |
| `LTX_TENSORRT_MAX_AUX_STREAMS` | `2` | Maximum auxiliary streams per engine. |
| `LTX_TENSORRT_DYNAMIC` | `0` | Use dynamic-shape tracing instead of static specialization. |
| `LTX_TENSORRT_ENGINE_CACHE` | `1` | Persist and reuse engines. |
| `LTX_TENSORRT_FAST_PARTITIONER` | `1` | Use the faster graph partitioner. |
| `LTX_TENSORRT_EXPERIMENTAL_DECOMPOSITIONS` | `0` | Opt into experimental operator decompositions. |
| `LTX_TENSORRT_STRICT` | `0` | Raise instead of falling back when compilation fails. |
| `LTX_TENSORRT_DEBUG` | `0` | Enable verbose compiler diagnostics. |
| `LTX_TENSORRT_ALLOW_SINGLE_GPU_COMPONENTS` | `0` | Permit VAE/upsampler engines on a shared GPU. |

The default single-GPU gate prevents TensorRT engine weights from competing with the resident FP8 transformer.
Use `--allow-single-gpu-components` only after measuring available VRAM.

## Engine cache behavior

The cache is namespaced by:

- PyTorch major/minor version;
- Torch-TensorRT major/minor version;
- TensorRT major/minor version;
- CUDA runtime version;
- GPU compute capability.

This prevents incompatible engines from being reused after a runtime or GPU-architecture change. Each new static
shape compiles on first use; later runs reuse the cached engine. Keep the cache on the local NVMe filesystem.

Clear it after changing TensorRT versions or when diagnosing a stale engine:

```bash
rm -rf ~/.cache/ltx/tensorrt
```

## Benchmarking

The repository includes a subprocess benchmark that runs identical baseline and TensorRT commands and writes a
JSON report:

```bash
python scripts/benchmark_tensorrt.py \
  --warmups 1 \
  --runs 3 \
  --result-json benchmark-1280x720-121f.json \
  -- \
  python -m ltx_pipelines.distilled \
    --distilled-checkpoint-path ./models/ltx-2.3-22b-dev.safetensors \
    --gemma-root ./models/gemma3 \
    --spatial-upsampler-path ./models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --quantization fp8-cast \
    --multi-gpu \
    --prompt "A cinematic aerial shot over an alpine lake" \
    --output-path benchmark.mp4 \
    --width 1280 --height 720 --num-frames 121
```

Compare the JSON median only after the TensorRT warmup has completed. Also inspect the generated logs for actual
TensorRT partitions and fallback warnings.

## Failure policy

Unsupported graph regions remain in PyTorch through hybrid execution. If lazy compilation or TensorRT runtime
execution raises an exception, the wrapper disables only that component and immediately retries the call through
its original PyTorch `forward`. Existing weights, state dictionaries, hooks, and pipeline APIs are preserved.

CUDA out-of-memory errors are not swallowed. They remain fatal because retrying the same allocation through eager
execution is unlikely to be safe. Reduce resolution, frame count, workspace size, or enabled components.

Use strict mode for validation:

```bash
python run_tensorrt.py web-v4 --strict --debug
```

Do not use strict mode for unattended queues until all production presets have been warmed and compared against the
PyTorch baseline.
