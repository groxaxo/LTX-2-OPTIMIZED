# LTX-2 Optimized (8GB VRAM Edition) + Web UI

This repository contains a **modified and optimized version of the LTX-2 Video Generation Model**, designed specifically to run on consumer hardware with as little as **8GB of VRAM**.

It includes a fully-featured **Gradio Web Interface** to make generating videos, managing presets, and applying LoRAs easy without needing to remember complex command-line arguments.

> **Compatibility:** This fork is synchronized with the upstream [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) repository (as of 2026-03-05) and is compatible with the latest LTX-2.3 model releases.

## Web UI Screenshots

### Web UI v2
<img width="2260" height="1078" alt="image" src="https://github.com/user-attachments/assets/5a9f5dce-f313-44a3-bbbe-10eccc002191" />

### Web UI v4
<img width="949" height="575" alt="scr221" src="https://github.com/user-attachments/assets/2e3a6c51-51b8-4487-b64d-a7d2c41d794a" />

### CinemaMaker UI
<img width="612" height="343" alt="cm" src="https://github.com/user-attachments/assets/0f5f2dca-bacd-4f5f-a627-505ad3751277" />

* https://youtu.be/eGOq0hUiri4
* https://youtu.be/HAQqzPdDIj0

### Music to Video UI (S2V, Lip Sync)
<img width="612" height="343" alt="cm" src="https://github.com/user-attachments/assets/852845f9-f113-41f7-a5e6-8a4e1dec0778" />

* https://youtu.be/HzK1nW-OVtQ

## 🚀 Fork Features & Optimizations

* **8GB VRAM Optimization:** Runs locally on cards like the RTX 3070/4060Ti using FP8 quantization and memory management tweaks.
* **Windows 11 support:** You can run it on Windows (not officially supported in the original model).
* **User-Friendly Web UI:** Control everything from your browser using `web_ui_v2.py` or `web_ui_v4.py`.
* **Smart "Safe Mode":** The UI automatically limits the frame count based on selected resolution to prevent Out-Of-Memory (OOM) errors.
* **Real-time Logging:** View the generation progress and console output directly in the web interface.
* **Optimized Transformer Code:** Increased max frames by ~40% for text-to-video; generation speed improved from 300–315 sec to 385–415 sec (RTX 3070 Ti laptop).
* **Stage 1 Video Preview, Task Queue, Prompt Constructor:** Available in Web UI v4.
* **Disable Audio option:** Speeds up inference by 10–30%.
* **Film Maker UI:** Full-featured film-making interface (`film_maker_ui_v4.py`).
* **Music to Video UI:** Audio-driven video generation with optional start/last frame conditioning (`music_maker_ui.py`, `music_maker_ui_v2.py`).
* **Advanced Features:** Image Conditioning, LoRA Support (checkbox selection for Camera Control), Seed Control for reproducible generations.

## 📥 Installation (Fork with Web UI)

**GPU Requirements:** NVIDIA GPU with at least 8GB VRAM (e.g., RTX 3070, 4060 Ti). For Windows, use WSL2 or native Python 3.12.

**1. Clone this repository:**
```bash
git clone https://github.com/groxaxo/LTX-2-OPTIMIZED.git
cd LTX-2-OPTIMIZED
```

**2. Install dependencies:**
```bash
pip install -e packages/ltx-pipelines
pip install -e packages/ltx-core
```

Verified working environment:
```
Python 3.12.8
accelerate==1.10.1
torch==2.8.0+cu128
torchaudio==2.8.0+cu128
torchvision==0.23.0+cu128
xformers==0.0.32.post2
```

**3. Create model directories and download models:**
```bash
mkdir -p models/loras
```

Download required files:
* [`ltx-2-19b-distilled-fp8.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors)
* [`ltx-2-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors)
* [`Gemma 3`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main) → place in `./models/gemma3/`

Expected directory layout:
```
./models/
    ltx-2-19b-distilled-fp8.safetensors
    ltx-2-spatial-upscaler-x2-1.0.safetensors
./models/gemma3/
    (Gemma 3 model files)
./models/loras/
    (LoRA files, optional)
```

**4. Launch Web UI:**
```bash
python web_ui_v2.py
# or
python web_ui_v4.py
```

**5. Launch other UIs:**
```bash
python film_maker_ui_v4.py       # CinemaMaker UI
python music_maker_ui.py         # Music to Video (distilled 2-step, fast)
python music_maker_ui_v2.py      # Music to Video (2-step, higher quality)
```

## 📊 Performance on 8GB VRAM (RTX 3070 Ti Laptop)

| Resolution   | Max Frames (i2v) | Max Frames (t2v) | Est. Time      |
|:-------------|:-----------------|:-----------------|:---------------|
| 1280 × 704   | 177              | 257              | ~300–400 sec   |
| 1536 × 1024  | 121              | 185              | ~300–400 sec   |
| 1920 × 1088  | 81               | 121              | ~300–400 sec   |
| 2560 × 1408  | 49               | 65               | ~300–400 sec   |
| 3840 × 2176  | 17               | 25               | ~300–400 sec   |

* +60 sec for prompt (if not empty/not cached)
* Stage 1 preview: 80–150 sec

## Credits

* Original Model: Lightricks (LTX-2 / LTX-2.3)
* Fork Optimizations & Web UI: nalexand, groxaxo
* Community contributions welcome!

---

# LTX-2

[![Website](https://img.shields.io/badge/Website-LTX-181717?logo=google-chrome)](https://ltx.io)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.3)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=vercel)](https://app.ltx.studio/ltx-2-playground/i2v)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2601.03233)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/ltxplatform)

**LTX-2** is the first DiT-based audio-video foundation model that contains all core capabilities of modern video generation in one model: synchronized audio and video, high fidelity, multiple performance modes, production-ready outputs, API access, and open access.

<div align="center">
  <video src="https://github.com/user-attachments/assets/4414adc0-086c-43de-b367-9362eeb20228" width="70%" poster=""> </video>
</div>

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2

# Set up the environment
uv sync --frozen
source .venv/bin/activate
```

### Required Models

Download the following models from the [LTX-2.3 HuggingFace repository](https://huggingface.co/Lightricks/LTX-2.3):

**LTX-2.3 Model Checkpoint** (choose and download one of the following)
  * [`ltx-2.3-22b-dev.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors)
  * [`ltx-2.3-22b-distilled.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled.safetensors)

**Spatial Upscaler** - Required for current two-stage pipeline implementations in this repository
  * [`ltx-2.3-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors)
  * [`ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors)

**Temporal Upscaler** - Supported by the model and will be required for future pipeline implementations
  * [`ltx-2.3-temporal-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors)

**Distilled LoRA** - Required for current two-stage pipeline implementations in this repository (except DistilledPipeline and ICLoraPipeline)
  * [`ltx-2.3-22b-distilled-lora-384.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-lora-384.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors)

**Gemma Text Encoder** (download all assets from the repository)
  * [`Gemma 3`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main)

**LoRAs**
  * [`LTX-2.3-22b-IC-LoRA-Union-Control`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) - [Download](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors)
  * [`LTX-2.3-22b-IC-LoRA-Inpainting`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Inpainting) - [Download](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Inpainting/resolve/main/ltx-2.3-22b-ic-lora-inpainting.safetensors)
  * [`LTX-2.3-22b-IC-LoRA-Motion-Track-Control`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) - [Download](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/resolve/main/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors)
  * [`LTX-2-19b-IC-LoRA-Detailer`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors)
  * [`LTX-2-19b-IC-LoRA-Pose-Control`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-In`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Left`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Out`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Right`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Jib-Down`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Jib-Up`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Static`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors)

### Available Pipelines

* **[TI2VidTwoStagesPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)** - Production-quality text/image-to-video with 2x upsampling (recommended)
* **[TI2VidTwoStagesHQPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py)** - Same two-stage flow as above but uses the res_2s second-order sampler (fewer steps, better quality)
* **[TI2VidOneStagePipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py)** - Single-stage generation for quick prototyping
* **[DistilledPipeline](packages/ltx-pipelines/src/ltx_pipelines/distilled.py)** - Fastest inference with 8 predefined sigmas
* **[ICLoraPipeline](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)** - Video-to-video and image-to-video transformations (uses distilled model.)
* **[KeyframeInterpolationPipeline](packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py)** - Interpolate between keyframe images
* **[A2VidPipelineTwoStage](packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py)** - Audio-to-video generation conditioned on an input audio file
* **[RetakePipeline](packages/ltx-pipelines/src/ltx_pipelines/retake.py)** - Regenerate a specific time region of an existing video

### ⚡ Optimization Tips

* **Use DistilledPipeline** - Fastest inference with only 8 predefined sigmas (8 steps stage 1, 4 steps stage 2)
* **Enable FP8 quantization** - Enables lower memory footprint: `--quantization fp8-cast` (CLI) or `quantization=QuantizationPolicy.fp8_cast()` (Python). For Hopper GPUs with TensorRT-LLM, use `--quantization fp8-scaled-mm` for FP8 scaled matrix multiplication.
* **Install attention optimizations** - Use xFormers (`uv sync --extra xformers`) or [Flash Attention 3](https://github.com/Dao-AILab/flash-attention) for Hopper GPUs
* **Use gradient estimation** - Reduce inference steps from 40 to 20-30 while maintaining quality (see [pipeline documentation](packages/ltx-pipelines/README.md#denoising-loop-optimization))
* **Skip memory cleanup** - If you have sufficient VRAM, disable automatic memory cleanup between stages for faster processing
* **Choose single-stage pipeline** - Use `TI2VidOneStagePipeline` for faster generation when high resolution isn't required

## ✍️ Prompting for LTX-2

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

- Start with main action in a single sentence
- Add specific details about movements and gestures
- Describe character/object appearances precisely
- Include background and environment details
- Specify camera angles and movements
- Describe lighting and colors
- Note any changes or sudden events

For additional guidance on writing a prompt please refer to <https://ltx.video/blog/how-to-prompt-for-ltx-2>

### Automatic Prompt Enhancement

LTX-2 pipelines support automatic prompt enhancement via an `enhance_prompt` parameter.

## 🔌 ComfyUI Integration

To use our model with ComfyUI, please follow the instructions at <https://github.com/Lightricks/ComfyUI-LTXVideo/>.

## 📦 Packages

This repository is organized as a monorepo with three main packages:

* **[ltx-core](packages/ltx-core/)** - Core model implementation, inference stack, and utilities
* **[ltx-pipelines](packages/ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and other generation modes
* **[ltx-trainer](packages/ltx-trainer/)** - Training and fine-tuning tools for LoRA, full fine-tuning, and IC-LoRA

Each package has its own README and documentation. See the [Documentation](#-documentation) section below.

## 📚 Documentation

Each package includes comprehensive documentation:

* **[LTX-Core README](packages/ltx-core/README.md)** - Core model implementation, inference stack, and utilities
* **[LTX-Pipelines README](packages/ltx-pipelines/README.md)** - High-level pipeline implementations and usage guides
* **[LTX-Trainer README](packages/ltx-trainer/README.md)** - Training and fine-tuning documentation with detailed guides
