import argparse
import logging
from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines._compat import DEFAULT_DISTILLED_LORA_FILENAME, has_lora_filename_pattern
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines.utils.media_io import encode_video


def _distilled_lora_from_args(checkpoint_path: str, stage_2_checkpoint_path: str | None) -> list[LoraPathStrengthAndSDOps]:
    candidates: list[Path] = []
    if stage_2_checkpoint_path:
        stage_2_path = Path(stage_2_checkpoint_path).resolve()
        if stage_2_path.exists() and not has_lora_filename_pattern(stage_2_path.as_posix()):
            logging.warning(
                "--stage-2-checkpoint-path exists but is not a LoRA .safetensors file. Falling back to default LoRA path."
            )
        else:
            candidates.append(stage_2_path)
    ckpt_dir = Path(checkpoint_path).resolve().parent
    candidates.append(ckpt_dir / DEFAULT_DISTILLED_LORA_FILENAME)

    for candidate in candidates:
        if candidate.exists() and has_lora_filename_pattern(candidate.as_posix()):
            return [LoraPathStrengthAndSDOps(candidate.as_posix(), 1.0)]
    return []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backward-compatible music-to-video-v2 entrypoint")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--stage-2-checkpoint-path", default=None)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--spatial-upsampler-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--frame-rate", type=float, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-fp8", action="store_true")
    parser.add_argument("--audio-input-path", default=None)
    parser.add_argument("--image", nargs=3, action="append", default=[])
    return parser


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = _build_parser()
    args = parser.parse_args()

    images = [(path, int(frame_idx), float(strength)) for path, frame_idx, strength in args.image]

    quantization = QuantizationPolicy.fp8_cast() if args.enable_fp8 else None
    distilled_lora = _distilled_lora_from_args(args.checkpoint_path, args.stage_2_checkpoint_path)
    if not distilled_lora:
        logging.warning(
            "No valid distilled LoRA found from --stage-2-checkpoint-path/default path. "
            "Proceeding without distilled LoRA for compatibility mode."
        )

    pipeline = A2VidPipelineTwoStage(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=[],
        quantization=quantization,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt="",
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(),
        images=images,
        tiling_config=tiling_config,
        enhance_prompt=False,
        audio_path=args.audio_input_path,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
