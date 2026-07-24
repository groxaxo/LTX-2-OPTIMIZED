#!/usr/bin/env python3
"""Launch an LTX UI or pipeline with the repository's Torch-TensorRT backend enabled."""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys
from pathlib import Path

UI_TARGETS = {
    "web-v4": "web_ui_v4.py",
    "web-v2": "web_ui_v2.py",
    "film-v4": "film_maker_ui_v4.py",
    "music": "music_maker_ui.py",
    "music-v2": "music_maker_ui_v2.py",
}
PIPELINE_TARGETS = {
    "distilled": "ltx_pipelines.distilled",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enable persistent Torch-TensorRT engines for LTX's VAE and spatial upsampler.",
    )
    parser.add_argument("target", choices=(*UI_TARGETS, *PIPELINE_TARGETS))
    parser.add_argument("--components", default="upsampler,vae")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "ltx" / "tensorrt")
    parser.add_argument("--cache-size-gb", type=float, default=64.0)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--min-block-size", type=int, default=5)
    parser.add_argument("--optimization-level", type=int, choices=range(6), default=3)
    parser.add_argument("--max-aux-streams", type=int, default=2)
    parser.add_argument("--dynamic-shapes", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-engine-cache", action="store_true")
    parser.add_argument("--global-partitioner", action="store_true")
    parser.add_argument("--experimental-decompositions", action="store_true")
    parser.add_argument("--allow-single-gpu-components", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.cache_size_gb < 1:
        raise SystemExit("--cache-size-gb must be at least 1")
    if args.workspace_gb < 0.25:
        raise SystemExit("--workspace-gb must be at least 0.25")
    if args.min_block_size < 1:
        raise SystemExit("--min-block-size must be at least 1")
    if args.max_aux_streams < 0:
        raise SystemExit("--max-aux-streams cannot be negative")


def _configure_environment(args: argparse.Namespace) -> None:
    settings = {
        "LTX_TENSORRT": "1",
        "LTX_TENSORRT_COMPONENTS": args.components,
        "LTX_TENSORRT_CACHE_DIR": str(args.cache_dir.expanduser().resolve()),
        "LTX_TENSORRT_CACHE_GB": str(args.cache_size_gb),
        "LTX_TENSORRT_WORKSPACE_GB": str(args.workspace_gb),
        "LTX_TENSORRT_MIN_BLOCK_SIZE": str(args.min_block_size),
        "LTX_TENSORRT_OPT_LEVEL": str(args.optimization_level),
        "LTX_TENSORRT_MAX_AUX_STREAMS": str(args.max_aux_streams),
        "LTX_TENSORRT_DYNAMIC": "1" if args.dynamic_shapes else "0",
        "LTX_TENSORRT_STRICT": "1" if args.strict else "0",
        "LTX_TENSORRT_DEBUG": "1" if args.debug else "0",
        "LTX_TENSORRT_ENGINE_CACHE": "0" if args.no_engine_cache else "1",
        "LTX_TENSORRT_FAST_PARTITIONER": "0" if args.global_partitioner else "1",
        "LTX_TENSORRT_EXPERIMENTAL_DECOMPOSITIONS": "1" if args.experimental_decompositions else "0",
        "LTX_TENSORRT_ALLOW_SINGLE_GPU_COMPONENTS": "1" if args.allow_single_gpu_components else "0",
    }
    os.environ.update(settings)
    for name, value in settings.items():
        logging.info("%s=%s", name, value)


def _target_args(target_args: list[str]) -> list[str]:
    return target_args[1:] if target_args[:1] == ["--"] else target_args


def main() -> None:
    args, remaining = _parser().parse_known_args()
    _validate(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _configure_environment(args)

    target_args = _target_args(remaining)
    if args.target in UI_TARGETS:
        target = Path(__file__).resolve().parent / UI_TARGETS[args.target]
        if not target.is_file():
            raise SystemExit(f"UI entry point not found: {target}")
        sys.argv = [str(target), *target_args]
        runpy.run_path(str(target), run_name="__main__")
        return

    module = PIPELINE_TARGETS[args.target]
    sys.argv = [module, *target_args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
