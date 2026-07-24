#!/usr/bin/env python3
"""Benchmark an LTX command with TensorRT disabled and enabled.

The benchmark runs locally in subprocesses, keeps the model arguments identical,
rewrites --output-path per run, and writes a machine-readable JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunResult:
    mode: str
    phase: str
    index: int
    elapsed_seconds: float
    return_code: int
    command: list[str]
    output_path: str | None
    log_path: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the same LTX generation command with PyTorch and TensorRT backends."
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--components", default="upsampler,vae")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "ltx" / "tensorrt")
    parser.add_argument("--cache-size-gb", type=float, default=64.0)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--result-json", type=Path, default=Path("benchmark-tensorrt.json"))
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _strip_separator(command: list[str]) -> list[str]:
    return command[1:] if command and command[0] == "--" else command


def _validate_args(args: argparse.Namespace, command: list[str]) -> None:
    if args.warmups < 0:
        raise SystemExit("--warmups must be zero or greater")
    if args.runs < 1:
        raise SystemExit("--runs must be one or greater")
    if args.cache_size_gb <= 0 or args.workspace_gb <= 0:
        raise SystemExit("cache and workspace sizes must be greater than zero")
    if not command:
        raise SystemExit("provide the LTX command after '--'")


def _rewrite_output_path(command: list[str], output_path: Path) -> tuple[list[str], bool]:
    rewritten = list(command)
    for index, token in enumerate(rewritten):
        if token == "--output-path":
            if index + 1 >= len(rewritten):
                raise SystemExit("--output-path is missing its value")
            rewritten[index + 1] = str(output_path)
            return rewritten, True
        if token.startswith("--output-path="):
            rewritten[index] = f"--output-path={output_path}"
            return rewritten, True
    return rewritten, False


def _environment(args: argparse.Namespace, enabled: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LTX_TENSORRT": "1" if enabled else "0",
            "LTX_TENSORRT_COMPONENTS": args.components,
            "LTX_TENSORRT_CACHE_DIR": str(args.cache_dir.expanduser().resolve()),
            "LTX_TENSORRT_CACHE_GB": str(args.cache_size_gb),
            "LTX_TENSORRT_WORKSPACE_GB": str(args.workspace_gb),
            "LTX_TENSORRT_ENGINE_CACHE": "1",
        }
    )
    return env


def _run_once(
    *,
    mode: str,
    phase: str,
    index: int,
    command: list[str],
    env: dict[str, str],
    artifacts_dir: Path,
) -> RunResult:
    output_path = artifacts_dir / f"{mode}-{phase}-{index}.mp4"
    run_command, had_output_path = _rewrite_output_path(command, output_path)
    log_path = artifacts_dir / f"{mode}-{phase}-{index}.log"

    print(f"[{mode}] {phase} {index + 1}: {' '.join(run_command)}", flush=True)  # noqa: T201
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            run_command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.perf_counter() - started
    print(  # noqa: T201
        f"[{mode}] return={completed.returncode} elapsed={elapsed:.3f}s log={log_path}",
        flush=True,
    )

    return RunResult(
        mode=mode,
        phase=phase,
        index=index,
        elapsed_seconds=elapsed,
        return_code=completed.returncode,
        command=run_command,
        output_path=str(output_path) if had_output_path else None,
        log_path=str(log_path),
    )


def _summary(results: list[RunResult]) -> dict[str, Any]:
    successful = [result.elapsed_seconds for result in results if result.phase == "measure" and result.return_code == 0]
    if not successful:
        return {"successful_runs": 0}
    return {
        "successful_runs": len(successful),
        "mean_seconds": statistics.fmean(successful),
        "median_seconds": statistics.median(successful),
        "min_seconds": min(successful),
        "max_seconds": max(successful),
        "stdev_seconds": statistics.stdev(successful) if len(successful) > 1 else 0.0,
    }


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
    }
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_runtime"] = torch.version.cuda
            info["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch"] = None
    return info


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    command = _strip_separator(args.command)
    _validate_args(args, command)

    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.clear_cache and args.cache_dir.exists():
        print(f"Removing TensorRT cache: {args.cache_dir}")  # noqa: T201
        shutil.rmtree(args.cache_dir)

    args.result_json = args.result_json.expanduser().resolve()
    if args.artifacts_dir is None:
        artifacts_dir = args.result_json.parent / f"{args.result_json.stem}-artifacts"
    else:
        artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for enabled, mode in ((False, "pytorch"), (True, "tensorrt")):
        env = _environment(args, enabled)
        for index in range(args.warmups):
            results.append(
                _run_once(
                    mode=mode,
                    phase="warmup",
                    index=index,
                    command=command,
                    env=env,
                    artifacts_dir=artifacts_dir,
                )
            )
        for index in range(args.runs):
            results.append(
                _run_once(
                    mode=mode,
                    phase="measure",
                    index=index,
                    command=command,
                    env=env,
                    artifacts_dir=artifacts_dir,
                )
            )

    pytorch_results = [result for result in results if result.mode == "pytorch"]
    tensorrt_results = [result for result in results if result.mode == "tensorrt"]
    pytorch_summary = _summary(pytorch_results)
    tensorrt_summary = _summary(tensorrt_results)

    speedup = None
    if "median_seconds" in pytorch_summary and "median_seconds" in tensorrt_summary:
        speedup = pytorch_summary["median_seconds"] / tensorrt_summary["median_seconds"]

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "system": _system_info(),
        "settings": {
            "warmups": args.warmups,
            "runs": args.runs,
            "components": args.components,
            "cache_dir": str(args.cache_dir),
            "cache_size_gb": args.cache_size_gb,
            "workspace_gb": args.workspace_gb,
            "command": command,
        },
        "summary": {
            "pytorch": pytorch_summary,
            "tensorrt": tensorrt_summary,
            "median_speedup": speedup,
        },
        "runs": [asdict(result) for result in results],
    }

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))  # noqa: T201
    print(f"Wrote benchmark report: {args.result_json}")  # noqa: T201

    if not args.keep_videos:
        for result in results:
            if result.output_path:
                Path(result.output_path).unlink(missing_ok=True)

    failed_measurements = [result for result in results if result.phase == "measure" and result.return_code != 0]
    if failed_measurements:
        raise SystemExit(f"{len(failed_measurements)} measurement run(s) failed; inspect the JSON report and logs")


if __name__ == "__main__":
    main()
