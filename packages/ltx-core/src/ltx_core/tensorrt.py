from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MethodType
from typing import Any

import torch

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_DEFAULT_COMPONENTS = frozenset({"upsampler", "vae"})
_GIB = 1024**3


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {value!r}")


def _env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    value = int(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    value = float(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_components(value: str | None) -> frozenset[str]:
    if value is None:
        return _DEFAULT_COMPONENTS
    components = frozenset(item.strip().lower().replace("-", "_") for item in value.split(",") if item.strip())
    if not components:
        raise ValueError("LTX_TENSORRT_COMPONENTS cannot be empty when TensorRT is enabled")
    aliases = {
        "video_vae": "vae",
        "vae_encoder": "vae",
        "vae_decoder": "vae",
        "spatial_upsampler": "upsampler",
    }
    normalized = frozenset(aliases.get(component, component) for component in components)
    unsupported = normalized - {"all", "upsampler", "vae"}
    if unsupported:
        raise ValueError(f"Unsupported TensorRT components: {sorted(unsupported)}")
    return normalized


def _default_cache_root() -> Path:
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return root / "ltx" / "tensorrt"


@dataclass(frozen=True)
class TensorRTConfig:
    """Environment-driven configuration for the optional Torch-TensorRT backend."""

    enabled: bool
    components: frozenset[str]
    cache_root: Path
    cache_size_bytes: int
    workspace_size_bytes: int
    min_block_size: int
    optimization_level: int
    max_aux_streams: int
    dynamic_shapes: bool
    strict: bool
    engine_cache: bool
    fast_partitioner: bool
    experimental_decompositions: bool
    debug: bool
    allow_single_gpu_components: bool

    @classmethod
    def from_env(cls) -> TensorRTConfig:
        enabled = _env_bool("LTX_TENSORRT", False)
        components = _parse_components(os.getenv("LTX_TENSORRT_COMPONENTS")) if enabled else _DEFAULT_COMPONENTS
        return cls(
            enabled=enabled,
            components=components,
            cache_root=Path(os.getenv("LTX_TENSORRT_CACHE_DIR", str(_default_cache_root()))).expanduser(),
            cache_size_bytes=int(_env_float("LTX_TENSORRT_CACHE_GB", 64.0, minimum=1.0) * _GIB),
            workspace_size_bytes=int(_env_float("LTX_TENSORRT_WORKSPACE_GB", 4.0, minimum=0.25) * _GIB),
            min_block_size=_env_int("LTX_TENSORRT_MIN_BLOCK_SIZE", 3, minimum=1),
            optimization_level=_env_int("LTX_TENSORRT_OPT_LEVEL", 3, minimum=0, maximum=5),
            max_aux_streams=_env_int("LTX_TENSORRT_MAX_AUX_STREAMS", 2, minimum=0),
            dynamic_shapes=_env_bool("LTX_TENSORRT_DYNAMIC", False),
            strict=_env_bool("LTX_TENSORRT_STRICT", False),
            engine_cache=_env_bool("LTX_TENSORRT_ENGINE_CACHE", True),
            fast_partitioner=_env_bool("LTX_TENSORRT_FAST_PARTITIONER", True),
            experimental_decompositions=_env_bool("LTX_TENSORRT_EXPERIMENTAL_DECOMPOSITIONS", True),
            debug=_env_bool("LTX_TENSORRT_DEBUG", False),
            allow_single_gpu_components=_env_bool("LTX_TENSORRT_ALLOW_SINGLE_GPU_COMPONENTS", False),
        )

    def includes(self, component: str) -> bool:
        normalized = component.strip().lower().replace("-", "_")
        return "all" in self.components or normalized in self.components


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _module_device(module: torch.nn.Module) -> torch.device | None:
    parameter = next(module.parameters(recurse=True), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(recurse=True), None)
    return buffer.device if buffer is not None else None


def _module_dtype(module: torch.nn.Module) -> torch.dtype | None:
    parameter = next(module.parameters(recurse=True), None)
    if parameter is not None:
        return parameter.dtype
    buffer = next(module.buffers(recurse=True), None)
    return buffer.dtype if buffer is not None else None


def _fail_or_fallback(config: TensorRTConfig, message: str, exception: Exception | None = None) -> None:
    if config.strict and exception is None:
        raise RuntimeError(message)
    if config.strict:
        raise RuntimeError(message) from exception
    if exception is None:
        logger.warning(message)
    else:
        logger.warning("%s: %s", message, exception)


def _load_backend(config: TensorRTConfig) -> bool:
    if not config.enabled:
        return False
    if not torch.cuda.is_available():
        _fail_or_fallback(config, "LTX TensorRT requested, but CUDA is not available; using PyTorch")
        return False

    try:
        importlib.import_module("torch_tensorrt")
        importlib.import_module("tensorrt")
    except (ImportError, OSError) as exc:
        _fail_or_fallback(
            config,
            "LTX TensorRT requested, but Torch-TensorRT/TensorRT could not be imported; "
            "install requirements-tensorrt.txt",
            exc,
        )
        return False

    torch_version = torch.__version__.split("+")[0]
    torch_trt_version = _package_version("torch-tensorrt")
    if torch_trt_version != "unknown" and torch_version.split(".")[:2] != torch_trt_version.split(".")[:2]:
        _fail_or_fallback(
            config,
            f"PyTorch {torch_version} and Torch-TensorRT {torch_trt_version} do not share a major/minor version; "
            "using PyTorch fallback",
        )
        return False

    return True


def _cache_namespace(config: TensorRTConfig, device: torch.device) -> Path:
    torch_version = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    torch_trt_version = ".".join(_package_version("torch-tensorrt").split(".")[:2])
    tensorrt_version = ".".join(_package_version("tensorrt").split(".")[:2])
    cuda_version = torch.version.cuda or "unknown"
    capability = torch.cuda.get_device_capability(device)
    runtime = (
        f"torch-{torch_version}_torchtrt-{torch_trt_version}_trt-{tensorrt_version}_"
        f"cuda-{cuda_version}_sm-{capability[0]}{capability[1]}"
    )
    namespace = config.cache_root / runtime
    namespace.mkdir(parents=True, exist_ok=True)
    return namespace


def _compile_options(config: TensorRTConfig, device: torch.device) -> dict[str, Any]:
    cache_dir = _cache_namespace(config, device)
    return {
        # Torch-TensorRT 2.8 explicit typing requires the sentinel FP32 set and then
        # respects the dtypes already present in the LTX module and its inputs.
        "enabled_precisions": {torch.float32},
        "use_explicit_typing": True,
        "device": device,
        "debug": config.debug,
        "workspace_size": config.workspace_size_bytes,
        "min_block_size": config.min_block_size,
        "pass_through_build_failures": config.strict,
        "max_aux_streams": config.max_aux_streams,
        "optimization_level": config.optimization_level,
        "use_fast_partitioner": config.fast_partitioner,
        "enable_experimental_decompositions": config.experimental_decompositions,
        "assume_dynamic_shape_support": config.dynamic_shapes,
        "truncate_double": True,
        "hardware_compatible": False,
        "timing_cache_path": str(cache_dir / "timing-cache.bin"),
        "cache_built_engines": config.engine_cache,
        "reuse_cached_engines": config.engine_cache,
        "engine_cache_dir": str(cache_dir / "engines"),
        "engine_cache_size": config.cache_size_bytes,
        # Refittable engines are required for weight-agnostic disk-cache reuse.
        "immutable_weights": False,
    }


def _is_oom(exception: Exception) -> bool:
    out_of_memory = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exception, out_of_memory) and "out of memory" in str(exception).lower()


def _dispatch_compiled_forward(module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
    compiled: Callable[..., Any] | None = object.__getattribute__(module, "_ltx_trt_compiled_forward")
    eager: Callable[..., Any] = object.__getattribute__(module, "_ltx_trt_eager_forward")
    device: torch.device = object.__getattribute__(module, "_ltx_trt_device")
    if compiled is None:
        return eager(*args, **kwargs)

    try:
        with torch.cuda.device(device):
            return compiled(*args, **kwargs)
    except Exception as exc:
        config: TensorRTConfig = object.__getattribute__(module, "_ltx_trt_config")
        component: str = object.__getattribute__(module, "_ltx_trt_component")
        if config.strict or _is_oom(exc):
            raise
        object.__setattr__(module, "_ltx_trt_compiled_forward", None)
        logger.warning(
            "TensorRT disabled for %s after a compile/runtime failure; continuing with PyTorch: %s",
            component,
            exc,
        )
        return eager(*args, **kwargs)


def _compile_forward(module: torch.nn.Module, component: str, config: TensorRTConfig) -> torch.nn.Module:
    if not config.enabled or not config.includes(component):
        return module
    if getattr(module, "_ltx_trt_initialized", False):
        return module
    if not _load_backend(config):
        return module

    device = _module_device(module)
    if device is None or device.type != "cuda":
        _fail_or_fallback(config, f"TensorRT component {component!r} is not resident on CUDA; using PyTorch")
        return module

    eager_forward = module.forward
    try:
        compiled_forward = torch.compile(
            eager_forward,
            backend="torch_tensorrt",
            dynamic=config.dynamic_shapes,
            fullgraph=False,
            options=_compile_options(config, device),
        )
    except Exception as exc:
        _fail_or_fallback(config, f"Could not initialize TensorRT compilation for {component}", exc)
        return module

    object.__setattr__(module, "_ltx_trt_initialized", True)
    object.__setattr__(module, "_ltx_trt_eager_forward", eager_forward)
    object.__setattr__(module, "_ltx_trt_compiled_forward", compiled_forward)
    object.__setattr__(module, "_ltx_trt_config", config)
    object.__setattr__(module, "_ltx_trt_component", component)
    object.__setattr__(module, "_ltx_trt_device", device)
    module.forward = MethodType(_dispatch_compiled_forward, module)
    logger.info(
        "Enabled lazy TensorRT compilation for %s (dtype=%s, device=%s, static_shapes=%s, cache=%s)",
        component,
        _module_dtype(module),
        device,
        not config.dynamic_shapes,
        config.cache_root,
    )
    return module


def optimize_component(
    module: torch.nn.Module,
    component: str,
    config: TensorRTConfig | None = None,
    *,
    multi_gpu: bool = False,
) -> torch.nn.Module:
    """Apply TensorRT to a VAE or upsampler while retaining eager fallback."""

    config = config or TensorRTConfig.from_env()
    normalized = component.strip().lower().replace("-", "_")
    if not config.enabled or normalized not in {"upsampler", "vae"} or not config.includes(normalized):
        return module
    if not multi_gpu and not config.allow_single_gpu_components:
        logger.info(
            "Skipping TensorRT %s on the shared single-GPU layout; set "
            "LTX_TENSORRT_ALLOW_SINGLE_GPU_COMPONENTS=1 to override",
            normalized,
        )
        return module
    return _compile_forward(module, normalized, config)


def enabled_components(config: TensorRTConfig | None = None) -> Iterable[str]:
    """Return configured component names for diagnostics and tests."""

    return tuple(sorted((config or TensorRTConfig.from_env()).components))


__all__ = [
    "TensorRTConfig",
    "enabled_components",
    "optimize_component",
]
