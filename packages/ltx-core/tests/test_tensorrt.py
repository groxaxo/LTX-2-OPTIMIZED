from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import pytest
import torch

import ltx_core.tensorrt as trt


def _config(**overrides: Any) -> trt.TensorRTConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "components": frozenset({"upsampler", "vae"}),
        "cache_root": Path("/tmp/ltx-test-trt-cache"),
        "cache_size_bytes": 1024**3,
        "workspace_size_bytes": 256 * 1024**2,
        "min_block_size": 5,
        "optimization_level": 3,
        "max_aux_streams": 2,
        "dynamic_shapes": False,
        "strict": False,
        "engine_cache": True,
        "fast_partitioner": True,
        "experimental_decompositions": True,
        "debug": False,
        "allow_single_gpu_components": False,
    }
    values.update(overrides)
    return trt.TensorRTConfig(**values)


def _clear_trt_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LTX_TENSORRT",
        "LTX_TENSORRT_COMPONENTS",
        "LTX_TENSORRT_CACHE_DIR",
        "LTX_TENSORRT_CACHE_GB",
        "LTX_TENSORRT_WORKSPACE_GB",
        "LTX_TENSORRT_MIN_BLOCK_SIZE",
        "LTX_TENSORRT_OPT_LEVEL",
        "LTX_TENSORRT_MAX_AUX_STREAMS",
        "LTX_TENSORRT_DYNAMIC",
        "LTX_TENSORRT_STRICT",
        "LTX_TENSORRT_ENGINE_CACHE",
        "LTX_TENSORRT_FAST_PARTITIONER",
        "LTX_TENSORRT_EXPERIMENTAL_DECOMPOSITIONS",
        "LTX_TENSORRT_DEBUG",
        "LTX_TENSORRT_ALLOW_SINGLE_GPU_COMPONENTS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_defaults_are_disabled_and_memory_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_trt_environment(monkeypatch)

    config = trt.TensorRTConfig.from_env()

    assert config.enabled is False
    assert config.components == frozenset({"upsampler", "vae"})
    assert config.dynamic_shapes is False
    assert config.min_block_size == 5
    assert config.experimental_decompositions is False
    assert config.allow_single_gpu_components is False


def test_config_parses_aliases_and_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_trt_environment(monkeypatch)
    monkeypatch.setenv("LTX_TENSORRT", "yes")
    monkeypatch.setenv("LTX_TENSORRT_COMPONENTS", "video-vae,spatial-upsampler")
    monkeypatch.setenv("LTX_TENSORRT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("LTX_TENSORRT_CACHE_GB", "12")
    monkeypatch.setenv("LTX_TENSORRT_OPT_LEVEL", "5")
    monkeypatch.setenv("LTX_TENSORRT_ALLOW_SINGLE_GPU_COMPONENTS", "1")

    config = trt.TensorRTConfig.from_env()

    assert config.enabled is True
    assert config.components == frozenset({"vae", "upsampler"})
    assert config.cache_root == tmp_path
    assert config.cache_size_bytes == 12 * 1024**3
    assert config.optimization_level == 5
    assert config.allow_single_gpu_components is True


def test_config_rejects_unknown_component_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_trt_environment(monkeypatch)
    monkeypatch.setenv("LTX_TENSORRT", "1")
    monkeypatch.setenv("LTX_TENSORRT_COMPONENTS", "transformer")

    with pytest.raises(ValueError, match="Unsupported TensorRT components"):
        trt.TensorRTConfig.from_env()


def test_stale_component_setting_is_ignored_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_trt_environment(monkeypatch)
    monkeypatch.setenv("LTX_TENSORRT", "0")
    monkeypatch.setenv("LTX_TENSORRT_COMPONENTS", "removed-component")

    assert trt.TensorRTConfig.from_env().components == frozenset({"upsampler", "vae"})


def test_disabled_or_unhandled_component_is_identity() -> None:
    module = torch.nn.Identity()

    assert trt.optimize_component(module, "vae", config=_config(enabled=False), multi_gpu=True) is module
    assert trt.optimize_component(module, "transformer", config=_config(), multi_gpu=True) is module


def test_vae_is_gated_on_single_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module = torch.nn.Identity()
    config = _config()
    called = False

    def fake_compile(model: torch.nn.Module, component: str, _: trt.TensorRTConfig) -> torch.nn.Module:
        nonlocal called
        called = True
        assert component == "vae"
        return model

    monkeypatch.setattr(trt, "_compile_forward", fake_compile)

    assert trt.optimize_component(module, "vae", config=config, multi_gpu=False) is module
    assert called is False

    assert trt.optimize_component(module, "vae", config=config, multi_gpu=True) is module
    assert called is True


def test_compile_options_use_explicit_model_typing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trt, "_package_version", lambda _: "2.8.0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 6))
    config = _config(cache_root=tmp_path)
    device = torch.device("cuda:1")

    options = trt._compile_options(config, device)

    assert options["enabled_precisions"] == {torch.float32}
    assert options["use_explicit_typing"] is True
    assert options["device"] == device
    assert options["pass_through_build_failures"] is True
    assert options["immutable_weights"] is False
    assert options["engine_cache_size"] == config.cache_size_bytes
    assert "sm-86" in options["engine_cache_dir"]


def test_compiled_forward_falls_back_and_disables_only_that_component() -> None:
    module = torch.nn.Identity()
    eager = module.forward

    def broken(_: torch.Tensor) -> torch.Tensor:
        raise ValueError("converter failure")

    object.__setattr__(module, "_ltx_trt_eager_forward", eager)
    object.__setattr__(module, "_ltx_trt_compiled_forward", broken)
    object.__setattr__(module, "_ltx_trt_config", _config())
    object.__setattr__(module, "_ltx_trt_component", "upsampler")
    object.__setattr__(module, "_ltx_trt_device", torch.device("cuda:0"))
    module.forward = MethodType(trt._dispatch_compiled_forward, module)

    input_tensor = torch.randn(2, 3)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.cuda, "device", lambda _: _NullContext())
        output = module(input_tensor)

    torch.testing.assert_close(output, input_tensor)
    assert object.__getattribute__(module, "_ltx_trt_compiled_forward") is None


def test_compiled_forward_does_not_swallow_out_of_memory() -> None:
    module = torch.nn.Identity()
    eager = module.forward

    def out_of_memory(_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("CUDA out of memory")

    object.__setattr__(module, "_ltx_trt_eager_forward", eager)
    object.__setattr__(module, "_ltx_trt_compiled_forward", out_of_memory)
    object.__setattr__(module, "_ltx_trt_config", _config())
    object.__setattr__(module, "_ltx_trt_component", "vae")
    object.__setattr__(module, "_ltx_trt_device", torch.device("cuda:0"))
    module.forward = MethodType(trt._dispatch_compiled_forward, module)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.cuda, "device", lambda _: _NullContext())
        with pytest.raises(RuntimeError, match="out of memory"):
            module(torch.randn(1))


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None
