"""Unit tests for device selection and hardware inspection."""

from __future__ import annotations

from sentiment_transformer.config import AppConfig, DeviceConfig
from sentiment_transformer.modeling.device import get_device, get_device_info


def test_get_device_cpu_fallback() -> None:
    cfg = AppConfig(device=DeviceConfig(use_cuda_if_available=False))
    device = get_device(cfg)
    assert device.type == "cpu"


def test_get_device_info() -> None:
    info = get_device_info()
    assert "platform" in info
    assert "python_version" in info
    assert "pytorch_version" in info
    assert "cuda_available" in info
    assert isinstance(info["cuda_available"], bool)
