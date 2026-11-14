"""Hardware device detection and resource auditing for CPU and CUDA GPU."""

from __future__ import annotations

import logging
import platform
from typing import Any

import psutil
import torch

from sentiment_transformer.config import AppConfig

logger = logging.getLogger(__name__)


def get_device(config: AppConfig | None = None) -> torch.device:
    """Determine and return the appropriate PyTorch device (CUDA GPU or CPU fallback).

    Args:
        config: Optional application configuration containing device preferences.

    Returns:
        torch.device instance ('cuda:0' or 'cpu').
    """
    use_cuda = True
    device_id = 0

    if config is not None:
        use_cuda = config.device.use_cuda_if_available
        device_id = config.device.device_id

    if use_cuda and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        valid_id = min(device_id, num_gpus - 1)
        device = torch.device(f"cuda:{valid_id}")
        gpu_name = torch.cuda.get_device_name(valid_id)
        logger.info("Selected hardware device: %s (%s)", device, gpu_name)
    else:
        device = torch.device("cpu")
        logger.info(
            "Selected hardware device: %s (CUDA available: %s, use_cuda_requested: %s)",
            device,
            torch.cuda.is_available(),
            use_cuda,
        )

    return device


def get_device_info() -> dict[str, Any]:
    """Inspect and collect detailed hardware, OS, and PyTorch environment information.

    Returns:
        Dictionary with hardware specifications.
    """
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if (cuda_available and torch.backends.cudnn.is_available())
            else None
        ),
    }

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpus: list[dict[str, Any]] = []
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_total_gb = round(props.total_memory / (1024**3), 2)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": mem_total_gb,
                    "major_capability": props.major,
                    "minor_capability": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        info["gpu_count"] = gpu_count
        info["gpus"] = gpus
    else:
        info["gpu_count"] = 0
        info["gpus"] = []

    return info
