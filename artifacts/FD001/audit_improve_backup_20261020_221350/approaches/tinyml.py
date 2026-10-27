"""CPU latency/size measurements and optional dynamic int8 quantization."""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import time
from typing import Any

import numpy as np

from approaches.neural_models import require_torch


def _state_dict_size_bytes(network: Any) -> int:
    torch, _ = require_torch()
    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)
    return buffer.tell()


def _latency_ms(network: Any, input_tensor: Any, repeats: int) -> float:
    torch, _ = require_torch()
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    network.eval()
    with torch.inference_mode():
        for _ in range(min(10, repeats)):
            network(input_tensor)
        started = time.perf_counter()
        for _ in range(repeats):
            network(input_tensor)
        elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / repeats


def measure_tinyml(
    network: Any,
    sample_input: np.ndarray,
    *,
    repeats: int = 100,
    quantize_int8: bool = True,
) -> dict[str, float | int | str | None]:
    """Measure a CPU PyTorch network and optionally compare dynamic int8 weights.

    This reports local model-state size and single-batch latency. It is not a
    substitute for target-device RAM, power, or real-time measurements.
    """
    torch, nn = require_torch()
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    values = np.asarray(sample_input, dtype=np.float32)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("sample_input must be non-empty and finite.")
    if not any(isinstance(module, nn.Linear) for module in network.modules()):
        raise ValueError("Dynamic int8 measurement requires at least one Linear layer.")

    network = network.cpu().eval()
    tensor = torch.as_tensor(values, dtype=torch.float32)
    report: dict[str, float | int | str | None] = {
        "backend": "PyTorch CPU",
        "float_state_dict_bytes": _state_dict_size_bytes(network),
        "float_latency_ms_per_batch": _latency_ms(network, tensor, repeats),
        "int8_state_dict_bytes": None,
        "int8_latency_ms_per_batch": None,
        "size_reduction_percent": None,
        "latency_change_percent": None,
    }
    if not quantize_int8:
        return report

    supported_engines = torch.backends.quantized.supported_engines
    if not any(engine in supported_engines for engine in ("x86", "fbgemm", "qnnpack", "onednn")):
        raise RuntimeError(
            f"No supported dynamic quantization engine is available: {supported_engines}"
        )
    quantized = torch.ao.quantization.quantize_dynamic(
        deepcopy(network),
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )
    int8_size = _state_dict_size_bytes(quantized)
    int8_latency = _latency_ms(quantized, tensor, repeats)
    float_size = int(report["float_state_dict_bytes"])
    float_latency = float(report["float_latency_ms_per_batch"])
    report.update(
        {
            "int8_state_dict_bytes": int8_size,
            "int8_latency_ms_per_batch": int8_latency,
            "size_reduction_percent": (
                100.0 * (float_size - int8_size) / float_size if float_size else 0.0
            ),
            "latency_change_percent": (
                100.0 * (int8_latency - float_latency) / float_latency
                if float_latency
                else 0.0
            ),
        }
    )
    return report
