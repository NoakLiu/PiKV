"""
PiKV distributed package — MoE EP, DeepSpeed, and fused training systems.

Lightweight exports (LAER / MoEBlaze / FSMoE) load eagerly.
Heavy DistributedPiKV / DeepSpeed symbols use lazy import via ``__getattr__``.
"""

from __future__ import annotations

from typing import Any

from .distributed_config import distributed_config
from .fused_moe_training import (
    FSMoELayer,
    FusedMoETrainingConfig,
    FusedMoETrainingLayer,
    FusedMoETrainingSystem,
    LAERMoELayer,
    MoEBlazeLayer,
    create_fsmoe,
    create_fused_moe_training,
    create_laer_moe,
    create_moeblaze,
)

__version__ = "3.2.0"

__all__ = [
    "distributed_config",
    # Fused MoE training (LAER / MoEBlaze / FSMoE)
    "FusedMoETrainingConfig",
    "FusedMoETrainingLayer",
    "FusedMoETrainingSystem",
    "create_fused_moe_training",
    "create_laer_moe",
    "create_moeblaze",
    "create_fsmoe",
    "LAERMoELayer",
    "MoEBlazeLayer",
    "FSMoELayer",
    # Lazy: Distributed PiKV
    "distributed_context",
    "DistributedPerformanceMonitor",
    "DistributedExpert",
    "DistributedKVCache",
    "DistributedPiKVMoE",
    "DistributedPiKVManager",
    # Lazy: DeepSpeed
    "PiKVDeepSpeedConfig",
    "PiKVDeepSpeedModel",
    "PiKVDeepSpeedManager",
    "create_pikv_deepspeed",
]

_LAZY_DISTRIBUTED = {
    "distributed_context",
    "DistributedPerformanceMonitor",
    "DistributedExpert",
    "DistributedKVCache",
    "DistributedPiKVMoE",
    "DistributedPiKVManager",
}

_LAZY_DEEPSPEED = {
    "PiKVDeepSpeedConfig",
    "PiKVDeepSpeedModel",
    "PiKVDeepSpeedManager",
    "create_pikv_deepspeed",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_DISTRIBUTED:
        from . import distributed_pikv as _dp

        return getattr(_dp, name)
    if name in _LAZY_DEEPSPEED:
        from . import deepspeed_integration as _ds

        return getattr(_ds, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(list(__all__) + ["__version__"])
