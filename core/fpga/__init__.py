"""
PiKV-FPGA: Hardware-aware FPGA offload for metadata-intensive KV cache stages.

Maps PiKV routing, compression, and scheduling to reconfigurable engines on a
CXL-attached SmartNIC (see paper §3.5). Falls back to software simulation when
no bitstream/driver is present.
"""

from .config import FPGAConfig, FPGAEngineMapping, estimate_bram_budget
from .pikv_fpga import (
    PiKVFPGA,
    PiKVFPGACTRL,
    create_pikv_fpga,
    is_fpga_available,
)

try:
    from .pikv_fpga_native import PiKVFPGANative, is_native_available
except ImportError:
    PiKVFPGANative = None  # type: ignore
    is_native_available = lambda: False  # type: ignore

__all__ = [
    "FPGAConfig",
    "FPGAEngineMapping",
    "estimate_bram_budget",
    "PiKVFPGA",
    "PiKVFPGACTRL",
    "create_pikv_fpga",
    "is_fpga_available",
    "PiKVFPGANative",
    "is_native_available",
]
