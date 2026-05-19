"""
ctypes bindings to libpikv_fpga.so (C host + hardware model).
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Optional, Tuple

import numpy as np

_LIB = None


def _lib_path() -> str:
    return os.path.join(os.path.dirname(__file__), "libpikv_fpga.so")


def load_native_library():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _lib_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"libpikv_fpga.so not found at {path}. Run: ./build_fpga.sh host"
        )
    _LIB = ctypes.CDLL(path)
    _LIB.pikv_fpga_open.argtypes = [ctypes.c_void_p]
    _LIB.pikv_fpga_open.restype = ctypes.c_void_p
    _LIB.pikv_fpga_close.argtypes = [ctypes.c_void_p]
    _LIB.pikv_fpga_process_token.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    _LIB.pikv_fpga_process_token.restype = ctypes.c_int
    _LIB.pikv_fpga_get_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return _LIB


class PiKVFPGANative:
    """Python wrapper around C PiKV-FPGA driver."""

    def __init__(
        self,
        num_experts: int = 64,
        hidden_dim: int = 128,
        top_k: int = 4,
        d_prime: int = 32,
        use_hw_model: bool = True,
    ):
        lib = load_native_library()

        class Cfg(ctypes.Structure):
            _fields_ = [
                ("num_experts", ctypes.c_uint32),
                ("top_k", ctypes.c_uint32),
                ("hidden_dim", ctypes.c_uint32),
                ("d_prime", ctypes.c_uint32),
                ("use_hw_model", ctypes.c_int),
                ("device_path", ctypes.c_char_p),
            ]

        cfg = Cfg(num_experts, top_k, hidden_dim, d_prime, int(use_hw_model), None)
        self._h = lib.pikv_fpga_open(ctypes.byref(cfg))
        if not self._h:
            raise RuntimeError("pikv_fpga_open failed")
        self._lib = lib
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.d_prime = d_prime

    def process_token(
        self,
        token_id: int,
        k: np.ndarray,
        v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        k16 = np.ascontiguousarray(k, dtype=np.int16).flatten()[: self.hidden_dim]
        v16 = np.ascontiguousarray(v, dtype=np.int16).flatten()[: self.hidden_dim]
        k_out = np.zeros(self.top_k * self.d_prime, dtype=np.int16)
        v_out = np.zeros(self.top_k * self.d_prime, dtype=np.int16)
        experts = (ctypes.c_uint32 * 4)()

        rc = self._lib.pikv_fpga_process_token(
            self._h,
            ctypes.c_uint32(token_id),
            k16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            v16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            k_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            v_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            experts,
            ctypes.c_uint32(4),
        )
        if rc != 0:
            raise RuntimeError("pikv_fpga_process_token failed")
        return k_out, v_out, [experts[i] for i in range(4)]

    def close(self):
        if self._h:
            self._lib.pikv_fpga_close(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def is_native_available() -> bool:
    return os.path.exists(_lib_path())
