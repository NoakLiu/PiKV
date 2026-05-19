# PiKV-FPGA Hardware Stack

Hardware-aware KV cache offload (paper §3.5): **Verilog RTL** + **C host driver** + **Python bindings**.

## Layout

| Path | Description |
|------|-------------|
| `rtl/pikv_defines.vh` | Parameters (E=64, S=256, k=4, d=128) and MMIO opcodes |
| `rtl/pikv_top.v` | Top-level: MMIO, CTRL, engines, DMA stub |
| `rtl/pikv_ctrl.v` | FSM orchestrating per-token pipeline |
| `rtl/pikv_page_table.v` | Module D+: Γ lookup/insert |
| `rtl/pikv_score_fuse.v` | Top-k expert routing |
| `rtl/pikv_codec_rho.v` | LoRA-style compress |
| `rtl/pikv_scheduler.v` | Score vs θ retention |
| `rtl/pikv_mmio.v` | Host register map |
| `host/pikv_fpga.c` | Device mmap or hw-model backend |
| `host/pikv_hw_model.c` | C cycle model (no FPGA required) |
| `pikv_fpga_native.py` | ctypes → `libpikv_fpga.so` |

## Build

```bash
# From repo root
./build_fpga.sh all    # C lib + test
./build_fpga.sh host   # libpikv_fpga.so only
./build_fpga.sh sim    # iverilog (optional)
```

Outputs:
- `libpikv_fpga.so` — shared library for Python/C
- `test_pikv_fpga` — host unit test

## MMIO map

| Offset | Name | R/W |
|--------|------|-----|
| 0x000 | STATUS | R |
| 0x004 | CTRL (start/reset) | W |
| 0x008 | CMD | W |
| 0x00C | TOKEN_ID | W |
| 0x010 | EXPERT_ID | W |
| 0x018 | ARG0 | W |
| 0x01C | THETA | R/W |
| 0x020 | HIT_CNT | R |
| 0x024 | MISS_CNT | R |
| 0x028 | EXPERTS[0:3] | R |

## Synthesis (Vivado / Quartus)

1. Add all `rtl/*.v` and `rtl/pikv_defines.vh` to project.
2. Set top module `pikv_top`.
3. Constrain clock (e.g. 300 MHz) and map MMIO to AXI-Lite.
4. Connect `dma_*` to CXL.mem controller IP.

## Device driver

For Linux, implement a character device `/dev/pikv_fpga0` that mmap's BAR0; the C driver in `pikv_fpga.c` already supports this path when `use_hw_model=0`.
