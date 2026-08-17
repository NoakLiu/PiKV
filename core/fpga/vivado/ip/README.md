# Xilinx/AMD IP Integration for PiKV-FPGA

## Topology

```
Host (GPU/CPU) --PCIe/CXL--> [XDMA] --AXI-Lite--> pikv_soc_top (PiKV-CTRL / MMIO)
                              |
                              +--AXI-MM----> [CXL Type-3 expander IP or HBM]
                                    ^
                                    pikv_cxl_dma (m_axi_*)
```

## Required IP (Vivado Catalog)

| IP | VLNV | Purpose |
|----|------|---------|
| **xdma** | `xilinx.com:ip:xdma` | PCIe host ↔ AXI (BAR0 MMIO, BAR1 DMA) |
| **clk_wiz** | `xilinx.com:ip:clk_wiz` | 100 MHz → 300 MHz `aclk` |
| **axi_interconnect** | `xilinx.com:ip:axi_interconnect` | Fan-out if multiple masters |
| **CXL Subsystem** | Board-specific | CXL.mem attachment (U55C CXL-enabled builds) |

## CXL Type-3 / DDR pool

PiKV routes KV **bodies** to disaggregated memory via `pikv_cxl_dma` AXI4 master:

- Default base: `0x0001_0000_0000` (programmable via MMIO `0x040`/`0x044`)
- Beat width: **128-bit** (K̂‖V̂ 32+32 per beat, paper §3.5)

**Option A — AMD CXL IP (datacenter card)**  
Connect `m_axi_*` from `pikv_soc_top` to the CXL memory expander IP per your shell (see *CXL-SpecKV*, FPGA'26).

**Option B — On-card HBM/DDR**  
Map `m_axi_*` to `axi_noc` / HBM controller on U55C for prototyping without CXL hardware.

**Option C — Simulation**  
Use `pikv_cxl_mem_model.v` as AXI slave (see `tb/tb_pikv_soc.v`).

## Build BD + bitstream

```bash
export PIKV_PART=xcu55c-fsvh2892-2L-e
./scripts/build_fpga.sh vivado      # create project
./scripts/build_fpga.sh bitstream   # synth + impl + .bit
./scripts/build_fpga.sh bd          # optional XDMA block design
```

## Program bitstream (U55C)

```bash
xbutil examine
xbutil program --device 0000:xx:00.0 --base pikv_fpga.bit
```

## Linux driver

Map XDMA user BAR to `/dev/pikv_fpga0` or use Xilinx `xdma` driver + `pikv_fpga.c` mmap at BAR offset 0.
