# PiKV-FPGA Hardware Stack

**Verilog RTL** + **AXI-Lite / CXL DMA** + **C host** + **Vivado bitstream** (paper §3.5).

## Architecture

```
Host PCIe ──► [XDMA] ──AXI-Lite──► pikv_soc_top ──► PiKV-CTRL / engines
                      └──AXI-MM──► pikv_cxl_dma ──► CXL Type-3 DDR (KV pool)
```

| Module | Interface | Role |
|--------|-----------|------|
| `pikv_axi_lite_slave` | AXI4-Lite | Host MMIO (BAR0) |
| `pikv_soc_top` | AXI-L + AXI-MM | SoC integration top |
| `pikv_cxl_dma` | AXI4 master | KV body DMA to CXL.mem |
| `pikv_axi_dma_master` | AXI4 | Single-beat read/write |
| `pikv_cxl_mem_model` | AXI4 slave | Simulation memory |

## Directory

```
rtl/           Core + AXI + CXL DMA
host/          C driver (libpikv_fpga.so)
tb/            Icarus testbenches
vivado/
  scripts/     create_project.tcl, build_bitstream.tcl, create_bd.tcl
  constraints/ pikv_u55c.xdc, pikv_generic.xdc
  ip/          XDMA / CXL integration guide
```

## Build

```bash
./build_fpga.sh all          # C lib + RTL sim + test
./build_fpga.sh sim-soc      # AXI-Lite + CXL mem model sim
./build_fpga.sh vivado       # Create Vivado project (pikv_soc_top)
./build_fpga.sh bitstream    # Synth + impl + .bit (needs Vivado + license)
./build_fpga.sh bd           # Optional XDMA block design
```

Env:
- `PIKV_PART=xcu55c-fsvh2892-2L-e` (Alveo U55C)
- `PIKV_JOBS=8`

Bitstream output: `vivado/project/pikv_fpga.runs/impl_1/*.bit`

## MMIO map

| Offset | Register |
|--------|----------|
| 0x000 | STATUS |
| 0x004 | CTRL |
| 0x008 | CMD |
| 0x00C | TOKEN_ID |
| 0x040 | CXL_BASE[31:0] |
| 0x044 | CXL_BASE[63:32] |
| 0x048 | DMA_XFER_CNT |
| 0x04C | DMA_BUSY |

## Vivado + XDMA + CXL

See [vivado/ip/README.md](vivado/ip/README.md) for:
- Xilinx **xdma** IP (PCIe)
- **CXL Type-3** memory expander hookup
- `xbutil program` for U55C

## Simulation without Vivado

```bash
brew install icarus-verilog   # macOS
./build_fpga.sh sim-soc
```
