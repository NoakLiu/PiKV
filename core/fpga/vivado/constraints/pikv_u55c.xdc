# AMD Alveo U55C — PiKV-FPGA placement hints
# Full pin LOC requires board system clock / PCIe refclk from XDMA BD.
# Use with: PIKV_PART=xcu55c-fsvh2892-2L-e

set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets aclk]

# 300 MHz user clock (when aclk driven from clk_wiz on 100 MHz SI570)
create_clock -name aclk -period 3.333 [get_ports aclk]

# CXL.mem / HBM region — timing budget for AXI-MM master (pikv_cxl_dma)
set_multicycle_path -setup 2 -from [get_cells -hierarchical -filter {NAME =~ *u_cxl_dma*}]
set_multicycle_path -hold 1 -from [get_cells -hierarchical -filter {NAME =~ *u_cxl_dma*}]

# BRAM inference for page table
set_property RAM_STYLE BLOCK [get_cells -hierarchical -filter {NAME =~ *u_pt*}]

# Bitstream options
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design]
