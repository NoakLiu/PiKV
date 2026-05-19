# Optional Block Design: Xilinx XDMA + AXI interconnect + pikv_soc_top
# Requires Vivado with XDMA IP licensed.
# Usage: vivado -mode batch -source vivado/scripts/create_bd.tcl
#
# After BD build, use BD wrapper as top OR keep pikv_soc_top for pure RTL sim.

set script_dir [file normalize [file dirname [info script]]]
source [file join $script_dir create_project.tcl]

set design_name pikv_bd
if {[get_bd_designs -quiet $design_name] != ""} {
  open_bd_design $design_name
} else {
  create_bd_design $design_name
}

# Clock / reset (300 MHz reference — adjust per board)
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
set_property -dict [list \
  CONFIG.PRIM_IN_FREQ {100.000} \
  CONFIG.CLKOUT1_USED {true} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {300.000} \
  CONFIG.RESET_TYPE {ACTIVE_LOW} \
] [get_bd_cells clk_wiz_0]

# XDMA (PCIe host interface) — 64-bit AXI-MM + AXI-Lite
if {[catch {create_bd_cell -type ip -vlnv xilinx.com:ip:xdma:4.1 xdma_0} err]} {
  puts "WARN: XDMA IP not available: $err"
  puts "Install Versal/UltraScale+ XDMA or use RTL-only pikv_soc_top with external PCIe bridge."
  save_bd_design
  exit 0
}

set_property -dict [list \
  CONFIG.pl_link_cap_max_link_width {X16} \
  CONFIG.axi_data_width {128_bit} \
  CONFIG.axisten_freq {300} \
  CONFIG.pf0_device_id {9028} \
  CONFIG.pf0_class_code {058000} \
  CONFIG.pf0_class_code_base {05} \
  CONFIG.pf0_class_code_interface {80} \
  CONFIG.pf0_sub_class_interface_menu {058000} \
] [get_bd_cells xdma_0]

# AXI interconnect: XDMA M_AXI -> pikv_soc_top (CXL path would be second master port)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ic_0
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {2}] [get_bd_cells axi_ic_0]

# RTL module (black box — reference pikv_soc_top.v in project)
create_bd_cell -type module -reference pikv_soc_top pikv_soc_0

# Connect clocks
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins pikv_soc_0/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins pikv_soc_0/aresetn]

# XDMA AXI-Lite -> pikv_soc AXI-Lite (simplified — map BAR2 to slave)
# Note: Full pin-level connection depends on XDMA IP version; customize per board guide.
# See vivado/ip/README.md for Alveo U55C pinout.

regenerate_bd_layout
save_bd_design

make_wrapper -files [get_files ${design_name}.bd] -top
add_files -norecurse [file join $proj_dir pikv_fpga.gen sources_1 bd $design_name hdl ${design_name}_wrapper.v]
set_property top ${design_name}_wrapper [current_project]

puts "Block design $design_name created. Review and connect XDMA AXI ports to pikv_soc_0 in GUI."
