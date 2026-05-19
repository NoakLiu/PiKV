# PiKV-FPGA Vivado project (RTL top: pikv_soc_top)
# Usage: vivado -mode batch -source vivado/scripts/create_project.tcl
# Env:   PIKV_PART (default xcu55c-fsvh2892-2L-e), PIKV_BOARD (u55c)

set script_dir [file normalize [file dirname [info script]]]
set fpga_root  [file normalize [file join $script_dir ../..]]
set rtl_dir    [file join $fpga_root rtl]
set constr_dir [file join $fpga_root vivado constraints]
set proj_dir   [file join $fpga_root vivado project]

set part_name $::env(PIKV_PART)
if {![info exists ::env(PIKV_PART)]} {
  set part_name "xcu55c-fsvh2892-2L-e"
}

file mkdir $proj_dir

set proj_name "pikv_fpga"
if {[file exists [file join $proj_dir ${proj_name}.xpr]]} {
  open_project [file join $proj_dir ${proj_name}.xpr]
} else {
  create_project $proj_name $proj_dir -part $part_name -force
}

set_property target_language Verilog [current_project]
set_property verilog_define {PIKV_FPGA} [current_project]

set rtl_files {
  pikv_defines.vh
  pikv_page_table.v
  pikv_score_fuse.v
  pikv_codec_rho.v
  pikv_scheduler.v
  pikv_mmio.v
  pikv_ctrl.v
  pikv_top.v
  pikv_axi_lite_slave.v
  pikv_axi_dma_master.v
  pikv_cxl_dma.v
  pikv_soc_top.v
}

foreach f $rtl_files {
  set path [file join $rtl_dir $f]
  if {[llength [get_files -quiet $path]] == 0} {
    if {[string match *.vh $f]} {
      add_files -fileset sources_1 $path
      set_property is_global_include true [get_files $path]
    } else {
      add_files -fileset sources_1 $path
    }
  }
}

set_property top pikv_soc_top [current_project]
set_property top_file [file join $rtl_dir pikv_soc_top.v] [current_project]

# Constraints
set xdc_u55c [file join $constr_dir pikv_u55c.xdc]
set xdc_gen  [file join $constr_dir pikv_generic.xdc]
if {[file exists $xdc_u55c]} {
  if {[llength [get_files -quiet $xdc_u55c]] == 0} {
    add_files -fileset constrs_1 $xdc_u55c
  }
}
if {[file exists $xdc_gen]} {
  if {[llength [get_files -quiet $xdc_gen]] == 0} {
    add_files -fileset constrs_1 $xdc_gen
  }
}

# Synthesis strategy
set_property strategy Vivado_Synthesis_Defaults [get_runs synth_1]
set_property strategy Vivado_Implementation_Defaults [get_runs impl_1]

update_compile_order -fileset sources_1
save_project_as $proj_name $proj_dir -force

puts "PiKV project ready: $proj_dir/${proj_name}.xpr (part=$part_name)"
