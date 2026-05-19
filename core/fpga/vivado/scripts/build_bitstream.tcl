# Synthesize, implement, and generate bitstream for PiKV-FPGA
# Usage: vivado -mode batch -source vivado/scripts/build_bitstream.tcl

set script_dir [file normalize [file dirname [info script]]]
source [file join $script_dir create_project.tcl]

set jobs 8
if {[info exists ::env(PIKV_JOBS)]} {
  set jobs $::env(PIKV_JOBS)
}

reset_run synth_1
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
  error "Synthesis failed"
}

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
  error "Implementation / bitstream failed"
}

set bitfile [glob -nocomplain [file join $proj_dir pikv_fpga.runs impl_1 *.bit]]
if {[llength $bitfile] > 0} {
  puts "Bitstream: [lindex $bitfile 0]"
} else {
  puts "Check: vivado/project/pikv_fpga.runs/impl_1/"
}

report_utilization -file [file join $proj_dir post_impl_util.rpt]
report_timing_summary -file [file join $proj_dir post_impl_timing.rpt]

save_project
puts "PiKV bitstream build complete."
