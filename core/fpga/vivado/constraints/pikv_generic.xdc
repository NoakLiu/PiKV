# Generic timing constraints for PiKV-FPGA (pikv_soc_top)
# Adjust clock port names when integrating with XDMA / board PLL

create_clock -name aclk -period 3.333 [get_ports aclk]
set_input_delay -clock aclk -max 1.0 [get_ports -filter {DIRECTION == IN}]
set_output_delay -clock aclk -max 1.0 [get_ports -filter {DIRECTION == OUT}]

set_false_path -from [get_ports aresetn]
set_max_fanout 256 [current_design]

# AXI interface false paths for async resets (if any)
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets -hierarchical -filter {NAME =~ "*aresetn*"}]
