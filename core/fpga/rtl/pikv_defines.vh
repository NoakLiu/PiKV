// PiKV-FPGA parameters (paper §3.5 default tile: E=64, S=256, k=4, d=128)
`ifndef PIKV_DEFINES_VH
`define PIKV_DEFINES_VH

// Geometry
`define PIKV_NUM_EXPERTS    64
`define PIKV_SHARD_SIZE     256
`define PIKV_TOP_K          4
`define PIKV_HIDDEN_DIM     128
`define PIKV_D_PRIME        32
`define PIKV_LORA_RANK      4
`define PIKV_MMIO_DEPTH     32

// Fixed-point: Q8.8 for scores, Q16.0 for KV samples
`define PIKV_DATA_W         16
`define PIKV_ACC_W          32
`define PIKV_ADDR_W         20
`define PIKV_TOKEN_W        32
`define PIKV_EXPERT_W       6

// MMIO opcodes (match core/fpga/pikv_fpga.py MMIOCommand)
`define PIKV_CMD_NOP        8'h00
`define PIKV_CMD_ROUTE      8'h01
`define PIKV_CMD_COMPRESS   8'h02
`define PIKV_CMD_SCHEDULE   8'h03
`define PIKV_CMD_PREFETCH   8'h04
`define PIKV_CMD_UPD_THETA  8'h05
`define PIKV_CMD_DMA_RD     8'h06
`define PIKV_CMD_DMA_WR     8'h07

// CTRL FSM states
`define PIKV_ST_IDLE        4'h0
`define PIKV_ST_ROUTE       4'h1
`define PIKV_ST_LOOKUP      4'h2
`define PIKV_ST_COMPRESS    4'h3
`define PIKV_ST_SCHEDULE    4'h4
`define PIKV_ST_DONE        4'h5

// Status bits
`define PIKV_STAT_BUSY      0
`define PIKV_STAT_DONE      1
`define PIKV_STAT_ERR       2

`endif
