// MMIO register file — host (C/GPU) <-> PiKV-CTRL
`include "pikv_defines.vh"

module pikv_mmio #(
    parameter ADDR_W = 12
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     mmio_wr,
    input  wire                     mmio_rd,
    input  wire [ADDR_W-1:0]        mmio_addr,
    input  wire [31:0]              mmio_wdata,
    output reg  [31:0]              mmio_rdata,
    // To CTRL
    output reg  [7:0]               cmd,
    output reg  [31:0]              token_id,
    output reg  [5:0]               expert_id,
    output reg  [19:0]              page_id,
    output reg  [31:0]              arg0,
    output reg                      ctrl_start,
    input  wire [31:0]              status,
    input  wire [31:0]              hit_cnt,
    input  wire [31:0]              miss_cnt,
    input  wire [`PIKV_EXPERT_W-1:0] expert0,
    input  wire [`PIKV_EXPERT_W-1:0] expert1,
    input  wire [`PIKV_EXPERT_W-1:0] expert2,
    input  wire [`PIKV_EXPERT_W-1:0] expert3,
    output wire [31:0]              theta_reg_out
);

  localparam REG_STATUS   = 12'h000;
  localparam REG_CTRL     = 12'h004;
  localparam REG_CMD      = 12'h008;
  localparam REG_TOKEN    = 12'h00C;
  localparam REG_EXPERT   = 12'h010;
  localparam REG_PAGE     = 12'h014;
  localparam REG_ARG0     = 12'h018;
  localparam REG_THETA    = 12'h01C;
  localparam REG_HIT      = 12'h020;
  localparam REG_MISS      = 12'h024;
  localparam REG_EXPERTS  = 12'h028;

  reg [31:0] theta_reg;
  reg ctrl_start_pulse;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cmd            <= `PIKV_CMD_NOP;
      token_id       <= 32'd0;
      expert_id      <= 6'd0;
      page_id        <= 20'd0;
      arg0           <= 32'd0;
      theta_reg      <= 32'sd0;
      ctrl_start     <= 1'b0;
      ctrl_start_pulse <= 1'b0;
      mmio_rdata     <= 32'd0;
    end else begin
      ctrl_start <= 1'b0;
      if (mmio_wr) begin
        case (mmio_addr)
          REG_CTRL: begin
            if (mmio_wdata[0]) ctrl_start <= 1'b1;
            if (mmio_wdata[1]) theta_reg <= 32'sd0;
          end
          REG_CMD:    cmd       <= mmio_wdata[7:0];
          REG_TOKEN:  token_id  <= mmio_wdata;
          REG_EXPERT: expert_id <= mmio_wdata[5:0];
          REG_PAGE:   page_id   <= mmio_wdata[19:0];
          REG_ARG0:   arg0      <= mmio_wdata;
          REG_THETA:  theta_reg <= mmio_wdata;
          default: ;
        endcase
      end
      if (mmio_rd) begin
        case (mmio_addr)
          REG_STATUS:  mmio_rdata <= status;
          REG_HIT:     mmio_rdata <= hit_cnt;
          REG_MISS:    mmio_rdata <= miss_cnt;
          REG_THETA:   mmio_rdata <= theta_reg;
          REG_EXPERTS: mmio_rdata <= {2'b0, expert3, 2'b0, expert2, 2'b0, expert1, 2'b0, expert0};
          default:     mmio_rdata <= 32'd0;
        endcase
      end
    end
  end

  assign theta_reg_out = theta_reg;

endmodule
