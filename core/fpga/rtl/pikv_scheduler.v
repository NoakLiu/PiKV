// Scheduler: score u_i vs threshold theta (H2O / AdaKV simplified)
`include "pikv_defines.vh"

module pikv_scheduler #(
    parameter DATA_W = `PIKV_DATA_W
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 score_en,
    input  wire [31:0]          attention_q8,   // Q8.8 fixed point
    input  wire [15:0]          recency,
    input  wire [31:0]          theta_q8,
    input  wire                 upd_theta,
    input  wire signed [31:0]  theta_delta,
    output reg                  retain,
    output reg  [31:0]          theta_out
);

  wire signed [31:0] ui;
  assign ui = attention_q8 + ($signed({16'b0, recency}) <<< 8);  // AdaKV-lite

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      retain    <= 1'b0;
      theta_out <= 32'sd0;
    end else begin
      if (upd_theta)
        theta_out <= theta_out + theta_delta;
      if (score_en)
        retain <= (ui >= theta_out);
    end
  end

endmodule
