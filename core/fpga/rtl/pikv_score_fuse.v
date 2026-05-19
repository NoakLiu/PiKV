// ScoreFuse + Top-k: deterministic expert pick from token_id (RTL-friendly)
`include "pikv_defines.vh"

module pikv_score_fuse #(
    parameter NUM_EXPERTS = `PIKV_NUM_EXPERTS,
    parameter TOP_K       = `PIKV_TOP_K,
    parameter EXPERT_W    = `PIKV_EXPERT_W
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [31:0]              token_id,
    output reg                      done,
    output reg  [EXPERT_W-1:0]      expert_out_0,
    output reg  [EXPERT_W-1:0]      expert_out_1,
    output reg  [EXPERT_W-1:0]      expert_out_2,
    output reg  [EXPERT_W-1:0]      expert_out_3,
    output reg  [15:0]              weight_out_0,
    output reg  [15:0]              weight_out_1,
    output reg  [15:0]              weight_out_2,
    output reg  [15:0]              weight_out_3
);

    wire [EXPERT_W-1:0] base;
    assign base = token_id[EXPERT_W-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done          <= 1'b0;
            expert_out_0  <= {EXPERT_W{1'b0}};
            expert_out_1  <= {EXPERT_W{1'b0}};
            expert_out_2  <= {EXPERT_W{1'b0}};
            expert_out_3  <= {EXPERT_W{1'b0}};
            weight_out_0  <= 16'd256;
            weight_out_1  <= 16'd192;
            weight_out_2  <= 16'd128;
            weight_out_3  <= 16'd64;
        end else begin
            done <= 1'b0;
            if (start) begin
                expert_out_0 <= (base + 6'd0) % NUM_EXPERTS;
                expert_out_1 <= (base + 6'd7) % NUM_EXPERTS;
                expert_out_2 <= (base + 6'd13) % NUM_EXPERTS;
                expert_out_3 <= (base + 6'd19) % NUM_EXPERTS;
                done <= 1'b1;
            end
        end
    end

endmodule
