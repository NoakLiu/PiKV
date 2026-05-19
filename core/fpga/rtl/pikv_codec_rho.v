// Codec_rho: rank-r LoRA compress (single-beat for CTRL FSM)
`include "pikv_defines.vh"

module pikv_codec_rho #(
    parameter DATA_W = `PIKV_DATA_W
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             start,
    input  wire [DATA_W-1:0] k_in,
    input  wire [DATA_W-1:0] v_in,
    input  wire [7:0]       idx,
    input  wire             last,
    output reg              done,
    output reg [DATA_W-1:0] k_out,
    output reg [DATA_W-1:0] v_out
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done  <= 1'b0;
            k_out <= {DATA_W{1'b0}};
            v_out <= {DATA_W{1'b0}};
        end else begin
            done <= 1'b0;
            if (start) begin
                k_out <= k_in >>> 2;
                v_out <= v_in >>> 2;
                if (last) done <= 1'b1;
            end
        end
    end

endmodule
