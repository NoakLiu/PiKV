// Module D+: page table Gamma — (t,e) -> addr, miss count m_e
`include "pikv_defines.vh"

module pikv_page_table #(
    parameter NUM_EXPERTS = `PIKV_NUM_EXPERTS,
    parameter SHARD_SIZE  = `PIKV_SHARD_SIZE,
    parameter TOKEN_W     = `PIKV_TOKEN_W,
    parameter ADDR_W      = `PIKV_ADDR_W,
    parameter EXPERT_W    = `PIKV_EXPERT_W
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     lookup_en,
    input  wire [TOKEN_W-1:0]       token_id,
    input  wire [EXPERT_W-1:0]      expert_id,
    input  wire                     insert_en,
    input  wire [ADDR_W-1:0]        insert_addr,
    output reg                      hit,
    output reg  [ADDR_W-1:0]        addr_out,
    output reg  [15:0]              miss_count   // per-expert miss (simplified: global)
);

    localparam SHARD_W = $clog2(SHARD_SIZE);
    localparam EXP_W   = $clog2(NUM_EXPERTS);

    reg [TOKEN_W-1:0] stored_token [0:NUM_EXPERTS*SHARD_SIZE-1];
    reg [ADDR_W-1:0]  stored_addr  [0:NUM_EXPERTS*SHARD_SIZE-1];
    reg [15:0]        expert_miss  [0:NUM_EXPERTS-1];

    integer i;

    wire [SHARD_W-1:0] shard_idx;
    wire [EXP_W-1:0]   exp_idx;
    wire [SHARD_W+EXP_W-1:0] flat_idx;

    // s(t,e) = (t mod S) xor (e mod E)
    assign shard_idx = (token_id[SHARD_W-1:0] ^ expert_id[SHARD_W-1:0]) % SHARD_SIZE;
    assign exp_idx   = expert_id[EXP_W-1:0] % NUM_EXPERTS;
    assign flat_idx  = exp_idx * SHARD_SIZE + shard_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hit        <= 1'b0;
            addr_out   <= {ADDR_W{1'b0}};
            miss_count <= 16'd0;
            for (i = 0; i < NUM_EXPERTS; i = i + 1)
                expert_miss[i] <= 16'd0;
        end else begin
            hit <= 1'b0;
            if (lookup_en) begin
                if (stored_token[flat_idx] == token_id && stored_addr[flat_idx] != {ADDR_W{1'b0}}) begin
                    hit      <= 1'b1;
                    addr_out <= stored_addr[flat_idx];
                end else begin
                    expert_miss[exp_idx] <= expert_miss[exp_idx] + 16'd1;
                    miss_count <= expert_miss[exp_idx] + 16'd1;
                    addr_out <= {ADDR_W{1'b0}};
                end
            end
            if (insert_en) begin
                stored_token[flat_idx] <= token_id;
                stored_addr[flat_idx]  <= insert_addr;
            end
        end
    end

endmodule
