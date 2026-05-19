// PiKV-CTRL: orchestrates ROUTE -> LOOKUP -> COMPRESS -> SCHEDULE per token
`include "pikv_defines.vh"

module pikv_ctrl (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [7:0]               cmd,
    input  wire [31:0]              token_id,
    input  wire [5:0]               expert_id_mmio,
    input  wire [31:0]              arg0,
    output reg  [31:0]              status,
    output reg  [31:0]              hit_cnt,
    output reg  [31:0]              miss_cnt,
    output reg  [`PIKV_EXPERT_W-1:0] out_expert0,
    output reg  [`PIKV_EXPERT_W-1:0] out_expert1,
    output reg  [`PIKV_EXPERT_W-1:0] out_expert2,
    output reg  [`PIKV_EXPERT_W-1:0] out_expert3,
    // Page table
    output reg                      pt_lookup,
    output reg                      pt_insert,
    output reg  [31:0]              pt_token,
    output reg  [5:0]               pt_expert,
    output reg  [`PIKV_ADDR_W-1:0] pt_insert_addr,
    input  wire                     pt_hit,
    input  wire [`PIKV_ADDR_W-1:0]  pt_addr,
    // Score fuse
    output reg                      route_start,
    input  wire                     route_done,
    input  wire [`PIKV_EXPERT_W-1:0] route_e0,
    input  wire [`PIKV_EXPERT_W-1:0] route_e1,
    input  wire [`PIKV_EXPERT_W-1:0] route_e2,
    input  wire [`PIKV_EXPERT_W-1:0] route_e3,
    // Codec
    output reg                      codec_start,
    output reg                      codec_last,
    input  wire                     codec_done,
    // Scheduler
    output reg                      sched_en,
    output reg  [31:0]              sched_attn,
    input  wire                     sched_retain,
    output reg  [31:0]              theta
);

    reg [3:0] state;
    reg [2:0] exp_idx;
    reg [5:0] cur_expert;
    reg [`PIKV_ADDR_W-1:0] next_pool_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= `PIKV_ST_IDLE;
            status        <= 32'd0;
            hit_cnt       <= 32'd0;
            miss_cnt      <= 32'd0;
            exp_idx       <= 3'd0;
            next_pool_addr<= {`PIKV_ADDR_W{1'b0}};
            pt_lookup     <= 1'b0;
            pt_insert     <= 1'b0;
            route_start   <= 1'b0;
            codec_start   <= 1'b0;
            codec_last    <= 1'b0;
            sched_en      <= 1'b0;
            out_expert0   <= 6'd0;
            out_expert1   <= 6'd0;
            out_expert2   <= 6'd0;
            out_expert3   <= 6'd0;
        end else begin
            pt_lookup   <= 1'b0;
            pt_insert   <= 1'b0;
            route_start <= 1'b0;
            codec_start <= 1'b0;
            codec_last  <= 1'b0;
            sched_en    <= 1'b0;
            status[0]   <= (state != `PIKV_ST_IDLE);
            status[1]   <= (state == `PIKV_ST_DONE);

            case (state)
                `PIKV_ST_IDLE: begin
                    if (start && cmd == `PIKV_CMD_ROUTE) begin
                        pt_token    <= token_id;
                        route_start <= 1'b1;
                        state       <= `PIKV_ST_ROUTE;
                    end else if (start && cmd == `PIKV_CMD_UPD_THETA) begin
                        theta <= theta + arg0;
                        state <= `PIKV_ST_DONE;
                    end
                end
                `PIKV_ST_ROUTE: begin
                    if (route_done) begin
                        out_expert0 <= route_e0;
                        out_expert1 <= route_e1;
                        out_expert2 <= route_e2;
                        out_expert3 <= route_e3;
                        exp_idx     <= 3'd0;
                        state       <= `PIKV_ST_LOOKUP;
                    end
                end
                `PIKV_ST_LOOKUP: begin
                    case (exp_idx)
                        3'd0: cur_expert <= route_e0;
                        3'd1: cur_expert <= route_e1;
                        3'd2: cur_expert <= route_e2;
                        default: cur_expert <= route_e3;
                    endcase
                    pt_token  <= token_id;
                    pt_expert <= cur_expert;
                    pt_lookup <= 1'b1;
                    if (pt_hit)
                        hit_cnt <= hit_cnt + 32'd1;
                    else
                        miss_cnt <= miss_cnt + 32'd1;
                    state <= `PIKV_ST_COMPRESS;
                end
                `PIKV_ST_COMPRESS: begin
                    codec_start <= 1'b1;
                    codec_last  <= 1'b1;
                    if (codec_done) begin
                        pt_insert_addr <= next_pool_addr;
                        next_pool_addr <= next_pool_addr + 1'b1;
                        pt_token       <= token_id;
                        pt_expert      <= cur_expert;
                        pt_insert      <= 1'b1;
                        sched_attn     <= arg0;
                        sched_en       <= 1'b1;
                        state          <= `PIKV_ST_SCHEDULE;
                    end
                end
                `PIKV_ST_SCHEDULE: begin
                    if (exp_idx >= `PIKV_TOP_K - 1)
                        state <= `PIKV_ST_DONE;
                    else begin
                        exp_idx <= exp_idx + 3'd1;
                        state   <= `PIKV_ST_LOOKUP;
                    end
                end
                `PIKV_ST_DONE: begin
                    state <= `PIKV_ST_IDLE;
                end
                default: state <= `PIKV_ST_IDLE;
            endcase
        end
    end

endmodule
