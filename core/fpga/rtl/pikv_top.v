// PiKV-FPGA top: MMIO + PiKV-CTRL + {D+, ScoreFuse, Codec_rho, Scheduler}
`include "pikv_defines.vh"

module pikv_top (
    input  wire         clk,
    input  wire         rst_n,
    // MMIO (AXI-Lite style simplified)
    input  wire         mmio_wr,
    input  wire         mmio_rd,
    input  wire [11:0]  mmio_addr,
    input  wire [31:0]  mmio_wdata,
    output wire [31:0]  mmio_rdata,
    // DMA / KV stream (to CXL DDR — stub)
    output wire         dma_req,
    output wire [19:0]  dma_addr
);

    wire [7:0]  cmd;
    wire [31:0] token_id, arg0;
    wire [31:0] status, hit_cnt, miss_cnt;
    wire [31:0] theta_reg;
    wire [5:0]  expert_id_mmio;
    wire [19:0] page_id;
    wire        ctrl_start;

    wire        route_start, route_done;
    wire [5:0]  route_e0, route_e1, route_e2, route_e3;
    wire [15:0] w0, w1, w2, w3;

    wire        pt_lookup, pt_insert, pt_hit;
    wire [31:0] pt_token;
    wire [5:0]  pt_expert;
    wire [19:0] pt_insert_addr, pt_addr;

    wire        codec_start, codec_last, codec_done;
    wire [15:0] k_in = mmio_wdata[15:0];
    wire [15:0] v_in = mmio_wdata[31:16];

    wire        sched_en, sched_retain;
    wire [31:0] sched_attn;
    wire [15:0] pt_miss;

    pikv_mmio u_mmio (
        .clk(clk), .rst_n(rst_n),
        .mmio_wr(mmio_wr), .mmio_rd(mmio_rd),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata),
        .cmd(cmd), .token_id(token_id), .expert_id(expert_id_mmio),
        .page_id(page_id), .arg0(arg0), .ctrl_start(ctrl_start),
        .status(status), .hit_cnt(hit_cnt), .miss_cnt(miss_cnt),
        .expert0(route_e0), .expert1(route_e1), .expert2(route_e2), .expert3(route_e3),
        .theta_reg_out(theta_reg)
    );

    pikv_score_fuse u_route (
        .clk(clk), .rst_n(rst_n),
        .start(route_start), .token_id(token_id), .done(route_done),
        .expert_out_0(route_e0), .expert_out_1(route_e1),
        .expert_out_2(route_e2), .expert_out_3(route_e3),
        .weight_out_0(w0), .weight_out_1(w1),
        .weight_out_2(w2), .weight_out_3(w3)
    );

    pikv_page_table u_pt (
        .clk(clk), .rst_n(rst_n),
        .lookup_en(pt_lookup), .token_id(pt_token), .expert_id(pt_expert),
        .insert_en(pt_insert), .insert_addr(pt_insert_addr),
        .hit(pt_hit), .addr_out(pt_addr), .miss_count(pt_miss)
    );

    pikv_codec_rho u_codec (
        .clk(clk), .rst_n(rst_n),
        .start(codec_start), .k_in(k_in), .v_in(v_in),
        .idx(8'd0), .last(codec_last), .done(codec_done),
        .k_out(), .v_out()
    );

    pikv_scheduler u_sched (
        .clk(clk), .rst_n(rst_n),
        .score_en(sched_en), .attention_q8(sched_attn),
        .recency(16'd0),         .theta_q8(theta_reg),
        .upd_theta(1'b0),
        .theta_delta(32'sd0),
        .retain(sched_retain), .theta_out()
    );

    pikv_ctrl u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(ctrl_start), .cmd(cmd), .token_id(token_id),
        .expert_id_mmio(expert_id_mmio), .arg0(arg0),
        .status(status), .hit_cnt(hit_cnt), .miss_cnt(miss_cnt),
        .out_expert0(route_e0), .out_expert1(route_e1),
        .out_expert2(route_e2), .out_expert3(route_e3),
        .pt_lookup(pt_lookup), .pt_insert(pt_insert),
        .pt_token(pt_token), .pt_expert(pt_expert), .pt_insert_addr(pt_insert_addr),
        .pt_hit(pt_hit), .pt_addr(pt_addr),
        .route_start(route_start), .route_done(route_done),
        .route_e0(route_e0), .route_e1(route_e1), .route_e2(route_e2), .route_e3(route_e3),
        .codec_start(codec_start), .codec_last(codec_last), .codec_done(codec_done),
        .sched_en(sched_en), .sched_attn(sched_attn), .sched_retain(sched_retain),
        .theta(theta_reg)
    );

    assign dma_req  = pt_insert;
    assign dma_addr = pt_insert_addr;

endmodule
