// CXL DMA bridge: PiKV core dma_req -> AXI-MM master (CXL.mem DDR pool)
`include "pikv_defines.vh"

module pikv_cxl_dma #(
    parameter ADDR_W   = 64,
    parameter DATA_W   = 128,
    parameter CXL_BASE = 64'h0000_0001_0000_0000
)(
    input  wire                     aclk,
    input  wire                     aresetn,
    input  wire                     core_dma_req,
    input  wire                     core_dma_we,
    input  wire [`PIKV_ADDR_W-1:0]  core_dma_addr,
    input  wire [DATA_W-1:0]        core_dma_wdata,
    output wire                     core_dma_ack,
    input  wire [ADDR_W-1:0]        cxl_base_addr,
    output wire [ADDR_W-1:0]        m_axi_araddr,
    output wire [7:0]               m_axi_arlen,
    output wire [2:0]               m_axi_arsize,
    output wire [1:0]               m_axi_arburst,
    output wire [2:0]               m_axi_arprot,
    output wire                     m_axi_arvalid,
    input  wire                     m_axi_arready,
    input  wire [DATA_W-1:0]        m_axi_rdata,
    input  wire [1:0]               m_axi_rresp,
    input  wire                     m_axi_rlast,
    input  wire                     m_axi_rvalid,
    output wire                     m_axi_rready,
    output wire [ADDR_W-1:0]        m_axi_awaddr,
    output wire [7:0]               m_axi_awlen,
    output wire [2:0]               m_axi_awsize,
    output wire [1:0]               m_axi_awburst,
    output wire [2:0]               m_axi_awprot,
    output wire                     m_axi_awvalid,
    input  wire                     m_axi_awready,
    output wire [DATA_W-1:0]        m_axi_wdata,
    output wire [(DATA_W/8)-1:0]    m_axi_wstrb,
    output wire                     m_axi_wlast,
    output wire                     m_axi_wvalid,
    input  wire                     m_axi_wready,
    input  wire [1:0]               m_axi_bresp,
    input  wire                     m_axi_bvalid,
    output wire                     m_axi_bready,
    output wire                     dma_busy,
    output reg  [31:0]              dma_xfer_cnt
);

    reg [2:0] state;
    localparam S_IDLE = 3'd0, S_RUN = 3'd1, S_WAIT = 3'd2;

    reg start_pulse;
    reg we_hold;
    reg [`PIKV_ADDR_W-1:0] addr_hold;
    reg [DATA_W-1:0] wdata_hold;
    wire dma_done, axi_busy;

  assign core_dma_ack = (state == S_WAIT) && dma_done;
  assign dma_busy = (state != S_IDLE) || axi_busy;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state <= S_IDLE;
            start_pulse <= 1'b0;
            we_hold <= 1'b1;
            addr_hold <= {`PIKV_ADDR_W{1'b0}};
            wdata_hold <= {DATA_W{1'b0}};
            dma_xfer_cnt <= 32'd0;
        end else begin
            start_pulse <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (core_dma_req) begin
                        we_hold <= core_dma_we;
                        addr_hold <= core_dma_addr;
                        wdata_hold <= core_dma_wdata;
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    start_pulse <= 1'b1;
                    state <= S_WAIT;
                end
                S_WAIT: begin
                    if (dma_done) begin
                        dma_xfer_cnt <= dma_xfer_cnt + 32'd1;
                        state <= S_IDLE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    wire [ADDR_W-1:0] eff_base = (cxl_base_addr != {ADDR_W{1'b0}}) ? cxl_base_addr : CXL_BASE;

    pikv_axi_dma_master #(
        .ADDR_W(ADDR_W),
        .DATA_W(DATA_W),
        .CXL_MEM_BASE_DEFAULT(CXL_BASE)
    ) u_axi_dma (
        .aclk(aclk),
        .aresetn(aresetn),
        .cxl_mem_base(eff_base),
        .dma_start(start_pulse),
        .dma_write(we_hold),
        .dma_addr({44'b0, addr_hold}),
        .dma_wdata(wdata_hold),
        .dma_done(dma_done),
        .dma_busy(axi_busy),
        .dma_rdata(),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready)
    );

endmodule
