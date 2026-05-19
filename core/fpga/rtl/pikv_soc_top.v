// PiKV SoC top: AXI-Lite (host control) + AXI-MM (CXL.mem KV DMA)
`include "pikv_defines.vh"

module pikv_soc_top #(
    parameter ADDR_W = 64,
    parameter DATA_W = 128,
    parameter CXL_BASE = 64'h0000_0001_0000_0000
)(
    input  wire         aclk,
    input  wire         aresetn,
    // AXI4-Lite slave (BAR0 / XDMA control)
    input  wire [31:0]  s_axi_awaddr,
    input  wire [2:0]   s_axi_awprot,
    input  wire         s_axi_awvalid,
    output wire         s_axi_awready,
    input  wire [31:0]  s_axi_wdata,
    input  wire [3:0]   s_axi_wstrb,
    input  wire         s_axi_wvalid,
    output wire         s_axi_wready,
    output wire [1:0]   s_axi_bresp,
    output wire         s_axi_bvalid,
    input  wire         s_axi_bready,
    input  wire [31:0]  s_axi_araddr,
    input  wire [2:0]   s_axi_arprot,
    input  wire         s_axi_arvalid,
    output wire         s_axi_arready,
    output wire [31:0]  s_axi_rdata,
    output wire [1:0]   s_axi_rresp,
    output wire         s_axi_rvalid,
    input  wire         s_axi_rready,
    // AXI4 master -> CXL Type-3 memory expander
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
    output wire                     m_axi_bready
);

    wire        mmio_wr, mmio_rd;
    wire [11:0] mmio_addr;
    wire [31:0] mmio_wdata, mmio_rdata_core, mmio_rdata;

    wire        dma_req, dma_we, dma_ack;
    wire [`PIKV_ADDR_W-1:0] dma_addr;
    wire [DATA_W-1:0] dma_wdata;

    reg [ADDR_W-1:0] cxl_base_reg;
    wire [31:0] dma_xfer_cnt;
    wire        dma_busy;

    // CXL.mem base address (MMIO 0x040 lo / 0x044 hi)
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            cxl_base_reg <= CXL_BASE;
        else if (mmio_wr && mmio_addr == 12'h040)
            cxl_base_reg[31:0] <= mmio_wdata;
        else if (mmio_wr && mmio_addr == 12'h044)
            cxl_base_reg[63:32] <= mmio_wdata;
    end

    assign mmio_rdata = (mmio_rd && mmio_addr == 12'h040) ? cxl_base_reg[31:0] :
                        (mmio_rd && mmio_addr == 12'h044) ? cxl_base_reg[63:32] :
                        (mmio_rd && mmio_addr == 12'h048) ? dma_xfer_cnt :
                        (mmio_rd && mmio_addr == 12'h04C) ? {30'b0, dma_busy, 1'b0} :
                        mmio_rdata_core;

    pikv_axi_lite_slave u_axil (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awprot(s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .mmio_wr(mmio_wr), .mmio_rd(mmio_rd),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata_core)
    );

    pikv_top u_core (
        .clk(aclk), .rst_n(aresetn),
        .mmio_wr(mmio_wr), .mmio_rd(mmio_rd),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata),
        .dma_req(dma_req), .dma_we(dma_we), .dma_addr(dma_addr),
        .dma_wdata(dma_wdata), .dma_ack(dma_ack)
    );

    pikv_cxl_dma #(
        .ADDR_W(ADDR_W), .DATA_W(DATA_W), .CXL_BASE(CXL_BASE)
    ) u_cxl_dma (
        .aclk(aclk), .aresetn(aresetn),
        .core_dma_req(dma_req), .core_dma_we(dma_we),
        .core_dma_addr(dma_addr), .core_dma_wdata(dma_wdata),
        .core_dma_ack(dma_ack),
        .cxl_base_addr(cxl_base_reg),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arprot(m_axi_arprot), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready), .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awprot(m_axi_awprot), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready), .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .dma_busy(dma_busy), .dma_xfer_cnt(dma_xfer_cnt)
    );

endmodule
