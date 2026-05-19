// AXI4 memory-mapped master for CXL Type-3 DDR pool (KV body DMA)
`include "pikv_defines.vh"

module pikv_axi_dma_master #(
    parameter ADDR_W        = 64,
    parameter DATA_W        = 128,
    parameter CXL_MEM_BASE_DEFAULT = 64'h0000_0001_0000_0000
)(
    input  wire                     aclk,
    input  wire                     aresetn,
    input  wire [ADDR_W-1:0]        cxl_mem_base,
    // Control
    input  wire                     dma_start,
    input  wire                     dma_write,     // 1=store KV, 0=load
    input  wire [ADDR_W-1:0]        dma_addr,
    input  wire [DATA_W-1:0]        dma_wdata,
    output reg                      dma_done,
    output reg                      dma_busy,
    output reg  [DATA_W-1:0]        dma_rdata,
    // AXI4 master
    output reg  [ADDR_W-1:0]        m_axi_araddr,
    output reg  [7:0]               m_axi_arlen,
    output reg  [2:0]               m_axi_arsize,
    output reg  [1:0]               m_axi_arburst,
    output reg  [2:0]               m_axi_arprot,
    output reg                      m_axi_arvalid,
    input  wire                     m_axi_arready,
    input  wire [DATA_W-1:0]        m_axi_rdata,
    input  wire [1:0]               m_axi_rresp,
    input  wire                     m_axi_rlast,
    input  wire                     m_axi_rvalid,
    output reg                      m_axi_rready,
    output reg  [ADDR_W-1:0]        m_axi_awaddr,
    output reg  [7:0]               m_axi_awlen,
    output reg  [2:0]               m_axi_awsize,
    output reg  [1:0]               m_axi_awburst,
    output reg  [2:0]               m_axi_awprot,
    output reg                      m_axi_awvalid,
    input  wire                     m_axi_awready,
    output reg  [DATA_W-1:0]        m_axi_wdata,
    output reg  [(DATA_W/8)-1:0]    m_axi_wstrb,
    output reg                      m_axi_wlast,
    output reg                      m_axi_wvalid,
    input  wire                     m_axi_wready,
    input  wire [1:0]               m_axi_bresp,
    input  wire                     m_axi_bvalid,
    output reg                      m_axi_bready
);

    localparam ST_IDLE = 3'd0;
    localparam ST_AW   = 3'd1;
    localparam ST_W    = 3'd2;
    localparam ST_B    = 3'd3;
    localparam ST_AR   = 3'd4;
    localparam ST_R    = 3'd5;

    reg [2:0] state;
    wire [ADDR_W-1:0] base_addr = (cxl_mem_base != {ADDR_W{1'b0}}) ? cxl_mem_base : CXL_MEM_BASE_DEFAULT;
    wire [ADDR_W-1:0] full_addr = base_addr + {{ADDR_W-`PIKV_ADDR_W{1'b0}}, dma_addr[`PIKV_ADDR_W-1:0]};

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state <= ST_IDLE;
            dma_done <= 1'b0;
            dma_busy <= 1'b0;
            dma_rdata <= {DATA_W{1'b0}};
            m_axi_arvalid <= 1'b0;
            m_axi_awvalid <= 1'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_wlast   <= 1'b0;
            m_axi_rready  <= 1'b0;
            m_axi_bready  <= 1'b0;
            m_axi_arlen   <= 8'd0;
            m_axi_awlen   <= 8'd0;
            m_axi_arsize  <= 3'd4;  // 16 bytes for 128-bit
            m_axi_awsize  <= 3'd4;
            m_axi_arburst <= 2'b01;
            m_axi_awburst <= 2'b01;
            m_axi_arprot  <= 3'b000;
            m_axi_awprot  <= 3'b000;
            m_axi_wstrb   <= {(DATA_W/8){1'b1}};
        end else begin
            dma_done <= 1'b0;
            case (state)
                ST_IDLE: begin
                    dma_busy <= 1'b0;
                    if (dma_start) begin
                        dma_busy <= 1'b1;
                        if (dma_write) state <= ST_AW;
                        else           state <= ST_AR;
                    end
                end
                ST_AW: begin
                    m_axi_awaddr  <= full_addr;
                    m_axi_awvalid <= 1'b1;
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                        state <= ST_W;
                    end
                end
                ST_W: begin
                    m_axi_wdata  <= dma_wdata;
                    m_axi_wvalid <= 1'b1;
                    m_axi_wlast  <= 1'b1;
                    if (m_axi_wready) begin
                        m_axi_wvalid <= 1'b0;
                        m_axi_wlast  <= 1'b0;
                        state <= ST_B;
                    end
                end
                ST_B: begin
                    m_axi_bready <= 1'b1;
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 1'b0;
                        dma_done <= 1'b1;
                        state <= ST_IDLE;
                    end
                end
                ST_AR: begin
                    m_axi_araddr  <= full_addr;
                    m_axi_arvalid <= 1'b1;
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        state <= ST_R;
                    end
                end
                ST_R: begin
                    m_axi_rready <= 1'b1;
                    if (m_axi_rvalid) begin
                        dma_rdata <= m_axi_rdata;
                        m_axi_rready <= 1'b0;
                        dma_done <= 1'b1;
                        state <= ST_IDLE;
                    end
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
