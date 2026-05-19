// Behavioral AXI4 slave: CXL Type-3 DDR pool (simulation / cosim)
`include "pikv_defines.vh"

module pikv_cxl_mem_model #(
    parameter ADDR_W = 64,
    parameter DATA_W = 128,
    parameter MEM_WORDS = 65536
)(
    input  wire                     aclk,
    input  wire                     aresetn,
    input  wire [ADDR_W-1:0]        s_axi_araddr,
    input  wire [7:0]               s_axi_arlen,
    input  wire [2:0]               s_axi_arsize,
    input  wire [1:0]               s_axi_arburst,
    input  wire [2:0]               s_axi_arprot,
    input  wire                     s_axi_arvalid,
    output reg                      s_axi_arready,
    output reg  [DATA_W-1:0]        s_axi_rdata,
    output reg  [1:0]               s_axi_rresp,
    output reg                      s_axi_rlast,
    output reg                      s_axi_rvalid,
    input  wire                     s_axi_rready,
    input  wire [ADDR_W-1:0]        s_axi_awaddr,
    input  wire [7:0]               s_axi_awlen,
    input  wire [2:0]               s_axi_awsize,
    input  wire [1:0]               s_axi_awburst,
    input  wire [2:0]               s_axi_awprot,
    input  wire                     s_axi_awvalid,
    output reg                      s_axi_awready,
    input  wire [DATA_W-1:0]        s_axi_wdata,
    input  wire [(DATA_W/8)-1:0]    s_axi_wstrb,
    input  wire                     s_axi_wlast,
    input  wire                     s_axi_wvalid,
    output reg                      s_axi_wready,
    output reg  [1:0]               s_axi_bresp,
    output reg                      s_axi_bvalid,
    input  wire                     s_axi_bready
);

    reg [DATA_W-1:0] mem [0:MEM_WORDS-1];
    wire [31:0] word_idx_w;
    assign word_idx_w = s_axi_awaddr[19:4] % MEM_WORDS;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            s_axi_arready <= 1'b0;
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_rlast   <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_bresp   <= 2'b00;
        end else begin
            s_axi_arready <= s_axi_arvalid;
            s_axi_awready <= s_axi_awvalid;
            s_axi_wready  <= s_axi_wvalid;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rdata <= mem[s_axi_araddr[19:4] % MEM_WORDS];
                s_axi_rvalid <= 1'b1;
                s_axi_rlast  <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
                s_axi_rlast  <= 1'b0;
            end
            if (s_axi_awvalid && s_axi_awready && s_axi_wvalid && s_axi_wready) begin
                mem[word_idx_w] <= s_axi_wdata;
                s_axi_bvalid <= 1'b1;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
        end
    end

endmodule
