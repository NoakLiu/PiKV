`timescale 1ns/1ps
`include "../rtl/pikv_defines.vh"

module tb_pikv_soc;
    reg aclk = 0;
    reg aresetn = 0;
    always #1.667 aclk = ~aclk;  // 300 MHz

    // AXI-Lite BFM wires
    reg [31:0] s_axi_awaddr;
    reg        s_axi_awvalid;
    wire       s_axi_awready;
    reg [31:0] s_axi_wdata;
    reg [3:0]  s_axi_wstrb;
    reg        s_axi_wvalid;
    wire       s_axi_wready;
    wire [1:0] s_axi_bresp;
    wire       s_axi_bvalid;
    reg        s_axi_bready;
    reg [31:0] s_axi_araddr;
    reg        s_axi_arvalid;
    wire       s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0] s_axi_rresp;
    wire       s_axi_rvalid;
    reg        s_axi_rready;

    // AXI-MM to CXL mem model
    wire [63:0] m_axi_araddr, m_axi_awaddr;
    wire [7:0]  m_axi_arlen, m_axi_awlen;
    wire [2:0]  m_axi_arsize, m_axi_awsize;
    wire [1:0]  m_axi_arburst, m_axi_awburst;
    wire [2:0]  m_axi_arprot, m_axi_awprot;
    wire        m_axi_arvalid, m_axi_awvalid, m_axi_wvalid, m_axi_wlast;
    wire        m_axi_arready, m_axi_awready, m_axi_wready;
    wire [127:0] m_axi_rdata, m_axi_wdata;
    wire [15:0]  m_axi_wstrb;
    wire [1:0]   m_axi_rresp, m_axi_bresp;
    wire        m_axi_rvalid, m_axi_rlast, m_axi_bvalid;
    reg         m_axi_rready, m_axi_bready;

    pikv_soc_top dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awprot(3'b0),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arprot(3'b0),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
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
        .m_axi_bready(m_axi_bready)
    );

    pikv_cxl_mem_model u_cxl_mem (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_araddr(m_axi_araddr), .s_axi_arlen(m_axi_arlen),
        .s_axi_arsize(m_axi_arsize), .s_axi_arburst(m_axi_arburst),
        .s_axi_arprot(m_axi_arprot), .s_axi_arvalid(m_axi_arvalid),
        .s_axi_arready(m_axi_arready), .s_axi_rdata(m_axi_rdata),
        .s_axi_rresp(m_axi_rresp), .s_axi_rlast(m_axi_rlast),
        .s_axi_rvalid(m_axi_rvalid), .s_axi_rready(m_axi_rready),
        .s_axi_awaddr(m_axi_awaddr), .s_axi_awlen(m_axi_awlen),
        .s_axi_awsize(m_axi_awsize), .s_axi_awburst(m_axi_awburst),
        .s_axi_awprot(m_axi_awprot), .s_axi_awvalid(m_axi_awvalid),
        .s_axi_awready(m_axi_awready), .s_axi_wdata(m_axi_wdata),
        .s_axi_wstrb(m_axi_wstrb), .s_axi_wlast(m_axi_wlast),
        .s_axi_wvalid(m_axi_wvalid), .s_axi_wready(m_axi_wready),
        .s_axi_bresp(m_axi_bresp), .s_axi_bvalid(m_axi_bvalid),
        .s_axi_bready(m_axi_bready)
    );

    task axil_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge aclk);
            s_axi_awaddr <= addr; s_axi_awvalid <= 1;
            s_axi_wdata <= data; s_axi_wstrb <= 4'hF; s_axi_wvalid <= 1;
            s_axi_bready <= 1;
            wait(s_axi_awready && s_axi_wready);
            @(posedge aclk);
            s_axi_awvalid <= 0; s_axi_wvalid <= 0;
            wait(s_axi_bvalid);
            @(posedge aclk);
            s_axi_bready <= 0;
        end
    endtask

    task axil_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge aclk);
            s_axi_araddr <= addr; s_axi_arvalid <= 1;
            s_axi_rready <= 1;
            wait(s_axi_arready);
            @(posedge aclk);
            s_axi_arvalid <= 0;
            wait(s_axi_rvalid);
            data = s_axi_rdata;
            @(posedge aclk);
            s_axi_rready <= 0;
        end
    endtask

    reg [31:0] rd;
    integer i;

    initial begin
        m_axi_rready = 1;
        m_axi_bready = 1;
        s_axi_awvalid = 0;
        s_axi_wvalid = 0;
        s_axi_arvalid = 0;
        #50 aresetn = 1;
        #30;

        axil_write(32'h040, 32'h0000_0001);
        axil_write(32'h044, 32'h0000_0000);

        axil_write(32'h00C, 32'd100);
        axil_write(32'h008, `PIKV_CMD_ROUTE);
        axil_write(32'h004, 32'd1);

        for (i = 0; i < 500; i = i + 1) begin
            #10;
            axil_read(32'h000, rd);
            if (rd[1]) begin
                axil_read(32'h028, rd);
                $display("SOC DONE experts=0x%h", rd);
                axil_read(32'h048, rd);
                $display("DMA xfers=%0d", rd);
                disable done_lbl;
            end
        end
        done_lbl: #20;
        $display("tb_pikv_soc finished");
        $finish;
    end
endmodule
