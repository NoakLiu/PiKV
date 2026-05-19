// AXI4-Lite slave -> PiKV MMIO register interface
`include "pikv_defines.vh"

module pikv_axi_lite_slave #(
    parameter ADDR_W = 12
)(
    input  wire         aclk,
    input  wire         aresetn,
    // AXI4-Lite slave
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
    // PiKV MMIO
    output reg          mmio_wr,
    output reg          mmio_rd,
    output reg [ADDR_W-1:0] mmio_addr,
    output reg [31:0]   mmio_wdata,
    input  wire [31:0]  mmio_rdata
);

    reg aw_hold, w_hold, ar_hold;
    reg [31:0] awaddr_r, araddr_r;
    reg b_pending, r_pending;

    assign s_axi_awready = s_axi_awvalid && !aw_hold;
    assign s_axi_wready  = s_axi_wvalid && !w_hold;
    assign s_axi_arready = s_axi_arvalid && !ar_hold;
    assign s_axi_bvalid  = b_pending;
    assign s_axi_rvalid  = r_pending;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_rresp   = 2'b00;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            aw_hold <= 1'b0;
            w_hold  <= 1'b0;
            ar_hold <= 1'b0;
            awaddr_r <= 32'd0;
            araddr_r <= 32'd0;
            b_pending <= 1'b0;
            r_pending <= 1'b0;
            mmio_wr <= 1'b0;
            mmio_rd <= 1'b0;
            mmio_addr <= {ADDR_W{1'b0}};
            mmio_wdata <= 32'd0;
        end else begin
            mmio_wr <= 1'b0;
            mmio_rd <= 1'b0;

            if (s_axi_awvalid && s_axi_awready) begin
                aw_hold  <= 1'b1;
                awaddr_r <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_hold <= 1'b1;
                mmio_wr   <= 1'b1;
                mmio_addr <= awaddr_r[ADDR_W-1:0];
                mmio_wdata<= s_axi_wdata;
            end
            if (aw_hold && w_hold) begin
                aw_hold <= 1'b0;
                w_hold  <= 1'b0;
                b_pending <= 1'b1;
            end
            if (b_pending && s_axi_bready)
                b_pending <= 1'b0;

            if (s_axi_arvalid && s_axi_arready) begin
                ar_hold  <= 1'b1;
                araddr_r <= s_axi_araddr;
            end
            if (ar_hold) begin
                ar_hold <= 1'b0;
                mmio_rd   <= 1'b1;
                mmio_addr <= araddr_r[ADDR_W-1:0];
                r_pending <= 1'b1;
            end
            if (r_pending && s_axi_rready)
                r_pending <= 1'b0;
        end
    end

    assign s_axi_rdata = mmio_rdata;

endmodule
