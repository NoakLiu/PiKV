`timescale 1ns/1ps
`include "../rtl/pikv_defines.vh"

module tb_pikv_top;
    reg clk = 0;
    reg rst_n = 0;
    reg mmio_wr = 0, mmio_rd = 0;
    reg [11:0] mmio_addr = 0;
    reg [31:0] mmio_wdata = 0;
    wire [31:0] mmio_rdata;
    wire dma_req;
    wire [19:0] dma_addr;

    always #5 clk = ~clk;

    pikv_top dut (
        .clk(clk), .rst_n(rst_n),
        .mmio_wr(mmio_wr), .mmio_rd(mmio_rd),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_rdata(mmio_rdata),
        .dma_req(dma_req), .dma_addr(dma_addr)
    );

    task mmio_write(input [11:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            mmio_wr = 1; mmio_rd = 0;
            mmio_addr = addr; mmio_wdata = data;
            @(posedge clk);
            mmio_wr = 0;
        end
    endtask

    task mmio_read(input [11:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            mmio_rd = 1; mmio_wr = 0;
            mmio_addr = addr;
            @(posedge clk);
            data = mmio_rdata;
            mmio_rd = 0;
        end
    endtask

    integer i;
    reg [31:0] rdata;

    initial begin
        $dumpfile("tb_pikv_top.vcd");
        $dumpvars(0, tb_pikv_top);
        #20 rst_n = 1;
        #30;

        mmio_write(12'h00C, 32'd42);       // token
        mmio_write(12'h008, `PIKV_CMD_ROUTE);
        mmio_write(12'h004, 32'd1);        // start

        for (i = 0; i < 200; i = i + 1) begin
            #10;
            mmio_read(12'h000, rdata);
            if (rdata[1]) begin
                $display("DONE status=0x%h experts reg next", rdata);
                mmio_read(12'h028, rdata);
                $display("experts=0x%h hits/miss next", rdata);
                mmio_read(12'h020, rdata);
                $display("hits=0x%h", rdata);
                mmio_read(12'h024, rdata);
                $display("misses=0x%h", rdata);
                disable finish;
            end
        end
        finish: #20;
        $display("tb_pikv_top finished");
        $finish;
    end
endmodule
