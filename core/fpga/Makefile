# PiKV-FPGA: Verilog simulation + C host library + Vivado hooks
ROOT     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RTL_DIR  := $(ROOT)rtl
TB_DIR   := $(ROOT)tb
HOST_DIR := $(ROOT)host
VIVADO   := vivado

CC       := gcc
CFLAGS   := -std=c11 -Wall -Wextra -fPIC -O2 -I$(HOST_DIR)
LDFLAGS  := -shared

IVERILOG := $(shell command -v iverilog 2>/dev/null)
VVP      := $(shell command -v vvp 2>/dev/null)

RTL_CORE := $(RTL_DIR)/pikv_top.v \
            $(RTL_DIR)/pikv_ctrl.v \
            $(RTL_DIR)/pikv_mmio.v \
            $(RTL_DIR)/pikv_page_table.v \
            $(RTL_DIR)/pikv_score_fuse.v \
            $(RTL_DIR)/pikv_codec_rho.v \
            $(RTL_DIR)/pikv_scheduler.v

RTL_SOC  := $(RTL_CORE) \
            $(RTL_DIR)/pikv_axi_lite_slave.v \
            $(RTL_DIR)/pikv_axi_dma_master.v \
            $(RTL_DIR)/pikv_cxl_dma.v \
            $(RTL_DIR)/pikv_cxl_mem_model.v \
            $(RTL_DIR)/pikv_soc_top.v

LIB      := $(ROOT)libpikv_fpga.so
TEST     := $(ROOT)test_pikv_fpga

.PHONY: all host sim sim-soc test vivado bitstream bd clean

all: host

host: $(LIB) $(TEST)

$(LIB): $(HOST_DIR)/pikv_fpga.c $(HOST_DIR)/pikv_hw_model.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TEST): $(HOST_DIR)/test_pikv_fpga.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $(HOST_DIR)/test_pikv_fpga.c -L$(ROOT) -lpikv_fpga -Wl,-rpath,$(ROOT)

sim:
ifneq ($(IVERILOG),)
ifneq ($(VVP),)
	iverilog -g2012 -I$(RTL_DIR) -o $(ROOT)tb_pikv_top.vvp $(RTL_CORE) $(TB_DIR)/tb_pikv_top.v
	cd $(ROOT) && vvp tb_pikv_top.vvp
else
	@echo "vvp not found; skip sim"
endif
else
	@echo "iverilog not found; skip sim"
endif

sim-soc:
ifneq ($(IVERILOG),)
ifneq ($(VVP),)
	iverilog -g2012 -I$(RTL_DIR) -o $(ROOT)tb_pikv_soc.vvp $(RTL_SOC) $(TB_DIR)/tb_pikv_soc.v
	cd $(ROOT) && vvp tb_pikv_soc.vvp
else
	@echo "vvp not found; skip sim-soc"
endif
else
	@echo "iverilog not found; skip sim-soc"
endif

test: host
	LD_LIBRARY_PATH=$(ROOT) $(TEST)

vivado:
	$(VIVADO) -mode batch -source vivado/scripts/create_project.tcl

bitstream:
	$(VIVADO) -mode batch -source vivado/scripts/build_bitstream.tcl

bd:
	$(VIVADO) -mode batch -source vivado/scripts/create_bd.tcl

clean:
	rm -f $(LIB) $(TEST) $(ROOT)tb_pikv_top.vvp $(ROOT)tb_pikv_soc.vvp $(ROOT)tb_*.vcd
