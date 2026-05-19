# PiKV-FPGA: Verilog simulation + C host library
ROOT     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RTL_DIR  := $(ROOT)rtl
TB_DIR   := $(ROOT)tb
HOST_DIR := $(ROOT)host

CC       := gcc
CFLAGS   := -std=c11 -Wall -Wextra -fPIC -O2 -I$(HOST_DIR)
LDFLAGS  := -shared

IVERILOG := $(shell command -v iverilog 2>/dev/null)
VVP      := $(shell command -v vvp 2>/dev/null)

RTL_SRCS := $(RTL_DIR)/pikv_top.v \
            $(RTL_DIR)/pikv_ctrl.v \
            $(RTL_DIR)/pikv_mmio.v \
            $(RTL_DIR)/pikv_page_table.v \
            $(RTL_DIR)/pikv_score_fuse.v \
            $(RTL_DIR)/pikv_codec_rho.v \
            $(RTL_DIR)/pikv_scheduler.v

LIB      := $(ROOT)libpikv_fpga.so
TEST     := $(ROOT)test_pikv_fpga
VCD      := $(ROOT)tb_pikv_top.vcd

.PHONY: all host sim test clean

all: host

host: $(LIB) $(TEST)

$(LIB): $(HOST_DIR)/pikv_fpga.c $(HOST_DIR)/pikv_hw_model.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TEST): $(HOST_DIR)/test_pikv_fpga.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $(HOST_DIR)/test_pikv_fpga.c -L$(ROOT) -lpikv_fpga -Wl,-rpath,$(ROOT)

sim:
ifneq ($(IVERILOG),)
ifneq ($(VVP),)
	iverilog -g2012 -I$(RTL_DIR) -o $(ROOT)tb_pikv_top.vvp $(RTL_SRCS) $(TB_DIR)/tb_pikv_top.v
	cd $(ROOT) && vvp tb_pikv_top.vvp
else
	@echo "vvp not found; skip sim"
endif
else
	@echo "iverilog not found; skip sim (install iverilog for RTL sim)"
endif

test: host
	LD_LIBRARY_PATH=$(ROOT) $(TEST)

clean:
	rm -f $(LIB) $(TEST) $(ROOT)tb_pikv_top.vvp $(VCD)
