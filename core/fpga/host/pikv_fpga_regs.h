/**
 * PiKV-FPGA MMIO register map (matches rtl/pikv_mmio.v and Python MMIOCommand).
 */
#ifndef PIKV_FPGA_REGS_H
#define PIKV_FPGA_REGS_H

#include <stdint.h>

#define PIKV_MMIO_BASE       0x0000

#define PIKV_REG_STATUS      0x000
#define PIKV_REG_CTRL        0x004
#define PIKV_REG_CMD         0x008
#define PIKV_REG_TOKEN       0x00C
#define PIKV_REG_EXPERT      0x010
#define PIKV_REG_PAGE        0x014
#define PIKV_REG_ARG0        0x018
#define PIKV_REG_THETA       0x01C
#define PIKV_REG_HIT         0x020
#define PIKV_REG_MISS        0x024
#define PIKV_REG_EXPERTS     0x028
#define PIKV_REG_K_IN        0x02C
#define PIKV_REG_V_IN        0x030
#define PIKV_REG_CXL_BASE_LO 0x040
#define PIKV_REG_CXL_BASE_HI 0x044
#define PIKV_REG_DMA_XFERS   0x048
#define PIKV_REG_DMA_BUSY    0x04C

#define PIKV_CTRL_START      (1u << 0)
#define PIKV_CTRL_RESET      (1u << 1)

#define PIKV_STAT_BUSY       (1u << 0)
#define PIKV_STAT_DONE       (1u << 1)
#define PIKV_STAT_ERR        (1u << 2)

#define PIKV_CMD_NOP         0
#define PIKV_CMD_ROUTE       1
#define PIKV_CMD_COMPRESS    2
#define PIKV_CMD_SCHEDULE    3
#define PIKV_CMD_PREFETCH    4
#define PIKV_CMD_UPD_THETA   5
#define PIKV_CMD_DMA_RD      6
#define PIKV_CMD_DMA_WR      7

#define PIKV_DEFAULT_EXPERTS 64
#define PIKV_DEFAULT_TOP_K   4
#define PIKV_DEFAULT_HIDDEN  128
#define PIKV_DEFAULT_DPRIME  32

typedef struct {
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t hidden_dim;
    uint32_t d_prime;
    int      use_hw_model;   /* 1 = C model, 0 = mmap device */
    const char *device_path;
} pikv_fpga_config_t;

typedef struct {
    uint32_t experts[4];
    uint32_t hits;
    uint32_t misses;
    uint32_t tokens;
    float    theta;
} pikv_fpga_stats_t;

#endif /* PIKV_FPGA_REGS_H */
