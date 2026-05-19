#ifndef PIKV_FPGA_H
#define PIKV_FPGA_H

#include "pikv_fpga_regs.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pikv_fpga_handle pikv_fpga_handle_t;

pikv_fpga_handle_t *pikv_fpga_open(const pikv_fpga_config_t *cfg);
void pikv_fpga_close(pikv_fpga_handle_t *h);

uint32_t pikv_fpga_reg_read(pikv_fpga_handle_t *h, uint32_t offset);
void pikv_fpga_reg_write(pikv_fpga_handle_t *h, uint32_t offset, uint32_t value);

/**
 * Submit token: writes MMIO, starts CTRL, returns compressed KV + experts.
 * k_in/v_in length = hidden_dim; k_out/v_out length = top_k * d_prime.
 */
int pikv_fpga_process_token(
    pikv_fpga_handle_t *h,
    uint32_t token_id,
    const int16_t *k_in,
    const int16_t *v_in,
    int16_t *k_out,
    int16_t *v_out,
    uint32_t *experts,
    uint32_t expert_count
);

void pikv_fpga_update_theta(pikv_fpga_handle_t *h, float target_hit, float observed_hit);
int pikv_fpga_get_stats(pikv_fpga_handle_t *h, pikv_fpga_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif
