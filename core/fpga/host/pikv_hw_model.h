#ifndef PIKV_HW_MODEL_H
#define PIKV_HW_MODEL_H

#include "pikv_fpga_regs.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pikv_hw_model pikv_hw_model_t;

pikv_hw_model_t *pikv_hw_model_create(const pikv_fpga_config_t *cfg);
void pikv_hw_model_destroy(pikv_hw_model_t *m);

void pikv_hw_model_reset(pikv_hw_model_t *m);
uint32_t pikv_hw_model_mmio_read(pikv_hw_model_t *m, uint32_t addr);
void pikv_hw_model_mmio_write(pikv_hw_model_t *m, uint32_t addr, uint32_t val);

/* Process one token through ROUTE->LOOKUP->COMPRESS->SCHEDULE pipeline */
void pikv_hw_model_update_theta(pikv_hw_model_t *m, int32_t delta);

int pikv_hw_model_process_token(
    pikv_hw_model_t *m,
    uint32_t token_id,
    const int16_t *k_in,
    const int16_t *v_in,
    uint32_t hidden_dim,
    int16_t *k_out,
    int16_t *v_out,
    uint32_t d_prime,
    uint32_t *experts_out,
    uint32_t top_k
);

#ifdef __cplusplus
}
#endif

#endif
