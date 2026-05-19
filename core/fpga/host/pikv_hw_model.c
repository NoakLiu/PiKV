/**
 * Cycle-accurate C model of PiKV-FPGA RTL (for sim / libpikv_fpga.so without hardware).
 * Mirrors: page_table, score_fuse, codec_rho, scheduler, ctrl FSM.
 */
#include "pikv_hw_model.h"
#include <stdlib.h>
#include <string.h>

#define MAX_EXPERTS 256
#define MAX_SHARD   512
struct pikv_hw_model {
    pikv_fpga_config_t cfg;
    uint32_t regs[64];
    uint32_t stored_token[MAX_EXPERTS * MAX_SHARD];
    uint32_t stored_addr[MAX_EXPERTS * MAX_SHARD];
    uint16_t expert_miss[MAX_EXPERTS];
    uint32_t next_addr;
    int32_t  theta;
};

static uint32_t shard_idx(uint32_t token, uint32_t expert, uint32_t shard_size) {
    return ((token % shard_size) ^ (expert % shard_size)) % shard_size;
}

static uint32_t flat_idx(uint32_t expert, uint32_t shard, uint32_t num_experts, uint32_t shard_size) {
    return (expert % num_experts) * shard_size + shard;
}

pikv_hw_model_t *pikv_hw_model_create(const pikv_fpga_config_t *cfg) {
    pikv_hw_model_t *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->cfg = *cfg;
    if (m->cfg.num_experts == 0) m->cfg.num_experts = PIKV_DEFAULT_EXPERTS;
    if (m->cfg.top_k == 0) m->cfg.top_k = PIKV_DEFAULT_TOP_K;
    if (m->cfg.hidden_dim == 0) m->cfg.hidden_dim = PIKV_DEFAULT_HIDDEN;
    if (m->cfg.d_prime == 0) m->cfg.d_prime = PIKV_DEFAULT_DPRIME;
    pikv_hw_model_reset(m);
    return m;
}

void pikv_hw_model_destroy(pikv_hw_model_t *m) {
    free(m);
}

void pikv_hw_model_reset(pikv_hw_model_t *m) {
    memset(m->regs, 0, sizeof(m->regs));
    memset(m->stored_token, 0, sizeof(m->stored_token));
    memset(m->stored_addr, 0, sizeof(m->stored_addr));
    memset(m->expert_miss, 0, sizeof(m->expert_miss));
    m->next_addr = 1;
    m->theta = 0;
}

uint32_t pikv_hw_model_mmio_read(pikv_hw_model_t *m, uint32_t addr) {
    switch (addr) {
        case PIKV_REG_STATUS:
            return m->regs[PIKV_REG_STATUS >> 2];
        case PIKV_REG_HIT:
            return m->regs[PIKV_REG_HIT >> 2];
        case PIKV_REG_MISS:
            return m->regs[PIKV_REG_MISS >> 2];
        case PIKV_REG_THETA:
            return (uint32_t)m->theta;
        case PIKV_REG_EXPERTS:
            return m->regs[PIKV_REG_EXPERTS >> 2];
        default:
            return 0;
    }
}

void pikv_hw_model_mmio_write(pikv_hw_model_t *m, uint32_t addr, uint32_t val) {
    uint32_t idx = addr >> 2;
    if (idx < 64) m->regs[idx] = val;
    if (addr == PIKV_REG_CTRL && (val & PIKV_CTRL_RESET)) {
        pikv_hw_model_reset(m);
    }
}

static void route_experts(uint32_t token_id, uint32_t num_experts, uint32_t top_k,
                          uint32_t *out) {
    uint32_t base = token_id % num_experts;
    uint32_t offs[] = {0, 7, 13, 19, 23, 29, 31, 37};
    for (uint32_t i = 0; i < top_k && i < 8; i++)
        out[i] = (base + offs[i]) % num_experts;
}

static void lora_compress(const int16_t *k, const int16_t *v, uint32_t d, uint32_t d_prime,
                          int16_t *k_out, int16_t *v_out) {
    uint32_t r = 4;
    if (r > d) r = d;
    for (uint32_t i = 0; i < d_prime; i++) {
        k_out[i] = (int16_t)(k[i % r] >> 2);
        v_out[i] = (int16_t)(v[i % r] >> 2);
    }
}

void pikv_hw_model_update_theta(pikv_hw_model_t *m, int32_t delta) {
    if (!m) return;
    m->theta += delta;
    pikv_hw_model_mmio_write(m, PIKV_REG_THETA, (uint32_t)m->theta);
}

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
    uint32_t top_k)
{
    uint32_t e, num_e = m->cfg.num_experts;
    uint32_t shard_size = 256;
    uint32_t experts[8];
    uint32_t hits = m->regs[PIKV_REG_HIT >> 2];
    uint32_t misses = m->regs[PIKV_REG_MISS >> 2];

    route_experts(token_id, num_e, top_k, experts);
    for (uint32_t i = 0; i < top_k; i++)
        experts_out[i] = experts[i];

    for (uint32_t i = 0; i < top_k; i++) {
        e = experts[i];
        uint32_t sh = shard_idx(token_id, e, shard_size);
        uint32_t fi = flat_idx(e, sh, num_e, shard_size);
        int hit = (m->stored_token[fi] == token_id && m->stored_addr[fi] != 0);
        if (hit) hits++;
        else {
            misses++;
            m->expert_miss[e % MAX_EXPERTS]++;
        }
        lora_compress(k_in, v_in, hidden_dim, d_prime, k_out + i * d_prime, v_out + i * d_prime);
        m->stored_token[fi] = token_id;
        m->stored_addr[fi] = m->next_addr++;
    }

    m->regs[PIKV_REG_HIT >> 2] = hits;
    m->regs[PIKV_REG_MISS >> 2] = misses;
    m->regs[PIKV_REG_EXPERTS >> 2] =
        (experts[0] & 0x3F) | ((experts[1] & 0x3F) << 8) |
        ((experts[2] & 0x3F) << 16) | ((experts[3] & 0x3F) << 24);
    m->regs[PIKV_REG_STATUS >> 2] = PIKV_STAT_DONE;
    return 0;
}
