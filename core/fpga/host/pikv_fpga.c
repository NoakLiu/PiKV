#include "pikv_fpga.h"
#include "pikv_hw_model.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MMIO_MAP_SIZE 0x1000

struct pikv_fpga_handle {
    pikv_fpga_config_t cfg;
    int fd;
    volatile uint32_t *mmio;
    pikv_hw_model_t *model;
};

static int use_model(const pikv_fpga_config_t *cfg) {
    if (cfg->use_hw_model) return 1;
    if (!cfg->device_path) return 1;
    return access(cfg->device_path, R_OK | W_OK) != 0;
}

pikv_fpga_handle_t *pikv_fpga_open(const pikv_fpga_config_t *cfg) {
    pikv_fpga_config_t c;
    pikv_fpga_handle_t *h = calloc(1, sizeof(*h));
    if (!h) return NULL;

    memset(&c, 0, sizeof(c));
    if (cfg) c = *cfg;
    if (c.num_experts == 0) c.num_experts = PIKV_DEFAULT_EXPERTS;
    if (c.top_k == 0) c.top_k = PIKV_DEFAULT_TOP_K;
    if (c.hidden_dim == 0) c.hidden_dim = PIKV_DEFAULT_HIDDEN;
    if (c.d_prime == 0) c.d_prime = PIKV_DEFAULT_DPRIME;
    if (!c.device_path) c.device_path = "/dev/pikv_fpga0";
    h->cfg = c;

    if (use_model(&c)) {
        h->cfg.use_hw_model = 1;
        h->model = pikv_hw_model_create(&h->cfg);
        if (!h->model) {
            free(h);
            return NULL;
        }
        return h;
    }

    h->fd = open(c.device_path, O_RDWR);
    if (h->fd < 0) {
        perror("pikv_fpga_open");
        free(h);
        return NULL;
    }
    h->mmio = mmap(NULL, MMIO_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd, 0);
    if (h->mmio == MAP_FAILED) {
        perror("mmap");
        close(h->fd);
        free(h);
        return NULL;
    }
    return h;
}

void pikv_fpga_close(pikv_fpga_handle_t *h) {
    if (!h) return;
    if (h->model) pikv_hw_model_destroy(h->model);
    if (h->mmio && h->mmio != MAP_FAILED) munmap((void *)h->mmio, MMIO_MAP_SIZE);
    if (h->fd >= 0) close(h->fd);
    free(h);
}

uint32_t pikv_fpga_reg_read(pikv_fpga_handle_t *h, uint32_t offset) {
    if (!h) return 0;
    if (h->model) return pikv_hw_model_mmio_read(h->model, offset);
    return h->mmio[offset >> 2];
}

void pikv_fpga_reg_write(pikv_fpga_handle_t *h, uint32_t offset, uint32_t value) {
    if (!h) return;
    if (h->model) {
        pikv_hw_model_mmio_write(h->model, offset, value);
        return;
    }
    h->mmio[offset >> 2] = value;
}

int pikv_fpga_process_token(
    pikv_fpga_handle_t *h,
    uint32_t token_id,
    const int16_t *k_in,
    const int16_t *v_in,
    int16_t *k_out,
    int16_t *v_out,
    uint32_t *experts,
    uint32_t expert_count)
{
    if (!h || !k_in || !v_in || !k_out || !v_out || !experts) return -1;

    if (h->model) {
        return pikv_hw_model_process_token(
            h->model, token_id, k_in, v_in, h->cfg.hidden_dim,
            k_out, v_out, h->cfg.d_prime, experts, expert_count);
    }

    /* Hardware path: MMIO sequence */
    pikv_fpga_reg_write(h, PIKV_REG_TOKEN, token_id);
    pikv_fpga_reg_write(h, PIKV_REG_CMD, PIKV_CMD_ROUTE);
    pikv_fpga_reg_write(h, PIKV_REG_CTRL, PIKV_CTRL_START);

    for (int i = 0; i < 10000; i++) {
        uint32_t st = pikv_fpga_reg_read(h, PIKV_REG_STATUS);
        if (st & PIKV_STAT_DONE) break;
        usleep(1);
    }

    uint32_t exp_reg = pikv_fpga_reg_read(h, PIKV_REG_EXPERTS);
    experts[0] = exp_reg & 0x3F;
    experts[1] = (exp_reg >> 8) & 0x3F;
    experts[2] = (exp_reg >> 16) & 0x3F;
    experts[3] = (exp_reg >> 24) & 0x3F;
    (void)expert_count;
    return 0;
}

void pikv_fpga_update_theta(pikv_fpga_handle_t *h, float target_hit, float observed_hit) {
    if (!h) return;
    int32_t delta = (int32_t)((target_hit - observed_hit) * 256.0f);
    pikv_fpga_reg_write(h, PIKV_REG_ARG0, (uint32_t)delta);
    pikv_fpga_reg_write(h, PIKV_REG_CMD, PIKV_CMD_UPD_THETA);
    pikv_fpga_reg_write(h, PIKV_REG_CTRL, PIKV_CTRL_START);
    if (h->model)
        pikv_hw_model_update_theta(h->model, delta);
}

int pikv_fpga_get_stats(pikv_fpga_handle_t *h, pikv_fpga_stats_t *stats) {
    if (!h || !stats) return -1;
    memset(stats, 0, sizeof(*stats));
    stats->hits = pikv_fpga_reg_read(h, PIKV_REG_HIT);
    stats->misses = pikv_fpga_reg_read(h, PIKV_REG_MISS);
    uint32_t exp = pikv_fpga_reg_read(h, PIKV_REG_EXPERTS);
    stats->experts[0] = exp & 0x3F;
    stats->experts[1] = (exp >> 8) & 0x3F;
    stats->experts[2] = (exp >> 16) & 0x3F;
    stats->experts[3] = (exp >> 24) & 0x3F;
    stats->theta = (float)pikv_fpga_reg_read(h, PIKV_REG_THETA) / 256.0f;
    return 0;
}
