#include "pikv_fpga.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    pikv_fpga_config_t cfg = {
        .num_experts = 64,
        .top_k = 4,
        .hidden_dim = 128,
        .d_prime = 32,
        .use_hw_model = 1,
        .device_path = NULL,
    };

    pikv_fpga_handle_t *h = pikv_fpga_open(&cfg);
    if (!h) {
        fprintf(stderr, "pikv_fpga_open failed\n");
        return 1;
    }

    int16_t k[128], v[128], k_out[128], v_out[128];
    uint32_t experts[4];

    for (int i = 0; i < 128; i++) {
        k[i] = (int16_t)(i & 0xFF);
        v[i] = (int16_t)((i * 3) & 0xFF);
    }

    for (uint32_t t = 0; t < 8; t++) {
        if (pikv_fpga_process_token(h, t, k, v, k_out, v_out, experts, 4) != 0) {
            fprintf(stderr, "process_token failed at t=%u\n", t);
            return 1;
        }
        printf("t=%u experts=[%u,%u,%u,%u] k_out[0]=%d\n",
               t, experts[0], experts[1], experts[2], experts[3], k_out[0]);
    }

    pikv_fpga_stats_t st;
    pikv_fpga_get_stats(h, &st);
    printf("hits=%u misses=%u\n", st.hits, st.misses);

    pikv_fpga_close(h);
    printf("test_pikv_fpga OK\n");
    return 0;
}
