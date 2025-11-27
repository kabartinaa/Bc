#include <stdint.h>

void speck_key_schedule_proxy(uint32_t *key_state) {
    uint32_t temp = key_state[0];
    key_state[0] = (key_state[1] + key_state[0]) ^ 0xAAAA;
    key_state[1] = (key_state[1] >> 8) | (key_state[1] << 24);
}
