#include <stdint.h>

void simon_four_args(uint32_t *x, uint32_t *y, uint32_t k1, uint32_t k2) {
    *x = (*x + *y) ^ k1;
    *y = (*y - k2) ^ (*x >> 5);
}


