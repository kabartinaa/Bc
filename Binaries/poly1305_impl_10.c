#include <stdint.h>

void poly1305_loop_continue(uint64_t *acc, int count) {
    for (int i = 0; i < count; i++) {
        if (i % 2 == 0) {
            *acc += i;
            continue; // Skip the subtraction step
        }
        *acc -= i;
    }
}
