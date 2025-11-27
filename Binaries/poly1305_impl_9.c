#include <stdint.h>

void poly1305_nested_checks(uint64_t *acc, uint8_t flag) {
    if (flag & 0x01) {
        *acc += 1;
        if (flag & 0x02) { // Nested check 1
            *acc *= 2;
            if (flag & 0x04) { // Nested check 2
                *acc %= 0x10000;
            }
        }
    }
}
