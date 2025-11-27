#include <stdint.h>
#define HUGE_BUFFER 4096 

void rsa_large_init(uint8_t *buffer) {
    // Forces the compiler to manage a very large static/stack buffer
    for (int i = 0; i < HUGE_BUFFER; i++) {
        buffer[i] = (uint8_t)(i % 256);
    }
}
