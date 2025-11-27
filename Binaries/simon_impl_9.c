#include <stdint.h>

void simon_break_loop(uint32_t *state, int rounds) {
    for (int i = 0; i < rounds; i++) {
        *state ^= i;
        if (i == rounds / 2) {
            break; // Conditional exit
        }
    }
}
