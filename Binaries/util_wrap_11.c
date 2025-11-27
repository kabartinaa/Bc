#include <stdint.h>
#include <stddef.h>
void util_clear_array(uint8_t *arr, size_t len) {
    for (size_t i = 0; i < len; i++) {
        arr[i] = 0; // Minimal loop, simple assignment
    }
}
