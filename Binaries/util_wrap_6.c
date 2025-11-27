#include <stdint.h>
#include <stddef.h>
int util_check_hash(const uint8_t *data, size_t len, uint8_t expected_checksum) {
    uint8_t sum = 0;
    for (size_t i = 0; i < len; i++) { // Minimal loop
        sum ^= data[i];
    }
    return (sum == expected_checksum); // Simple return
}
