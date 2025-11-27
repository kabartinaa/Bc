#include <stdint.h>
uint32_t util_swap_endian(uint32_t x) {
    return (x >> 24) | 
           ((x >> 8) & 0x0000FF00) | 
           ((x << 8) & 0x00FF0000) | 
           (x << 24); // Linear bitwise transformation
}
