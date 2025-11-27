#include <stdint.h>
#define ROR(x, k) (((x) >> (k)) | ((x) << (32 - (k))))

void simon_macro_rotate(uint32_t *x, uint32_t k) {
    *x = ROR(*x, 3) ^ k;
}
