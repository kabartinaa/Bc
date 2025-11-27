#include <stdint.h>
#include <stddef.h>
void util_ptr_to_ptr(uint32_t **ptr_ref, uint32_t val) {
    if (*ptr_ref != NULL) {
        **ptr_ref = val; // Double dereference
    }
}
