#include <stdint.h>
#include <stddef.h>
typedef enum { ERROR_NONE, ERROR_NULL, ERROR_LEN } ErrorCode;

ErrorCode util_check_args(const uint8_t *data, size_t len) {
    if (data == NULL) return ERROR_NULL;
    if (len == 0) return ERROR_LEN;
    return ERROR_NONE; // Multiple simple exit paths
}
