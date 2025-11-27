#include <stdint.h>

int rsa_recursive_check(uint64_t n, uint64_t divisor, int depth) {
    if (depth > 5) return 0;
    if (n % divisor == 0) return 1;
    return rsa_recursive_check(n + 1, divisor, depth + 1); // Arithmetic recursion
}
