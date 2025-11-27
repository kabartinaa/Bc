#include <stdint.h>
// Uses the Extended Euclidean Algorithm approach
uint64_t rsa_mod_inverse_proxy(uint64_t a, uint64_t m) {
    uint64_t m0 = m, t, q;
    uint64_t x0 = 0, x1 = 1;

    if (m == 1) return 0;

    while (a > 1) { // Main loop
        q = a / m;
        t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0; // Arithmetic operations
        x1 = t;
    }

    if (x1 < 0) x1 += m0;
    return x1;
}
