#include <stdint.h>
int util_nested_if(int a, int b) {
    if (a > 10) {
        if (b < 5) {
            return a + b;
        } else {
            return a - b;
        }
    }
    return 0; // Simple branches, no loops
}
