#include <iostream>
#include <vector>
#include "ue8m0_scale.h"

int main() {
    std::vector<float> block = {1.2f, -3.7f, 0.5f, 2.1f};
    float amax = 0.0f;
    for (float v : block) {
        amax = std::max(amax, std::abs(v));
    }

    float destmax = 448.0f; // e4m3 max representable value
    int8_t exp = compute_ue8m0_exp(amax, destmax);
    std::cout << "Stored exponent: " << int(exp) << std::endl;

    float scale = decode_ue8m0_scale(exp);
    std::cout << "Decoded scale (2^E): " << scale << std::endl;

    std::cout << "Scaled values (V / scale):" << std::endl;
    for (float v : block) {
        float q = apply_ue8m0_scale(v, exp);
        std::cout << q << " ";
    }
    std::cout << std::endl;
    return 0;
}
