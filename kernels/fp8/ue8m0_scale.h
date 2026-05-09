#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

inline int8_t compute_ue8m0_exp(float amax, float destmax) {
    float s = amax / destmax;
    const float min_scale = 0x1p-127f;
    if (s < min_scale) s = min_scale;
    uint32_t bits;
    std::memcpy(&bits, &s, sizeof(bits));

    // IEEE 754: (sign, exponent, mantissa): (1 bit,8 bits, 23 bits)
    int exp = ((bits >> 23) & 0xFF) - 127; // move exponent bits to the right and mask out
    uint32_t mantissa = bits & 0x7FFFFF;

    if (mantissa != 0) {
        exp += 1; // rounds up to the next power of two (e8m0 has no mantissa)
    }

    exp = std::max(exp, -127);
    exp = std::min(exp, 127);
    return static_cast<int8_t>(exp);
}

inline float decode_ue8m0_scale(int8_t exp) {
    uint32_t bits = static_cast<uint32_t>((exp + 127) << 23); // +127 for exponent bias
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    return scale;
}

inline float apply_ue8m0_scale(float value, int8_t exp) {
    return std::ldexp(value, -exp);
}
