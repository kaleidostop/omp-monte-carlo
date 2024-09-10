

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}


float random_float_in_range(pcg32_random_t* rng, const float & min, const float & max) {
    union {
        uint32_t i;
        float f;
    } u;
    u.i = (pcg32_random_r(rng) >> 9) | 0x3F800000;

    return (u.f - 1.f) * (max - min) + min;
}
