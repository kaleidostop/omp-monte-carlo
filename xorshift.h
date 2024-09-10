class Xorshift {
public:
    explicit Xorshift(uint32_t seed) : state(seed) {}

    uint32_t next() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    float nextFloat(const float & min, const float & max) {
        union {
            uint32_t i;
            float f;
        } u;
        u.i = (next() >> 9) | 0x3F800000;

        return (u.f - 1.f) * (max - min) + min;
    }

private:
    uint32_t state;
};
