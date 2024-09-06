#include "hit.h"

// фигура piriform с параметром a = 2
// (x^4 - 2 * x^3) + 4 * (y^2 + z^2) < 0

bool hit_test(float x, float y, float z) {
    float x2 = x * x;
    return ((x2 * x2 - 2 * x * x2) + 4 * (y * y + z * z)) < 0.0f;
}

// x_min, x_max, y_min, y_max, z_min, z_max
const float* get_axis_range() {
    static const float axis_range[6] = {0.0, 2.0, -0.64952, 0.64952, -0.64952, 0.64952};
    return axis_range;
}
