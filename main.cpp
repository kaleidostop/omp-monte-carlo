#include "parser.h"
#include "hit.h"

#include <iostream>
#include <omp.h>
#include <random>
#include "pcg.h"


void oneThread(long long & hit, const long long N, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
    static std::mt19937 generator(omp_get_wtime());

    std::uniform_real_distribution<float> x_distribution(x_min, x_max);
    std::uniform_real_distribution<float> y_distribution(y_min, y_max);
    std::uniform_real_distribution<float> z_distribution(z_min, z_max);

    float x, y, z;
    x = x_distribution(generator);
    y = y_distribution(generator);
    z = z_distribution(generator);
    for (int i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            x = x_distribution(generator);
        } else if (i % 3 == 1) {
            y = y_distribution(generator);
        } else {
            z = z_distribution(generator);
        }
        if (hit_test(x, y, z)) {
            ++hit;
        }
    }
}

void MultiThreadSimple(long long & hit, const long long N, int n_threads, int chunk_size, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
#pragma omp parallel num_threads(n_threads)
    {
        static thread_local std::mt19937 generator(omp_get_wtime() + omp_get_thread_num());

        std::uniform_real_distribution<float> x_distribution(x_min, x_max);
        std::uniform_real_distribution<float> y_distribution(y_min, y_max);
        std::uniform_real_distribution<float> z_distribution(z_min, z_max);

        long long local_hit = 0;
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < N; ++i) {
            float x = x_distribution(generator);
            float y = y_distribution(generator);
            float z = z_distribution(generator);
            if (hit_test(x, y, z)) {
                ++local_hit;
            }
        }
#pragma omp atomic
        hit += local_hit;
    }
}

void MultiThreadMT19937(long long & hit, const long long N, int n_threads, int chunk_size, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
#pragma omp parallel num_threads(n_threads)
    {
        static thread_local std::mt19937 generator(omp_get_wtime() + omp_get_thread_num());

        std::uniform_real_distribution<float> x_distribution(x_min, x_max);
        std::uniform_real_distribution<float> y_distribution(y_min, y_max);
        std::uniform_real_distribution<float> z_distribution(z_min, z_max);

        long long local_hit = 0;
        float x, y, z;
        x = x_distribution(generator);
        y = y_distribution(generator);
        z = z_distribution(generator);

#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < N; ++i) {
            if (i % 3 == 0) {
                x = x_distribution(generator);
            } else if (i % 3 == 1) {
                y = y_distribution(generator);
            } else {
                z = z_distribution(generator);
            }

            if (hit_test(x, y, z)) {
                ++local_hit;
            }
        }
#pragma omp atomic
        hit += local_hit;
    }
}


void MultiThreadPCG(long long & hit, const long long N, int n_threads, int chunk_size, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
#pragma omp parallel num_threads(n_threads)
    {
        static thread_local pcg32_random_t rng_x, rng_y, rng_z;

        thread_local const int thread_n = omp_get_thread_num();

        rng_x.state = 20u + thread_n;
        rng_x.inc = (15u + thread_n * 2) | 1;

        rng_y.state = 35u + omp_get_wtime() + omp_get_thread_num();
        rng_y.inc = (111u + thread_n * 4) | 1;

        rng_z.state = 11u + omp_get_wtime() + omp_get_thread_num();
        rng_z.inc = (129u + thread_n * 5) | 1;

        long long local_hit = 0;
        float x, y, z;
        x = random_float_in_range(&rng_x, x_min, x_max);
        y = random_float_in_range(&rng_y, y_min, y_max);
        z = random_float_in_range(&rng_z, z_min, z_max);

#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < N; ++i) {
            if (i % 3 == 0) {
                x = random_float_in_range(&rng_x, x_min, x_max);
            } else if (i % 3 == 1) {
                y = random_float_in_range(&rng_y, y_min, y_max);
            } else {
                z = random_float_in_range(&rng_z, z_min, z_max);
            }

            if (hit_test(x, y, z)) {
                ++local_hit;
            }
        }
#pragma omp atomic
        hit += local_hit;
    }
}


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

    float nextFloat() {
        return static_cast<float>(next()) / static_cast<float>(UINT32_MAX);
    }

private:
    uint32_t state;
};

int main(int argc, char** argv) {
    Parser P = Parser();
    Arguments Args = P.Parse(omp_get_max_threads(), argc, argv);

    const long long N = readN(Args.input_file);

    const float* axis_range = get_axis_range();
    const float x_min = axis_range[0], x_max = axis_range[1],
            y_min = axis_range[2], y_max = axis_range[3],
            z_min = axis_range[4], z_max = axis_range[5];

    long long hit = 0;

    const double start = omp_get_wtime();
    if (!Args.is_omp) {
        oneThread(hit, N, x_min, x_max, y_min, y_max, z_min, z_max);
    } else {
        MultiThreadPCG(hit, N, Args.num_threads, Args.chunk_size, x_min, x_max, y_min, y_max, z_min, z_max);
    }
    const double end = omp_get_wtime();

    const float V = (x_max - x_min) * (y_max - y_min) * (z_max - z_min) * (float)hit / (float)N;

    writeV(V, Args.output_file);

    printf("Time(%i thread(s)): %g ms\n", Args.num_threads, (end - start) * 1000.f);

    return 0;
}