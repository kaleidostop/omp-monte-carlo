#include <omp.h>
#include <random>

#include "hit.h"
#include "pcg.h"
#include "xorshift.h"

void oneThreadPCG(long long & hit, const long long N, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
    static thread_local pcg32_random_t rng_x, rng_y, rng_z;

    thread_local const int thread_n = omp_get_thread_num();

    rng_x.state = 20u + thread_n;
    rng_x.inc = (15u + thread_n * 123) | 1;

    rng_y.state = 35u + omp_get_wtime() + omp_get_thread_num();
    rng_y.inc = (111u + thread_n * 11) | 1;

    rng_z.state = 11u + omp_get_wtime() + omp_get_thread_num();
    rng_z.inc = (129u + thread_n * 15) | 1;

    float x, y, z;
    x = random_float_in_range(&rng_x, x_min, x_max);
    y = random_float_in_range(&rng_y, y_min, y_max);
    z = random_float_in_range(&rng_z, z_min, z_max);

    for (int i = 0; i < N; ++i) {
        if (i % 3 == 0) {
            x = random_float_in_range(&rng_x, x_min, x_max);
        } else if (i % 3 == 1) {
            y = random_float_in_range(&rng_y, y_min, y_max);
        } else {
            z = random_float_in_range(&rng_z, z_min, z_max);
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

#pragma omp for schedule(static, chunk_size) nowait
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

void MultiThreadMT19937_no_chunk(long long & hit, const long long N, int n_threads, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
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

#pragma omp for schedule(static) nowait
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


void MultiThreadPCG_static(long long & hit, const long long N, int n_threads, int chunk_size, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
#pragma omp parallel num_threads(n_threads)
    {
        static thread_local pcg32_random_t rng_x, rng_y, rng_z;

        thread_local const int thread_n = omp_get_thread_num();

        rng_x.state = 20u + thread_n;
        rng_x.inc = (15u + thread_n * 123) | 1;

        rng_y.state = 35u + omp_get_wtime() + omp_get_thread_num();
        rng_y.inc = (111u + thread_n * 11) | 1;

        rng_z.state = 11u + omp_get_wtime() + omp_get_thread_num();
        rng_z.inc = (129u + thread_n * 15) | 1;

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

void MultiThreadPCG_static_no_chunk(long long & hit, const long long N, int n_threads, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
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

#pragma omp for schedule(static) nowait
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


void MultiThreadPCG_dynamic(long long & hit, const long long N, int n_threads, int chunk_size, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
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

#pragma omp for schedule(dynamic, chunk_size) nowait
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

void MultiThreadPCG_dynamic_no_chunk(long long & hit, const long long N, int n_threads, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
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

#pragma omp for schedule(dynamic) nowait
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


void MultiThreadXorshift(long long & hit, const long long N, int n_threads, int chunk_size, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
#pragma omp parallel num_threads(n_threads)
    {
        thread_local const int thread_n = omp_get_thread_num();

        static thread_local Xorshift rng_x(123456789 + thread_n), rng_y(353553 + thread_n), rng_z(11111 + thread_n);

        long long local_hit = 0;
        float x, y, z;
        x = rng_x.nextFloat(x_min, x_max);
        y = rng_y.nextFloat(y_min, y_max);
        z = rng_z.nextFloat(z_min, z_max);

#pragma omp for schedule(static, chunk_size) nowait
        for (int i = 0; i < N; ++i) {
            if (i % 3 == 0) {
                x = rng_x.nextFloat(x_min, x_max);
            } else if (i % 3 == 1) {
                y = rng_y.nextFloat(y_min, y_max);
            } else {
                z = rng_z.nextFloat(z_min, z_max);
            }

            if (hit_test(x, y, z)) {
                ++local_hit;
            }
        }
#pragma omp atomic
        hit += local_hit;
    }
}

void MultiThreadXorshift_no_chunk(long long & hit, const long long N, int n_threads, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max) {
#pragma omp parallel num_threads(n_threads)
    {
        thread_local const int thread_n = omp_get_thread_num();

        static thread_local Xorshift rng_x(123456789 + thread_n), rng_y(353553 + thread_n), rng_z(11111 + thread_n);

        long long local_hit = 0;
        float x, y, z;
        x = rng_x.nextFloat(x_min, x_max);
        y = rng_y.nextFloat(y_min, y_max);
        z = rng_z.nextFloat(z_min, z_max);

#pragma omp for schedule(static) nowait
        for (int i = 0; i < N; ++i) {
            if (i % 3 == 0) {
                x = rng_x.nextFloat(x_min, x_max);
            } else if (i % 3 == 1) {
                y = rng_y.nextFloat(y_min, y_max);
            } else {
                z = rng_z.nextFloat(z_min, z_max);
            }

            if (hit_test(x, y, z)) {
                ++local_hit;
            }
        }
#pragma omp atomic
        hit += local_hit;
    }
}

