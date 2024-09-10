#include "parser.h"
#include "parallel.h"


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
        oneThreadPCG(hit, N, x_min, x_max, y_min, y_max, z_min, z_max);
    } else {
        MultiThreadPCG_static(hit, N, Args.num_threads, 1, x_min, x_max, y_min, y_max, z_min, z_max);
    }
    const double end = omp_get_wtime();

    const float V = (x_max - x_min) * (y_max - y_min) * (z_max - z_min) * (float)hit / (float)N;

    writeV(V, Args.output_file);

    printf("Time(%i thread(s)): %g ms\n", Args.num_threads, (end - start) * 1000.f);

    return 0;
}