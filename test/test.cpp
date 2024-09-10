#include "../parallel.h"

#include <iostream>
#include <fstream>
#include <numeric>

static const long long N = 100000000;

static const float* axis_range = get_axis_range();
static const float x_min = axis_range[0], x_max = axis_range[1],
        y_min = axis_range[2], y_max = axis_range[3],
        z_min = axis_range[4], z_max = axis_range[5];

template<typename Func, typename... Args>
double measureTime(Func func, Args... args) {
    double start = omp_get_wtime();
    func(args...);
    double end = omp_get_wtime();
    return (end - start) * 1000.f;
}

void testOneThread() {
    long long hit = 0;
    std::vector<double> times_oneThread;
    for (int i = 0; i < 10; ++i) {
        times_oneThread.push_back(measureTime(oneThreadPCG, hit, N, x_min, x_max, y_min, y_max, z_min, z_max));
    }
    double avg_oneThread = std::accumulate(times_oneThread.begin(), times_oneThread.end(), 0.0) / times_oneThread.size();
    std::ofstream oneThreadFile("../test/onethread.txt");
    oneThreadFile << avg_oneThread << std::endl;
    oneThreadFile.close();
    std::cout << "One thread tested\n";
}

void test_PCG_static_num_threads() {
    long long hit = 0;

    std::ofstream pcgStaticNumThreadsFile("../test/pcg_static_num_threads.txt");
    for (int num_threads = 1; num_threads <= 5; ++num_threads) {
        std::vector<double> times_pcgStaticNumThreads;
        for (int i = 0; i < 4; ++i) {
            times_pcgStaticNumThreads.push_back(measureTime(MultiThreadPCG_static_no_chunk, hit, N, num_threads, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_pcgStaticNumThreads = std::accumulate(times_pcgStaticNumThreads.begin(), times_pcgStaticNumThreads.end(), 0.0) / times_pcgStaticNumThreads.size();

        pcgStaticNumThreadsFile << avg_pcgStaticNumThreads << "\n";
    }
    pcgStaticNumThreadsFile.close();
    std::cout << "PCG static num threads tested\n";
}

void test_PCG_static_chunk_size() {
    long long hit = 0;
    std::vector<int> chunks = {1, 2, 4, 6, 8, 10, 100, 1000, 10000, 1000000, 25000000, 50000000, 100000000};

    std::ofstream pcgStaticChunkSizeFile("../test/pcg_static_chunk_size.txt");
    for (int chunk_size : chunks) {
        std::vector<double> times_pcgStaticChunkSize;
        for (int i = 0; i < 4; ++i) {
            times_pcgStaticChunkSize.emplace_back(measureTime(MultiThreadPCG_static, hit, N, 4, chunk_size, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_pcgStaticChunkSize = std::accumulate(times_pcgStaticChunkSize.begin(), times_pcgStaticChunkSize.end(), 0.0) / times_pcgStaticChunkSize.size();

        pcgStaticChunkSizeFile << chunk_size << " : " << avg_pcgStaticChunkSize << "\n";
    }
    pcgStaticChunkSizeFile.close();
    std::cout << "PCG static chunk size tested\n";
}

void test_PCG_static_chunk_size_one() {
    long long hit = 0;
    std::vector<int> chunks = {1, 2, 4, 6, 8, 10, 100, 1000, 10000, 1000000, 25000000, 50000000, 100000000};

    std::ofstream pcgStaticChunkSizeFile("../test/pcg_static_chunk_size_one.txt");
    for (int chunk_size : chunks) {
        std::vector<double> times_pcgStaticChunkSize;
        for (int i = 0; i < 4; ++i) {
            times_pcgStaticChunkSize.emplace_back(measureTime(MultiThreadPCG_static, hit, N, 1, chunk_size, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_pcgStaticChunkSize = std::accumulate(times_pcgStaticChunkSize.begin(), times_pcgStaticChunkSize.end(), 0.0) / times_pcgStaticChunkSize.size();

        pcgStaticChunkSizeFile << chunk_size << " : " << avg_pcgStaticChunkSize << "\n";
    }
    pcgStaticChunkSizeFile.close();
    std::cout << "PCG static chunk size tested\n";
}

void test_PCG_dynamic_chunk_size() {
    long long hit = 0;
    std::vector<int> chunks = {1, 2, 4, 6, 8, 10, 100, 1000, 10000, 1000000, 25000000, 50000000, 100000000};

    std::ofstream pcgDynamicChunkSizeFile("../test/pcg_dynamic_chunk_size.txt");
    for (int chunk_size : chunks) {
        std::vector<double> times_pcgDynamicChunkSize;
        for (int i = 0; i < 4; ++i) {
            times_pcgDynamicChunkSize.emplace_back(measureTime(MultiThreadPCG_dynamic, hit, N, 4, chunk_size, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_pcgDynamicChunkSize = std::accumulate(times_pcgDynamicChunkSize.begin(), times_pcgDynamicChunkSize.end(), 0.0) / times_pcgDynamicChunkSize.size();

        pcgDynamicChunkSizeFile << chunk_size << " : " << avg_pcgDynamicChunkSize << "\n";
    }
    pcgDynamicChunkSizeFile.close();
    std::cout << "PCG dynamic chunk size tested\n";
}

void test_PCG_dynamic_chunk_size_one() {
    long long hit = 0;
    std::vector<int> chunks = {1, 2, 4, 6, 8, 10, 100, 1000, 10000, 1000000, 25000000, 50000000, 100000000};

    std::ofstream pcgDynamicChunkSizeFile("../test/pcg_dynamic_chunk_size_one.txt");
    for (int chunk_size : chunks) {
        std::vector<double> times_pcgDynamicChunkSize;
        for (int i = 0; i < 4; ++i) {
            times_pcgDynamicChunkSize.emplace_back(measureTime(MultiThreadPCG_dynamic, hit, N, 1, chunk_size, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_pcgDynamicChunkSize = std::accumulate(times_pcgDynamicChunkSize.begin(), times_pcgDynamicChunkSize.end(), 0.0) / times_pcgDynamicChunkSize.size();

        pcgDynamicChunkSizeFile << chunk_size << " : " << avg_pcgDynamicChunkSize << "\n";
    }
    pcgDynamicChunkSizeFile.close();
    std::cout << "PCG dynamic chunk size tested\n";
}


void test_PCG_dynamic_num_threads() {
    long long hit = 0;

    std::ofstream pcgDynamicNumThreadsFile("../test/pcg_dynamic_num_threads.txt");
    for (int num_threads = 1; num_threads <= 5; ++num_threads) {
        std::vector<double> times_pcgDynamicNumThreads;
        for (int i = 0; i < 4; ++i) {
            times_pcgDynamicNumThreads.push_back(measureTime(MultiThreadPCG_dynamic_no_chunk, hit, N, num_threads, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_pcgDynamicNumThreads = std::accumulate(times_pcgDynamicNumThreads.begin(), times_pcgDynamicNumThreads.end(), 0.0) / times_pcgDynamicNumThreads.size();

        pcgDynamicNumThreadsFile << avg_pcgDynamicNumThreads << "\n";
    }
    pcgDynamicNumThreadsFile.close();
    std::cout << "PCG dynamic num threads tested\n";
}

void test_MT_num_threads() {
    long long hit = 0;

    std::ofstream MTNumThreadsFile("../test/mt_num_threads.txt");
    for (int num_threads = 1; num_threads <= 5; ++num_threads) {
        std::vector<double> times_MTNumThreads;
        for (int i = 0; i < 4; ++i) {
            times_MTNumThreads.push_back(measureTime(MultiThreadMT19937_no_chunk, hit, N, num_threads, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_mtNumThreads = std::accumulate(times_MTNumThreads.begin(), times_MTNumThreads.end(), 0.0) / times_MTNumThreads.size();

        MTNumThreadsFile << avg_mtNumThreads << "\n";
    }
    MTNumThreadsFile.close();
    std::cout << "MT num threads tested\n";
}

void test_MT_chunk_size() {
    long long hit = 0;
    std::vector<int> chunks = {1, 2, 4, 6, 8, 10, 100, 1000, 10000, 100000, 1000000, 2500000, 5000000, 10000000};

    std::ofstream MTChunkSizeFile("../test/mt_chunk_size.txt");

    for (int chunk_size : chunks) {
        std::vector<double> times_MTChunkSize;
        for (int i = 0; i < 4; ++i) {
            times_MTChunkSize.push_back(measureTime(MultiThreadMT19937, hit, N, 4, chunk_size, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_mtChunkSize = std::accumulate(times_MTChunkSize.begin(), times_MTChunkSize.end(), 0.0) / times_MTChunkSize.size();

        MTChunkSizeFile << avg_mtChunkSize << "\n";
    }
    MTChunkSizeFile.close();
    std::cout << "MT chunk size tested\n";
}


void test_Xorshift_num_threads() {
    long long hit = 0;

    std::ofstream XorshiftNumThreadsFile("../test/xorshift_num_threads.txt");
    for (int num_threads = 1; num_threads <= 5; ++num_threads) {
        std::vector<double> times_XorshiftNumThreads;
        for (int i = 0; i < 4; ++i) {
            times_XorshiftNumThreads.push_back(measureTime(MultiThreadXorshift_no_chunk, hit, N, num_threads, x_min, x_max, y_min, y_max, z_min, z_max));
        }
        double avg_pcgStaticNumThreads = std::accumulate(times_XorshiftNumThreads.begin(), times_XorshiftNumThreads.end(), 0.0) / times_XorshiftNumThreads.size();

        XorshiftNumThreadsFile << avg_pcgStaticNumThreads << "\n";
    }
    XorshiftNumThreadsFile.close();
    std::cout << "Xorshift num threads tested\n";
}

int main() {
    testOneThread();
    test_MT_num_threads();
    test_Xorshift_num_threads();
    test_PCG_static_num_threads();
    test_PCG_dynamic_num_threads();
    test_PCG_static_chunk_size();
    test_PCG_dynamic_chunk_size();
    test_PCG_static_chunk_size_one();
    test_PCG_dynamic_chunk_size_one();
}
