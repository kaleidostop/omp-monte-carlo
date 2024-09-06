#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

struct Arguments {
    bool is_omp;
    int num_threads;
    std::string input_file;
    std::string output_file;
    int chunk_size;

    Arguments() {
        is_omp = true;
        num_threads = 0;
        chunk_size = 1;
    }
};

struct Parser {
    Arguments Parse(int max_threads, int argc, char** argv) {
        Arguments Args = Arguments();
        for (int i = 1; i < argc; i++) {
            char * arg = argv[i];
            if (strncmp(arg, "--input", strlen("--input")) == 0) {
                if (i + 1 < argc) {
                    Args.input_file = argv[++i];
                } else {
                    std::cerr << "Not enough arguments (input)\n";
                    exit(1);
                }
            } else if (strncmp(arg, "--output", strlen("--output")) == 0) {
                if (i + 1 < argc) {
                    Args.output_file = argv[++i];
                } else {
                    std::cerr << "Not enough arguments (output)\n";
                    exit(1);
                }
            }  else if (strncmp(arg, "--omp-threads", strlen("--omp-threads")) == 0) {
                if (i + 1 < argc) {
                    i++;
                    if (strncmp(argv[i], "default", strlen("default")) == 0)  {
                        Args.num_threads = max_threads;
                    } else {
                        try {
                            Args.num_threads = std::stoi(argv[i]);
                        } catch (std::invalid_argument const &ex) {
                            std::cerr << "Invalid argument: " << argv[i] << " " << ex.what() << std::endl;
                            exit(1);
                        } catch (std::out_of_range const &ex) {
                            std::cerr << "Out of range: " << ex.what() << std::endl;
                            exit(1);
                        }
                    }
                } else {
                    std::cerr << "Not enough arguments (omp-threads)\n";
                    exit(1);
                }
            } else if (strncmp(arg, "--no-omp", strlen("--no-omp")) == 0)  {
                Args.is_omp = false;
            } else if (strncmp(arg, "--chunk", strlen("--chunk")) == 0) {
                if (i + 1 < argc) {
                    i++;
                    try {
                        Args.chunk_size = std::stoi(argv[i]);
                    } catch (std::invalid_argument const &ex) {
                        std::cerr << "Invalid argument: " << argv[i] << " " << ex.what() << std::endl;
                        exit(1);
                    }
                } else {
                    std::cerr << "Not enough arguments (input)\n";
                    exit(1);
                }
            } else {
                std::cerr << "Wrong argument: " << arg <<  "\n";
                exit(1);
            }
        }

        if (Args.input_file.empty() || Args.output_file.empty() || (Args.is_omp && Args.num_threads == 0)) {
            std::cerr << "Not enough arguments\n";
            exit(1);
        }

        return Args;
    }
};

long long readN(const std::string & input_file) {
    long long N;

    std::ifstream file(input_file);

    if (!file.is_open()) {
        std::cerr << "Input file not found\n";
        exit(1);
    } else {
        std::string line;
        std::getline(file, line);

        try {
            N = std::stol(line);
            file.close();
        } catch (std::invalid_argument const &ex) {
            std::cerr << "Invalid argument in file: " << N << " " << ex.what() << "\n";
            file.close();
            exit(1);
        } catch (std::out_of_range const &ex) {
            std::cerr << "Out of range number in file: " << N << " " << ex.what() << "\n";
            file.close();
            exit(1);
        }
    }

    return N;
}


void writeV(const float & V, const std::string & output_file) {
    FILE* file = fopen(output_file.c_str(), "w");

    if (file) {
        fprintf(file, "%g\n", V);
        fclose(file);
    } else {
        std::cerr << "Error. File cannot be opened\n";
        exit(1);
    }
}