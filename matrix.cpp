#include <vector>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <random>

class Matrix {
private:
    std::vector<double> data;
    size_t rows, cols;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        
        #pragma omp parallel for
        for (size_t i = 0; i < rows * cols; ++i) {
            data[i] = dis(gen);
        }
    }

    double& operator()(size_t i, size_t j) { 
        return data[i * cols + j]; 
    }
    
    const double& operator()(size_t i, size_t j) const { 
        return data[i * cols + j]; 
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
};

Matrix multiplySequential(const Matrix& A, const Matrix& B) {
    if (A.getCols() != B.getRows()) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication");
    }

    Matrix C(A.getRows(), B.getCols());
    const size_t N = A.getRows();
    const size_t M = B.getCols();
    const size_t K = A.getCols();

    // Cache blocking
    const size_t BLOCK_SIZE = 32;
    
    for (size_t ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (size_t jj = 0; jj < M; jj += BLOCK_SIZE) {
            for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, N); ++i) {
                    for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, M); ++j) {
                        double sum = C(i, j);
                        for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k) {
                            sum += A(i, k) * B(k, j);
                        }
                        C(i, j) = sum;
                    }
                }
            }
        }
    }

    return C;
}

Matrix multiplyParallel(const Matrix& A, const Matrix& B) {
    if (A.getCols() != B.getRows()) {
        throw std::runtime_error("Invalid matrix dimensions for multiplication");
    }

    Matrix C(A.getRows(), B.getCols());
    const size_t N = A.getRows();
    const size_t M = B.getCols();
    const size_t K = A.getCols();

    // Cache blocking
    const size_t BLOCK_SIZE = 32;

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic) collapse(2)
        for (size_t ii = 0; ii < N; ii += BLOCK_SIZE) {
            for (size_t jj = 0; jj < M; jj += BLOCK_SIZE) {
                for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
                    for (size_t i = ii; i < std::min(ii + BLOCK_SIZE, N); ++i) {
                        for (size_t j = jj; j < std::min(jj + BLOCK_SIZE, M); ++j) {
                            double sum = C(i, j);
                            for (size_t k = kk; k < std::min(kk + BLOCK_SIZE, K); ++k) {
                                sum += A(i, k) * B(k, j);
                            }
                            C(i, j) = sum;
                        }
                    }
                }
            }
        }
    }

    return C;
}

void benchmarkMultiplication(size_t size) {
    Matrix A(size, size), B(size, size);
    A.randomize();
    B.randomize();

    // Warm up the cache
    Matrix warmup = multiplyParallel(A, B);

    // Benchmark sequential multiplication
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C1 = multiplySequential(A, B);
    auto end = std::chrono::high_resolution_clock::now();
    double sequential_time = std::chrono::duration<double>(end - start).count();

    // Benchmark parallel multiplication
    start = std::chrono::high_resolution_clock::now();
    Matrix C2 = multiplyParallel(A, B);
    end = std::chrono::high_resolution_clock::now();
    double parallel_time = std::chrono::duration<double>(end - start).count();

    std::cout << "Matrix size: " << size << "x" << size << "\n";
    std::cout << "Sequential time: " << std::fixed << std::setprecision(4) 
              << sequential_time << " seconds\n";
    std::cout << "Parallel time: " << parallel_time << " seconds\n";
    std::cout << "Speedup: " << sequential_time / parallel_time << "x\n\n";
}

int main() {
    // Set number of threads for OpenMP
    omp_set_num_threads(omp_get_max_threads());
    std::cout << "Using " << omp_get_max_threads() << " threads\n\n";

    // Benchmark with different matrix sizes
    std::vector<size_t> sizes = {100, 500, 1000, 2000};
    for (size_t size : sizes) {
        benchmarkMultiplication(size);
    }

    return 0;
}