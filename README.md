## README for Matrix Multiplication Code

### Overview

This C++ program implements matrix multiplication using both sequential and parallel approaches. It utilizes the OpenMP library to parallelize the matrix multiplication process, allowing for improved performance on multi-core processors. The program also benchmarks the performance of both methods across different matrix sizes.

### Features

- **Matrix Class**: A simple class to represent matrices with functionalities for initialization, randomization, and accessing elements.
- **Sequential Multiplication**: A method to multiply matrices without parallelization.
- **Parallel Multiplication**: A method to multiply matrices using OpenMP for parallel execution.
- **Benchmarking**: Compares the execution time of sequential and parallel matrix multiplication.

### Dependencies

- **C++ Standard Library**: Utilizes standard libraries such as `<vector>`, `<iostream>`, `<random>`, `<chrono>`, and `<iomanip>`.
- **OpenMP**: Used for parallelizing the matrix multiplication process.

### Usage

1. **Compilation**: Ensure you have a compiler that supports OpenMP. Compile the program using a command like:
   ```bash
   g++ -o matrix_multiplication matrix.cpp -fopenmp
   ```

2. **Execution**: Run the compiled program:
   ```bash
   ./matrix_multiplication
   ```

3. **Output**: The program will output the number of threads used and benchmark results for different matrix sizes, displaying sequential and parallel execution times along with the speedup achieved.

### Code Structure

- **Matrix Class**: 
  - **Constructor**: Initializes a matrix with given dimensions and sets all elements to zero.
  - **Randomize Method**: Fills the matrix with random values between 0 and 1.
  - **Element Access Operators**: Provides access to matrix elements using `operator()`.

- **Multiplication Methods**:
  - `multiplySequential`: Performs matrix multiplication without parallelization.
  - `multiplyParallel`: Performs matrix multiplication using OpenMP for parallel execution.

- **Benchmarking Function**:
  - `benchmarkMultiplication`: Tests both multiplication methods on matrices of various sizes and measures their execution times.

- **Main Function**:
  - Sets the number of threads based on system capabilities.
  - Benchmarks both multiplication methods for matrices of sizes 100x100, 500x500, 1000x1000, and 2000x2000.

### Notes

- Ensure that OpenMP is supported by your compiler to utilize parallel processing capabilities.
- The program uses cache blocking techniques to optimize memory access patterns during multiplication.
