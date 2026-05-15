/**
 * Multiplicación de Matrices con MPI — versión por bloques (Block-Column)
 * Mejora la localidad de caché respecto a la versión por filas.
 *
 * Compilar: mpic++ -O2 -o matmul_mpi_block matmul_mpi_block.cpp
 * Ejecutar:  mpirun -np <P> --hostfile hosts ./matmul_mpi_block <N>
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

using Matrix = std::vector<double>;

static void fill_random(Matrix& M, int n, int seed) {
    std::srand(seed);
    for (auto& v : M) v = static_cast<double>(std::rand()) / RAND_MAX;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1000;
    if (argc >= 2) N = std::atoi(argv[1]);

    std::vector<int> row_counts(size), row_displs(size);
    int base = N / size, rem = N % size;
    for (int p = 0; p < size; p++) {
        row_counts[p] = (base + (p < rem ? 1 : 0)) * N;
        row_displs[p] = (p == 0) ? 0 : row_displs[p-1] + row_counts[p-1];
    }
    int local_rows = row_counts[rank] / N;

    Matrix A, B(N * N), C;
    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N, 0.0);
        fill_random(A, N * N, 1);
        fill_random(B, N * N, 2);
    }

    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Matrix local_A(local_rows * N);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr,
                 row_counts.data(), row_displs.data(), MPI_DOUBLE,
                 local_A.data(), local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    Matrix local_C(local_rows * N, 0.0);

    double t0 = MPI_Wtime();

    /* Multiplicación ikj (mejor uso de caché) */
    for (int i = 0; i < local_rows; i++)
        for (int k = 0; k < N; k++) {
            double a = local_A[i * N + k];
            for (int j = 0; j < N; j++)
                local_C[i * N + j] += a * B[k * N + j];
        }

    double elapsed = MPI_Wtime() - t0;

    double max_t;
    MPI_Reduce(&elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_C.data(), local_rows * N, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                row_counts.data(), row_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gflops = (2.0 * N * N * N) / (max_t * 1e9);
        std::cout << "BLOCK N=" << N
                  << " P=" << size
                  << " tiempo=" << max_t * 1000.0 << " ms"
                  << " GFLOP/s=" << gflops
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
