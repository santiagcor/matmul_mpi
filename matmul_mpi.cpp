/**
 * Multiplicación de Matrices con MPI
 * Estrategia: distribución por filas (row-wise scatter/gather)
 *
 * Compilar: mpic++ -O2 -o matmul_mpi matmul_mpi.cpp
 * Ejecutar:  mpirun -np <P> --hostfile hosts ./matmul_mpi <N>
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

using Matrix = std::vector<double>;

static void fill_random(Matrix& M, int rows, int cols, int seed = 42) {
    std::srand(seed);
    for (int i = 0; i < rows * cols; i++)
        M[i] = static_cast<double>(std::rand()) / RAND_MAX;
}

static double max_error(const Matrix& A, const Matrix& B, int n) {
    double err = 0.0;
    for (int i = 0; i < n * n; i++)
        err = std::max(err, std::abs(A[i] - B[i]));
    return err;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1000;
    if (argc >= 2) N = std::atoi(argv[1]);

    /* ── Distribución de filas ── */
    // Cada proceso recibe base_rows filas; los primeros (N % size) reciben una extra
    std::vector<int> sendcounts(size), displs(size);
    int base = N / size, rem = N % size;
    for (int p = 0; p < size; p++) {
        sendcounts[p] = (base + (p < rem ? 1 : 0)) * N;
        displs[p] = (p == 0) ? 0 : displs[p-1] + sendcounts[p-1];
    }
    int local_rows = sendcounts[rank] / N;

    Matrix A, B(N * N), C;
    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
        fill_random(A, N, N, 1);
        fill_random(B, N, N, 2);
    }

    /* Broadcast de B completa (todos la necesitan) */
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Scatter de filas de A */
    Matrix local_A(local_rows * N);
    MPI_Scatterv(rank == 0 ? A.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* ── Cómputo local ── */
    Matrix local_C(local_rows * N, 0.0);

    double t_start = MPI_Wtime();

    for (int i = 0; i < local_rows; i++)
        for (int k = 0; k < N; k++) {
            double a_ik = local_A[i * N + k];
            for (int j = 0; j < N; j++)
                local_C[i * N + j] += a_ik * B[k * N + j];
        }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    /* Recolección de tiempos */
    double max_time, min_time, avg_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Gather de resultados */
    MPI_Gatherv(local_C.data(), local_rows * N, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        avg_time /= size;
        double gflops = (2.0 * N * N * N) / (max_time * 1e9);

        std::cout << "N=" << N
                  << " P=" << size
                  << " tiempo_max=" << max_time * 1000.0 << " ms"
                  << " tiempo_avg=" << avg_time * 1000.0 << " ms"
                  << " GFLOP/s=" << gflops
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
