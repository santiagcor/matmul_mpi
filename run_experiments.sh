#!/bin/bash
# Script de experimentos — Cluster MPI
# Uso: bash run_experiments.sh 2>&1 | tee resultados.txt

HOSTFILE="hosts"          # archivo con IPs/hostnames del cluster
EXEC_ROW="./matmul_mpi"
EXEC_BLK="./matmul_mpi_block"
SIZES=(500 1000 2000 3000)
PROCS=(1 2 3 4 6 8 12)
REPS=5                    # repeticiones por configuración

echo "======================================================"
echo "  Experimentos MPI — Multiplicación de Matrices"
echo "  Fecha: $(date)"
echo "======================================================"

# Compilar
echo "[INFO] Compilando..."
mpic++ -O2 -o $EXEC_ROW matmul_mpi.cpp
mpic++ -O2 -o $EXEC_BLK matmul_mpi_block.cpp
echo "[INFO] Compilación completada"
echo ""

echo "version,N,P,rep,tiempo_ms,gflops"

for N in "${SIZES[@]}"; do
  for P in "${PROCS[@]}"; do
    for rep in $(seq 1 $REPS); do

      # Versión por filas
      out=$(mpirun -np $P --hostfile $HOSTFILE $EXEC_ROW $N 2>/dev/null)
      tiempo=$(echo "$out" | grep -oP 'tiempo_max=\K[0-9.]+')
      gflops=$(echo "$out" | grep -oP 'GFLOP/s=\K[0-9.]+')
      echo "row,$N,$P,$rep,$tiempo,$gflops"

      # Versión bloque
      out2=$(mpirun -np $P --hostfile $HOSTFILE $EXEC_BLK $N 2>/dev/null)
      tiempo2=$(echo "$out2" | grep -oP 'tiempo=\K[0-9.]+')
      gflops2=$(echo "$out2" | grep -oP 'GFLOP/s=\K[0-9.]+')
      echo "block,$N,$P,$rep,$tiempo2,$gflops2"

    done
  done
done

echo ""
echo "[INFO] Experimentos completados."
