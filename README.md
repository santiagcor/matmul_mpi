# matmul_mpi — Multiplicación de Matrices con MPI

Implementación paralela de multiplicación de matrices usando **MPI (Message Passing Interface)** sobre cluster computacional. Tercer caso de estudio de la serie de Computación de Alto Rendimiento.

**Repos previos:**
- [MultiplicacionMatrices](https://github.com/santiagcor/MultiplicacionMatrices) — Secuencial, Pthreads, Fork
- [matmul_openmp](https://github.com/santiagcor/matmul_openmp) — OpenMP

---

## Descripción

Este proyecto implementa la multiplicación de matrices **C = A × B** distribuida sobre múltiples nodos físicos usando MPI. A diferencia de Pthreads y OpenMP (memoria compartida dentro de un nodo), MPI permite escalar a múltiples máquinas comunicadas por red.

**Estrategia:** distribución por filas (row-wise scatter/gather).

```
        ┌─────────┐         ┌─────────┐       ┌─────────┐
A  =    │ filas 0 │   B =   │ matriz  │  C =  │ filas 0 │
        │ filas 1 │         │completa │       │ filas 1 │
        │ filas 2 │         │ (cada   │       │ filas 2 │
        │   ...   │         │ proceso)│       │   ...   │
        └─────────┘         └─────────┘       └─────────┘
       (Scatterv)            (Bcast)          (Gatherv)
```

---

## Estructura del proyecto

```
matmul_mpi/
├── matmul_mpi.cpp          # Implementación principal (row-wise)
├── matmul_mpi_block.cpp    # Variante con optimización ikj
├── run_experiments.sh      # Script de barrido automático
├── plot_results.py         # Genera gráficas de speedup/eficiencia
├── hosts.example           # Plantilla del hostfile MPI
└── README.md
```

---

## Compilación

```bash
mpic++ -O2 -o matmul_mpi matmul_mpi.cpp
mpic++ -O2 -o matmul_mpi_block matmul_mpi_block.cpp
```

---

## Uso

### Ejecución básica (1 nodo)
```bash
mpirun -np 4 ./matmul_mpi 1000
```
Sale: `N=1000 P=4 tiempo_max=XXX ms tiempo_avg=XXX ms GFLOP/s=X.XX`

### Ejecución en cluster (multi-nodo)
```bash
# Crear archivo hosts
cat > hosts << EOF
10.128.0.2 slots=4
10.128.0.3 slots=4
10.128.0.4 slots=4
EOF

# Ejecutar con 12 procesos en 3 nodos
mpirun -np 12 --hostfile hosts ./matmul_mpi 2000
```

### Experimento completo
```bash
bash run_experiments.sh 2>&1 | tee resultados.txt
```
Prueba: 4 tamaños × 7 configuraciones × 5 repeticiones × 2 versiones = 280 corridas.

### Generar gráficas
```bash
python3 plot_results.py resultados.txt
# → mpi_results.png
```

---

## Configuración del cluster (Google Cloud Platform)

### 1. Crear 3 VMs idénticas
- Tipo: `e2-standard-4` (4 vCPU, 16 GB RAM)
- SO: Ubuntu 22.04 LTS
- Misma zona (ej. `us-central1-a`)
- Nombrar: `nodo1`, `nodo2`, `nodo3`

### 2. Instalar dependencias en cada nodo
```bash
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev g++ git
```

### 3. SSH sin contraseña entre nodos
En **nodo1** generar clave:
```bash
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
```

Copiar la salida en GCP → **Compute Engine → Metadatos → Claves SSH → Agregar**.
La clave se propaga automáticamente a los 3 nodos.

Verificar:
```bash
ssh usuario@10.128.0.3 hostname
```

### 4. Distribuir binarios a los otros nodos
```bash
ssh usuario@10.128.0.3 "mkdir -p ~/matmul_mpi"
ssh usuario@10.128.0.4 "mkdir -p ~/matmul_mpi"
scp matmul_mpi usuario@10.128.0.3:~/matmul_mpi/
scp matmul_mpi usuario@10.128.0.4:~/matmul_mpi/
```

---

## Algoritmo

### Distribución
Cada proceso `p` recibe `local_rows` filas de A:
```
base = N / P
local_rows(p) = base + 1   si p < (N mod P)
local_rows(p) = base       en otro caso
```

### Cómputo local (orden ikj para localidad de caché)
```cpp
for (int i = 0; i < local_rows; i++)
    for (int k = 0; k < N; k++) {
        double a_ik = local_A[i*N + k];
        for (int j = 0; j < N; j++)
            local_C[i*N + j] += a_ik * B[k*N + j];
    }
```

### Comunicación
| Operación | Datos | Complejidad |
|---|---|---|
| `MPI_Bcast(B)` | N² doubles | O(N²·log P) |
| `MPI_Scatterv(A)` | N²/P doubles | O(N²) |
| `MPI_Gatherv(C)` | N²/P doubles | O(N²) |

---

## Métricas evaluadas

- **Tiempo de cómputo** (ms) — máximo entre todos los procesos
- **Speedup**: S(P) = T(1) / T(P)
- **Eficiencia**: E(P) = S(P) / P
- **GFLOP/s**: (2 · N³) / (T · 10⁹)

---

## Resultados preliminares

Primera ejecución exitosa en cluster GCP de 3 nodos:

```
N=500  P=12  tiempo_max=19.12 ms  tiempo_avg=16.86 ms  GFLOP/s=13.07
```

> Resultados completos: ver `resultados.txt` y `mpi_results.png`.

---

## Comparativa con casos anteriores (N=1000)

| Implementación | Paradigma | Tiempo (ms) | Speedup | Nodos |
|---|---|---|---|---|
| Secuencial -O0 | — | 9,217 | 1.00× | 1 |
| Secuencial -O3 | — | 1,788 | 5.15× | 1 |
| Transposición | Caché | 433 | 21.3× | 1 |
| Pthreads (4 hilos) | Mem. compartida | 398 | 23.2× | 1 |
| Fork (4 procesos) | Mem. compartida | 374 | 24.6× | 1 |
| OpenMP (4 hilos) | Mem. compartida | 432 | 21.3× | 1 |
| **MPI (12 procesos, 3 nodos)** | **Dist. memoria** | **TBD** | **TBD** | **3** |

---

## Requisitos

- **OpenMPI** ≥ 4.0
- **g++** con soporte C++11
- **Python 3** + matplotlib (solo para gráficas)
- 3+ nodos Linux con SSH sin contraseña entre ellos

---

## Autor

Santiago Castaño Cortés
Caso de Estudio 3 — Computación de Alto Rendimiento — Mayo 2026

---

## Licencia

MIT
