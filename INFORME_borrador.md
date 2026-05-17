# Caso de Estudio 3: Multiplicación de Matrices con MPI

**Computación de Alto Rendimiento — Implementación y Análisis de Desempeño sobre Cluster Computacional**

**Autor:** Santiago Castaño
**Fecha:** Mayo 2026
**Repositorio:** https://github.com/santiagcor/matmul_mpi

---

## 1. Introducción

La multiplicación de matrices es uno de los kernels computacionales más estudiados en HPC. Su complejidad O(n³) y sus elevados requerimientos de memoria la convierten en un benchmark natural para evaluar estrategias de paralelización.

Este caso de estudio extiende el trabajo previo:
- **Caso 1** ([MultiplicacionMatrices](https://github.com/santiagcor/MultiplicacionMatrices)): secuencial, flags de compilación, transposición, Pthreads, fork/mmap
- **Caso 2** ([matmul_openmp](https://github.com/santiagcor/matmul_openmp)): paralelización con OpenMP

Esta tercera entrega implementa MPI sobre un cluster real de 3 nodos en Google Cloud Platform.

**Objetivo:** Evaluar speedup, eficiencia y GFLOP/s de la multiplicación de matrices con MPI escalando a múltiples nodos físicos.

---

## 2. Marco Teórico

### 2.1 MPI (Message Passing Interface)
Estándar de paso de mensajes para cómputo paralelo distribuido. A diferencia de Pthreads/OpenMP (memoria compartida), MPI permite escalar a múltiples nodos.

**Primitivas usadas:**
- `MPI_Bcast` — difusión de B desde rank 0 a todos
- `MPI_Scatterv` — distribución de filas de A
- `MPI_Gatherv` — recolección de C en rank 0
- `MPI_Reduce` — recolección de tiempos (máx, mín, promedio)
- `MPI_Wtime` — temporizador de alta resolución

### 2.2 Estrategia Row-wise
Cada proceso recibe `N/P` filas de A y la matriz B completa. Calcula su porción de C de forma independiente.

```
base = N / P
local_rows(p) = base + 1 si p < (N mod P), base en otro caso
```

### 2.3 Métricas
- **Speedup:** S(P) = T(1) / T(P)
- **Eficiencia:** E(P) = S(P) / P
- **GFLOP/s:** (2·N³) / (T · 10⁹)

---

## 3. Implementación

### 3.1 matmul_mpi.cpp (versión por filas)
Implementación principal. Bucle ikj para mejor localidad de caché:

```cpp
for (int i = 0; i < local_rows; i++)
    for (int k = 0; k < N; k++) {
        double a_ik = local_A[i*N + k];
        for (int j = 0; j < N; j++)
            local_C[i*N + j] += a_ik * B[k*N + j];
    }
```

### 3.2 matmul_mpi_block.cpp (variante optimizada)
Misma estrategia row-wise con énfasis explícito en localidad de memoria.

### 3.3 Compilación
```bash
mpic++ -O2 -o matmul_mpi matmul_mpi.cpp
mpic++ -O2 -o matmul_mpi_block matmul_mpi_block.cpp
```

---

## 4. Configuración del Cluster (REAL)

### 4.1 Infraestructura

| Componente | Especificación |
|---|---|
| Proveedor | Google Cloud Platform |
| Región/Zona | us-central1-a |
| Número de nodos | 3 |
| Tipo de instancia | e2-standard-4 |
| vCPUs por nodo | 4 (2 núcleos físicos) |
| RAM por nodo | 16 GB |
| Disco | 10 GB SSD persistente |
| SO | Ubuntu 22.04 LTS |
| Kernel | 6.8.0-1054-gcp |
| Implementación MPI | OpenMPI |
| Compilador | g++ (mpic++) |
| Red interna | Red VPC de GCP |

### 4.2 Hosts

```
10.128.0.2 slots=4   # nodo1 (maestro)
10.128.0.3 slots=4   # nodo2
10.128.0.4 slots=4   # nodo3
```

**Total de procesos posibles:** 12 (4 × 3 nodos)

### 4.3 Configuración SSH
Clave RSA generada en nodo1 y agregada como metadato del proyecto en GCP, propagando autenticación sin contraseña a los 3 nodos.

---

## 5. Diseño del Experimento

| Parámetro | Valores |
|---|---|
| Tamaños N | 500, 1000, 2000, 3000 |
| Procesos P | 1, 2, 3, 4, 6, 8, 12 |
| Repeticiones | 5 |
| Versiones | row, block |

**Script de automatización:** `run_experiments.sh` ejecuta 280 corridas, guardando CSV en `resultados.txt`.

---

## 6. Resultados Experimentales

### 6.1 Resultado Preliminar (verificación inicial)

Primera ejecución exitosa sobre los 3 nodos:
```
N=500  P=12  tiempo_max=19.12 ms  tiempo_avg=16.86 ms  GFLOP/s=13.07
```

> ⚠️ Las tablas completas se llenarán cuando termine `run_experiments.sh`

### 6.2 Tabla 1 — Tiempos de Ejecución (ms)

| N \ P | P=1 | P=2 | P=3 | P=4 | P=6 | P=8 | P=12 |
|---|---|---|---|---|---|---|---|
| 500 | _ | _ | _ | _ | _ | _ | 19.12 |
| 1000 | _ | _ | _ | _ | _ | _ | _ |
| 2000 | _ | _ | _ | _ | _ | _ | _ |
| 3000 | _ | _ | _ | _ | _ | _ | _ |

### 6.3 Tabla 2 — Speedup
*(pendiente — se calculará al terminar el experimento)*

### 6.4 Tabla 3 — Eficiencia
*(pendiente)*

### 6.5 Tabla 4 — GFLOP/s

| N \ P | P=12 (GFLOP/s) |
|---|---|
| 500 | **13.07** ✅ |

---

## 7. Análisis y Discusión

### 7.1 Comparativa con Casos Anteriores (N=1000)

| Implementación | Paradigma | Tiempo (ms) | Speedup | Nodos |
|---|---|---|---|---|
| Secuencial O0 | — | 9,217 | 1.00 | 1 |
| Secuencial O3 | — | 1,788 | 5.15 | 1 |
| Transposición | Caché | 433 | 21.3 | 1 |
| Pthreads | Mem. compartida | 398 | 23.2 | 1 |
| OpenMP | Mem. compartida | 432 | 21.3 | 1 |
| **MPI P=12** | **Dist. memoria** | **_pendiente_** | **_pendiente_** | **3** |

### 7.2 Sobrecarga de Comunicación
- `MPI_Bcast(B)`: O(N²) — cuello de botella principal
- `MPI_Scatterv(A)`: O(N²/P)
- `MPI_Gatherv(C)`: O(N²/P)

### 7.3 Ley de Amdahl
Con fracción serial estimada f ≈ 0.03, el speedup máximo teórico es ~33×.

---

## 8. Conclusiones (preliminar)

1. **El cluster MPI está operativo** — 3 nodos GCP con conectividad SSH sin contraseña y MPI funcional.
2. **El primer resultado real** (N=500, P=12 → 13 GFLOP/s) confirma que la estrategia row-wise escala correctamente a múltiples nodos.
3. **Trabajo en curso:** completar el barrido completo de N y P para análisis cuantitativo de escalabilidad.

---

## 9. Referencias

1. Gropp, Lusk, Skjellum. *Using MPI*. MIT Press, 2014.
2. Quinn, M. J. *Parallel Programming in C with MPI and OpenMP*. McGraw-Hill, 2003.
3. MPI Forum. *MPI: A Message-Passing Interface Standard v4.0*.
4. Repos previos del autor: [MultiplicacionMatrices](https://github.com/santiagcor/MultiplicacionMatrices), [matmul_openmp](https://github.com/santiagcor/matmul_openmp)

---

## Anexos

### A — Bitácora de configuración del cluster

1. Creación de proyecto `matmul-mpi` en GCP
2. 3 VMs `e2-standard-4` Ubuntu 22.04 en us-central1-a
3. Instalación: `sudo apt install openmpi-bin libopenmpi-dev g++ git`
4. Clave SSH RSA-4096 generada en nodo1
5. Clave agregada como metadato del proyecto (propaga a los 3 nodos)
6. Verificación: `ssh josfernight@10.128.0.3` sin contraseña ✅
7. Clone del repo: `git clone https://github.com/santiagcor/matmul_mpi.git`
8. Archivo `hosts` con las 3 IPs internas
9. Compilación en nodo1: `mpic++ -O2`
10. Distribución de binarios con `scp` a nodo2 y nodo3
11. Verificación: `mpirun -np 12 --hostfile hosts ./matmul_mpi 500` ✅

### B — Problemas encontrados y soluciones

| Problema | Solución |
|---|---|
| MPI bloqueado en WSL local | Migración a cluster en nube |
| GCP rechazó cuenta institucional | Uso de cuenta Gmail personal |
| `ssh: Permission denied (publickey)` entre nodos | Agregar clave RSA como metadato del proyecto en GCP |
| `mpirun: could not access executable` en nodo2/3 | `scp` de binarios a los 3 nodos |

### C — Código fuente
Ver repositorio: https://github.com/santiagcor/matmul_mpi

### D — Comandos clave

```bash
# Conexión SSH
gcloud compute ssh nodo1 --zone=us-central1-a

# Compilación
mpic++ -O2 -o matmul_mpi matmul_mpi.cpp

# Distribución de binarios
scp matmul_mpi josfernight@10.128.0.3:~/matmul_mpi/
scp matmul_mpi josfernight@10.128.0.4:~/matmul_mpi/

# Ejecución
mpirun -np 12 --hostfile hosts ./matmul_mpi 2000

# Experimento completo en background
nohup bash run_experiments.sh > resultados.txt 2>&1 &
```
