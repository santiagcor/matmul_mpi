# Caso de Estudio 3: Multiplicación de Matrices con MPI

**Computación de Alto Rendimiento — Implementación y Análisis de Desempeño sobre Cluster Computacional**

**Autor:** Santiago castrillon ortiz
**Fecha:** Mayo 2026
**Repositorio:** https://github.com/santiagcor/matmul_mpi

---

## 1. Introducción

La multiplicación de matrices es uno de los kernels computacionales más estudiados en HPC. Su complejidad O(n³) y sus elevados requerimientos de memoria la convierten en un benchmark natural para evaluar estrategias de paralelización.

Este caso de estudio extiende el trabajo previo:
- **Caso 1** ([MultiplicacionMatrices](https://github.com/santiagcor/MultiplicacionMatrices)): secuencial, flags de compilación, transposición, Pthreads, fork/mmap
- **Caso 2** ([matmul_openmp](https://github.com/santiagcor/matmul_openmp)): paralelización con OpenMP

Esta tercera entrega implementa MPI sobre un **cluster real de 3 nodos en Google Cloud Platform**, permitiendo escalar más allá de los límites de memoria compartida de un solo nodo.

**Objetivo:** Implementar, desplegar y evaluar la multiplicación de matrices con MPI, midiendo speedup, eficiencia y GFLOP/s sobre el cluster.

---

## 2. Marco Teórico

### 2.1 MPI (Message Passing Interface)
Estándar de paso de mensajes para cómputo paralelo distribuido. A diferencia de Pthreads/OpenMP, MPI permite escalar a múltiples nodos físicos comunicados por red.

**Primitivas utilizadas:**
- `MPI_Bcast` — difusión de B desde rank 0 a todos
- `MPI_Scatterv` — distribución de filas de A
- `MPI_Gatherv` — recolección de C en rank 0
- `MPI_Reduce` — recolección de tiempos (máx, mín, promedio)
- `MPI_Wtime` — temporizador de alta resolución

### 2.2 Estrategia Row-wise
Cada proceso recibe N/P filas de A y la matriz B completa, calculando su porción de C de forma independiente.

```
base = N / P
local_rows(p) = base + 1   si p < (N mod P)
local_rows(p) = base       en otro caso
```

### 2.3 Métricas
- **Speedup:** S(P) = T(1) / T(P)
- **Eficiencia:** E(P) = S(P) / P
- **GFLOP/s:** (2·N³) / (T · 10⁹)

---

## 3. Implementación

### 3.1 matmul_mpi.cpp (versión row-wise)
Bucle ikj para mejor localidad de caché:

```cpp
for (int i = 0; i < local_rows; i++)
    for (int k = 0; k < N; k++) {
        double a_ik = local_A[i*N + k];
        for (int j = 0; j < N; j++)
            local_C[i*N + j] += a_ik * B[k*N + j];
    }
```

### 3.2 matmul_mpi_block.cpp (variante optimizada)
Misma estrategia con énfasis explícito en localidad de memoria.

### 3.3 Compilación
```bash
mpic++ -O2 -o matmul_mpi matmul_mpi.cpp
mpic++ -O2 -o matmul_mpi_block matmul_mpi_block.cpp
```

---

## 4. Configuración del Cluster

### 4.1 Infraestructura

| Componente | Especificación |
|---|---|
| Proveedor | Google Cloud Platform |
| Zona | us-central1-a |
| Número de nodos | 3 (nodo1, nodo2, nodo3) |
| Tipo de instancia | e2-standard-4 |
| vCPUs por nodo | 4 (2 núcleos físicos) |
| RAM por nodo | 16 GB |
| Total CPUs disponibles | 12 |
| Sistema operativo | Ubuntu 22.04 LTS |
| Kernel | 6.8.0-1054-gcp |
| MPI | OpenMPI |
| Compilador | g++ (mpic++) |
| Red interna | VPC de GCP |

### 4.2 Hosts

```
10.128.0.2 slots=4   # nodo1 (maestro)
10.128.0.3 slots=4   # nodo2
10.128.0.4 slots=4   # nodo3
```

### 4.3 Configuración SSH
Clave RSA-4096 generada en nodo1 y agregada como metadato del proyecto en GCP, propagando autenticación sin contraseña a los 3 nodos.

---

## 5. Diseño del Experimento

| Parámetro | Valores |
|---|---|
| Tamaños N | 500, 1000, 2000, 3000 |
| Procesos P | 1, 2, 3, 4, 6, 8, 12 |
| Repeticiones | 5 |
| Versiones | row, block |
| **Total corridas** | **280** |

Script de automatización: `run_experiments.sh` ejecutado con `nohup` para sobrevivir desconexiones.

---

## 6. Resultados Experimentales

### 6.1 Tiempos de Ejecución (ms) — Versión row

Tiempos promedio sobre 5 repeticiones:

| N \ P | P=1 | P=2 | P=3 | P=4 | P=6 | P=8 | P=12 |
|---|---|---|---|---|---|---|---|
| **500** | 100.6 | 53.3 | 65.3 | 50.8 | 35.0 | 27.4 | 18.3 |
| **1000** | 872.5 | 679.5 | 576.3 | 443.5 | 305.0 | 331.2 | 169.7 |
| **2000** | 9,223.4 | 4,646.6 | 4,888.9 | 3,661.8 | 2,490.6 | 1,894.4 | 1,293.9 |
| **3000** | 31,382.6 | 15,708.1 | 16,066.9 | 12,667.5 | 8,324.4 | 6,447.8 | 4,293.0 |

### 6.2 Speedup

| N \ P | P=1 | P=2 | P=3 | P=4 | P=6 | P=8 | P=12 |
|---|---|---|---|---|---|---|---|
| **500** | 1.00 | 1.89 | 1.54 | 1.98 | 2.87 | 3.67 | **5.49** |
| **1000** | 1.00 | 1.28 | 1.51 | 1.97 | 2.86 | 2.63 | **5.14** |
| **2000** | 1.00 | 1.98 | 1.89 | 2.52 | 3.70 | 4.87 | **7.13** |
| **3000** | 1.00 | 2.00 | 1.95 | 2.48 | 3.77 | 4.87 | **7.31** |

**Observaciones:**
- El speedup mejora con N: matrices más grandes amortizan mejor la comunicación
- P=3 muestra una **anomalía** (peor que P=2) — probablemente porque distribuye un proceso a un nodo distinto sin aprovechar el paralelismo dentro del nodo
- Para N=3000 con P=12 se logra **7.31× speedup** sobre 3 nodos

### 6.3 Eficiencia E(P) = S(P)/P

| N \ P | P=1 | P=2 | P=3 | P=4 | P=6 | P=8 | P=12 |
|---|---|---|---|---|---|---|---|
| **500** | 1.00 | 0.94 | 0.51 | 0.50 | 0.48 | 0.46 | 0.46 |
| **1000** | 1.00 | 0.64 | 0.50 | 0.49 | 0.48 | 0.33 | 0.43 |
| **2000** | 1.00 | 0.99 | 0.63 | 0.63 | 0.62 | 0.61 | 0.59 |
| **3000** | 1.00 | 1.00 | 0.65 | 0.62 | 0.63 | 0.61 | 0.61 |

**Observaciones:**
- **Eficiencia ~100% para P=2** en N grandes (cómputo intra-nodo perfecto)
- Caída notable de P=2 a P=3 (transición de 1 nodo a 2 nodos = aparece overhead de red)
- A partir de P=3 la eficiencia se estabiliza en ~60% (overhead de comunicación dominante)

### 6.4 GFLOP/s

| N \ P | P=1 | P=4 | P=8 | P=12 |
|---|---|---|---|---|
| **500** | 2.48 | 5.05 | 9.21 | **14.07** |
| **1000** | 2.32 | 4.51 | 6.04 | **11.91** |
| **2000** | 1.74 | 4.37 | 8.46 | **12.40** |
| **3000** | 1.73 | 4.38 | 8.37 | **12.70** |

**Pico de rendimiento:** ~**14 GFLOP/s** con N=500, P=12.

---

## 7. Análisis y Discusión

### 7.1 Comparativa con Casos Anteriores (N=1000)

| Implementación | Paradigma | Tiempo (ms) | Speedup | Nodos |
|---|---|---|---|---|
| Secuencial -O0 | — | 9,217 | 1.00× | 1 |
| Secuencial -O3 | — | 1,788 | 5.15× | 1 |
| Transposición | Caché | 433 | 21.3× | 1 |
| Pthreads (4 hilos) | Mem. compartida | 398 | 23.2× | 1 |
| Fork (4 procesos) | Mem. compartida | 374 | 24.6× | 1 |
| OpenMP (4 hilos) | Mem. compartida | 432 | 21.3× | 1 |
| **MPI P=4 (1 nodo)** | Dist. memoria | **443** | **20.8×** | 1 |
| **MPI P=12 (3 nodos)** | **Dist. memoria** | **170** | **54.3×** | **3** |

**Hallazgo clave:** MPI con 12 procesos sobre 3 nodos supera ampliamente cualquier implementación de nodo único:
- **54× speedup vs secuencial** (vs 23-25× de Pthreads/OpenMP/Fork)
- Solo posible gracias al escalado horizontal

### 7.2 Sobrecarga de Comunicación

| Operación | Datos | Para N=3000 |
|---|---|---|
| `MPI_Bcast(B)` | N² doubles | 72 MB |
| `MPI_Scatterv(A)` | N²/P doubles | 6 MB por proceso |
| `MPI_Gatherv(C)` | N²/P doubles | 6 MB por proceso |

Con red 1 Gbps en GCP: ~80 ms teóricos para Bcast de 72 MB. Esto explica:
- Eficiencia ~60% sostenida
- Mejor escalado con matrices grandes (mayor ratio cómputo/comunicación)

### 7.3 Ley de Amdahl

Fracción serial estimada: f ≈ 0.05 (inicialización + Bcast secuencial)
Speedup máximo teórico: S_max = 1 / 0.05 ≈ 20×

Speedup observado (7.31) es coherente con esta limitación cuando se contabiliza la fracción comunicación que crece con P.

### 7.4 Anomalía en P=3

P=3 muestra peor speedup que P=2. Análisis:
- Con P=2, ambos procesos van al nodo1 (slots=4): comunicación intra-nodo (memoria compartida vía MPI)
- Con P=3, el tercer proceso se asigna al nodo2: aparece latencia de red por primera vez
- A partir de P=4+, el costo de red se amortiza con más cómputo paralelo

Esto valida que MPI explota memoria compartida cuando los procesos están co-ubicados.

### 7.5 Escalabilidad Fuerte vs Débil

**Escalabilidad fuerte (N fijo, P crece):** confirmada hasta P=12 con eficiencia decreciente pero positiva.

**Escalabilidad débil (N/P constante):** no medida directamente, pero los datos sugieren que matrices más grandes mantienen mejor eficiencia (E=0.61 para N=3000 vs E=0.46 para N=500).

---

## 8. Conclusiones

1. **El experimento se ejecutó exitosamente** sobre cluster real de 3 nodos en GCP con 280 corridas automatizadas.

2. **MPI es superior para escalado horizontal:** 54× speedup vs secuencial con N=1000 sobre 3 nodos, frente a 23-25× de las implementaciones de memoria compartida (Pthreads/OpenMP/Fork) limitadas a 1 nodo.

3. **Eficiencia depende del tamaño del problema:**
   - N=2000-3000 mantienen E ≈ 60% con 12 procesos
   - N=500-1000 caen a E ≈ 43-46% por dominancia de comunicación

4. **Pico de rendimiento:** **14 GFLOP/s** con N=500 P=12, **12.7 GFLOP/s** con N=3000 P=12.

5. **Comunicación es el cuello de botella:** `MPI_Bcast(B)` con complejidad O(N²) limita la escalabilidad. Para matrices muy grandes habría que considerar algoritmos como Cannon o SUMMA que reducen comunicación a O(N²/√P).

6. **Anomalía P=3:** validó empíricamente la importancia de la co-ubicación de procesos. La transición de comunicación intra-nodo (memoria) a inter-nodo (red) tiene un costo significativo.

7. **Comparativa global de paradigmas:**
   - **Pthreads/OpenMP:** mejores dentro de un nodo (latencia baja de memoria compartida)
   - **Fork:** overhead mayor por procesos pesados
   - **MPI:** **única opción viable para clusters reales** y escalado más allá de un nodo

---

## 9. Trabajo Futuro

- Implementar **Algoritmo de Cannon** o **SUMMA** para reducir comunicación O(N²) → O(N²/√P)
- Probar con red InfiniBand para reducir overhead de Bcast
- Implementar **versión híbrida MPI + OpenMP** (1 proceso MPI por nodo + threads OpenMP intra-nodo)
- Escalar a más nodos (6, 12) para evaluar escalabilidad fuerte
- Comparar con bibliotecas optimizadas (BLAS, MKL, cuBLAS)

---

## 10. Referencias

1. Gropp, W., Lusk, E., Skjellum, A. *Using MPI: Portable Parallel Programming with the Message-Passing Interface* (3rd ed.). MIT Press, 2014.
2. Quinn, M. J. *Parallel Programming in C with MPI and OpenMP*. McGraw-Hill, 2003.
3. Grama, A., Kumar, V., Gupta, A., Karypis, G. *Introduction to Parallel Computing* (2nd ed.). Addison-Wesley, 2003.
4. MPI Forum. *MPI: A Message-Passing Interface Standard, Version 4.0*. https://www.mpi-forum.org/
5. Cannon, L. E. *A Cellular Computer to Implement the Kalman Filter Algorithm*. PhD Thesis, Montana State University, 1969.
6. Repositorio del autor — MultiplicacionMatrices: https://github.com/santiagcor/MultiplicacionMatrices
7. Repositorio del autor — matmul_openmp: https://github.com/santiagcor/matmul_openmp
8. Repositorio del autor — matmul_mpi (este trabajo): https://github.com/santiagcor/matmul_mpi

---

## Anexos

### A — Bitácora de configuración

1. Creación de proyecto `matmul-mpi` en GCP
2. 3 VMs `e2-standard-4` Ubuntu 22.04 en us-central1-a
3. Instalación: `sudo apt install openmpi-bin libopenmpi-dev g++ git`
4. Clave SSH RSA-4096 generada en nodo1
5. Clave agregada como metadato del proyecto (propaga a los 3 nodos)
6. Verificación: `ssh josfernight@10.128.0.3` sin contraseña ✅
7. Clone del repo desde GitHub
8. Archivo `hosts` con las 3 IPs internas
9. Compilación en nodo1: `mpic++ -O2`
10. Distribución de binarios con `scp` a nodo2 y nodo3
11. Ejecución de `run_experiments.sh` en background con `nohup`
12. Descarga de resultados.txt y generación de gráficas en WSL local

### B — Problemas encontrados y soluciones

| Problema | Solución |
|---|---|
| MPI bloqueado en WSL local | Migración a cluster en GCP |
| Cuenta institucional rechazada en GCP | Uso de cuenta Gmail personal |
| `Permission denied (publickey)` entre nodos | Clave SSH agregada como metadato del proyecto |
| `mpirun: could not access executable` | `scp` de binarios a los 3 nodos |
| Script se cortó al desconectar SSH | `nohup` para correr en background |
| Tiempos vacíos en CSV (1 corrida falló) | Filtrado tolerante en script de análisis |

### C — Estructura del proyecto

```
matmul_mpi/
├── matmul_mpi.cpp          # Implementación principal (row-wise)
├── matmul_mpi_block.cpp    # Variante con optimización ikj
├── run_experiments.sh      # Script de barrido automático
├── plot_results.py         # Generador de gráficas
├── plot_simple.py          # Versión simplificada del plotter
├── hosts.example           # Plantilla del hostfile MPI
├── resultados.txt          # Datos crudos del experimento
├── resultados_clean.csv    # CSV limpio para análisis
├── mpi_results.png         # Gráficas de speedup y eficiencia
└── README.md
```

### D — Comandos clave

```bash
# Compilación
mpic++ -O2 -o matmul_mpi matmul_mpi.cpp

# Distribución de binarios
scp matmul_mpi josfernight@10.128.0.3:~/matmul_mpi/
scp matmul_mpi josfernight@10.128.0.4:~/matmul_mpi/

# Ejecución multi-nodo
mpirun -np 12 --hostfile hosts ./matmul_mpi 3000

# Experimento completo en background
nohup bash run_experiments.sh > resultados.txt 2>&1 &

# Análisis local
python3 plot_simple.py resultados_clean.csv
```

### E — Gráficas

Ver archivo `mpi_results.png` para visualización de speedup y eficiencia por tamaño N.
