"""
Genera gráficas de speedup y eficiencia a partir de resultados.txt (CSV).
Uso: python plot_results.py resultados.txt
"""

import sys, csv, statistics, math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

def load(path):
    data = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["version"], int(row["N"]), int(row["P"]))
            data[key].append(float(row["tiempo_ms"]))
    return {k: statistics.mean(v) for k, v in data.items()}

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "resultados.txt"
    data = load(path)

    versions = sorted({k[0] for k in data})
    sizes    = sorted({k[1] for k in data})
    procs    = sorted({k[2] for k in data})

    colors = {"row": "#2196F3", "block": "#FF5722"}
    markers = {"row": "o", "block": "s"}

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, len(sizes), figure=fig)

    for col, N in enumerate(sizes):
        # --- Speedup ---
        ax_sp = fig.add_subplot(gs[0, col])
        ax_sp.set_title(f"Speedup — N={N}", fontsize=11)
        ax_sp.set_xlabel("Procesos (P)")
        ax_sp.set_ylabel("Speedup" if col == 0 else "")
        ax_sp.plot(procs, procs, "k--", linewidth=0.8, label="Ideal")

        for ver in versions:
            t1 = data.get((ver, N, 1))
            if t1 is None:
                continue
            sp = [t1 / data[(ver, N, p)] for p in procs if (ver, N, p) in data]
            pp = [p for p in procs if (ver, N, p) in data]
            ax_sp.plot(pp, sp, marker=markers[ver], color=colors[ver], label=ver)

        ax_sp.legend(fontsize=8)
        ax_sp.grid(True, alpha=0.3)

        # --- Eficiencia ---
        ax_ef = fig.add_subplot(gs[1, col])
        ax_ef.set_title(f"Eficiencia — N={N}", fontsize=11)
        ax_ef.set_xlabel("Procesos (P)")
        ax_ef.set_ylabel("Eficiencia" if col == 0 else "")
        ax_ef.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="Ideal")

        for ver in versions:
            t1 = data.get((ver, N, 1))
            if t1 is None:
                continue
            ef = [(t1 / data[(ver, N, p)]) / p for p in procs if (ver, N, p) in data]
            pp = [p for p in procs if (ver, N, p) in data]
            ax_ef.plot(pp, ef, marker=markers[ver], color=colors[ver], label=ver)

        ax_ef.set_ylim(0, 1.1)
        ax_ef.legend(fontsize=8)
        ax_ef.grid(True, alpha=0.3)

    plt.suptitle("MPI — Multiplicación de Matrices: Speedup y Eficiencia", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("mpi_results.png", dpi=150, bbox_inches="tight")
    print("Guardado: mpi_results.png")

    # --- Tabla de tiempos ---
    print("\n=== Tiempos promedio (ms) ===")
    print(f"{'version':8} {'N':6} {'P':4}  {'tiempo_ms':>12}")
    for (ver, N, P), t in sorted(data.items()):
        print(f"{ver:8} {N:6} {P:4}  {t:12.1f}")

if __name__ == "__main__":
    main()
