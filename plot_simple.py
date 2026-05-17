"""Generador rápido de gráficas y tabla resumen."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, statistics, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "resultados_clean.csv"

data = defaultdict(list)
with open(path) as f:
    for row in csv.DictReader(f):
        try:
            data[(row["version"], int(row["N"]), int(row["P"]))].append(float(row["tiempo_ms"]))
        except: pass

avg = {k: statistics.mean(v) for k, v in data.items()}

sizes = sorted({k[1] for k in avg})
procs = sorted({k[2] for k in avg})
versions = sorted({k[0] for k in avg})

print(f"\n{'Ver':6} {'N':>5} {'P':>4} {'Tiempo(ms)':>12} {'Speedup':>9} {'Eficiencia':>11} {'GFLOP/s':>9}")
print("-" * 65)
for ver in versions:
    for N in sizes:
        t1 = avg.get((ver, N, 1))
        for P in procs:
            t = avg.get((ver, N, P))
            if t is None: continue
            sp = t1 / t if t1 else 0
            ef = sp / P
            gf = (2 * N**3) / (t * 1e6)
            print(f"{ver:6} {N:5} {P:4} {t:12.2f} {sp:9.2f} {ef:11.2f} {gf:9.2f}")
    print()

# ── Gráficas ──
fig, axes = plt.subplots(2, len(sizes), figsize=(5*len(sizes), 8))
colors = {"row": "#1976D2", "block": "#E64A19"}

for col, N in enumerate(sizes):
    ax_sp, ax_ef = axes[0, col], axes[1, col]
    ax_sp.plot(procs, procs, "k--", lw=0.8, label="Ideal")
    ax_ef.axhline(1.0, color="k", ls="--", lw=0.8, label="Ideal")
    for ver in versions:
        t1 = avg.get((ver, N, 1))
        if not t1: continue
        pp = [p for p in procs if (ver, N, p) in avg]
        sp = [t1 / avg[(ver, N, p)] for p in pp]
        ef = [s/p for s, p in zip(sp, pp)]
        ax_sp.plot(pp, sp, "o-", color=colors[ver], label=ver)
        ax_ef.plot(pp, ef, "o-", color=colors[ver], label=ver)
    ax_sp.set_title(f"Speedup — N={N}")
    ax_sp.set_xlabel("P"); ax_sp.set_ylabel("Speedup")
    ax_sp.legend(fontsize=8); ax_sp.grid(alpha=0.3)
    ax_ef.set_title(f"Eficiencia — N={N}")
    ax_ef.set_xlabel("P"); ax_ef.set_ylabel("Eficiencia")
    ax_ef.set_ylim(0, 1.1); ax_ef.legend(fontsize=8); ax_ef.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mpi_results.png", dpi=120)
print("\n[OK] Guardado: mpi_results.png")
