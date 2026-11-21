r"""Experimentos ilustrativos para Fernández Bonder–Groisman–Rossi, *On numerical blow-up sets*.

El paper es teórico (no tiene experimentos).  Problema:
:math:`u_t=u_{xx}` en :math:`(0,1)`, :math:`u_x(0)=0`, :math:`u_x(1)=f(u(1))`,
semidiscretizado con P1 + mass lumping.  Teorema 1: el número de nodos que
explotan depende sólo de :math:`f`, no de :math:`h`.

Se integra hasta muy cerca de :math:`T_h` con el reescalamiento del tiempo
:math:`dt/ds = \min(1, u/f(u))` (versión continua del paso adaptativo), que
permite llegar a :math:`T_h-t\sim10^{-60}`–:math:`10^{-200}` y calcular
:math:`T_h-t` sin cancelación.

Experimentos (``--only`` para correr uno):

* ``count``: :math:`f=s^p`, :math:`p\in\{3,2,1.5,1.3,1.25\}`,
  :math:`N\in\{10,20,40,80\}`: cantidad de nodos que explotan.  Un nodo
  "explota" si sigue creciendo en las últimas 10 décadas de :math:`T_h-t`.
* ``rates``: :math:`f=s^{1.5}`, :math:`N=20`: tasas
  :math:`u_{N-k}\sim(T_h-t)^{-2+k}` (:math:`k=0,1`), :math:`u_{N-2}\sim-\ln(T_h-t)`,
  :math:`u_{N-3}` acotado; y :math:`u_i/u_{i+1}\to0` (Teorema 3).
* ``profiles``: perfiles cerca de :math:`T_h` para :math:`p=1.5` y varios
  :math:`N`: el conjunto de explosión :math:`[1-Kh,1]` se achica con :math:`h`.
* ``log``: :math:`f=s(\ln s)^2`: explosión numérica global (el continuo
  explota sólo en :math:`x=1`).
* ``exp``: :math:`f=e^s`: explota sólo :math:`u_N`, con
  :math:`u_N+\ln(T_h-t)` acotado.

Uso::

    python -m experiments.parabolic.blowup_sets [--only count|rates|profiles|log|exp]
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from blowup.parabolic import HeatFlux1D, flux_power, solve_to_blowup
from experiments.common import save

PS = [3.0, 2.0, 1.5, 1.3, 1.25]
NS = [10, 20, 40, 80]


def power_run(p: float, N: int):
    prob = HeatFlux1D(N, *flux_power(p))
    # u_stop tal que f(u) = u^p no desborde en punto flotante.
    return prob, solve_to_blowup(prob, np.ones(N + 1), lambda m: m ** (p - 1), u_stop=10 ** (290 / p))


def blowing_nodes(sol, decades: float = 10.0) -> np.ndarray:
    """Nodos que siguen creciendo en las últimas ``decades`` décadas de T_h - t."""
    R = sol.remaining
    j = int(np.searchsorted(-R, -R[-1] * 10**decades))
    du = sol.U[-1] - sol.U[j]
    return du > 1e-6 * (1 + np.abs(sol.U[-1]))


def exp_count() -> None:
    print("Cantidad de nodos que explotan, f(s) = s^p")
    print(f"{'p':>6} {'1/(p-1)':>8} " + " ".join(f"{'N=' + str(N):>6}" for N in NS) + "   [1/(p-1)]+1")
    table = np.zeros((len(PS), len(NS)), int)
    for a, p in enumerate(PS):
        for b, N in enumerate(NS):
            _, sol = power_run(p, N)
            table[a, b] = blowing_nodes(sol).sum()
        print(f"{p:6g} {1 / (p - 1):8.3f} " + " ".join(f"{v:6d}" for v in table[a])
              + f"   {math.floor(1 / (p - 1) + 1e-12) + 1:6d}")


def exp_rates() -> None:
    p, N = 1.5, 20
    prob, sol = power_run(p, N)
    R = sol.remaining
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = plt.get_cmap("tab10")
    for k in range(4):
        u = sol.U[:, N - k]
        ax1.loglog(R, u, color=colors(k), label=f"$u_{{N-{k}}}$" if k else "$u_N$")
    for k, expo in [(0, -2), (1, -1)]:
        ref = sol.U[-1, N - k] * (R / R[-1]) ** expo
        ax1.loglog(R, ref, ":", color=colors(k), lw=1)
    ax1.invert_xaxis()
    ax1.set_xlabel("$T_h - t$")
    ax1.set_ylabel("$u_i$")
    ax1.set_title(r"Tasas ($p=1.5$; punteado: pendientes $-2$, $-1$)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    ax2.semilogx(R, sol.U[:, N - 2], color=colors(2), label="$u_{N-2}$")
    ax2.semilogx(R, sol.U[:, N - 3], color=colors(3), label="$u_{N-3}$")
    ax2.invert_xaxis()
    ax2.set_xlabel("$T_h - t$")
    ax2.set_title(r"$u_{N-2}\sim-\ln(T_h-t)$, $u_{N-3}$ acotado", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    for k in range(3):
        ax3.loglog(R, sol.U[:, N - k - 1] / sol.U[:, N - k], color=colors(k),
                   label=f"$u_{{N-{k + 1}}}/u_{{N-{k}}}$" if k else "$u_{N-1}/u_N$")
    ax3.invert_xaxis()
    ax3.set_xlabel("$T_h - t$")
    ax3.set_title(r"Teorema 3: $u_i/u_{i+1}\to0$", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, which="both")
    fig.suptitle(rf"$u_x(1)=u^{{1.5}}$, $N={N}$, $T_h\approx{sol.T:.6f}$")
    fig.tight_layout()
    save(fig, "parabolic", "blowup_sets_rates")

    # Pendientes medidas en el tramo final.
    i0 = int(np.searchsorted(-R, -R[-1] * 1e40))
    for k, expo in [(0, -2), (1, -1)]:
        slope = np.polyfit(np.log(R[i0:]), np.log(sol.U[i0:, N - k]), 1)[0]
        print(f"pendiente de log u_(N-{k}) vs log(T_h - t): {slope:.4f}   (teórica {expo})")
    c = np.polyfit(-np.log(R[i0:]), sol.U[i0:, N - 2], 1)[0]
    print(f"u_(N-2) ≈ c (-ln(T_h - t)) con c = {c:.4f}")


def exp_profiles() -> None:
    p = 1.5
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for N in [10, 20, 40, 80]:
        prob, sol = power_run(p, N)
        mask = blowing_nodes(sol)
        line, = ax.semilogy(prob.x, sol.U[-1], ".-", ms=4, label=f"$N={N}$ (explotan {mask.sum()})")
        ax.semilogy(prob.x[mask], sol.U[-1][mask], "o", mfc="none", color=line.get_color(), ms=8)
    ax.set_xlim(0.6, 1.01)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x, t)$ cerca de $T_h$")
    ax.set_title(r"$f=s^{1.5}$: explotan los nodos $x_N, x_{N-1}, x_{N-2}$ (círculos)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save(fig, "parabolic", "blowup_sets_profiles")


def exp_log() -> None:
    p, N = 2.0, 20
    F = lambda s: s * math.log(s) ** p  # noqa: E731
    dF = lambda s: math.log(s) ** p + p * math.log(s) ** (p - 1)  # noqa: E731
    prob = HeatFlux1D(N, F, dF)
    sol = solve_to_blowup(prob, np.full(N + 1, math.e), lambda m: math.log(max(m, math.e)) ** p,
                          u_stop=1e290)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    for i in range(0, N + 1, 2):
        ax1.plot(sol.t, np.log10(sol.U[:, i]), color=cmap(i / N), lw=1)
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(r"$\log_{10} u_i$")
    ax1.set_title(r"Todos los nodos explotan (color: de $x_0$ a $x_N$)", fontsize=10)
    ax1.grid(alpha=0.3)
    for frac in [0.5, 0.8, 0.9, 0.97, 1.0]:
        j = int(frac * (len(sol.s) - 1))
        ax2.plot(prob.x, np.log10(sol.U[j]), label=rf"$\max u = 10^{{{np.log10(sol.U[j].max()):.0f}}}$")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel(r"$\log_{10} u$")
    ax2.set_title("Perfiles: la explosión es global", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle(rf"$u_x(1)=u\,(\ln u)^2$, $N={N}$ (el problema continuo explota sólo en $x=1$)")
    fig.tight_layout()
    save(fig, "parabolic", "blowup_sets_log")
    print(f"s ln(s)^2: al final min u = 10^{np.log10(sol.U[-1].min()):.1f}, "
          f"max u = 10^{np.log10(sol.U[-1].max()):.1f}")


def exp_exp() -> None:
    N = 20
    E = lambda s: math.exp(min(s, 700.0))  # noqa: E731  (evita desbordes en pasos de prueba)
    prob = HeatFlux1D(N, E, E)
    sol = solve_to_blowup(prob, np.ones(N + 1), lambda m: E(m) / max(m, 1.0), u_stop=600.0)
    R = sol.remaining
    mask = blowing_nodes(sol)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.semilogx(R, sol.U[:, N] + np.log(R), label=r"$u_N + \ln(T_h-t)$")
    ax1.invert_xaxis()
    ax1.set_xlabel("$T_h-t$")
    ax1.set_title(r"$u_N\sim-\ln(T_h-t)$", fontsize=10)
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")
    for k in range(1, 4):
        ax2.semilogx(R, sol.U[:, N - k], label=f"$u_{{N-{k}}}$")
    ax2.invert_xaxis()
    ax2.set_xlabel("$T_h-t$")
    ax2.set_title("Los demás nodos quedan acotados", fontsize=10)
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle(rf"$u_x(1)=e^u$, $N={N}$: explota sólo $x_N$ (nodos que explotan: {mask.sum()})")
    fig.tight_layout()
    save(fig, "parabolic", "blowup_sets_exp")
    print(f"e^s: nodos que explotan = {mask.sum()}")


EXPERIMENTS = {"count": exp_count, "rates": exp_rates, "profiles": exp_profiles, "log": exp_log,
               "exp": exp_exp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=list(EXPERIMENTS))
    args = ap.parse_args()
    for name, fn in EXPERIMENTS.items():
        if args.only in (None, name):
            print(f"\n=== {name} ===")
            fn()


if __name__ == "__main__":
    main()
