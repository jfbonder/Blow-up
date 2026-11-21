r"""Experimentos ilustrativos para Acosta–Fernández Bonder–Groisman–Rossi,
*Numerical approximation of a parabolic problem with a nonlinear boundary
condition in several space dimensions*.

El paper es teórico.  Problema :math:`u_t=\Delta u` en :math:`\Omega=(0,1)^2`,
:math:`\partial u/\partial\eta = u^p` en :math:`\partial\Omega`,
semidiscretizado con diferencias finitas (Ejemplo 3 del paper) en la forma
general :math:`MU'=-AU+BU^p`.  Dato :math:`u_0=1+\tfrac12(x^2+y^2)`, con
máximo en la esquina :math:`(1,1)`.

Se integra hasta :math:`T_h-t\sim10^{-70}`–:math:`10^{-170}` con el tiempo
reescalado :math:`dt/ds=\min(1,\|U\|_\infty^{1-p})` (``solve_to_blowup``).

Experimentos (``--only``):

* ``propagation``: Teorema 1.5.  Se identifica :math:`F` (nodos con
  :math:`(T_h-t)^{1/(p-1)}u_k\to l_k=(m_k/((p-1)b_k))^{1/(p-1)}`), se calcula
  la distancia en el grafo de :math:`A` y se verifica que explotan
  exactamente los nodos con :math:`d(k)\le K`,
  :math:`\frac{K+2}{K+1}<p\le\frac{K+1}{K}`, para varios :math:`h`.
* ``rate``: Teorema 1.4.  La tasa numérica es
  :math:`(T_h-t)^{-1/(p-1)}`, distinta de la continua
  :math:`(T-t)^{-1/(2(p-1))}`.  La pendiente local de
  :math:`\log\|U\|_\infty` vs :math:`\log(T_h-t)` pasa de :math:`-1/(2(p-1))`
  a :math:`-1/(p-1)` cuando :math:`T_h-t\approx h^2`.
* ``times``: Teorema 1.3, :math:`T_h\to T`.

Uso::

    python -m experiments.parabolic.several_dims [--only propagation|rate|times]
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from blowup.parabolic import solve_to_blowup, square_fd
from experiments.common import save


def run(n: int, p: float, u_stop: float | None = None):
    S = square_fd(n, p)
    X, Y = S.coords.T
    u0 = 1 + 0.5 * (X**2 + Y**2)
    sol = solve_to_blowup(S, u0, lambda m: m ** (p - 1), u_stop=u_stop or 10 ** (250 / p),
                          method="BDF", rtol=1e-9, atol=1e-12)
    return S, sol


def analyse(S, sol, decades: float = 10.0):
    """Devuelve (F, nodos que explotan, distancia a F)."""
    R = sol.remaining
    j = int(np.searchsorted(-R, -R[-1] * 10**decades))
    grow = sol.U[-1] - sol.U[j] > 1e-6 * (1 + np.abs(sol.U[-1]))
    l = S.limit_profile()
    y = R[-1] ** (1 / (S.p - 1)) * sol.U[-1]
    F = np.isfinite(l) & (np.abs(y / l - 1) < 1e-2)
    return F, grow, S.graph_distance(F)


def K_of(p: float) -> int:
    # (K+2)/(K+1) < p <= (K+1)/K  <=>  K <= 1/(p-1) < K+1
    return math.floor(1 / (p - 1) + 1e-12)


def exp_propagation() -> None:
    PS, NS = [3.0, 2.0, 1.8, 1.5, 1.4], [10, 20, 40]
    print(f"{'p':>5} {'K':>3} {'N':>4} {'|F|':>4} {'explotan':>9} {'#{d<=K}':>8} {'max d':>6}")
    for p in PS:
        for n in NS:
            S, sol = run(n, p)
            F, grow, d = analyse(S, sol)
            K = K_of(p)
            print(f"{p:5g} {K:3d} {n:4d} {F.sum():4d} {grow.sum():9d} {(d <= K).sum():8d} "
                  f"{d[grow].max():6.0f}" + ("" if np.array_equal(grow, d <= K) else "   <-- distinto"),
                  flush=True)

    # Figura: nodos que explotan cerca de la esquina, p = 1.4 (K = 2).
    p = 1.4
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, n in zip(axes, [10, 20]):
        S, sol = run(n, p)
        F, grow, d = analyse(S, sol)
        X, Y = S.coords.T
        logu = np.log10(sol.U[-1])
        sc = ax.scatter(X, Y, c=logu, s=900 / n, cmap="viridis", marker="s")
        ax.scatter(X[grow], Y[grow], s=2400 / n, facecolors="none", edgecolors="r", lw=1.5,
                   label=f"explotan ({grow.sum()} nodos)")
        for k in np.flatnonzero(d <= 3):
            ax.annotate(f"{d[k]:.0f}", (X[k], Y[k]), ha="center", va="center", fontsize=7, color="w")
        ax.set_xlim(1 - 5.5 / n, 1 + 0.5 / n)
        ax.set_ylim(1 - 5.5 / n, 1 + 0.5 / n)
        ax.set_aspect("equal")
        ax.set_title(f"$N={n}$ ($h=1/{n}$)", fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
        fig.colorbar(sc, ax=ax, label=r"$\log_{10}u$ cerca de $T_h$")
    fig.suptitle(rf"$p={p}$ ($K={K_of(p)}$): explotan los nodos a distancia $\leq K$ de la esquina "
                 r"(número = distancia en el grafo)", fontsize=10)
    fig.tight_layout()
    save(fig, "parabolic", "several_dims_propagation")


def exp_rate() -> None:
    p = 2.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for n in [10, 20, 40, 80]:
        S, sol = run(n, p, u_stop=1e60)
        R = sol.remaining
        m = sol.U.max(axis=1)
        sel = R < 0.5 * sol.T
        lr, lm = np.log(R[sel]), np.log(m[sel])
        slope = np.gradient(lm, lr)
        ax1.loglog(R[sel], m[sel], label=f"$N={n}$")
        ax2.semilogx(R[sel] * n**2, slope, label=f"$N={n}$")
        print(f"N={n}: T_h={sol.T:.6f}; pendiente final {slope[-1]:.4f}")
    ax1.invert_xaxis()
    ax1.set_xlabel("$T_h-t$")
    ax1.set_ylabel(r"$\|U(t)\|_\infty$")
    ax1.set_title("Tasa de explosión", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    ax2.axhline(-1 / (2 * (p - 1)), color="k", ls="--", lw=0.8, label=r"continua $-\frac{1}{2(p-1)}$")
    ax2.axhline(-1 / (p - 1), color="k", ls=":", lw=0.8, label=r"numérica $-\frac{1}{p-1}$")
    ax2.invert_xaxis()
    ax2.set_xlim(1e4, 1e-8)
    ax2.set_ylim(-1.15, -0.2)
    ax2.set_xlabel(r"$(T_h-t)/h^2$")
    ax2.set_ylabel(r"pendiente local $d\log\|U\|_\infty/d\log(T_h-t)$")
    ax2.set_title(r"El cambio de tasa ocurre en $T_h-t\approx h^2$", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle(rf"$(0,1)^2$, $\partial u/\partial\eta=u^{{{p:g}}}$, $u_0=1+\frac{{1}}{{2}}(x^2+y^2)$")
    fig.tight_layout()
    save(fig, "parabolic", "several_dims_rate")


def exp_times() -> None:
    p = 2.0
    print("Tiempo de explosión T_h (p = 2)")
    Ts = []
    for n in [10, 20, 40, 80, 160]:
        _, sol = run(n, p, u_stop=1e20)
        Ts.append(sol.T)
        extra = ""
        if len(Ts) > 1:
            extra = f"   diferencia con N={n // 2}: {Ts[-2] - Ts[-1]:.3e}"
        if len(Ts) > 2:
            extra += f"   orden observado {math.log2((Ts[-3] - Ts[-2]) / (Ts[-2] - Ts[-1])):.2f}"
        print(f"  N={n:4d}: T_h = {sol.T:.6f}{extra}", flush=True)
    order = math.log2((Ts[-3] - Ts[-2]) / (Ts[-2] - Ts[-1]))
    print(f"  T (Richardson) ≈ {Ts[-1] - (Ts[-2] - Ts[-1]) / (2**order - 1):.6f}")


EXPERIMENTS = {"propagation": exp_propagation, "rate": exp_rate, "times": exp_times}


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
