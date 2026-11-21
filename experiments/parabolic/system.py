r"""Experimentos ilustrativos para Acosta–Fernández Bonder–Groisman–Rossi,
*Simultaneous vs. non-simultaneous blow-up in numerical approximations of a
parabolic system with nonlinear boundary conditions*.

El paper es teórico.  Sistema en :math:`(0,1)`:

.. math::  u_t=u_{xx},\quad v_t=v_{xx},\qquad
           \partial_\eta u=u^{p_{11}}v^{p_{12}},\quad \partial_\eta v=u^{p_{21}}v^{p_{22}}
           \quad (x=0,1),

con P1 + mass lumping.  Se integra hasta la explosión con el tiempo
reescalado genérico :math:`dt/ds=\min(1,\|Y\|_\infty/\|Y'\|_\infty)`.

Experimentos (``--only``):

* ``existence``: Teorema 1.2.  En el plano :math:`p_{11}=p_{22}=a`,
  :math:`p_{12}=p_{21}=b`, hay explosión sii :math:`a>1` o :math:`b>1-a`.
  Criterio numérico: se comparan los tiempos que tarda :math:`\|Y\|_\infty`
  en pasar de :math:`10^{30}` a :math:`10^{60}` y de :math:`10^{60}` a
  :math:`10^{90}`; el cociente es :math:`<1` si hay explosión (tipo potencia),
  1 si el crecimiento es exponencial y enorme si es polinomial.
* ``nonsimultaneous``: Teoremas 1.4–1.5.  Con :math:`p_{11}=3`,
  :math:`p_{12}=1`, :math:`p_{22}=0` y dato :math:`(U_0,V_0)=(2,1)`: :math:`V`
  queda acotada sii :math:`p_{21}<p_{11}-1=2`, para todo :math:`h`.  En el
  caso no simultáneo :math:`\|U\|_\infty\sim(T_h-t)^{-1/(p_{11}-1)}`.

Uso::

    python -m experiments.parabolic.system [--only existence|nonsimultaneous]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from blowup.parabolic import CoupledFlux1D, solve_to_blowup
from experiments.common import save


def run(S: CoupledFlux1D, U0: float, V0: float, u_stop: float = 1e100):
    Y0 = np.concatenate([np.full(S.N + 1, U0), np.full(S.N + 1, V0)])
    return solve_to_blowup(S, Y0, None, u_stop=u_stop, n_out=6000)


def blows_up(S: CoupledFlux1D) -> tuple[bool, float]:
    """Explota si el tiempo para pasar de 1e60 a 1e90 es mucho menor que el
    de 1e30 a 1e60 (para crecimiento exponencial el cociente es 1, para
    polinomial es enorme, para explosión tipo potencia es < 1)."""
    try:
        sol = run(S, 2.0, 0.5, u_stop=1e90)  # dato asimétrico: U != V
    except RuntimeError:  # crecimiento polinomial lento: no llega al umbral
        return False, np.inf
    m = sol.U.max(axis=1)
    # Se usa el tiempo restante (sin cancelación) en lugar de restar tiempos.
    r30, r60 = (sol.remaining[int(np.searchsorted(m, v))] for v in (1e30, 1e60))
    ratio = (r60 - sol.remaining[-1]) / (r30 - r60)
    return ratio < 0.5, sol.T


def exp_existence() -> None:
    vals = np.linspace(0.0, 1.4, 15)
    N = 10
    res = np.zeros((len(vals), len(vals)), bool)
    wrong = 0
    for i, a in enumerate(vals):
        for j, b in enumerate(vals):
            S = CoupledFlux1D(N, a, b, b, a)
            res[i, j], _ = blows_up(S)
            wrong += res[i, j] != S.blows_up_ode()
    print(f"puntos clasificados: {res.size}; distintos de la predicción del Teorema 1.2: {wrong}")
    A, B = np.meshgrid(vals, vals, indexing="ij")
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.scatter(A[res], B[res], marker="x", color="tab:red", label="explota")
    ax.scatter(A[~res], B[~res], marker="o", facecolors="none", color="tab:blue", label="global")
    aa = np.linspace(0, 1, 50)
    ax.plot(aa, 1 - aa, "k-", lw=1.2, label=r"$(a-1)^2=b^2$")
    ax.plot([1, 1], [0, 1.45], "k--", lw=1, label="$a=1$")
    ax.set_xlabel(r"$a = p_{11} = p_{22}$")
    ax.set_ylabel(r"$b = p_{12} = p_{21}$")
    ax.set_title(r"Teorema 1.2: explota sii $a>1$ o $(a-1)^2<b^2$" + f"\n($N={N}$, dato $U_0=2$, $V_0=0.5$)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.05, 1.45)
    ax.set_ylim(-0.05, 1.45)
    ax.grid(alpha=0.3)
    save(fig, "parabolic", "system_existence")


def exp_nonsimultaneous() -> None:
    p11, p12, p22 = 3.0, 1.0, 0.0
    P21 = [0.0, 1.0, 1.5, 2.5, 3.0, 4.0]
    NS = [10, 20, 40, 80]
    print(f"p11={p11}, p12={p12}, p22={p22}; dato (U0, V0) = (2, 1).  Umbral: p21 < p11 - 1 = {p11 - 1:g}")
    print(f"{'p21':>5} " + " ".join(f"{'N=' + str(N):>22}" for N in NS))
    for p21 in P21:
        row = []
        for N in NS:
            sol = run(CoupledFlux1D(N, p11, p12, p21, p22), 2.0, 1.0, u_stop=1e60)
            R = sol.remaining
            V = sol.U[:, N + 1:].max(axis=1)
            j = int(np.searchsorted(-R, -R[-1] * 1e10))
            bounded = V[-1] - V[j] < 1e-8 * V[-1]
            row.append(("V acotada " if bounded else "V explota ") + f"(T={sol.T:.4f})")
        print(f"{p21:5g} " + " ".join(f"{r:>22}" for r in row), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    N = 40
    for ax, p21 in zip(axes, [1.0, 3.0]):
        sol = run(CoupledFlux1D(N, p11, p12, p21, p22), 2.0, 1.0, u_stop=1e60)
        R = sol.remaining
        U = sol.U[:, : N + 1].max(axis=1)
        V = sol.U[:, N + 1:].max(axis=1)
        ax.loglog(R, U, label=r"$\|U\|_\infty$")
        ax.loglog(R, V, label=r"$\|V\|_\infty$")
        ref = U[-1] * (R / R[-1]) ** (-1 / (p11 - 1))
        ax.loglog(R, ref, "k:", lw=1, label=r"pendiente $-\frac{1}{p_{11}-1}$")
        ax.invert_xaxis()
        ax.set_xlabel("$T_h-t$")
        kind = "no simultáneo" if p21 < p11 - 1 else "simultáneo"
        ax.set_title(rf"$p_{{21}}={p21:g}$ ({kind}), $N={N}$", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(rf"$p_{{11}}={p11:g}$, $p_{{12}}={p12:g}$, $p_{{22}}={p22:g}$: "
                 r"$V$ queda acotada sii $p_{21}<p_{11}-1$")
    fig.tight_layout()
    save(fig, "parabolic", "system_nonsimultaneous")


EXPERIMENTS = {"existence": exp_existence, "nonsimultaneous": exp_nonsimultaneous}


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
