r"""Figura 1 de Acosta–Fernández Bonder–Rossi: solución con :math:`\lambda` apenas supercrítico.

:math:`N=100`, :math:`\lambda=0.9771` (como en el paper; aquí
:math:`\lambda_{c,h}\approx0.977024`).  La solución se aplana rápidamente
cerca de :math:`u\equiv1`, permanece ahí un tiempo largo (se aleja por la
dirección inestable con tasa :math:`\mu\approx1.44`) y luego explota,
primero en :math:`x=1`.

Uso::

    python -m experiments.parabolic.stable_manifold_fig1 [--lam 0.9771] [--t-end 6]
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from blowup.parabolic import HeatFlux1D, flux_power_minus_linear
from experiments.common import save
from experiments.parabolic.stable_manifold_table1 import P, u0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lam", type=float, default=0.9771)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--t-end", type=float, default=6.0)
    args = ap.parse_args()

    prob = HeatFlux1D(args.N, *flux_power_minus_linear(P))
    t_eval = np.linspace(0, args.t_end, 241)

    def stop(t, u):  # se corta al llegar a u(1)=1.7, como el gráfico del paper
        return u[-1] - 1.7

    stop.terminal = True
    sol = prob.solve(args.lam * prob.nodal(u0), args.t_end, t_eval=t_eval, events=stop)
    print(f"t final = {sol.t[-1]:.3f}   u(1) = {sol.y[-1, -1]:.4f}   u(0) = {sol.y[0, -1]:.4f}")

    X, T = np.meshgrid(prob.x, sol.t)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(X, T, sol.y.T, rstride=4, cstride=4, linewidth=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$u(x,t)$")
    ax.view_init(elev=22, azim=-55)
    ax.set_title(rf"$N={args.N}$, $\lambda={args.lam}$")
    save(fig, "parabolic", "stable_manifold_fig1")


if __name__ == "__main__":
    main()
