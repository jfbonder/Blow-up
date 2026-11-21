r"""Experimento sin figura de Fernández Bonder–Groisman: :math:`\sigma \le 5` hasta :math:`t=1000`.

El paper dice: "corrimos el código con :math:`\sigma\le5` hasta :math:`t=1000`
y no observamos explosiones sino un comportamiento metaestable".  Para
:math:`\sigma = 3, 4, 5` se simula una trayectoria hasta :math:`t = 1000` y se
grafica :math:`\|u(\cdot,t)\|_\infty`, junto con el máximo de la solución
estacionaria (:math:`\approx 11.8`), que hay que superar para explotar.

Tarda varios minutos (≈6·10⁶ pasos por trayectoria).

Uso::

    python -m experiments.spde.exp0_metastable [--seed N] [--sigmas 3 4 5] [--t-max 1000]
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from blowup.spde import Scheme, simulate_path, stationary_max
from experiments.common import parser, save


def main() -> None:
    p = parser(__doc__.splitlines()[0], seed=6)
    p.add_argument("--sigmas", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    p.add_argument("--t-max", type=float, default=1000.0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(len(args.sigmas), 1, figsize=(8, 2.4 * len(args.sigmas)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sigma in zip(axes, args.sigmas):
        path = simulate_path(sigma, rng, Scheme(), t_max=args.t_max, record_dt=1.0, store_stride=200)
        status = f"explotó en t={path.T:.2f}" if math.isfinite(path.T) else "no explotó"
        print(f"sigma={sigma:g}: {status};  máx. de ||u||_inf = {path.umax.max():.2f}")
        ax.plot(path.t, path.umax, lw=0.4)
        ax.axhline(stationary_max(), color="k", ls="--", lw=0.8)
        ax.set_ylabel(r"$\|u\|_\infty$")
        ax.set_title(rf"$\sigma={sigma:g}$: {status}", fontsize=10)
        ax.set_ylim(0, 1.15 * max(stationary_max(), min(path.umax.max(), 40)))
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("$t$")
    fig.suptitle(r"Comportamiento metaestable para $\sigma\leq5$ ($u_0\equiv0$; línea: máx. de la estacionaria)")
    fig.tight_layout()
    save(fig, "spde", "exp0_metastable")


if __name__ == "__main__":
    main()
