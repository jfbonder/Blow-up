r"""Figura 3 de Dávila et al.: densidad de :math:`R_h^M` para distintos :math:`h`.

Estimador de núcleo (gaussiano, ``scipy.stats.gaussian_kde``) de la densidad
de :math:`R_h^M` con 1000 trayectorias para :math:`h = 1, 0.5, 0.1`
(ilustra el Teorema 1.4).

Elecciones propias (el paper no las especifica):

* :math:`h = 0.1` (el texto dice 0.1 y lo confirma :math:`\tau_1 \simeq 0.008`;
  la leyenda original dice 0.01).
* :math:`M = 10^4`: con ese umbral la densidad tiene el pico en
  :math:`t\approx4.0` y soporte :math:`\approx[3.4, 4.9]` como en la figura
  original; con :math:`M=10^5` el pico queda en :math:`\approx4.8`.
  Se puede cambiar con ``--M``.
* Números aleatorios independientes para cada :math:`h`.

Uso::

    python -m experiments.sode.fig3_kde [--seed N] [--M 1e4] [--n-paths 1000]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from blowup.sode import Z0_PAPER, b_paper, sigma_paper, stopping_times
from experiments.common import parser, save

HS = [1.0, 0.5, 0.1]
STYLES = [("tab:blue", "-"), ("tab:green", "--"), ("tab:red", "-.")]


def main() -> None:
    p = parser(__doc__.splitlines()[0], seed=3)
    p.add_argument("--M", type=float, default=1e4, help="umbral (default: %(default)g)")
    p.add_argument("--n-paths", type=int, default=1000, help="trayectorias por h (default: %(default)s)")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    grid = np.linspace(2.5, 7.0, 900)
    for h, (c, ls) in zip(HS, STYLES):
        R = stopping_times(b_paper, sigma_paper, Z0_PAPER, h, args.M, args.n_paths, rng)
        tau1 = h / float(b_paper(Z0_PAPER))
        print(f"h={h:<4}  tau_1={tau1:.4f}  media={R.mean():.4f}  desv={R.std(ddof=1):.4f}  "
              f"min={R.min():.3f}  max={R.max():.3f}")
        kde = gaussian_kde(R)
        ax.plot(grid, kde(grid), color=c, ls=ls, label=f"$h={h:g}$")
    if args.M == 1e4:
        ax.set_xlim(3.2, 5.2)  # mismo rango que la figura original
    ax.set_xlabel("$R_h^M$")
    ax.set_ylabel("densidad estimada")
    ax.set_title(rf"Estimador de núcleo de $R_h^M$ ({args.n_paths} trayectorias, $M={args.M:g}$)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "sode", "fig3_kde")


if __name__ == "__main__":
    main()
