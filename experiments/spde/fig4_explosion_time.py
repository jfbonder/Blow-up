r"""Figura 4 de Fernández Bonder–Groisman: distribución del tiempo de explosión.

832 trayectorias con :math:`\sigma = 6.36`, :math:`u_0\equiv0`, :math:`h=0.02`,
explosión cuando :math:`\max u \ge 10^{13}`.  Estimador de núcleo de la
densidad y box-plot.  El paper reporta media 46.8834 y desvío 43.8857.

La corrida tarda del orden de 20 minutos; los tiempos se guardan en
``results/spde/`` y se reutilizan (``--recompute`` para rehacerlos).

Uso::

    python -m experiments.spde.fig4_explosion_time [--seed N] [--n 832] [--recompute]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from blowup.spde import SIGMA_PAPER, Scheme
from experiments.common import parser, save
from experiments.spde.cache import get_times


def main() -> None:
    p = parser(__doc__.splitlines()[0], seed=5)
    p.add_argument("--n", type=int, default=832)
    p.add_argument("--sigma", type=float, default=SIGMA_PAPER)
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()

    T = get_times(args.sigma, args.n, args.seed, Scheme(), recompute=args.recompute)
    T = T[np.isfinite(T)]
    q1, med, q3 = np.percentile(T, [25, 50, 75])
    print(f"n={T.size}  media={T.mean():.4f}  desvío={T.std(ddof=1):.4f}  "
          f"mediana={med:.2f}  Q1={q1:.2f}  Q3={q3:.2f}  máx={T.max():.1f}")
    print("paper:     media=46.8834  desvío=43.8857")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    grid = np.linspace(0, T.max() * 1.05, 800)
    # Reflexión en t=0 para que el núcleo no pierda masa en el borde.
    kde = gaussian_kde(np.r_[T, -T])
    ax1.plot(grid, 2 * kde(grid), label="estimador de núcleo")
    ax1.plot(grid, np.exp(-grid / T.mean()) / T.mean(), "k--", lw=1,
             label=rf"Exp(media $={T.mean():.1f}$)")
    ax1.set_xlabel("tiempo de explosión")
    ax1.set_ylabel("densidad")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.boxplot(T, whis=1.5)
    ax2.set_xticks([])
    ax2.set_ylabel("tiempo de explosión")
    ax2.grid(alpha=0.3)
    fig.suptitle(rf"Tiempo de explosión: {T.size} trayectorias, $\sigma={args.sigma}$ "
                 rf"(media {T.mean():.1f}, desvío {T.std(ddof=1):.1f})")
    fig.tight_layout()
    save(fig, "spde", "fig4_explosion_time")


if __name__ == "__main__":
    main()
