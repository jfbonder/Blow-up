r"""Experimento adicional (no está en el paper): ¿es exponencial el tiempo de explosión?

Fernández Bonder–Groisman conjeturan que :math:`T_\sigma/\mathbb E(T_\sigma)`
converge a una exponencial de media 1 cuando :math:`\sigma\to0` (pérdida de
memoria, típica de la metaestabilidad).  Groisman y Saglietti estudiaron el
problema para la discretización espacial (J. Math. Anal. Appl. 385, 2012) y
Groisman, Saglietti y Saintier para la SPDE (Stoch. Proc. Appl. 128, 2018).

Para varios valores de :math:`\sigma` se simulan ``n`` tiempos de explosión
(:math:`u_0\equiv0`, :math:`h=0.02`) y se compara :math:`T/\bar T` con Exp(1):

* coeficiente de variación (= 1 para la exponencial);
* test de Kolmogorov–Smirnov contra una exponencial con media estimada, con
  p-valor por Monte Carlo (``scipy.stats.goodness_of_fit``, que tiene en
  cuenta que el parámetro se estimó);
* gráfico Q-Q y funciones de supervivencia en escala logarítmica.

Para :math:`\sigma` grande se espera un desvío de la exponencial: la
explosión deja de ser un evento raro y aparece un tiempo mínimo de tránsito.

Los tiempos se guardan en ``results/spde/`` (el valor :math:`\sigma=6.36`
comparte la corrida de la Fig. 4).

Uso::

    python -m experiments.spde.exponential_law [--sigmas 6.36 7 8 10] [--n 832] [--seed N]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from blowup.spde import Scheme
from experiments.common import parser, save
from experiments.spde.cache import get_times


def main() -> None:
    p = parser(__doc__.splitlines()[0], seed=5)
    p.add_argument("--sigmas", type=float, nargs="+", default=[6.36, 7.0, 8.0, 10.0])
    p.add_argument("--n", type=int, default=832)
    args = p.parse_args()
    scheme = Scheme()
    rng = np.random.default_rng(12345)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    n_q = args.n
    q_exp = -np.log(1 - (np.arange(1, n_q + 1) - 0.5) / n_q)  # cuantiles de Exp(1)
    ax1.plot(q_exp, q_exp, "k--", lw=1, label="Exp(1)")
    ax2.plot(q_exp, np.exp(-q_exp), "k--", lw=1, label="Exp(1)")

    print(f"{'sigma':>6} {'n':>5} {'media':>9} {'desvío':>9} {'CV':>6} {'mín':>7} {'KS':>6} {'p-valor':>8}")
    for sigma in args.sigmas:
        T = get_times(sigma, args.n, args.seed, scheme)
        T = np.sort(T[np.isfinite(T)])
        S = T / T.mean()
        res = stats.goodness_of_fit(stats.expon, S, known_params={"loc": 0.0}, statistic="ks",
                                    n_mc_samples=2000, rng=rng)
        cv = T.std(ddof=1) / T.mean()
        print(f"{sigma:6g} {T.size:5d} {T.mean():9.3f} {T.std(ddof=1):9.3f} {cv:6.3f} "
              f"{T.min():7.3f} {res.statistic:6.3f} {res.pvalue:8.3f}")
        q = -np.log(1 - (np.arange(1, S.size + 1) - 0.5) / S.size)
        ax1.plot(q, S, ".", ms=3, label=rf"$\sigma={sigma:g}$ (CV$={cv:.2f}$)")
        ax2.semilogy(S, 1 - (np.arange(1, S.size + 1) - 0.5) / S.size, lw=1.2,
                     label=rf"$\sigma={sigma:g}$")

    ax1.set_xlabel("cuantiles de Exp(1)")
    ax1.set_ylabel(r"cuantiles empíricos de $T/\bar T$")
    ax1.set_title("Gráfico Q-Q")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel(r"$s$")
    ax2.set_ylabel(r"$\mathbb{P}(T/\bar T > s)$")
    ax2.set_title("Función de supervivencia")
    ax2.set_xlim(0, 6)
    ax2.set_ylim(1e-3, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle(r"Tiempo de explosión normalizado vs. Exp(1) ($u_0\equiv0$, $h=0.02$)")
    fig.tight_layout()
    save(fig, "spde", "exponential_law")


if __name__ == "__main__":
    main()
