r"""Experimento adicional: un corte de la variedad estable de :math:`u\equiv1`.

En el plano de datos iniciales :math:`u_0 = a + b\cos(2\pi x)` se busca, para
cada :math:`b`, el valor crítico :math:`a^*(b)` por bisección en :math:`a`
(válida por el principio de comparación: aumentar :math:`a` aumenta el dato).
La curva :math:`a^*(b)` es la intersección de la variedad estable discreta
con ese plano; se compara para varios :math:`N`.  El rayo de la Tabla 1,
:math:`\lambda(1+\tfrac12\cos 2\pi x)`, corta la curva en
:math:`(b, a) = (\lambda_c/2, \lambda_c)`.

Uso::

    python -m experiments.parabolic.stable_manifold_slice
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blowup.parabolic import HeatFlux1D, critical_parameter, flux_power_minus_linear
from experiments.common import save
from experiments.parabolic.stable_manifold_table1 import P

NS = [4, 10, 100]
BS = np.linspace(-0.6, 0.6, 25)
LAMBDA_C = 0.97703  # límite de la Tabla 1 (ver stable_manifold_table1)


def main() -> None:
    F, dF = flux_power_minus_linear(P)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for N in NS:
        prob = HeatFlux1D(N, F, dF)
        cosx = np.cos(2 * np.pi * prob.x)
        a_star = []
        for b in BS:
            # a = |b|: dato >= 0 (debe decaer); a = 1 + |b|: dato >= 1 (explota).
            lo, hi = abs(b), 1.05 + abs(b)
            a_star.append(critical_parameter(prob, lambda a, b=b: a + b * cosx, lo, hi, tol=1e-6))
        a_star = np.array(a_star)
        ax.plot(BS, a_star, "o-", ms=3, label=f"$N={N}$")
        print(f"N={N}: a*(b=0)={a_star[len(BS) // 2]:.6f}  a*(b=0.5)={np.interp(0.5, BS, a_star):.5f}",
              flush=True)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.scatter([LAMBDA_C / 2], [LAMBDA_C], marker="*", s=140, color="k", zorder=5,
               label=r"rayo de la Tabla 1: $\lambda_c(1+\frac{1}{2}\cos 2\pi x)$")
    ax.set_xlabel("$b$")
    ax.set_ylabel("$a^*(b)$")
    ax.set_title(r"Variedad estable de $u\equiv1$ $\cap$ $\{a + b\cos(2\pi x)\}$ ($p=2$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "parabolic", "stable_manifold_slice")


if __name__ == "__main__":
    main()
