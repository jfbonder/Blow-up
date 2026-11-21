r"""Tabla 1 de Acosta–Fernández Bonder–Rossi: valor crítico :math:`\lambda_{c,h}`.

Problema :math:`u_t=u_{xx}` en :math:`(0,1)`, :math:`u_x(0)=0`,
:math:`u_x(1)=u^2-u`, dato :math:`\lambda u_0`, :math:`u_0=\tfrac12\cos(2\pi x)+1`
(interpolante nodal).  Para cada :math:`N` se calcula por bisección el
:math:`\lambda_{c,h}` tal que :math:`\lambda U_0` decae a 0 si
:math:`\lambda<\lambda_{c,h}` y explota si :math:`\lambda>\lambda_{c,h}`.

Criterios de clasificación (rigurosos por comparación, Teorema 3.1):
decaimiento si :math:`\max u<0.99`, explosión si :math:`\min u>1.01`.
Integración: LSODA con jacobiano exacto, ``rtol=1e-10``.

Además: orden de convergencia observado, extrapolación de Richardson para
:math:`\lambda_c` y autovalor inestable de la linealización en
:math:`(1,\dots,1)` comparado con el continuo (:math:`k^2`, :math:`k\tanh k=p-1`).

Uso::

    python -m experiments.parabolic.stable_manifold_table1
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from blowup.parabolic import (HeatFlux1D, critical_lambda, flux_power_minus_linear,
                              unstable_eigenvalue_continuous)
from experiments.common import save

P = 2.0
NS = [10, 20, 50, 100, 200, 400]
PAPER = {10: 0.9755, 20: 0.9766, 50: 0.9770, 100: 0.9770}


def u0(x):
    return 0.5 * np.cos(2 * np.pi * x) + 1.0


def main() -> None:
    F, dF = flux_power_minus_linear(P)
    mu_c = unstable_eigenvalue_continuous(P)
    lams, mus = [], []
    print(f"{'N':>5} {'lambda_ch':>12} {'paper':>8} {'mu_h':>9}")
    for N in NS:
        prob = HeatFlux1D(N, F, dF)
        lam = critical_lambda(prob, prob.nodal(u0), 0.9, 1.05, tol=1e-9)
        mu = np.linalg.eigvals(prob.jac_dense(np.ones(N + 1))).real.max()
        lams.append(lam)
        mus.append(mu)
        print(f"{N:5d} {lam:12.8f} {PAPER.get(N, float('nan')):8.4f} {mu:9.5f}", flush=True)
    lams = np.array(lams)
    # Orden observado con tres mallas consecutivas que se duplican (200, 400 y 100).
    i100, i200, i400 = NS.index(100), NS.index(200), NS.index(400)
    order = np.log2((lams[i200] - lams[i100]) / (lams[i400] - lams[i200]))
    lam_c = lams[i400] + (lams[i400] - lams[i200]) / (2**order - 1)
    print(f"\norden observado: {order:.3f}   lambda_c (Richardson) ≈ {lam_c:.8f}")
    print(f"autovalor inestable continuo: {mu_c:.5f}")

    h = 1.0 / np.array(NS)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.loglog(h, np.abs(lams - lam_c), "o-", label=r"$|\lambda_{c,h}-\lambda_c|$ (este código)")
    hp = np.array([1 / N for N in PAPER])
    ax.loglog(hp[:2], [lam_c - PAPER[10], lam_c - PAPER[20]], "s--", label="Tabla 1 del paper (N=10, 20)")
    ax.loglog(h, 0.5 * np.abs(lams[0] - lam_c) * (h / h[0]) ** 2, "k:", label=r"pendiente 2")
    ax.set_xlabel("$h$")
    ax.set_ylabel("error")
    ax.set_title(rf"Convergencia de $\lambda_{{c,h}}$ ($\lambda_c\approx{lam_c:.6f}$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save(fig, "parabolic", "stable_manifold_lambda_convergence")


if __name__ == "__main__":
    main()
