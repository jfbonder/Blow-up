r"""Experimento adicional: umbral de explosión para :math:`u_t=u_{xx}+u^p` con Dirichlet.

Análogo del paper de la variedad estable (Acosta–Fernández Bonder–Rossi)
para

.. math::  u_t = u_{xx} + u^p \ \text{en } (0,1),\qquad u(0,t)=u(1,t)=0,
           \qquad u(x,0) = \lambda u_0(x).

Hay una única solución estacionaria positiva :math:`v` (para :math:`p=2`,
:math:`\max v\approx11.797`).  Para :math:`\lambda<\lambda_c` la solución
tiende a 0, para :math:`\lambda>\lambda_c` explota y para
:math:`\lambda=\lambda_c` converge a :math:`v`: :math:`\lambda_c u_0` está en
la variedad estable de :math:`v`, que tiene codimensión uno (la
linealización :math:`\phi''+pv^{p-1}\phi` tiene un único autovalor positivo).

Semidiscretización: diferencias finitas (= P1 + mass lumping) en
:math:`x_i=i/N`.  Se calcula la estacionaria discreta :math:`V_h` por Newton
y :math:`\lambda_{c,h}` por bisección, con criterios rigurosos por
comparación: decaimiento si :math:`U<0.99\,V_h`, explosión si
:math:`U>1.01\,V_h` (en todos los nodos).

Salida:

* tabla de :math:`\lambda_{c,h}` para dos datos (:math:`\sin\pi x` y uno
  asimétrico, :math:`\tfrac{27}{4}x(1-x)^2`), orden observado y extrapolación;
* figura con :math:`\|u(\cdot,t)\|_\infty` para :math:`\lambda=\lambda_{c,h}(1\pm10^{-k})`:
  cuanto más cerca del umbral, más tiempo pasa la solución cerca de
  :math:`v` (del orden de :math:`k\ln10/\mu`) antes de decaer o explotar;
* figura con :math:`V_h`, la estacionaria continua y los datos críticos.

Uso::

    python -m experiments.parabolic.dirichlet_threshold [--p 2]
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from blowup.parabolic import (ReactionDirichlet1D, critical_lambda, power_steady_max,
                              power_steady_profile)
from experiments.common import save

NS = [10, 20, 50, 100, 200, 400]
DATA = {
    r"\sin(\pi x)": lambda x: np.sin(np.pi * x),
    r"\frac{27}{4}x(1-x)^2": lambda x: 27 / 4 * x * (1 - x) ** 2,
}


def make_problem(N: int, p: float) -> tuple[ReactionDirichlet1D, np.ndarray]:
    prob = ReactionDirichlet1D(N, lambda u: np.abs(u) ** p,
                               lambda u: p * np.abs(u) ** (p - 1) * np.sign(u))
    V = prob.steady_state(power_steady_profile(p, prob.xi))
    return prob, V


def lambda_c(prob, V, u0) -> float:
    r = V / u0
    return critical_lambda(prob, u0, 0.9 * r.min(), 1.1 * r.max(), tol=1e-9, equilibrium=V)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--p", type=float, default=2.0)
    args = ap.parse_args()
    p = args.p
    print(f"p = {p}:  max de la estacionaria continua = {power_steady_max(p):.6f}\n")

    # --- Tabla ---------------------------------------------------------------
    results = {name: [] for name in DATA}
    print(f"{'N':>5} {'max V_h':>10} {'mu_h':>9} " + " ".join(f"{'lambda_ch  ' + n:>24}" for n in ["dato 1", "dato 2"]))
    for N in NS:
        prob, V = make_problem(N, p)
        mu = np.linalg.eigvalsh(prob.jac_dense(V)).max()
        row = []
        for name, g in DATA.items():
            lam = lambda_c(prob, V, prob.nodal(g))
            results[name].append(lam)
            row.append(lam)
        print(f"{N:5d} {V.max():10.6f} {mu:9.5f} " + " ".join(f"{v:24.9f}" for v in row), flush=True)
    for name, lams in results.items():
        l1, l2, l3 = lams[-3:]
        order = math.log2((l2 - l1) / (l3 - l2))
        lim = l3 + (l3 - l2) / (2**order - 1)
        print(f"dato ${name}$: orden observado {order:.3f}, lambda_c ≈ {lim:.8f}")

    # --- Dinámica cerca del umbral (N = 100, dato sin(pi x)) -------------------
    N = 100
    prob, V = make_problem(N, p)
    u0 = prob.nodal(DATA[r"\sin(\pi x)"])
    lam_c = lambda_c(prob, V, u0)
    mu = np.linalg.eigvalsh(prob.jac_dense(V)).max()

    def stop(t, u):
        return u.max() - 5 * V.max()

    stop.terminal = True
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("viridis")
    ks = [2, 4, 6, 8]
    for j, k in enumerate(ks):
        for sign, ls in [(+1, "-"), (-1, "--")]:
            lam = lam_c * (1 + sign * 10.0**-k)
            sol = prob.solve(lam * u0, 6.0, events=stop, dense_output=False, max_step=0.01)
            ax.plot(sol.t, sol.y.max(axis=0), ls, color=cmap(j / len(ks)),
                    label=rf"$\lambda_{{c,h}}(1{'+' if sign > 0 else '-'}10^{{-{k}}})$")
    ax.axhline(V.max(), color="k", lw=0.8, ls=":", label=r"$\max V_h$")
    ax.set_ylim(0, 3 * V.max())
    ax.set_xlim(0, 4)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|u(\cdot,t)\|_\infty$")
    ax.set_title(rf"$u_t=u_{{xx}}+u^{{{p:g}}}$, Dirichlet, $u_0=\lambda\sin\pi x$, $N={N}$ "
                 rf"($\lambda_{{c,h}}={lam_c:.6f}$, $\mu={mu:.3f}$)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, "parabolic", "dirichlet_threshold_dynamics")

    # --- Estacionaria y datos críticos --------------------------------------
    xs = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, power_steady_profile(p, xs), "k-", lw=2, alpha=0.4, label="estacionaria continua $v$")
    ax.plot(prob.x, prob.full(V), "k.", ms=3, label=f"$V_h$ ($N={N}$)")
    for name, g in DATA.items():
        u = prob.nodal(g)
        ax.plot(prob.x, prob.full(lambda_c(prob, V, u) * u), label=rf"$\lambda_{{c,h}}\,{name}$")
    ax.set_xlabel("$x$")
    ax.set_title("Estacionaria positiva y datos críticos (en la variedad estable)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "parabolic", "dirichlet_threshold_profiles")


if __name__ == "__main__":
    main()
