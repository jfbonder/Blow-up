r"""Experimentos ilustrativos para Fernández Bonder–Rossi, *Blow-up vs. spurious steady solutions*.

El paper es teórico.  Problema:

.. math::  u_t = u_{xx} - \lambda u^p,\qquad u_x(0)=0,\qquad u_x(1)=u^q,

semidiscretizado con P1 + mass lumping (ec. (1.3) del paper, la absorción
entra en todos los nodos).  Resultados:

* el continuo explota (para datos grandes) si :math:`p<2q-1`;
* el semidiscreto explota sólo si :math:`p\le q` (Teorema 1.3);
* para :math:`q<p<2q-1` aparece una solución estacionaria espuria atractiva
  con :math:`w_N\approx(2/(\lambda h))^{1/(p-q)}` (Teorema 1.4).

Valores: :math:`q=2`, :math:`\lambda=1`; :math:`p=2.5` (régimen espurio),
:math:`p=1.5` (explosión) y :math:`p=4` (global).  Aquí :math:`N` es la
cantidad de subintervalos, :math:`h=1/N` (en el paper hay :math:`N` nodos).

Experimentos (``--only``):

* ``steady``: todas las estacionarias positivas (por *shooting* desde
  :math:`x=0`, ec. (3.2), con el :math:`F_2` corregido), su :math:`w_N`,
  :math:`c_h=w_Nh^{1/(p-q)}\to(2/\lambda)^{1/(p-q)}` y su estabilidad
  (autovalores del jacobiano).
* ``dynamics``: :math:`p=2.5`, dato grande: para cada :math:`N` la solución
  sigue a la del continuo hasta saturar en la espuria, cada vez más alta.
* ``regimes``: :math:`p=1.5` (explota, :math:`T_h` converge) y :math:`p=4`
  (global, el límite no depende de :math:`h`).

Uso::

    python -m experiments.parabolic.spurious [--only steady|dynamics|regimes]
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from blowup.parabolic import HeatFlux1D, flux_power
from experiments.common import save

Q, LAM = 2.0, 1.0
NS = [10, 20, 40, 80, 160]


def problem(N: int, p: float, q: float = Q, lam: float = LAM) -> HeatFlux1D:
    return HeatFlux1D(N, *flux_power(q), reaction=lambda u: -lam * np.abs(u) ** p,
                      dreaction=lambda u: -lam * p * np.abs(u) ** (p - 1) * np.sign(u))


def shoot(w1: float, N: int, p: float, q: float = Q, lam: float = LAM):
    """Recursión de (3.1) desde x=0.  Devuelve (W, G(w1)); W=None si desborda."""
    h = 1.0 / N
    w = np.empty(N + 1)
    w[0] = w1
    d = 0.5 * lam * h * h * w1**p  # w_1 - w_0 (F_2 corregido: lambda h^2 / 2)
    with np.errstate(over="raise", invalid="raise"):
        try:
            for k in range(1, N + 1):
                w[k] = w[k - 1] + d
                d = d + lam * h * h * w[k] ** p
            # Última ecuación: 2(w_{N-1}-w_N)/h^2 - lam w_N^p + (2/h) w_N^q
            dN = w[N] - w[N - 1]
            G = -2.0 * dN / h**2 - lam * w[N] ** p + 2.0 / h * w[N] ** q
        except FloatingPointError:
            return None, -math.inf
    return w, G


def steady_states(N: int, p: float) -> list[np.ndarray]:
    """Todas las estacionarias positivas: ceros de G en una malla logarítmica de w_1."""
    grid = np.logspace(-6, 6, 4000)
    G = np.array([shoot(w, N, p)[1] for w in grid])
    roots = []
    for a, b, ga, gb in zip(grid[:-1], grid[1:], G[:-1], G[1:]):
        if np.isfinite(ga) and np.sign(ga) != np.sign(gb):
            # tanh acota G (que puede valer -inf por desborde) sin cambiar su signo.
            f = lambda s: np.tanh(shoot(math.exp(s), N, p)[1] / 1e6)  # noqa: E731
            s = brentq(f, math.log(a), math.log(b), xtol=1e-15)
            W, _ = shoot(math.exp(s), N, p)
            if W is not None:
                roots.append(W)
    # Refinamiento por Newton sobre el sistema completo.
    prob = problem(N, p)
    out = []
    for W in roots:
        for _ in range(30):
            dW = np.linalg.solve(prob.jac_dense(W), -prob.rhs(0, W))
            W = W + dW
            if np.abs(dW).max() < 1e-13 * W.max():
                break
        out.append(W)
    return out


def exp_steady() -> None:
    p = 2.5
    r = 1 / (p - Q)
    c_lim = (2 / LAM) ** r
    print(f"p={p}, q={Q}, lambda={LAM}: (2/lambda)^(1/(p-q)) = {c_lim:.4f}")
    print(f"{'N':>5} {'#est.':>6} {'w_1':>10} {'w_N (espuria)':>14} {'c_h':>8} {'cota estab.':>12}"
          f" {'max Re mu':>11}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    hs, wN = [], []
    for N in NS:
        prob = problem(N, p)
        h = 1 / N
        Ws = sorted(steady_states(N, p), key=lambda W: W[-1])
        mus = [np.linalg.eigvals(prob.jac_dense(W)).real.max() for W in Ws]
        sp = Ws[-1]
        bound = (2 * Q / (p * LAM * h)) ** r
        print(f"{N:5d} {len(Ws):6d} {sp[0]:10.6f} {sp[-1]:14.4f} {sp[-1] * h**r:8.4f} {bound:12.4f}"
              f" {mus[-1]:11.4f}")
        hs.append(h)
        wN.append(sp[-1])
        ax2.semilogy(prob.x, sp, ".-", ms=3, label=f"$N={N}$")
    hs, wN = np.array(hs), np.array(wN)
    ax1.loglog(hs, wN, "o-", label="$w_N$ de la espuria")
    ax1.loglog(hs, (2 / (LAM * hs)) ** r, "k--", label=r"$(2/(\lambda h))^{1/(p-q)}$")
    ax1.set_xlabel("$h$")
    ax1.set_title(r"La espuria crece como $h^{-1/(p-q)} = h^{-2}$", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$w$")
    ax2.set_title("Perfiles de la estacionaria espuria (crecientes)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle(rf"$u_t=u_{{xx}}-u^{{{p}}}$, $u_x(1)=u^{{{Q:g}}}$: estacionaria espuria ($q<p<2q-1$)")
    fig.tight_layout()
    save(fig, "parabolic", "spurious_steady")


def exp_dynamics() -> None:
    p = 2.5
    K = 3.0  # dato constante grande: el continuo explota
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for N in NS:
        prob = problem(N, p)
        sol = prob.solve(np.full(N + 1, K), 2.0, t_eval=np.r_[np.linspace(0, 0.1, 2000), 2.0])
        Ws = sorted(steady_states(N, p), key=lambda W: W[-1])
        ax.semilogy(sol.t[:-1], sol.y[-1, :-1], label=f"$N={N}$")
        ax.axhline(Ws[-1][-1], color=ax.lines[-1].get_color(), ls=":", lw=0.8)
        print(f"N={N}: u_N(t=2) = {sol.y[-1, -1]:.4f}   w_N espuria = {Ws[-1][-1]:.4f}")
    ax.set_xlim(0, 0.1)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u(1,t)$")
    ax.set_title(rf"$p={p}$, dato $u_0\equiv{K:g}$: el continuo explota, el discreto satura en la espuria (punteado)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save(fig, "parabolic", "spurious_dynamics")


def exp_regimes() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # p = 1.5 <= q: explosión, T_h converge.
    p, K = 1.5, 3.0
    print(f"p={p} (p <= q): tiempo de explosión semidiscreto")

    def big(t, u):
        return u[-1] - 1e8

    big.terminal = True
    for N in NS:
        prob = problem(N, p)
        sol = prob.solve(np.full(N + 1, K), 10.0, events=big)
        Th = sol.t_events[0][0] if sol.t_events[0].size else math.nan
        # Cota (2.2): T_h - t <= u_N^{1-q}/(q-1) cuando u_N^{q-1} > 2/h.
        print(f"  N={N:4d}: T_h ≈ {Th:.6f}")
        ax1.semilogy(sol.t, sol.y[-1], label=f"$N={N}$, $T_h\\approx{Th:.4f}$")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$u(1,t)$")
    ax1.set_title(rf"$p={p}\leq q$: explosión, $T_h$ converge", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    # p = 4 > 2q-1: global, el límite no depende de h.
    p, K = 4.0, 3.0
    print(f"p={p} (p > 2q-1): estado final")
    for N in NS:
        prob = problem(N, p)
        sol = prob.solve(np.full(N + 1, K), 20.0, t_eval=np.linspace(0, 20, 2001))
        print(f"  N={N:4d}: u_N(t=20) = {sol.y[-1, -1]:.6f}, u_0(t=20) = {sol.y[0, -1]:.6f}")
        ax2.plot(sol.t, sol.y[-1], label=f"$N={N}$")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("$u(1,t)$")
    ax2.set_xlim(0, 5)
    ax2.set_title(rf"$p={p}>2q-1$: global, límite independiente de $h$", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle(rf"$u_t=u_{{xx}}-u^p$, $u_x(1)=u^{{{Q:g}}}$, $u_0\equiv{K:g}$")
    fig.tight_layout()
    save(fig, "parabolic", "spurious_regimes")


EXPERIMENTS = {"steady": exp_steady, "dynamics": exp_dynamics, "regimes": exp_regimes}


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
