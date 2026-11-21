r"""Validación del paso adaptativo totalmente discreto (:mod:`blowup.adaptive`).

Regla :math:`\tau_n=\tau/\max(1,\psi(U_n))`; tiempo de explosión numérico
:math:`T_\tau=\sum_n\tau_n`.

Experimentos (``--only``):

* ``ode``: :math:`u'=u^2`, :math:`u(0)=1` (:math:`T=1`).  Euler explícito e
  implícito con :math:`\tau_n=\tau\min(1,u_n^{-1})`: :math:`T_\tau\to T` con
  orden 1 (para Euler explícito, :math:`T_\tau=1+\tau` exactamente).  Con paso
  fijo :math:`t_n=n\tau\to\infty`: la solución discreta está definida para
  todo :math:`n` y crece doblemente exponencial en :math:`n`, en lugar de
  geométricamente.
* ``pde``: :math:`u_t=u_{xx}`, :math:`u_x(0)=0`, :math:`u_x(1)=u^2`, :math:`N=20`
  (P1 + lumping), regla genérica :math:`\psi=\|U'\|/\|U\|`.  Euler explícito
  (con :math:`\tau_n\le0.4h^2`), IMEX e implícito: :math:`T_\tau\to T_h`
  con orden 1 (:math:`T_h` de referencia: versión continua con
  integrador de alto orden).  El esquema discreto reproduce la tasa
  :math:`u_N\sim(T_\tau-t_n)^{-1}` y la explosión de exactamente dos nodos
  (:math:`u_{N-1}\sim-\ln(T_\tau-t_n)`).

Uso::

    python -m experiments.adaptive.discrete [--only ode|pde]
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from blowup.adaptive import integrate, psi_power, psi_rhs, rescaled
from blowup.parabolic import HeatFlux1D, flux_power
from experiments.common import save


class ScalarPower:
    """u' = u^p."""

    def __init__(self, p):
        self.p = p

    def rhs(self, t, u):
        return u**self.p

    def jac(self, t, u):
        return np.array([[self.p * u[0] ** (self.p - 1)]])

    def linear_operator(self):
        return np.zeros((1, 1))


def exp_ode() -> None:
    P = ScalarPower(2.0)
    taus = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    print(f"{'tau':>7} {'T explícito':>14} {'T implícito':>14}")
    errs = {"explicit": [], "implicit": []}
    for tau in taus:
        row = []
        for m in errs:
            r = integrate(P, np.array([1.0]), tau, psi_power(2.0), method=m, u_stop=1e150)
            errs[m].append(abs(r.T - 1.0))
            row.append(r.T)
        print(f"{tau:7g} {row[0]:14.10f} {row[1]:14.10f}")
    for m, lab in [("explicit", "Euler explícito"), ("implicit", "Euler implícito")]:
        ax1.loglog(taus, errs[m], "o-", label=lab)
    ax1.loglog(taus, taus, "k:", label=r"$\tau$")
    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"$|T_\tau - T|$")
    ax1.set_title(r"$u'=u^2$: error en el tiempo de explosión (orden 1)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    tau = 0.05
    ra = integrate(P, np.array([1.0]), tau, psi_power(2.0), method="explicit", u_stop=1e150)
    rf = integrate(P, np.array([1.0]), tau, None, method="explicit", u_stop=1e150)
    ax2.loglog(np.arange(1, len(ra.t) + 1), ra.U[:, 0], label=rf"adaptativo: $T_\tau={ra.T:.4f}$")
    ax2.loglog(np.arange(1, len(rf.t) + 1), np.minimum(rf.U[:, 0], 1e150), label=rf"paso fijo: $t_n=n\tau$ (llega a $10^{{150}}$ en $t={rf.t[-1]:.2f}$)")
    ax2.set_xlabel("$n+1$ (paso)")
    ax2.set_ylabel("$u_n$")
    ax2.set_title(rf"Crecimiento de $u_n$ ($\tau={tau}$): geométrico vs. doblemente exponencial", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "adaptive", "discrete_ode")
    print(f"paso fijo tau={tau}: {len(rf.t) - 1} pasos hasta 1e150; adaptativo: {ra.n_steps} pasos")


def exp_pde() -> None:
    N, p = 20, 2.0
    P = HeatFlux1D(N, *flux_power(p))
    u0 = np.ones(N + 1)
    ref = rescaled(P, u0, psi_power(p), u_stop=1e100).T
    print(f"T_h de referencia (versión continua, LSODA rtol=1e-10): {ref:.10f}")
    psi = psi_rhs(P)
    configs = {
        "explicit": dict(taus=[0.004, 0.002, 0.001, 0.0005], tau_max=0.4 / N**2),
        "imex": dict(taus=[0.1, 0.05, 0.02, 0.01, 0.005], tau_max=None),
        "implicit": dict(taus=[0.1, 0.05, 0.02, 0.01], tau_max=None),
    }
    labels = {"explicit": r"Euler explícito ($\tau_n\leq0.4h^2$)", "imex": "IMEX", "implicit": "Euler implícito"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, cfg in configs.items():
        errs = []
        for tau in cfg["taus"]:
            r = integrate(P, u0, tau, psi, method=m, u_stop=1e100, tau_max=cfg["tau_max"])
            errs.append(abs(r.T - ref))
            print(f"{m:9s} tau={tau:<6g} T_tau={r.T:.8f}  error={r.T - ref:+.3e}  pasos={r.n_steps}", flush=True)
        ax1.loglog(cfg["taus"], errs, "o-", label=labels[m])
        order = np.polyfit(np.log(cfg["taus"]), np.log(errs), 1)[0]
        print(f"  orden observado ({m}): {order:.2f}")
    t = np.array(configs["imex"]["taus"])
    ax1.loglog(t, 0.3 * t, "k:", label=r"$\propto\tau$")
    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"$|T_\tau-T_h|$")
    ax1.set_title(rf"$u_x(1)=u^2$, $N={N}$: convergencia de $T_\tau$", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    r = integrate(P, u0, 0.01, psi, method="imex", u_stop=1e100)
    R = r.remaining
    for k, lab in [(N, "$u_N$"), (N - 1, "$u_{N-1}$"), (N - 2, "$u_{N-2}$")]:
        ax2.loglog(R, r.U[:, k], label=lab)
    ax2.loglog(R, r.U[-1, N] * R[-1] / R, "k:", lw=1, label="pendiente $-1$")
    ax2.invert_xaxis()
    ax2.set_xlabel(r"$T_\tau-t_n$")
    ax2.set_title(r"IMEX, $\tau=0.01$: tasa $u_N\sim(T_\tau-t_n)^{-1}$; explotan $u_N$, $u_{N-1}$", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "adaptive", "discrete_pde")
    i0 = int(np.searchsorted(-R, -R[-1] * 1e40))
    slope = np.polyfit(np.log(R[i0:]), np.log(r.U[i0:, N]), 1)[0]
    j = int(np.searchsorted(-R, -R[-1] * 1e10))
    grow = r.U[-1] - r.U[j] > 1e-6 * (1 + np.abs(r.U[-1]))
    print(f"IMEX tau=0.01: pendiente de u_N = {slope:.4f}; nodos que explotan = {grow.sum()}")


EXPERIMENTS = {"ode": exp_ode, "pde": exp_pde}


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
