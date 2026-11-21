r"""Semidiscretizaciones de la ecuación del calor con flujo no lineal en el borde.

Problema modelo en una dimensión:

.. math::

    u_t = u_{xx} + r(u) \quad\text{en } (0,1),\qquad
    u_x(0,t) = 0,\qquad u_x(1,t) = F(u(1,t)),

con :math:`r` un término de reacción/absorción opcional (por ejemplo
:math:`-\lambda u^p` en Fernández Bonder–Rossi, *Blow-up vs. spurious steady
solutions*) y :math:`F` el flujo (por ejemplo :math:`u^p`, :math:`u^p-u`,
:math:`e^u`).

Elementos finitos P1 con *mass lumping* en la malla uniforme
:math:`x_i = i/N`, :math:`h = 1/N`, :math:`i = 0..N`:

.. math::

    u_0' &= \tfrac{2}{h^2}(u_1-u_0) + r(u_0),\\
    u_i' &= \tfrac{1}{h^2}(u_{i+1}-2u_i+u_{i-1}) + r(u_i),\\
    u_N' &= \tfrac{2}{h^2}(u_{N-1}-u_N) + \tfrac{2}{h}F(u_N) + r(u_N),

es decir :math:`MU' = -AU + B\,F(U) + M\,r(U)` con
:math:`M=\mathrm{diag}(h/2, h, \dots, h, h/2)`,
:math:`A=\frac1h\mathrm{tridiag}(-1,2,-1)` (esquinas :math:`1/h`) y
:math:`B=e_N`.  Es el esquema (3.3) de Acosta–Fernández Bonder–Rossi y
(con :math:`r=0`) el de *On numerical blow-up sets*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

Func = Callable[[np.ndarray], np.ndarray]


@dataclass
class HeatFlux1D:
    """Semidiscretización P1 + mass lumping de :math:`u_t=u_{xx}+r(u)`,
    :math:`u_x(0)=0`, :math:`u_x(1)=F(u(1))`.

    Parameters
    ----------
    N:
        Cantidad de subintervalos (``N + 1`` nodos, :math:`h=1/N`).
    flux, dflux:
        :math:`F` y :math:`F'` (escalares).
    reaction, dreaction:
        :math:`r` y :math:`r'` (vectorizadas), opcionales.
    """

    N: int
    flux: Callable[[float], float]
    dflux: Callable[[float], float]
    reaction: Func | None = None
    dreaction: Func | None = None

    def __post_init__(self):
        self.h = 1.0 / self.N
        self.x = np.linspace(0.0, 1.0, self.N + 1)
        h = self.h
        self.mass = np.full(self.N + 1, h)
        self.mass[[0, -1]] = h / 2
        main = np.full(self.N + 1, 2.0 / h)
        main[[0, -1]] = 1.0 / h
        off = np.full(self.N, -1.0 / h)
        self.stiffness = sparse.diags([off, main, off], [-1, 0, 1], format="csr")
        # M^{-1} A como matriz rala (parte lineal del jacobiano).
        self._MinvA = sparse.diags(1.0 / self.mass) @ self.stiffness

    # ------------------------------------------------------------------
    def rhs(self, t: float, u: np.ndarray) -> np.ndarray:
        h2 = self.h * self.h
        d = np.empty_like(u)
        d[0] = 2.0 * (u[1] - u[0]) / h2
        d[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / h2
        d[-1] = 2.0 * (u[-2] - u[-1]) / h2 + 2.0 / self.h * self.flux(u[-1])
        if self.reaction is not None:
            d += self.reaction(u)
        return d

    def jac(self, t: float, u: np.ndarray):
        J = (-self._MinvA).tolil()
        J[self.N, self.N] += 2.0 / self.h * self.dflux(u[-1])
        J = J.tocsr()
        if self.dreaction is not None:
            J = J + sparse.diags(self.dreaction(u))
        return J

    def jac_dense(self, u: np.ndarray) -> np.ndarray:
        return self.jac(0.0, u).toarray()

    def nodal(self, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Interpolante nodal de ``f``."""
        return np.asarray(f(self.x), dtype=float)

    def lyapunov(self, u: np.ndarray, G: Callable[[float], float]) -> float:
        r""":math:`V_h(U) = \tfrac12 U^TAU - G(u_N)` con :math:`G'=F` (sin reacción)."""
        return 0.5 * float(u @ (self.stiffness @ u)) - G(u[-1])

    # ------------------------------------------------------------------
    def solve(self, u0: np.ndarray, t_end: float, method: str = "LSODA", rtol: float = 1e-10,
              atol: float = 1e-12, events=None, **kw):
        """Integra con ``scipy.integrate.solve_ivp``.

        Para los métodos implícitos (``BDF``, ``Radau``, ``LSODA``) se pasa el
        jacobiano exacto.
        """
        if method in ("BDF", "Radau"):
            kw.setdefault("jac", self.jac)
        elif method == "LSODA":
            kw.setdefault("jac", lambda t, u: self.jac_dense(u))
        return solve_ivp(self.rhs, (0.0, t_end), u0, method=method, rtol=rtol, atol=atol,
                         events=events, **kw)


# ---------------------------------------------------------------------------
# Flujos usados en los papers
# ---------------------------------------------------------------------------

def flux_power_minus_linear(p: float):
    r""":math:`F(s) = s^p - s` (stable manifold) y su derivada."""
    return (lambda s: s**p - s), (lambda s: p * s ** (p - 1) - 1.0)


def flux_power(p: float):
    r""":math:`F(s) = s^p` y su derivada."""
    return (lambda s: s**p), (lambda s: p * s ** (p - 1))


# ---------------------------------------------------------------------------
# Umbral de explosión (variedad estable)
# ---------------------------------------------------------------------------

Outcome = Literal["decay", "blowup", "undecided"]


def classify(problem: HeatFlux1D, u0: np.ndarray, equilibrium: float = 1.0, margin: float = 0.01,
             blowup_level: float | None = None, t_end: float = 1e4, **solve_kw) -> tuple[Outcome, float]:
    r"""Clasifica la solución con dato ``u0`` en decaimiento o explosión.

    Criterios (rigurosos por el principio de comparación, Teorema 3.1 de
    Acosta–Fernández Bonder–Rossi):

    * decaimiento si :math:`\max_i u_i < 1 - \text{margin}`;
    * explosión si :math:`\min_i u_i > 1 + \text{margin}` o
      :math:`u_N >` ``blowup_level``.

    Devuelve ``(resultado, tiempo en que se decidió)``.
    """
    lo, hi = equilibrium - margin, equilibrium + margin
    if u0.max() < lo:
        return "decay", 0.0
    if u0.min() > hi or (blowup_level is not None and u0[-1] > blowup_level):
        return "blowup", 0.0

    def ev_decay(t, u):
        return u.max() - lo

    def ev_blow(t, u):
        return u.min() - hi

    ev_decay.terminal = ev_blow.terminal = True
    events = [ev_decay, ev_blow]
    if blowup_level is not None:
        def ev_level(t, u):
            return u[-1] - blowup_level
        ev_level.terminal = True
        events.append(ev_level)

    sol = problem.solve(u0, t_end, events=events, **solve_kw)
    if sol.status == 1:
        if sol.t_events[0].size:
            return "decay", float(sol.t_events[0][0])
        return "blowup", float(sol.t[-1])
    if sol.status == -1:  # el integrador falló: típicamente por explosión
        return ("blowup", float(sol.t[-1])) if sol.y[-1, -1] > hi else ("undecided", float(sol.t[-1]))
    return "undecided", float(sol.t[-1])


def critical_parameter(problem: HeatFlux1D, family: Callable[[float], np.ndarray], lo: float, hi: float,
                       tol: float = 1e-7, **classify_kw) -> float:
    r"""Bisección sobre una familia monótona de datos ``family(a)``: devuelve el
    :math:`a^*` tal que el dato decae para :math:`a<a^*` y explota para
    :math:`a>a^*` (el dato ``family(a*)`` está en la variedad estable)."""
    out_lo, _ = classify(problem, family(lo), **classify_kw)
    out_hi, _ = classify(problem, family(hi), **classify_kw)
    if out_lo != "decay" or out_hi != "blowup":
        raise ValueError(f"intervalo inicial inválido: {lo}->{out_lo}, {hi}->{out_hi}")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        out, _ = classify(problem, family(mid), **classify_kw)
        if out == "decay":
            lo = mid
        elif out == "blowup":
            hi = mid
        else:
            raise RuntimeError(f"no se pudo clasificar el parámetro {mid}")
    return 0.5 * (lo + hi)


def critical_lambda(problem: HeatFlux1D, u0: np.ndarray, lo: float, hi: float, tol: float = 1e-7,
                    **classify_kw) -> float:
    r""":math:`\lambda_{c,h}`: :math:`\lambda U_0` decae si
    :math:`\lambda<\lambda_{c,h}` y explota si :math:`\lambda>\lambda_{c,h}`."""
    return critical_parameter(problem, lambda lam: lam * u0, lo, hi, tol, **classify_kw)


def unstable_eigenvalue_continuous(p: float) -> float:
    r"""Autovalor positivo de la linealización continua en :math:`u\equiv1`:
    :math:`\mu=k^2` con :math:`k\tanh k = p-1`."""
    from scipy.optimize import brentq

    k = brentq(lambda k: k * math.tanh(k) - (p - 1), 1e-9, 50)
    return k * k
