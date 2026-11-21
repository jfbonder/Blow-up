r"""Paso temporal adaptativo para problemas con blow-up.

Implementa la regla de paso de los artículos del grupo (Acosta–Durán–Rossi,
Dávila–Fernández Bonder–Groisman–Rossi–Sued, Fernández Bonder–Groisman):

.. math::  \tau_n = \frac{\tau}{\max\big(1, \psi(U_n)\big)},

donde :math:`\psi` mide la velocidad de crecimiento de la solución.  Para
:math:`u'=f(u)` lo natural es :math:`\psi(U)=f(\|U\|)/\|U\|`; para
:math:`f=u^p` queda la regla clásica :math:`\tau_n=\tau\min(1,\|U_n\|^{1-p})`.
Con esa elección el cambio relativo de la solución en cada paso es
:math:`O(\tau)` y el tiempo de explosión numérico

.. math::  T_\tau = \sum_{n\ge0}\tau_n

es finito, mientras que con paso fijo :math:`t_n=n\tau\to\infty` y la
explosión no se captura.

Dos variantes:

* **Totalmente discreta** (:func:`integrate`): Euler explícito, Euler
  semi-implícito (IMEX: parte lineal implícita, no lineal explícita) o Euler
  implícito (Newton), con la regla de paso anterior.
* **Continua** (:func:`rescaled`): la misma regla como cambio de variable
  :math:`dt/ds = 1/\max(1,\psi(U))`, integrada con un método de alto orden.

El problema es cualquier objeto con ``rhs(t, u)``; para los métodos
implícitos además ``jac(t, u)`` (rala o densa) y, para IMEX,
``linear_operator()`` (la parte lineal :math:`L`, de modo que
:math:`U'=LU+N(U)`).  Las clases de :mod:`blowup.parabolic` cumplen esto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

Method = Literal["explicit", "imex", "implicit"]
Psi = Callable[[np.ndarray], float]


# ---------------------------------------------------------------------------
# Reglas de paso (funciones psi)
# ---------------------------------------------------------------------------

def psi_power(p: float, norm: str = "inf") -> Psi:
    r""":math:`\psi(U)=\|U\|^{p-1}`: regla :math:`\tau_n=\tau\min(1,\|U_n\|^{1-p})`.

    ``norm``: ``"inf"`` (máximo) o ``"l1"`` (suma de valores absolutos; en
    ese caso conviene que el problema provea los pesos, ver :func:`psi_l1`).
    """
    if norm == "inf":
        return lambda U: float(np.max(np.abs(U))) ** (p - 1)
    if norm == "l1":
        return lambda U: float(np.sum(np.abs(U))) ** (p - 1)
    raise ValueError(norm)


def psi_growth(f: Callable[[float], float]) -> Psi:
    r""":math:`\psi(U)=f(m)/m` con :math:`m=\|U\|_\infty` (paso :math:`\tau m/f(m)`).

    Para la EDE :math:`dx=b(x)dt+\dots` de Dávila et al. es :math:`\tau_k=h/b(X_k)`
    con :math:`\psi = b(x)` (ver :func:`psi_drift`).
    """
    def psi(U):
        m = float(np.max(np.abs(U)))
        return f(m) / m if m > 0 else 0.0
    return psi


def psi_drift(b: Callable[[float], float]) -> Psi:
    r""":math:`\psi(X)=b(X)`: la regla :math:`\tau_k=h/b(X_k)` de Dávila et al."""
    return lambda X: float(b(float(np.max(X))))


def psi_l1(weights: np.ndarray, p: float = 2.0) -> Psi:
    r""":math:`\psi(U)=\|U\|_{L^1}^{p-1}` con :math:`\|U\|_{L^1}=\sum_k w_k|u_k|`
    (para :math:`p=2`, la regla :math:`\tau_j=\tau/\|u^j\|_{L^1}` de
    Fernández Bonder–Groisman)."""
    return lambda U: float(np.dot(weights, np.abs(U))) ** (p - 1)


def psi_rhs(problem) -> Psi:
    r"""Regla genérica :math:`\psi(U)=\|U'\|_\infty/\|U\|_\infty` (cambio relativo
    acotado), que no requiere conocer la no linealidad."""
    def psi(U):
        m = float(np.max(np.abs(U)))
        return float(np.max(np.abs(problem.rhs(0.0, U)))) / m if m > 0 else 0.0
    return psi


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveSolution:
    r"""Resultado de una integración adaptativa.

    Attributes
    ----------
    t, U:
        Tiempos y soluciones guardados (cada ``store_every`` pasos y siempre
        el último).
    tau:
        Pasos :math:`\tau_n` en los instantes guardados.
    remaining:
        :math:`T_\tau - t_n`, calculado sumando los pasos posteriores (sin
        cancelación) más una cola geométrica.
    T:
        Tiempo de explosión numérico :math:`T_\tau` (``inf`` si no explotó).
    exploded:
        Si se alcanzó ``u_stop``.
    n_steps:
        Cantidad total de pasos.
    """

    t: np.ndarray
    U: np.ndarray
    tau: np.ndarray
    remaining: np.ndarray
    T: float
    exploded: bool
    n_steps: int


# ---------------------------------------------------------------------------
# Integración totalmente discreta
# ---------------------------------------------------------------------------

def integrate(problem, u0: np.ndarray, tau: float, psi: Psi | None, method: Method = "imex",
              u_stop: float = 1e100, t_max: float = math.inf, max_steps: int = 10**7,
              tau_max: float | None = None, store_every: int = 1, newton_tol: float = 1e-12,
              newton_maxit: int = 30) -> AdaptiveSolution:
    r"""Integra con paso :math:`\tau_n=\min(\tau_{\max},\ \tau/\max(1,\psi(U_n)))`.

    Parameters
    ----------
    problem:
        Objeto con ``rhs(t, u)`` (y ``jac`` / ``linear_operator`` según el método).
    tau:
        Parámetro de paso :math:`\tau`.
    psi:
        Regla de paso (ver ``psi_*``).  ``None`` = paso fijo :math:`\tau`
        (para comparar: con paso fijo no hay tiempo de explosión finito).
    method:
        ``"explicit"`` (Euler explícito: requiere :math:`\tau\lesssim h^2/2`
        por la difusión), ``"imex"`` (:math:`(I-\tau_nL)U_{n+1}=U_n+\tau_nN(U_n)`,
        sin restricción de estabilidad por la difusión) o ``"implicit"``
        (Euler implícito, Newton).
    u_stop:
        Se detiene cuando :math:`\|U_n\|_\infty\ge` ``u_stop``.
    tau_max:
        Cota superior opcional para el paso.
    """
    U = np.array(u0, dtype=float)
    n = U.size
    I = sparse.identity(n, format="csc")
    L = None
    if method == "imex":
        L = sparse.csc_matrix(problem.linear_operator())
    lu_cache: dict[float, object] = {}

    def step_size(U):
        dt = tau if psi is None else tau / max(1.0, psi(U))
        return min(dt, tau_max) if tau_max is not None else dt

    ts, Us, taus = [0.0], [U.copy()], []
    all_tau = []
    t = 0.0
    exploded = False
    k = 0
    while k < max_steps and t < t_max:
        dt = step_size(U)
        if method == "explicit":
            U = U + dt * problem.rhs(t, U)
        elif method == "imex":
            Nl = problem.rhs(t, U) - L @ U
            # Con paso fijo se factoriza una sola vez; con paso variable, cada vez.
            key = dt if psi is None else None
            lu = lu_cache.get(key) if key is not None else None
            if lu is None:
                lu = splu((I - dt * L).tocsc())
                if key is not None:
                    lu_cache[key] = lu
            U = lu.solve(U + dt * Nl)
        elif method == "implicit":
            V = U + dt * problem.rhs(t, U)  # predictor explícito
            for _ in range(newton_maxit):
                G = V - U - dt * problem.rhs(t + dt, V)
                J = problem.jac(t + dt, V)
                J = sparse.csc_matrix(J) if not sparse.issparse(J) else J.tocsc()
                dV = splu((I - dt * J).tocsc()).solve(-G)
                V = V + dV
                if np.max(np.abs(dV)) <= newton_tol * max(1.0, np.max(np.abs(V))):
                    break
            U = V
        else:
            raise ValueError(method)
        t += dt
        k += 1
        all_tau.append(dt)
        m = float(np.max(np.abs(U)))
        if not np.isfinite(m):
            raise FloatingPointError("la solución desbordó: bajar u_stop o tau")
        if m >= u_stop:
            exploded = True
        if k % store_every == 0 or exploded:
            ts.append(t)
            Us.append(U.copy())
        if exploded:
            break
    all_tau = np.array(all_tau)
    # Pasos asociados a cada instante guardado: tau en t_j es el paso que sigue.
    stored_idx = np.r_[0, np.arange(store_every, k + 1, store_every)]
    if stored_idx[-1] != k:
        stored_idx = np.r_[stored_idx, k]
    # Tiempo restante: suma de los pasos posteriores + cola geométrica.
    tail = 0.0
    if exploded and psi is not None and len(all_tau) >= 3:
        r = all_tau[-1] / all_tau[-2]
        tail = all_tau[-1] * r / (1 - r) if 0 < r < 1 else 0.0
    suffix = np.r_[np.cumsum(all_tau[::-1])[::-1], 0.0]  # suffix[j] = sum_{i>=j} tau_i
    remaining = suffix[stored_idx] + tail
    T = (t + tail) if exploded else math.inf
    tau_stored = np.r_[all_tau, np.nan][stored_idx]
    return AdaptiveSolution(np.array(ts), np.array(Us), tau_stored, remaining, T, exploded, k)


# ---------------------------------------------------------------------------
# Versión continua
# ---------------------------------------------------------------------------

def rescaled(problem, u0: np.ndarray, psi: Psi | None = None, **kw):
    r"""Versión continua de la regla: :math:`dt/ds=1/\max(1,\psi(U))`.

    Envuelve :func:`blowup.parabolic.solve_to_blowup` (integración de alto
    orden en :math:`s`, cálculo de :math:`T_h-t` sin cancelación).  Con
    ``psi=None`` usa la regla genérica :math:`\|U'\|/\|U\|`.
    """
    from blowup.parabolic import solve_to_blowup

    return solve_to_blowup(problem, u0, None, psi=psi, **kw)
