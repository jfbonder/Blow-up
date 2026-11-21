r"""Euler–Maruyama adaptativo para EDEs con explosión.

Implementa el esquema de

    J. Dávila, J. Fernández Bonder, P. Groisman, J. D. Rossi, M. Sued,
    *Numerical analysis of stochastic differential equations with explosions*
    (ver ``papers/notes/DFGRS_sde-explosions.md``).

Para la EDE

.. math::  dx = b(x)\,dt + \sigma(x)\,dW,\qquad x(0) = z > 0,

el esquema es

.. math::

    X_{k+1} = X_k + \tau_k b(X_k) + \sigma(X_k)\,\Delta W_k,\qquad
    \tau_k = \frac{h}{b(X_k)},\qquad \Delta W_k = \sqrt{\tau_k}\,Z_k,

con :math:`Z_k \sim N(0,1)` i.i.d.  Con esta elección el incremento de deriva
es exactamente :math:`h` en cada paso.  El algoritmo se detiene cuando
:math:`|X_k| \ge M`; el tiempo :math:`t_k` en ese momento es
:math:`R_h^M = \inf\{t : |X(t)| \ge M\}`, que aproxima el tiempo de explosión
(Teorema 1.4 del paper).

Convenciones (el paper es inconsistente con los índices): :math:`t_0 = 0`,
:math:`X_0 = z` y el primer paso es :math:`\tau_0 = h/b(z)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayFunc = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Coeficientes del paper (Sección 6)
# ---------------------------------------------------------------------------

def b_paper(x):
    r""":math:`b(\xi) = |\xi|^{1.1} + 0.1` (Sección 6 del paper)."""
    return np.abs(x) ** 1.1 + 0.1


def sigma_paper(x):
    r""":math:`\sigma(\xi) = \sqrt{|\xi|^{1.1} + 0.1}` (Sección 6 del paper)."""
    return np.sqrt(np.abs(x) ** 1.1 + 0.1)


Z0_PAPER = 10.0


# ---------------------------------------------------------------------------
# Una trayectoria
# ---------------------------------------------------------------------------

@dataclass
class SDEPath:
    """Trayectoria del esquema adaptativo.

    Attributes
    ----------
    t, X:
        Tiempos :math:`t_k` y valores :math:`X_k`, ``k = 0..n``.
    exploded:
        ``True`` si se alcanzó ``|X| >= M`` (entonces ``t[-1]`` es
        :math:`R_h^M`); ``False`` si se agotó ``max_steps``.
    h, M:
        Parámetros usados.
    """

    t: np.ndarray
    X: np.ndarray
    exploded: bool
    h: float
    M: float

    @property
    def stopping_time(self) -> float:
        r""":math:`R_h^M` (``nan`` si la trayectoria no llegó a ``M``)."""
        return float(self.t[-1]) if self.exploded else math.nan

    @property
    def n_steps(self) -> int:
        return len(self.t) - 1


def simulate_path(
    b: ArrayFunc,
    sigma: ArrayFunc,
    z: float,
    h: float,
    M: float,
    rng: np.random.Generator,
    max_steps: int = 10**8,
    block: int = 2**16,
) -> SDEPath:
    """Simula una trayectoria del esquema hasta que ``|X| >= M``.

    Guarda todos los :math:`(t_k, X_k)`.  Los coeficientes ``b`` y ``sigma``
    deben aceptar arrays de NumPy; se evalúan sobre escalares, así que el
    costo es del orden de unos pocos microsegundos por paso.
    """
    ts = [0.0]
    xs = [float(z)]
    t, x = 0.0, float(z)
    sqrt_h = math.sqrt(h)
    exploded = abs(x) >= M
    k = 0
    while not exploded and k < max_steps:
        Z = rng.standard_normal(block)
        for zk in Z:
            bx = float(b(x))
            sx = float(sigma(x))
            tau = h / bx
            # sigma * sqrt(tau) * Z = sqrt(h) * sigma / sqrt(b) * Z
            x = x + h + sqrt_h * (sx / math.sqrt(bx)) * zk
            t = t + tau
            ts.append(t)
            xs.append(x)
            k += 1
            if abs(x) >= M:
                exploded = True
                break
            if k >= max_steps:
                break
    return SDEPath(np.array(ts), np.array(xs), exploded, h, M)


# ---------------------------------------------------------------------------
# Monte Carlo de tiempos de parada
# ---------------------------------------------------------------------------

def stopping_times(
    b: ArrayFunc,
    sigma: ArrayFunc,
    z: float,
    h: float,
    M: float,
    n_paths: int,
    rng: np.random.Generator,
    max_steps: int = 10**8,
) -> np.ndarray:
    r"""Muestras independientes de :math:`R_h^M` para ``n_paths`` trayectorias.

    Vectorizado sobre trayectorias (sólo se avanzan las que siguen activas).
    Las trayectorias que no llegan a ``M`` en ``max_steps`` pasos devuelven
    ``nan``.
    """
    X = np.full(n_paths, float(z))
    T = np.zeros(n_paths)
    active = np.flatnonzero(np.abs(X) < M)
    sqrt_h = math.sqrt(h)
    k = 0
    while active.size and k < max_steps:
        xa = X[active]
        bx = b(xa)
        sx = sigma(xa)
        T[active] += h / bx
        xa = xa + h + sqrt_h * (sx / np.sqrt(bx)) * rng.standard_normal(active.size)
        X[active] = xa
        active = active[np.abs(xa) < M]
        k += 1
    T[active] = np.nan
    return T


def deterministic_tail(b: Callable[[float], float], M: float) -> float:
    r""":math:`\int_M^\infty ds / b(s)`: tiempo que le falta a la EDO
    :math:`\bar x' = b(\bar x)` para explotar desde :math:`M`.

    Útil para estimar cuánto se aleja :math:`R_h^M` del tiempo de explosión.
    """
    from scipy.integrate import quad

    val, _ = quad(lambda s: 1.0 / float(b(s)), M, np.inf, limit=200)
    return val
