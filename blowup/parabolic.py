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


@dataclass
class ReactionDirichlet1D:
    r"""Semidiscretización de :math:`u_t = u_{xx} + f(u)` en :math:`(0,1)` con
    :math:`u(0)=u(1)=0`.

    P1 + mass lumping (= diferencias finitas de segundo orden) en
    :math:`x_i=i/N`; las incógnitas son los nodos interiores
    :math:`u_1,\dots,u_{N-1}`:

    .. math::  u_i' = \tfrac{1}{h^2}(u_{i+1}-2u_i+u_{i-1}) + f(u_i),
               \qquad u_0 = u_N = 0.
    """

    N: int
    f: Func
    df: Func

    def __post_init__(self):
        self.h = 1.0 / self.N
        self.x = np.linspace(0.0, 1.0, self.N + 1)
        self.xi = self.x[1:-1]
        n = self.N - 1
        self._lap = sparse.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1],
                                 format="csr") / self.h**2

    def rhs(self, t: float, u: np.ndarray) -> np.ndarray:
        d = -2.0 * u
        d[1:] += u[:-1]
        d[:-1] += u[1:]
        return d / self.h**2 + self.f(u)

    def jac(self, t: float, u: np.ndarray):
        return (self._lap + sparse.diags(self.df(u))).tocsr()

    def jac_dense(self, u: np.ndarray) -> np.ndarray:
        return self.jac(0.0, u).toarray()

    def nodal(self, g: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Interpolante nodal de ``g`` en los nodos interiores."""
        return np.asarray(g(self.xi), dtype=float)

    def full(self, u: np.ndarray) -> np.ndarray:
        """Agrega los valores de borde (cero)."""
        return np.concatenate(([0.0], u, [0.0]))

    solve = HeatFlux1D.solve

    def steady_state(self, guess: np.ndarray, tol: float = 1e-12, maxit: int = 50) -> np.ndarray:
        """Solución estacionaria discreta por Newton desde ``guess``."""
        from scipy.sparse.linalg import spsolve

        u = np.array(guess, dtype=float)
        for _ in range(maxit):
            F = self.rhs(0.0, u)
            du = spsolve(self.jac(0.0, u).tocsc(), -F)
            u += du
            if np.abs(du).max() < tol * max(1.0, np.abs(u).max()):
                return u
        raise RuntimeError("Newton no convergió")


def power_steady_max(p: float) -> float:
    r"""Máximo de la solución positiva de :math:`v''+v^p=0`, :math:`v(0)=v(1)=0`:

    .. math::  M = \Big(2\sqrt{\tfrac{p+1}{2}}\,J\Big)^{2/(p-1)},\qquad
               J = \int_0^1\frac{ds}{\sqrt{1-s^{p+1}}} = \frac{1}{p+1}B\Big(\frac{1}{p+1},\frac12\Big).

    (Para :math:`p=2`, :math:`M\approx11.7967`.)
    """
    from scipy.special import beta

    J = beta(1 / (p + 1), 0.5) / (p + 1)
    return (2 * math.sqrt((p + 1) / 2) * J) ** (2 / (p - 1))


def power_steady_profile(p: float, x: np.ndarray) -> np.ndarray:
    r"""Perfil continuo :math:`v(x)` de la solución positiva de :math:`v''+v^p=0`
    (Dirichlet), por integración de la ecuación de energía."""
    from scipy.integrate import quad
    from scipy.optimize import brentq

    M = power_steady_max(p)
    c = math.sqrt((p + 1) / 2) * M ** ((1 - p) / 2)

    def x_of(v):  # x(v) para la mitad izquierda; sustitución s = v/M
        if v <= 0:
            return 0.0
        return c * quad(lambda s: (1 - s ** (p + 1)) ** -0.5, 0, v / M)[0]

    out = np.empty_like(np.asarray(x, dtype=float))
    for k, xk in enumerate(np.asarray(x, dtype=float)):
        y = min(xk, 1 - xk)
        out[k] = 0.0 if y <= 0 else (M if y >= 0.5 else brentq(lambda v: x_of(v) - y, 0, M))
    return out


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


def classify(problem, u0: np.ndarray, equilibrium: float | np.ndarray = 1.0, margin: float = 0.01,
             blowup_level: float | None = None, t_end: float = 1e4, **solve_kw) -> tuple[Outcome, float]:
    r"""Clasifica la solución con dato ``u0`` en decaimiento o explosión
    comparándola con un equilibrio positivo :math:`W` (escalar o vector).

    Criterios (rigurosos por el principio de comparación cuando :math:`W` es
    el único equilibrio positivo y separa la cuenca de 0 de la explosión,
    p. ej. Teorema 3.1 de Acosta–Fernández Bonder–Rossi):

    * decaimiento si :math:`u < (1-\text{margin})\,W` en todos los nodos;
    * explosión si :math:`u > (1+\text{margin})\,W` en todos los nodos o
      :math:`\max u >` ``blowup_level``.

    Devuelve ``(resultado, tiempo en que se decidió)``.
    """
    lo = (1 - margin) * np.asarray(equilibrium, dtype=float)
    hi = (1 + margin) * np.asarray(equilibrium, dtype=float)
    if np.all(u0 < lo):
        return "decay", 0.0
    if np.all(u0 > hi) or (blowup_level is not None and u0.max() > blowup_level):
        return "blowup", 0.0

    def ev_decay(t, u):
        return np.max(u - lo)

    def ev_blow(t, u):
        return np.min(u - hi)

    ev_decay.terminal = ev_blow.terminal = True
    events = [ev_decay, ev_blow]
    if blowup_level is not None:
        def ev_level(t, u):
            return u.max() - blowup_level
        ev_level.terminal = True
        events.append(ev_level)

    sol = problem.solve(u0, t_end, events=events, **solve_kw)
    if sol.status == 1:
        if sol.t_events[0].size:
            return "decay", float(sol.t_events[0][0])
        return "blowup", float(sol.t[-1])
    if sol.status == -1:  # el integrador falló: típicamente por explosión
        blew = np.all(sol.y[:, -1] > hi) or sol.y[:, -1].max() > 10 * np.max(hi)
        return ("blowup" if blew else "undecided"), float(sol.t[-1])
    return "undecided", float(sol.t[-1])


def critical_parameter(problem, family: Callable[[float], np.ndarray], lo: float, hi: float,
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
            # La solución quedó numéricamente sobre la variedad estable hasta
            # t_end: el umbral está a menos de la precisión del integrador.
            return mid
    return 0.5 * (lo + hi)


def critical_lambda(problem, u0: np.ndarray, lo: float, hi: float, tol: float = 1e-7,
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


# ---------------------------------------------------------------------------
# Integración hasta la explosión con reescalamiento del tiempo
# ---------------------------------------------------------------------------

@dataclass
class BlowupSolution:
    r"""Solución integrada hasta muy cerca del tiempo de explosión.

    Attributes
    ----------
    s:
        Tiempo reescalado (equiespaciado).
    t:
        Tiempo físico :math:`t(s)`.
    U:
        Solución, ``U[j]`` es el vector en ``s[j]``.
    remaining:
        :math:`T_h - t(s)`, calculado como :math:`\int_s^\infty g` (sin
        cancelación, aunque :math:`T_h-t` sea :math:`10^{-100}`).
    T:
        Tiempo de explosión estimado :math:`T_h`.
    """

    s: np.ndarray
    t: np.ndarray
    U: np.ndarray
    remaining: np.ndarray
    T: float


def solve_to_blowup(problem, u0: np.ndarray, growth: Callable[[float], float], u_stop: float = 1e100,
                    n_out: int = 4000, s_max: float = 1e6, rtol: float = 1e-10, atol: float = 1e-12,
                    method: str = "LSODA") -> BlowupSolution:
    r"""Integra :math:`U'=\Phi(U)` hasta que :math:`\max U\ge` ``u_stop``.

    Usa el reescalamiento del tiempo

    .. math::  \frac{dt}{ds} = g(U) = \frac{1}{\max(1, \text{growth}(\max U))},
               \qquad \frac{dU}{ds} = g(U)\,\Phi(U),

    que es la versión continua de la regla de paso adaptativo
    :math:`\tau_n=\tau\min(1, \max U_n/f(\max U_n))` (para :math:`f=s^p`,
    :math:`\tau_n=\tau\min(1,\|U_n\|_\infty^{1-p})`).  ``growth(m)`` es
    :math:`f(m)/m`.  En la variable :math:`s` la solución crece
    exponencialmente y :math:`t(s)\to T_h` cuando :math:`s\to\infty`.

    El tiempo restante :math:`T_h-t(s)=\int_s^\infty g` se calcula integrando
    :math:`g` hacia atrás desde el final (más una cola geométrica), de modo
    que tiene precisión relativa completa.
    """

    def g_of(U):
        m = float(np.max(U))
        return 1.0 / max(1.0, growth(m))

    def F(s, y):
        U = y[:-1]
        g = g_of(U)
        return np.append(g * problem.rhs(0.0, U), g)

    def stop(s, y):
        return np.max(y[:-1]) - u_stop

    stop.terminal = True
    y0 = np.append(np.asarray(u0, float), 0.0)
    sol = solve_ivp(F, (0.0, s_max), y0, method=method, rtol=rtol, atol=atol, events=stop,
                    dense_output=True)
    if sol.status != 1:
        raise RuntimeError(f"no se alcanzó u_stop={u_stop:g}: {sol.message}")
    s_end = float(sol.t[-1])
    s = np.linspace(0.0, s_end, n_out)
    Y = sol.sol(s)
    U = Y[:-1].T
    t = Y[-1]
    g = np.array([g_of(u) for u in U])
    # g es casi exponencial en s: se integra cada tramo exactamente para una
    # exponencial, ds (g1 - g2) / ln(g1 / g2), y la cola como g_end / tasa.
    ds = s[1] - s[0]
    lr = np.log(g[:-1]) - np.log(g[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        seg = np.where(np.abs(lr) > 1e-12, ds * (g[:-1] - g[1:]) / lr, 0.5 * ds * (g[:-1] + g[1:]))
    rate = lr[-1] / ds
    tail = g[-1] / rate if rate > 0 else 0.0
    rem_int = np.append(np.cumsum(seg[::-1])[::-1], 0.0) + tail
    T = float(t[-1] + tail)
    # Lejos de T_h la diferencia directa es exacta; cerca, la integral evita
    # la cancelación.
    rem_direct = T - t
    remaining = np.where(rem_direct > 1e-6 * T, rem_direct, rem_int)
    return BlowupSolution(s, t, U, remaining, T)
