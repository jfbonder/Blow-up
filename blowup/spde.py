r"""Ecuación de reacción-difusión con ruido blanco espacio-temporal.

Implementa el esquema de la Sección 5 de

    J. Fernández Bonder, P. Groisman, *Time–space white noise eliminates
    global solutions in reaction diffusion equations*
    (ver ``papers/notes/FG_white-noise-blowup.md``)

para

.. math::  u_t = u_{xx} + f(u) + \sigma \dot W(x,t),\qquad x\in(0,1),
           \qquad u(0,t)=u(1,t)=0,

con :math:`f(u) = (u_+)^2` por defecto.

Discretización
--------------
* Espacio: diferencias finitas de segundo orden en :math:`x_k = kh`,
  :math:`k=0,\dots,N`, :math:`N=1/h`; incógnitas :math:`u_1,\dots,u_{N-1}`
  (:math:`u_0=u_N=0`).
* Ruido: :math:`\frac{\sigma}{\sqrt h}\,dw_k` con :math:`w_k` movimientos
  brownianos estándar independientes, es decir, por paso
  :math:`\sigma\sqrt{\tau_j/h}\,\xi_k`, :math:`\xi_k\sim N(0,1)` (escala
  :math:`\sqrt{1/(h\,\tau_j)}` para :math:`\dot W`).  Es la lectura coherente
  con la prueba del Teorema 4.1 y reproduce la amplitud de las fluctuaciones
  de la Fig. 1 del paper.
* Tiempo: Euler explícito con paso adaptativo

  .. math::  \tau_j = \min\Big(\tau_{\max},\ \frac{\tau}{\|u^j\|_{L^1}}\Big),
             \qquad \|u\|_{L^1} = h\sum_k |u_k|.

  El paper usa :math:`\tau_j = \tau/\sum_k h u^j_k`; agregamos el valor
  absoluto y la cota :math:`\tau_{\max}` (por defecto :math:`0.4h^2`, dentro
  de la condición de estabilidad :math:`h^2/2` de Euler explícito), que el
  paper no especifica.  Sin la cota el paso sería infinito en :math:`t=0`
  (:math:`u_0\equiv0`).
* Explosión: se detiene cuando :math:`\max_k u_k \ge` ``u_max``
  (:math:`10^{13}` en el paper); el tiempo en ese momento es el tiempo de
  explosión numérico.

Cerca de la explosión :math:`\|u\|_{L^1}\approx h\max u`, así que
:math:`\tau_j \max u \approx \tau/h`: en cada paso el máximo crece en un
factor :math:`\approx 1+\tau/h` y la suma de los pasos converge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

H_PAPER = 0.02
SIGMA_PAPER = 6.36
U_MAX_PAPER = 1e13


def f_square(u: np.ndarray) -> np.ndarray:
    r""":math:`f(u) = (u_+)^2`."""
    v = np.maximum(u, 0.0)
    return v * v


def grid(h: float = H_PAPER) -> np.ndarray:
    """Nodos :math:`x_k = kh`, ``k = 0..N`` (incluye los bordes)."""
    N = round(1 / h)
    if not math.isclose(N * h, 1.0):
        raise ValueError("1/h debe ser entero")
    return np.linspace(0.0, 1.0, N + 1)


def _laplacian(u: np.ndarray, h: float) -> np.ndarray:
    """Laplaciano discreto Dirichlet sobre la última dimensión (sólo nodos interiores)."""
    lap = -2.0 * u
    lap[..., 1:] += u[..., :-1]
    lap[..., :-1] += u[..., 1:]
    return lap / (h * h)


@dataclass
class Scheme:
    """Parámetros del esquema totalmente discreto."""

    h: float = H_PAPER
    tau: float = 5e-4
    tau_max: float | None = None  # por defecto 0.4 h^2
    u_max: float = U_MAX_PAPER
    f: Callable[[np.ndarray], np.ndarray] = field(default=f_square)

    def __post_init__(self):
        if self.tau_max is None:
            self.tau_max = 0.4 * self.h**2
        self.N = round(1 / self.h)

    def step_size(self, u: np.ndarray) -> np.ndarray:
        l1 = self.h * np.abs(u).sum(axis=-1)
        with np.errstate(divide="ignore"):
            return np.minimum(self.tau_max, self.tau / l1)

    def step(self, u: np.ndarray, sigma, rng: np.random.Generator):
        """Un paso de Euler explícito.  Devuelve ``(u_nuevo, dt)``."""
        dt = self.step_size(u)
        dtc = dt[..., None]
        noise = (np.asarray(sigma)[..., None] if np.ndim(sigma) else sigma) * np.sqrt(dtc / self.h)
        u_new = u + dtc * (_laplacian(u, self.h) + self.f(u)) + noise * rng.standard_normal(u.shape)
        return u_new, dt


# ---------------------------------------------------------------------------
# Una trayectoria con registro
# ---------------------------------------------------------------------------

@dataclass
class SPDEPath:
    """Trayectoria registrada.

    Attributes
    ----------
    x:
        Nodos (con bordes).
    t_snap, u_snap:
        Perfiles completos (con bordes) a tiempos aproximadamente
        equiespaciados cada ``record_dt`` y en todos los pasos con
        ``max u >= dense_from`` (fase final).
    t, umax:
        Tiempo y :math:`\\max_k u_k` en cada paso (o cada ``store_stride`` pasos).
    T:
        Tiempo de explosión numérico (``nan`` si no explotó antes de ``t_max``).
    """

    x: np.ndarray
    t_snap: np.ndarray
    u_snap: np.ndarray
    t: np.ndarray
    umax: np.ndarray
    T: float

    def profile_at_level(self, level: float):
        """Primer perfil registrado con ``max u >= level``: ``(t, u)``."""
        m = self.u_snap.max(axis=1)
        i = int(np.argmax(m >= level))
        if m[i] < level:
            raise ValueError("nivel no alcanzado")
        return self.t_snap[i], self.u_snap[i]

    def profile_at_time(self, time: float):
        """Perfil registrado más cercano a ``time``: ``(t, u)``."""
        i = int(np.argmin(np.abs(self.t_snap - time)))
        return self.t_snap[i], self.u_snap[i]


def simulate_path(
    sigma: float,
    rng: np.random.Generator,
    scheme: Scheme | None = None,
    u0: np.ndarray | float = 0.0,
    t_max: float = np.inf,
    record_dt: float = 0.1,
    dense_from: float = 15.0,
    store_stride: int = 1,
) -> SPDEPath:
    """Simula una trayectoria hasta la explosión (o ``t_max``) registrando perfiles.

    ``store_stride``: guarda :math:`(t, \\max u)` sólo cada tantos pasos (salvo
    en la fase final, ``max u >= dense_from``), para corridas largas.
    """
    s = scheme or Scheme()
    x = grid(s.h)
    if np.ndim(u0):  # perfil completo en los N+1 nodos (se ignoran los bordes)
        u = np.asarray(u0, float)[1:-1].copy()
    else:
        u = np.full(s.N - 1, float(u0))

    def full(v):
        return np.concatenate(([0.0], v, [0.0]))

    t = 0.0
    ts, ms = [0.0], [float(u.max())]
    t_snap, u_snap = [0.0], [full(u)]
    next_rec = record_dt
    T = math.nan
    k = 0
    while t < t_max:
        u, dt = s.step(u, sigma, rng)
        t += float(dt)
        m = float(u.max())
        k += 1
        if k % store_stride == 0 or m >= dense_from:
            ts.append(t)
            ms.append(m)
        if t >= next_rec or m >= dense_from:
            t_snap.append(t)
            u_snap.append(full(u))
            while next_rec <= t:
                next_rec += record_dt
        if m >= s.u_max:
            T = t
            break
    return SPDEPath(x, np.array(t_snap), np.array(u_snap), np.array(ts), np.array(ms), T)


# ---------------------------------------------------------------------------
# Monte Carlo de tiempos de explosión
# ---------------------------------------------------------------------------

def explosion_times(
    sigma,
    n_paths: int,
    rng: np.random.Generator,
    scheme: Scheme | None = None,
    u0: float = 0.0,
    t_max: float = np.inf,
    progress: bool = False,
) -> np.ndarray:
    """Tiempos de explosión numéricos de ``n_paths`` trayectorias independientes.

    ``sigma`` puede ser un escalar o un array de largo ``n_paths`` (una
    intensidad por trayectoria, para barrer varios valores en una sola
    corrida).  Vectorizado sobre trayectorias; las que no explotan antes de
    ``t_max`` devuelven ``nan``.
    """
    s = scheme or Scheme()
    sig = np.broadcast_to(np.asarray(sigma, float), (n_paths,)).copy()
    U = np.full((n_paths, s.N - 1), float(u0))
    t = np.zeros(n_paths)
    T = np.full(n_paths, np.nan)
    idx = np.arange(n_paths)  # trayectorias activas
    k = 0
    while idx.size:
        U, dt = s.step(U, sig, rng)
        t += dt
        m = U.max(axis=1)
        done = m >= s.u_max
        if done.any():
            T[idx[done]] = t[done]
        keep = ~done & (t < t_max)
        if not keep.all():
            idx, U, t, sig = idx[keep], U[keep], t[keep], sig[keep]
        k += 1
        if progress and k % 200_000 == 0:
            print(f"  paso {k}: activas {idx.size}, t min {t.min() if t.size else 0:.1f}", flush=True)
    return T


def stationary_max() -> float:
    r"""Máximo de la solución estacionaria positiva de :math:`v''+v^2=0`,
    :math:`v(0)=v(1)=0`: :math:`6\,I^2`, :math:`I=\tfrac13B(\tfrac13,\tfrac12)`
    (:math:`\approx 11.7967`; el paper dice "de tamaño 12")."""
    from scipy.special import beta

    I = beta(1 / 3, 1 / 2) / 3
    return 6 * I**2
