r"""Figuras 1–3 y Tabla 1 de Fernández Bonder–Groisman: una trayectoria muestral.

:math:`u_t = u_{xx} + (u_+)^2 + \sigma\dot W`, :math:`\sigma = 6.36`,
:math:`u_0 \equiv 0`, :math:`h = 0.02`, Dirichlet homogéneo.

* Fig. 1: perfiles :math:`u(\cdot,t)` en 9 instantes.  Como cada trayectoria
  explota en un tiempo distinto, en lugar de los tiempos del paper usamos
  :math:`t=1`, un tiempo intermedio de la fase metaestable y los primeros
  instantes en que :math:`\max u` supera los niveles de la Tabla 1
  (15.5, 18.3, 38.6, 82.9, 203, 2270, 1.8e12).
* Tabla 1: :math:`\|u(\cdot,t)\|_\infty` en esos instantes (se imprime).
* Fig. 2: evolución de :math:`\|u(\cdot,t)\|_\infty` (cortada en 90, como en el paper).
* Fig. 3: superficie :math:`u(x,t)` hasta que :math:`\max u \approx 45`.

Uso::

    python -m experiments.spde.fig1_3_sample_path [--seed N] [--sigma 6.36]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blowup.spde import SIGMA_PAPER, Scheme, simulate_path, stationary_max
from experiments.common import parser, save

LEVELS = [15.5104, 18.2885, 38.5848, 82.8705, 203.0799, 2.2695e3, 1.8128e12]


def main() -> None:
    p = parser(__doc__.splitlines()[0], seed=11)
    p.add_argument("--sigma", type=float, default=SIGMA_PAPER)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    path = simulate_path(args.sigma, rng, Scheme())
    T = path.T
    print(f"tiempo de explosión: {T:.4f}   pasos: {len(path.t) - 1}")

    # --- Fig. 1 / Tabla 1 -------------------------------------------------
    t_mid = 50.0 if T > 60 else 0.6 * T
    snaps = [path.profile_at_time(1.0), path.profile_at_time(t_mid)]
    snaps += [path.profile_at_level(L) for L in LEVELS]
    print("\nTabla 1: máximo de la solución a distintos tiempos")
    print(f"{'t':>12}  {'||u||_inf':>12}")
    for t, u in snaps:
        print(f"{t:12.4f}  {np.abs(u).max():12.5g}")

    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    for ax, (t, u) in zip(axes.flat, snaps):
        ax.plot(path.x, u, lw=0.9)
        ax.set_title(rf"$t = {t:.4f}$", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("$x$")
    fig.suptitle(rf"Perfiles de una trayectoria ($\sigma={args.sigma}$, $u_0\equiv0$, $h=0.02$)")
    fig.tight_layout()
    save(fig, "spde", "fig1_profiles")

    # --- Fig. 2 -----------------------------------------------------------
    stride = max(1, len(path.t) // 200_000)
    sel = np.flatnonzero(path.umax <= 90)
    end = sel[-1] + 1 if sel.size else len(path.t)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(path.t[:end:stride], path.umax[:end:stride], lw=0.5)
    ax.axhline(stationary_max(), color="k", ls="--", lw=0.8,
               label=f"máx. de la estacionaria ($\\approx{stationary_max():.2f}$)")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|u(\cdot,t)\|_\infty$")
    ax.set_title(r"Evolución del máximo de una trayectoria, $u_0\equiv0$")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    save(fig, "spde", "fig2_max")

    # --- Fig. 3 -----------------------------------------------------------
    m = path.u_snap.max(axis=1)
    k_end = int(np.argmax(m >= 45)) if (m >= 45).any() else len(m)
    # Perfiles equiespaciados en t (la fase final está registrada paso a paso).
    t_end = path.t_snap[k_end - 1]
    keep = np.unique(np.searchsorted(path.t_snap[:k_end], np.linspace(0, t_end, 200)).clip(0, k_end - 1))
    keep = np.r_[keep[:-1], np.arange(keep[-1] - 40, k_end).clip(0)]
    keep = np.unique(keep)
    tt, xx = np.meshgrid(path.t_snap[keep], path.x, indexing="ij")
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xx, tt, path.u_snap[keep], cmap="viridis", linewidth=0, antialiased=True,
                    rcount=len(keep), ccount=len(path.x))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$u$")
    ax.view_init(elev=25, azim=-60)
    ax.set_title(r"Una trayectoria $u(x,t)$ con $u_0\equiv0$")
    save(fig, "spde", "fig3_surface")


if __name__ == "__main__":
    main()
