# Blow-up: aproximaciones numéricas de problemas con explosión

Reimplementación en Python (NumPy / SciPy / Matplotlib) de los experimentos
numéricos de una serie de artículos sobre aproximaciones numéricas de
problemas con *blow-up* (explosión en tiempo finito): ecuaciones parabólicas
con condiciones de borde no lineales, ecuaciones diferenciales estocásticas
(SODEs) y una ecuación en derivadas parciales estocástica (SPDE).

Los códigos originales estaban escritos en MATLAB y se perdieron. El objetivo
es reconstruirlos a partir de los artículos y, en los trabajos puramente
teóricos, agregar experimentos que ilustren los resultados.

## Artículos

Los PDFs (versiones de autor) están en [`papers/`](papers/). Para cada uno hay
una nota detallada en [`papers/notes/`](papers/notes/) con el problema, el
esquema numérico, los teoremas, los experimentos y todo lo que el artículo no
especifica.

### Problemas parabólicos

Todos estudian semidiscretizaciones en espacio (elementos finitos P1 con
*mass lumping* / diferencias finitas) de la ecuación del calor con flujo no
lineal en el borde.

| Artículo | Archivo | Nota | Experimentos en el paper |
|---|---|---|---|
| J. Fernández Bonder, P. Groisman, J. D. Rossi, *On numerical blow-up sets* | [PDF](papers/FGR_numerical-blowup-sets.pdf) | [nota](papers/notes/FGR_numerical-blowup-sets.md) | No (teórico) |
| G. Acosta, J. Fernández Bonder, P. Groisman, J. D. Rossi, *Numerical approximation of a parabolic problem with a nonlinear boundary condition in several space dimensions* | [PDF](papers/AFGR_nonlinear-bc-several-dims.pdf) | [nota](papers/notes/AFGR_nonlinear-bc-several-dims.md) | No (teórico) |
| G. Acosta, J. Fernández Bonder, P. Groisman, J. D. Rossi, *Simultaneous vs. non-simultaneous blow-up in numerical approximations of a parabolic system with nonlinear boundary conditions* | [PDF](papers/AFGR_simultaneous-blowup-system.pdf) | [nota](papers/notes/AFGR_simultaneous-blowup-system.md) | No (teórico) |
| J. Fernández Bonder, J. D. Rossi, *Blow-up vs. spurious steady solutions* | [PDF](papers/FR_blowup-vs-spurious-steady.pdf) | [nota](papers/notes/FR_blowup-vs-spurious-steady.md) | No (teórico) |
| G. Acosta, J. Fernández Bonder, J. D. Rossi, *Stable manifold approximation for the heat equation with nonlinear boundary condition* | [PDF](papers/AFR_stable-manifold.pdf) | [nota](papers/notes/AFR_stable-manifold.md) | Tabla 1 ($\lambda_{c,h}$), Fig. 1 |

### Ecuaciones estocásticas

| Artículo | Archivo | Nota | Experimentos en el paper |
|---|---|---|---|
| J. Dávila, J. Fernández Bonder, P. Groisman, J. D. Rossi, M. Sued, *Numerical analysis of stochastic differential equations with explosions* | [PDF](papers/DFGRS_sde-explosions.pdf) ([PS original](papers/DFGRS_sde-explosions.ps)) | [nota](papers/notes/DFGRS_sde-explosions.md) | Figs. 1–3 |
| J. Fernández Bonder, P. Groisman, *Time–space white noise eliminates global solutions in reaction diffusion equations* | [PDF](papers/FG_white-noise-blowup.pdf) | [nota](papers/notes/FG_white-noise-blowup.md) | Figs. 1–4, Tabla 1 |

> Los PDFs son preprints y no incluyen los datos de publicación (revista,
> año, volumen). Completar las referencias definitivas.

## Estructura del repositorio

```
papers/            artículos originales (PDF / PS)
papers/notes/      notas de implementación de cada artículo
blowup/            paquete Python con los esquemas numéricos
experiments/       un script por figura / tabla / experimento
figures/           figuras generadas por los scripts
tests/             tests (pytest)
```

Módulos del paquete:

- `blowup/sode.py` ✅: Euler–Maruyama con paso adaptativo $\tau_k = h/b(X_k)$,
  trayectorias individuales y Monte Carlo vectorizado de $R_h^M$.
- `blowup/spde.py` (pendiente): diferencias finitas en espacio con ruido
  blanco espacio-temporal y paso adaptativo.
- `blowup/parabolic.py` (pendiente): semidiscretizaciones (P1 + *mass
  lumping*), pasos temporales adaptativos, detección de blow-up y
  estimación de $T_h$.

## Instalación y uso

Requiere Python ≥ 3.10. Todos los comandos se corren desde la raíz del repo.

```bash
pip install -r requirements.txt
```

```bash
python -m pytest
```

### SDEs con explosión (Dávila et al.)

$b(\xi)=|\xi|^{1.1}+0.1$, $\sigma=\sqrt b$, $z=10$. Cada script acepta
`--seed N`. Las figuras se guardan en `figures/sode/`.

| Script | Figura del paper | Parámetros | Tiempo |
|---|---|---|---|
| `python -m experiments.sode.fig1_paths` | Fig. 1 (y ampliaciones 1.1, 1.2) | $h=0.25$\*, $M=10^5$, 3 trayectorias | ~2 s |
| `python -m experiments.sode.fig2_ratio` | Fig. 2: $X_k/(hk)\to1$ | las mismas trayectorias de la Fig. 1 | ~2 s |
| `python -m experiments.sode.fig3_kde` | Fig. 3: densidad de $R_h^M$ | $h=1,\,0.5,\,0.1$, $M=10^4$\*, 1000 trayectorias | ~10 s |

\* Valores elegidos por nosotros porque el paper no los especifica:

- **$h=0.25$ en las Figs. 1 y 2.** Con ese valor cada trayectoria da
  $\approx M/h = 4\times10^5$ pasos, que es el rango del eje $k$ de la
  Fig. 2 original.
- **$M=10^4$ en la Fig. 3.** Con $M=10^4$ la densidad queda centrada en
  $\approx4.0$ con soporte $\approx[3.4,\,4.9]$, como en la figura original.
  Con $M=10^5$ quedaría centrada en $\approx4.8$. Se puede cambiar con `--M`.
- **$h=0.1$ y no $0.01$ en la Fig. 3.** La leyenda original dice $0.01$,
  pero el texto y $\tau_1\simeq0.008$ corresponden a $h=0.1$.

Resultado de referencia de la Fig. 3 (semilla por defecto):

| $h$ | media $R_h^M$ | desvío |
|---|---|---|
| 1 | 4.032 | 0.248 |
| 0.5 | 4.020 | 0.237 |
| 0.1 | 4.006 | 0.249 |

Para estos $M$, $R_h^M$ todavía está lejos del tiempo de explosión $T$: la
cola determinista $\int_M^\infty ds/b \approx 10\,M^{-0.1}$ vale $\approx4$
para $M=10^4$. La convergencia en $M$ es muy lenta porque $b\sim s^{1.1}$.
