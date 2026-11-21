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
blowup/            paquete Python con los esquemas numéricos   (en construcción)
experiments/       un script por figura / tabla / experimento   (en construcción)
```

Estructura prevista del paquete:

- `blowup/parabolic.py`: semidiscretizaciones (P1 + *mass lumping*), pasos
  temporales adaptativos, detección de blow-up y estimación de $T_h$.
- `blowup/sode.py`: Euler–Maruyama con paso adaptativo $\tau_k = h/b(X_k)$ y
  tiempos de explosión numéricos.
- `blowup/spde.py`: diferencias finitas en espacio con ruido blanco
  espacio-temporal y paso adaptativo.

## Instalación y uso

Requiere Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

Las instrucciones para correr cada experimento se irán agregando a medida que
se implementen.
