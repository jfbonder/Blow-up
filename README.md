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
- `blowup/spde.py` ✅: diferencias finitas en espacio con ruido blanco
  espacio-temporal, Euler explícito con paso adaptativo, trayectorias
  registradas y Monte Carlo vectorizado de tiempos de explosión.
- `blowup/parabolic.py` (en construcción): semidiscretización P1 + *mass
  lumping* de $u_t=u_{xx}+r(u)$ con $u_x(0)=0$ y $u_x(1)=F(u(1))$, con
  jacobiano exacto para integradores implícitos. Incluye clasificación
  decaimiento/explosión y bisección para umbrales (variedad estable). Falta
  el paso temporal adaptativo para blow-up.

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

### SPDE con ruido blanco (Fernández Bonder–Groisman)

$u_t=u_{xx}+(u_+)^2+\sigma\dot W$ en $(0,1)$, Dirichlet homogéneo,
$u_0\equiv0$, $h=0.02$, explosión cuando $\max u\ge10^{13}$. Las figuras
se guardan en `figures/spde/`. Los tiempos de explosión se guardan en
`results/spde/` y se reutilizan entre scripts.

| Script | Qué reproduce | Tiempo |
|---|---|---|
| `python -m experiments.spde.fig1_3_sample_path` | Figs. 1–3 y Tabla 1: una trayectoria ($\sigma=6.36$) | ~10 s |
| `python -m experiments.spde.fig4_explosion_time` | Fig. 4: densidad y box-plot del tiempo de explosión, 832 trayectorias | ~25 min |
| `python -m experiments.spde.exp0_metastable` | Experimento sin figura: $\sigma=3,4,5$ hasta $t=1000$, sin explosión | ~10 min |
| `python -m experiments.spde.exponential_law` | **Adicional:** $T/\bar T$ vs. Exp(1) para $\sigma=6.36,\,7,\,8,\,10$ | ~1 h sin caché |

**Elecciones propias**, porque el paper no las especifica:

- **Escala del ruido.** Uso $\frac{\sigma}{\sqrt h}\,dw_k$ con $w_k$
  brownianos estándar, es decir, $\sigma\sqrt{\tau_j/h}\,\xi_k$ por paso.
  Es la lectura coherente con la prueba del Teorema 4.1 y reproduce la
  amplitud de las fluctuaciones de la Fig. 1 original.
- **Paso adaptativo.** Uso
  $\tau_j=\min\big(\tau_{\max},\,\tau/\|u^j\|_{L^1}\big)$ con
  $\tau=5\cdot10^{-4}$ y $\tau_{\max}=0.4h^2$, dentro de la condición de
  estabilidad de Euler explícito. El paper escribe $\tau/\sum_k hu_k$: sin
  valor absoluto ni cota, el paso no está definido en $t=0$.

**Resultados.**

- **Figs. 1–3 y Tabla 1.** La trayectoria de la semilla por defecto explota
  en $t\approx69.94$; la del paper, en $72.51$. Los máximos coinciden con
  los de la Tabla 1 (5.0 en $t=1$ y 3.7 en $t=50$, contra 5.6 y 3.4).
- **Fig. 4: el tiempo medio da el doble que en el paper.** Obtengo media
  94.0 y desvío 94.9; el paper reporta 46.9 y 43.9. La media no depende
  del paso temporal: con 100 trayectorias da ≈104 para
  $\tau_{\max}=0.1h^2,\ 0.25h^2$ y $0.49h^2$. Como el tiempo depende
  exponencialmente de $1/\sigma^2$, la diferencia equivale a usar
  $\sigma\approx6.8$ en lugar de $6.36$ (con $\sigma=7$ obtengo media
  36.5). Probablemente viene de alguna convención no especificada del
  código original: la celda del ruido, la indexación de la malla o el
  tratamiento de $\sum_k hu_k\le0$. Lo cualitativo coincide: media ≈
  desvío, densidad de forma exponencial.
- **Metaestabilidad para $\sigma\le5$.** Ninguna explosión hasta $t=1000$,
  igual que en el paper.

**Ley exponencial** (experimento adicional, motivado por Groisman–Saglietti,
*J. Math. Anal. Appl.* 385, 2012, y Groisman–Saglietti–Saintier,
*Stoch. Proc. Appl.* 128, 2018). Comparo $T/\bar T$ con Exp(1), 832
trayectorias por valor de $\sigma$. El p-valor es de Kolmogorov–Smirnov,
calculado por Monte Carlo porque la media es estimada.

| $\sigma$ | media | desvío | CV | KS | p-valor |
|---|---|---|---|---|---|
| 6.36 | 94.01 | 94.92 | 1.01 | 0.031 | 0.18 |
| 7 | 36.52 | 35.92 | 0.98 | 0.021 | 0.68 |
| 8 | 13.57 | 13.61 | 1.00 | 0.044 | 0.006 |
| 10 | 4.43 | 3.96 | 0.89 | 0.074 | <0.001 |

Para $\sigma\le7$ no se rechaza la exponencialidad. Para $\sigma$ grande la
explosión deja de ser un evento raro y la ley se aparta de la exponencial:
la cola es más liviana y aparece un tiempo mínimo de tránsito. Es lo
esperable si $T_\sigma/\mathbb E T_\sigma\to\mathrm{Exp}(1)$ cuando
$\sigma\to0$.

### Variedad estable (Acosta–Fernández Bonder–Rossi)

Problema: $u_t=u_{xx}$ en $(0,1)$, $u_x(0)=0$, $u_x(1)=u^2-u$, con dato
$\lambda u_0$, $u_0=\tfrac12\cos(2\pi x)+1$. Las figuras se guardan en
`figures/parabolic/`.

| Script | Qué reproduce | Tiempo |
|---|---|---|
| `python -m experiments.parabolic.stable_manifold_table1` | Tabla 1: $\lambda_{c,h}$ para $N=10,\dots,400$, orden de convergencia y autovalor inestable | ~10 s |
| `python -m experiments.parabolic.stable_manifold_fig1` | Fig. 1: solución con $N=100$, $\lambda=0.9771$ | ~5 s |
| `python -m experiments.parabolic.stable_manifold_slice` | **Adicional:** corte de la variedad estable en el plano $a+b\cos(2\pi x)$ | ~2 min |

**Método.** $\lambda_{c,h}$ se calcula por bisección hasta $10^{-9}$.
Clasificación: la solución decae si $\max u<0.99$ y explota si
$\min u>1.01$; los dos criterios son rigurosos por el Teorema 3.1. Integro
con LSODA, jacobiano exacto y `rtol=1e-10`.

| $N$ | $\lambda_{c,h}$ (Python) | Tabla 1 del paper |
|---|---|---|
| 10 | 0.976472 | 0.9755 |
| 20 | 0.976894 | 0.9766 |
| 50 | 0.977008 | 0.9770 |
| 100 | 0.977024 | 0.9770 |
| 200 | 0.977028 | — |
| 400 | 0.977029 | — |

- **$N=50$ y $N=100$ coinciden** con la tabla del paper. Para $N=10$ y
  $N=20$ hay una diferencia de ~$10^{-3}$ que no logré explicar. Probé
  tolerancias tipo `ode45` de MATLAB, bordes de primer orden, matriz de masa
  consistente y otra discretización del dato, y ninguna la reproduce.
- **Orden de convergencia $h^2$**, confirmado: el orden observado es 2.00.
  El paper lo sugería sin probarlo. Por extrapolación de Richardson,
  $\lambda_c\approx0.9770294$.
- **Autovalor inestable** de la linealización en $(1,\dots,1)$: 1.43920 para
  $N=100$; el valor continuo es 1.43923 ($k^2$ con $k\tanh k=1$).
- **Fig. 1:** la solución se aplana cerca de $u\equiv1$, queda ahí hasta
  $t\approx4.5$ y después despega, primero en $x=1$, como en el paper.
