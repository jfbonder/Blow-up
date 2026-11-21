# Acosta, Fernández Bonder, Rossi — *Stable manifold approximation for the heat equation with nonlinear boundary condition*

> **Referencia publicada:** G. Acosta, J. Fernández Bonder and J.D. Rossi. Stable manifold approximation for the heat equation with nonlinear boundary conditions. *J. Dynam. Differential Equations*, 12 (2000), no. 3, 557–578.

Notas para reimplementar (Python/NumPy/SciPy) todos los cálculos numéricos del artículo sin volver al PDF.
Archivo fuente: `papers/AFR_stable-manifold.pdf` (20 páginas, preprint compilado con TeX el 2001-10-31).

---

## 1. Referencia bibliográfica

- **Autores:** Gabriel Acosta, Julián Fernández Bonder, Julio D. Rossi (Departamento de Matemática, FCEyN, UBA, (1428) Buenos Aires, Argentina; e-mails `gacosta@dm.uba.ar`, `jfbonder@dm.uba.ar`, `jrossi@dm.uba.ar`).
- **Título:** *Stable manifold approximation for the heat equation with nonlinear boundary condition.*
- **Revista, año, volumen, páginas:** **no figuran en el PDF.** El campo "Date:" de la primera página está vacío. El PDF es un preprint (metadatos: "TeX output 2001.10.31"). No se indica revista.
- 1991 MSC: 35K55, 35B40, 65M20. Palabras clave: numerical approximations, nonlinear boundary conditions, stable manifold.
- Financiación: UBA (subsidios TX47, TX48), ANPCyT PICT No. 03-00000-00137; J. D. Rossi también por Fundación Antorchas.
- Agradecimientos: N. Guelman, P. Felmer, S. Ponce Dawson, R. G. Durán.

---

## 2. Problema continuo

$$
\begin{cases}
u_t = u_{xx} & \text{en } (0,1)\times[0,T),\\
u_x(1,t) = g(u(1,t)) & \text{en } [0,T),\\
u_x(0,t) = 0 & \text{en } [0,T),\\
u(x,0) = u_0(x)\ge 0 & \text{en } [0,1],
\end{cases}
\qquad g(s)=s^p-s,\ \ p>1. \tag{1.1}
$$

- Es decir: flujo de Neumann homogéneo en $x=0$, flujo no lineal $u_x=u^p-u$ en $x=1$.
- Se trabaja con soluciones en $H^1$; (1.1) define un semiflujo local compacto en $C^+=\{v\in H^1: v\ge 0\}$ (Lema 2.1, vía Teorema 14.5 de Amann [4]). Si $\sup_{0\le t<t_{max}}\|u(\cdot,t)\|_{H^1}<\infty$ entonces $t_{max}=\infty$.
- **Estados estacionarios:** sólo $u\equiv 0$ y $u\equiv 1$ (Corolario 2.1: son los únicos $\omega$-límites posibles de soluciones globales). $u\equiv1$ es el único estado estacionario positivo.
- Hechos citados en la introducción: si $u_0>1$ la solución explota en tiempo finito; si $0\le u_0<1$ la solución tiende a $0$ uniformemente (y en $H^1$). $u\equiv1$ es hiperbólico, con variedad estable de codimensión uno.
- **Funcional de Lyapunov (Lema 2.2):**
$$V(u(\cdot,t))=\frac12\int_0^1 u_x^2\,dx-G(u(1,t)),\qquad G(s)=\frac{s^{p+1}}{p+1}-\frac{s^2}{2},$$
$$\frac{dV}{dt}=-\int_0^1 u_t^2\,dx. \tag{2.2}$$
  $V(0)=0$, $V(1)=\tfrac12-\tfrac1{p+1}>0$. Identidad de energía: $\int_0^t\!\int_0^1 u_t^2+V(u(\cdot,t))=V(u_0)$.
- **Lema 2.3** (Fila [12]): o bien $t_{max}(u_0)<\infty$, o bien $u$ está acotada en $H^1$. Si $t_{max}<\infty$ entonces $u(1,t)\to\infty$ cuando $t\nearrow t_{max}$ (Observación).
- **Lema 2.4:** si $u\not\equiv0$ y existe $t_0$ con $V(u(\cdot,t_0))\le0$, entonces $u$ explota en tiempo finito.
- **Familia de datos iniciales:** $u_\lambda$ = solución con dato $\lambda u_0$, $u_0$ fijo positivo. Se definen $\lambda_+=\inf\{\lambda: u_\lambda \text{ explota}\}$, $\lambda_-=\sup\{\lambda: u_\lambda\to0\}$. Se prueba $\lambda_+=\lambda_-\equiv\lambda_c$.
- Linealización en $1$ (prueba del Teorema 3.5): $v_t=v_{xx}$, $v_x(0,t)=0$, $v_x(1,t)=g'(1)v(1,t)$ con $g'(1)=p-1$. Todos los autovalores salvo uno son negativos; uno es positivo.
- Rango de parámetros: teoría para todo $p>1$. Experimentos sólo con $p=2$.

---

## 3. Esquema numérico

### 3.1 Semidiscretización (elementos finitos P1 + mass lumping)

- Malla uniforme: $x_i=i/N$, $0\le i\le N$, $h=1/N$, subintervalos $I_i=(x_i,x_{i+1})$.
- $V_h$ = funciones continuas afines a trozos; base nodal $\varphi_i(x_j)=\delta_{ij}$; $u_h(x,t)=\sum_{i=0}^N u_i(t)\varphi_i(x)$ (3.1).
- Formulación débil ⇒ $MU'=-AU+g(U)$ (3.2), con $U=(u_0,\dots,u_N)$, $M$ matriz de masa, $A$ matriz de rigidez, $g(U)=g(u_N)\,e_N$, $e_N=(0,\dots,0,1)$.
- **Mass lumping:** $M$ diagonal con $M_{ii}=h$ para $1\le i\le N-1$ y $M_{00}=M_{NN}=h/2$. Resulta el sistema que se integra:

$$
\begin{aligned}
u_0' &= 2h^{-2}\,(-u_0+u_1),\\
u_i' &= h^{-2}\,(u_{i+1}-2u_i+u_{i-1}),\qquad 1\le i\le N-1,\\
u_N' &= 2h^{-2}\,\big(-u_N+u_{N-1}+h\,g(u_N)\big),\qquad g(s)=s^p-s.
\end{aligned}\tag{3.3}
$$

  (La matriz de rigidez P1 implícita es $A=\frac1h\,\mathrm{tridiag}(-1,2,-1)$ con $A_{00}=A_{NN}=1/h$; con $M$ agrupada da exactamente (3.3). El artículo no escribe $A$ entrada a entrada; esto es consecuencia directa.)
- **Principio del máximo discreto (Lema 3.1):** si $U$ es subsolución de (3.3) (desigualdades $\le$), entonces $\max_k u_k(t)\le\max\{\max_k u_k(0),\ \sup_{0<\tau<t}u_N(\tau)\}$.
- **Lema 3.2:** para $h$ fijo, toda solución global de (3.3) está acotada en $\mathbb R^{N+1}$ (si no, $u_N\to\infty$ y $u_N'\ge C_1(h)u_N^p-C_2(h)u_N\ge C_3(h)u_N^p$, contradicción).
- **Funcional de Lyapunov discreto (Lema 3.3):** $V_h(U)=\tfrac12\langle A^{1/2}U,A^{1/2}U\rangle-G(u_N)$ $(=\tfrac12 U^TAU-G(u_N))$. (Derivado, no escrito en el artículo: $\frac{d}{dt}V_h=-U'^TMU'$.)

### 3.2 Estados estacionarios del sistema discreto

$u_0=u_1=\dots=u_N=c$ con $g(c)=0$: $U=0$ y $U=(1,\dots,1)$. $(1,\dots,1)$ es "la única solución estacionaria positiva" de (3.3) (Teorema 3.5).

### 3.3 Linealización y variedad estable

- Linealización discreta en $(1,\dots,1)$ (no escrita explícitamente; consecuencia directa): $MZ'=-AZ+g'(1)z_Ne_N$, $g'(1)=p-1$. Es decir, Jacobiano $J=M^{-1}\big(-A+(p-1)e_Ne_N^T\big)$: igual que (3.3) linealizado, con la última fila $z_N'=2h^{-2}(-z_N+z_{N-1}+h(p-1)z_N)$.
- **Teorema 3.5:** $(1,\dots,1)$ es hiperbólico y su variedad estable es una hipersuperficie (dimensión $N$ en $\mathbb R^{N+1}$; el texto dice "dimension $N-1$ as can be expected" justo antes del enunciado — inconsistencia del texto con $N+1$ incógnitas; el enunciado dice "hypersurface"). Prueba: un autovalor positivo y el resto negativos, por convergencia de autovalores discretos a continuos (Osborn [16]) usando $\|v-v_h\|_{H^1}\le Ch\|f\|_{H^1}$ para el problema elíptico $v_{xx}=f$, $v_x(0)=0$, $v_x(1)=g'(1)v(1)$.
- **Cómo se aproxima la variedad estable:** el artículo NO calcula la variedad estable directamente (ni autovectores, ni parametrización). Sólo calcula, para la dirección fija $u_0$, el valor crítico $\lambda_{c,h}$ tal que $\lambda_{c,h}U_0$ pertenece a la variedad estable discreta (Teorema 3.2(3)); la intersección de la variedad con el rayo $\{\lambda u_0\}$ es ese único punto.

### 3.4 Integración temporal, bisección, criterios

- Lo único que dice el artículo: "The results were obtained integrating the ODE system (3.3) by using an adaptive Runge–Kutta method." (Figura 1 es un `mesh` de MATLAB, por lo que presumiblemente `ode45` o similar; no se dice.)
- "Table 1 gives the approximate $\lambda_{c,h}$ with four exact decimals."
- **No se describe** el procedimiento para hallar $\lambda_{c,h}$ (bisección u otro), ni tolerancias, ni tiempo final, ni criterio de explosión/decaimiento. Ver "No especificado".
- Criterios rigurosos que se deducen de los teoremas del artículo (útiles para clasificar cada $\lambda$ en una bisección):
  - **Decaimiento:** si en algún $t_0$ se tiene $U(t_0)\le(1,\dots,1)$ componente a componente y $U(t_0)\neq(1,\dots,1)$, entonces $U\to0$ (Teorema 3.1(1) con la solución estacionaria $1$). La prueba del Teorema 3.2 usa concretamente $\|U(t_0)\|_\infty<3/4<1$ (y $<1/2$ para $\lambda_0$).
  - **Explosión:** si $U(t_0)\ge(1,\dots,1)$, $\neq$, entonces explota (Teorema 3.1(2)). Alternativamente, en la prueba del Teorema 3.2: de $u_N'\ge2h^{-2}(-u_N+hg(u_N))$, si $u_N(t_0)>K(h)$, donde $K(h)$ = última raíz positiva de $-s+hg(s)$, la solución no es globalmente acotada ⇒ explota. Explícitamente $K(h)=\big((1+h)/h\big)^{1/(p-1)}$; para $p=2$, $K(h)=1+1/h$.
- Bisección (Teorema 3.2): $\lambda<\lambda_{c,h}\Rightarrow U_\lambda\to0$; $\lambda>\lambda_{c,h}\Rightarrow$ explosión en tiempo finito; $\lambda=\lambda_{c,h}\Rightarrow U_\lambda\to(1,\dots,1)$. Monotonía (Teorema 3.1) garantiza que la bisección está bien planteada.

---

## 4. Resultados principales

- **Teorema 2.1.** Sea $u_0$ tal que $u(\cdot,t)\to1$ en $H^1$. (1) Si $v_0\le u_0$, $v_0\ne u_0$, entonces $v\to0$ en $H^1$. (2) Si $v_0\ge u_0$, $v_0\ne u_0$, entonces $v$ explota en tiempo finito. ⇒ $\lambda_-=\lambda_+\equiv\lambda_c$.
- **Lema 2.5, 2.6, 2.7, Corolario 2.2, Proposición 2.1, Lema 2.8:** cotas a priori de soluciones globales con $\|u_0\|_\infty\le\mathcal A$: $\|u(\cdot,t)\|_\infty\le C(V(u(\cdot,t_0))^\alpha+V^\beta+V^\gamma)$ para $t\ge t_0$ (Prop. 2.1), y $V(u(\cdot,t_0))\le C(t_0,p,\mathcal A)$ (Lema 2.8). (Teóricos; sin uso numérico.)
- **Teorema 2.2.** $u_0>0$ en $[0,1]$, $u_\lambda$ con dato $\lambda u_0$. Existe $\lambda_c$ tal que: (1) $\lambda<\lambda_c\Rightarrow u_\lambda\to0$ en $H^1$; (2) $\lambda>\lambda_c\Rightarrow$ explosión en tiempo finito; (3) $\lambda=\lambda_c\Rightarrow u_\lambda\to1$ en $H^1$.
- **Lema 2.9.** Para todo $M>1$ existe $L>0$ tal que si $u(1,t)>M$ $\forall t\in[0,L]$, existe $T_0<L$ con $u(x,T_0)>1$ $\forall x$, y por tanto $u$ explota.
- **Teorema 2.3.** $\lambda_c=\lambda_c(u_0)$ es continua respecto del dato inicial en norma $L^\infty$.
- **Teorema 3.1** (análogo discreto del 2.1), **Teorema 3.2** (existencia y unicidad de $\lambda_{c,h}$ para $h$ fijo, con las tres alternativas anteriores con $U_\lambda\to0$, explosión, $U_\lambda\to(1,\dots,1)$).
- **Teorema 3.3** (de Durán–Etcheverry–Rossi [9]). Si $u\in C^{2,1}([0,1]\times[0,T])$, $T<t_{max}(u_0)$, y $u_h$ es la aproximación semidiscreta (3.1), entonces para $h$ pequeño $\|u-u_h\|_{L^\infty([0,1]\times[0,T])}\le Ch^{3/2}$, con $C=C(T,u)$. Observación: esa regularidad se obtiene con datos compatibles $u_0'(0)=0$, $u_0'(1)=g(u_0(1))$ (densos en $H^1$).
- **Lema 3.4** (versión discreta del 2.9): para todo $M>1$ existen $h_0,L>0$ tales que si $u_h(1,t)>M$ $\forall t\in[0,L]$ hay $T_0<L$ con $u_h(x,T_0)>1$ y $u_h$ explota, $\forall h<h_0$.
- **Teorema 3.4.** $\lambda_{c,h}\to\lambda_c$ cuando $h\to0$. **No se da tasa.** En la Sección 4 se dice que los valores numéricos "suggest that the convergence rate is of order $h^2$ however we were not able to prove it".
- **Teorema 3.5.** $(1,\dots,1)$ hiperbólico, variedad estable hipersuperficie (ver 3.3).

---

## 5. Experimentos numéricos (Sección 4)

Configuración común:
- Problema semidiscreto (3.3) con $p=2$, es decir $g(s)=s^2-s$.
- Dato $u_0(x)=\tfrac12\cos(2\pi x)+1$ (rango $[1/2,3/2]$, máximo en $x=0$ y $x=1$, mínimo $1/2$ en $x=1/2$); dato discreto $U_\lambda(0)=\lambda U_0$. (Que $U_0$ sea la evaluación nodal $u_0(x_i)$ no está dicho explícitamente; es lo natural.)
- Integrador: Runge–Kutta adaptativo.
- Nota: $u_0$ no es compatible en $x=1$ ($u_0'(1)=0\neq g(\lambda u_0(1))$).

### Tabla 1 — valores críticos discretos

| $N$ | $\lambda_{c,h}$ |
|---|---|
| 10 | 0.9755 |
| 20 | 0.9766 |
| 50 | 0.9770 |
| 100 | 0.9770 |

($h=1/N$; "four exact decimals"). Sugiere orden $h^2$ (no probado).

### Figura 1 — evolución cerca del umbral

- $h=1/100$ ($N=100$), $\lambda=0.9771>\lambda_{c,h}$ (=0.9770 según la Tabla 1).
- Gráfico 3D tipo `mesh` de MATLAB de $u(x,t)$: eje $x\in[0,1]$, eje $t\in[0,6]$ (marcas 0–6), eje vertical "u(x,t)" con marcas 0.5, 1, 1.5. Vista: origen $x=0,t=0$ en la esquina frontal; $x=1$ hacia la izquierda-atrás.
- Lo que se ve: en $t=0$ el perfil $\lambda u_0$ (≈1.47 en $x=0,1$; ≈0.49 en $x=1/2$); se aplana muy rápido (t ≲ 0.5–1) a casi $u\equiv1$; permanece cerca de 1 durante mucho tiempo; hacia $t\approx 5$–$6$ crece, más fuerte en $x=1$ (alcanza visiblemente por encima de 1.5, ~1.6–1.7 al final) y algo menos en $x=0$ (~1.05). El dibujo termina en $t\approx6$, sin llegar al tiempo de explosión. Las líneas en $t$ son muy densas (probablemente los pasos de salida del integrador adaptativo).
- Texto: "the solution stays close to 1 for a long period of time, after which it rapidly goes to infinity (this happens because we have chosen a value of lambda greater but close to the critical one)".

### No especificado

- Método RK exacto (orden, par embebido), tolerancias relativa/absoluta, paso inicial/máximo.
- Procedimiento para obtener $\lambda_{c,h}$ (¿bisección?), intervalo inicial, tolerancia en $\lambda$, número de iteraciones.
- Criterio de explosión (umbral en $u_N$ o $\|U\|_\infty$) y de decaimiento; tiempo final de integración.
- Cómo se discretiza el dato inicial (se supone evaluación nodal).
- Valor de referencia de $\lambda_c$ continuo (no se da).
- Para la Figura 1: tiempo final exacto, instantes de salida, puntos de $x$ dibujados (¿los 101 nodos?).
- Ningún cálculo de autovalores/autovectores de la linealización ni de la variedad estable aparte de $\lambda_{c,h}$.

---

## 6. Notas de implementación

1. **RHS vectorizado** (p genérico):
   ```python
   d[0]    = 2/h**2*(u[1]-u[0])
   d[1:-1] = (u[2:]-2*u[1:-1]+u[:-2])/h**2
   d[-1]   = 2/h**2*(u[-2]-u[-1]+h*(u[-1]**p-u[-1]))
   ```
   Sistema rígido ($\sim h^{-2}$): con `solve_ivp` usar `LSODA`/`BDF`/`Radau` (con jacobiano tridiagonal) para rapidez; `RK45` reproduce el espíritu del artículo pero es lento para $N=100$ y tiempos largos.
2. **Clasificación de cada $\lambda$** con eventos terminales: decaimiento si $\max_i u_i<1$ (o, más conservador, $<0.99$); explosión si $\min_i u_i>1$ (o $>1.01$), o si $u_N>K(h)=((1+h)/h)^{1/(p-1)}$. Ambos son rigurosos por los Teoremas 3.1/3.2. Cerca del umbral la solución permanece mucho tiempo cerca de 1 (salida a lo largo de la dirección inestable con tasa $\mu_1\approx1.44$ para $p=2$), por lo que hace falta $T_{final}$ grande (p.ej. $10^4$) y tolerancias estrictas (rtol ~1e-10) si se quiere $|\lambda-\lambda_{c,h}|\sim10^{-6}$.
3. **Bisección** en $\lambda$, p.ej. en $[0.9,1.05]$ (para este $u_0$, $\lambda=2/3$ da $\lambda u_0\le1$ ⇒ decae y $\lambda=2$ da $\lambda u_0\ge1$ ⇒ explota, así que $[2/3,2]$ es un intervalo garantizado).
4. **Linealización** discreta: $J=M^{-1}(-A+(p-1)e_Ne_N^T)$; un autovalor positivo. Por mi cuenta (no en el artículo): continuo, autovalor positivo $\mu=k^2$ con $k\tanh k=p-1$, negativos $\mu=-k^2$ con $-k\tan k=p-1$ (autofunciones $\cosh kx$, $\cos kx$). Para $p=2$: $\mu_1\approx1.4392$ ($k\approx1.1997$), $\mu_2\approx-7.83$.
5. **Comprobación propia (no del artículo)**, con el esquema (3.3) tal cual, dato nodal, LSODA rtol=1e-10, eventos $\max U<0.99$ / $\min U>1.01$, bisección a $10^{-6}$: $\lambda_{c,h}\approx$ 0.976473 ($N=10$), 0.976894 ($N=20$), 0.977008 ($N=50$), 0.977024 ($N=100$). Coincide con la Tabla 1 para $N=50,100$ (0.9770) pero **no** para $N=10$ (0.9755 en el artículo) ni $N=20$ (0.9766). Las diferencias de mis valores al límite también decrecen como $h^2$ (factor ≈4), con constante ~3 veces menor que en la tabla. Probé masa consistente (0.97781, 0.97723) y "N = número de nodos" (0.976336, 0.976879): tampoco reproducen. La causa de la discrepancia en mallas gruesas es desconocida (posiblemente distinta discretización del dato o tolerancias del RK adaptativo). Autovalor positivo discreto obtenido: 1.43617 ($N=10$), 1.43846 (20), 1.43911 (50), 1.43920 (100).
6. **Figura 1:** integrar con $N=100$, $\lambda=0.9771$, $t\in[0,6]$, y dibujar superficie (`plot_wireframe`/`plot_surface`) con $x$ en un eje y $t$ en el otro, límites verticales ~[0.5, 1.7]. Debe verse la meseta cerca de 1 y el despegue en $x=1$ hacia $t\approx5$–6. (El instante del despegue depende sensiblemente de $\lambda-\lambda_{c,h}$; con el $\lambda_{c,h}$ de mi comprobación, $\lambda-\lambda_{c,h}\approx7.6\times10^{-5}$.)
7. Funcional de Lyapunov discreto $V_h=\tfrac12U^TAU-G(u_N)$ sirve como chequeo: debe decrecer; $V_h\le0$ en algún instante (con $U\ne0$) indica explosión por analogía con el Lema 2.4 (el análogo discreto del Lema 2.4 no está enunciado en el artículo).
