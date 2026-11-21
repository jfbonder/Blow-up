# Notas: Acosta, Fernández Bonder, Groisman, Rossi — *Simultaneous vs. non-simultaneous blow-up in numerical approximations of a parabolic system with nonlinear boundary conditions*

> **Referencia publicada:** G. Acosta, J. Fernández Bonder, P. Groisman and J.D. Rossi. Simultaneous vs. non-simultaneous blow-up in numerical approximations of a parabolic system with nonlinear boundary conditions. *M2AN Math. Model. Numer. Anal.*, 36 (2002), no. 1, 55–68.

> **Aviso principal.** Este artículo es **puramente teórico**: no contiene ningún experimento numérico, **ninguna figura y ninguna tabla** (revisadas las 14 páginas del PDF, incluidas las imágenes de cada página). No se menciona MATLAB, ni pasos de tiempo, ni tolerancias, ni datos iniciales concretos. Todo lo que aparece abajo en la sección 5 bajo "experimentos" es, por tanto, vacío; en la sección 6 se proponen (marcados explícitamente como **propuesta propia, no del paper**) cálculos que permitirían ilustrar los teoremas.

---

## 1. Referencia bibliográfica

- **Autores:** G. Acosta, J. Fernández Bonder, P. Groisman, J. D. Rossi.
- **Título:** *Simultaneous vs. non-simultaneous blow-up in numerical approximations of a parabolic system with nonlinear boundary conditions.*
- **Revista, año, volumen, páginas:** **No figuran en el PDF.** El PDF es un preprint (metadatos: "TeX output 2001.10.15", es decir, compilado el 15-oct-2001), 14 páginas numeradas 1–14. No se indica revista ni año de publicación; no se adivinan aquí.
- 2000 MSC: 65M60, 65M20, 35K60, 35B40.
- Palabras clave: blow-up, parabolic equations, semidiscretization in space, asymptotic behavior, nonlinear boundary conditions.
- Financiación: Universidad de Buenos Aires (TX48), ANPCyT PICT No. 03-00000-05009. J. D. Rossi miembro de CONICET.
- Afiliaciones: G. Acosta (Instituto de Ciencias, UNGS, Los Polvorines); J. Fernández Bonder y J. D. Rossi (Depto. de Matemática, FCEyN, UBA); P. Groisman (Universidad de San Andrés, Victoria).
- Referencias clave citadas: [15] J.P. Pinasco y J.D. Rossi, *Simultaneous vs. nonsimultaneous blow-up*, New Zealand J. Math. 29 (2000), no. 1, 55–59 (caso continuo, no simultáneo); [16] J.D. Rossi, *On existence and nonexistence in the large for an N dimensional system of heat equations with nontrivial coupling at the boundary*, New Zealand J. Math. 26 (1997), no. 2, 275–285 (condiciones de blow-up); [3] Acosta, Fernández Bonder, Groisman, Rossi, *Numerical approximation of a parabolic problem with nonlinear boundary condition in several space dimensions*, preprint; [10] Durán, Etcheverry, Rossi, DCDS 4 (3) (1998) 497–506.

---

## 2. Problema continuo

Sistema de dos ecuaciones del calor acopladas por el borde:

$$u_t = \Delta u,\qquad v_t = \Delta v \qquad \text{en } \Omega\times(0,T), \tag{1.1}$$

$$\frac{\partial u}{\partial\eta} = u^{p_{11}} v^{p_{12}},\qquad \frac{\partial v}{\partial\eta} = u^{p_{21}} v^{p_{22}} \qquad \text{en } \partial\Omega\times(0,T), \tag{1.2}$$

$$u(x,0)=u_0(x),\quad v(x,0)=v_0(x)\quad\text{en }\Omega.$$

- $\partial/\partial\eta$ = derivada normal **exterior** unitaria.
- $\Omega\subset\mathbb{R}^d$ acotado y suave.
- Parámetros: $p_{ij}\ge 0$ (cuatro exponentes $p_{11},p_{12},p_{21},p_{22}$).
- $u_0,v_0$ positivos, acotados, compatibles con el dato de borde y suficientemente suaves para que $u,v$ sean regulares. Existencia local [4] (Amann).
- $T$ = tiempo maximal de existencia (finito o infinito). Si $T<\infty$: $\limsup_{t\nearrow T}(\|u(\cdot,t)\|_{L^\infty}+\|v(\cdot,t)\|_{L^\infty})=+\infty$.

**Resultados continuos conocidos (citados, no probados aquí):**
- [16]: existen soluciones que explotan en tiempo finito **si y sólo si** se cumple alguna de: $p_{11}>1$, $p_{22}>1$, o $(p_{11}-1)(p_{22}-1)<p_{12}p_{21}$.
- [15]: bajo hipótesis adecuadas, existen datos iniciales tales que $u$ explota y $v$ queda acotada **si y sólo si** $p_{11}>1$ y $p_{21}<p_{11}-1$ ("non-simultaneous blow-up").

En 1D con $\Omega=(0,1)$ (caso no tratado explícitamente en el paper, pero es la especialización directa): $u_t=u_{xx}$, $v_t=v_{xx}$ en $(0,1)$, con
$-u_x(0,t)=u^{p_{11}}v^{p_{12}}(0,t)$, $u_x(1,t)=u^{p_{11}}v^{p_{12}}(1,t)$, y análogo para $v$ con $p_{21},p_{22}$.

---

## 3. Esquema numérico

### 3.1 Semidiscretización general (sólo en espacio; $t$ continuo)

Para cada $h>0$ hay nodos $\{x_1,\dots,x_N\}\subset\overline\Omega$, $N=N(h)$, y $U(t)=(u_1,\dots,u_N)$, $V(t)=(v_1,\dots,v_N)$ aproximan $u(x_k,t)$, $v(x_k,t)$. El sistema EDO es

$$MU'(t) = -AU(t) + B\,(U^{p_{11}}V^{p_{12}}),\qquad MV'(t) = -AV(t) + B\,(U^{p_{21}}V^{p_{22}}), \tag{1.3}$$

con $u_k(0)=u_0(x_k)$, $v_k(0)=v_0(x_k)$. Todas las operaciones entre vectores (potencias, productos) son **componente a componente**.

Hipótesis sobre las matrices (que en general dependen de $h$):
- **(H1)** $M$ diagonal con entradas positivas $m_k$.
- **(H2)** $B$ diagonal con entradas no negativas $b_k$.
- **(H3)** $A$ simétrica, no negativa, con $a_{ij}\le0$ si $i\ne j$ y $a_{ii}>0$.
- **(H4)** $\sum_{j=1}^N a_{ij}=0$ (filas suman cero; implica que no hay estado estacionario no trivial, Lema 3.2).
- **(H5)** El grafo $\mathcal G$ con vértices $x_k$, donde $x_k,x_j$ están conectados sii $a_{kj}\ne0$, es conexo.

(H1)–(H5) implican principio del máximo. Un nodo $k$ es **nodo de borde** sii $b_k\ne0$.

Forma explícita (1.4):

$$m_k u_k'(t) = -\sum_{j=1}^N a_{kj}u_j(t) + b_k\,u_k^{p_{11}}(t)\,v_k^{p_{12}}(t),\qquad 1\le k\le N,$$
$$m_k v_k'(t) = -\sum_{j=1}^N a_{kj}v_j(t) + b_k\,u_k^{p_{21}}(t)\,v_k^{p_{22}}(t),\qquad 1\le k\le N. \tag{1.4}$$

### 3.2 Ejemplo 1 del paper: elementos finitos lineales con mass lumping

Triangulación regular **acutángula** de $\Omega$ (ver Ciarlet [9]); $W_h$ = funciones lineales a trozos en $H^1(\Omega)$. Se busca $u_h,v_h:[0,T_h)\to W_h$ con, para todo $w\in W_h$,

$$\int_\Omega ((u_h)_t w)^I = -\int_\Omega \nabla u_h\cdot\nabla w + \int_{\partial\Omega}\big((u_h)^{p_{11}}(v_h)^{p_{12}}w\big)^I,$$
$$\int_\Omega ((v_h)_t w)^I = -\int_\Omega \nabla v_h\cdot\nabla w + \int_{\partial\Omega}\big((u_h)^{p_{21}}(v_h)^{p_{22}}w\big)^I,$$

donde $(\cdot)^I$ es la interpolación de Lagrange lineal en los nodos. Así $M$ = matriz de masa concentrada (lumped), $A$ = matriz de rigidez, $B$ = "masa de borde" concentrada (diagonal). Nodo de borde $\Leftrightarrow x_k\in\partial\Omega$. La condición de borde entra **sólo** a través de $B$ (término de flujo natural, tipo Neumann), no se impone fuertemente.

Con funciones base nodales $\varphi_k$ esto da (derivación estándar, consistente con el texto):
$m_k=\int_\Omega\varphi_k$, $a_{kj}=\int_\Omega\nabla\varphi_k\cdot\nabla\varphi_j$, $b_k=\int_{\partial\Omega}\varphi_k$.

### 3.3 Ejemplo 2 del paper: diferencias finitas

Si $\Omega=(0,1)^d$ se puede usar un método de diferencias finitas semidiscreto que da un sistema de la forma (1.4); nodo de borde sii $x_k\in\partial\Omega$. El paper **no** escribe las fórmulas.

### 3.4 Matrices explícitas en 1D (derivación propia a partir de 3.2; el paper no las escribe)

$\Omega=(0,1)$, malla uniforme $x_k=(k-1)h$, $k=1,\dots,N$, $h=1/(N-1)$. P1 + lumping:

- $M=\mathrm{diag}(m_k)$: $m_1=m_N=h/2$, $m_k=h$ para $2\le k\le N-1$.
- $A=\frac1h\begin{pmatrix}1&-1&&&\\-1&2&-1&&\\&\ddots&\ddots&\ddots&\\&&-1&2&-1\\&&&-1&1\end{pmatrix}$, es decir $a_{11}=a_{NN}=1/h$, $a_{kk}=2/h$ ($2\le k\le N-1$), $a_{k,k\pm1}=-1/h$, resto 0. Cumple (H3)–(H5) (filas suman 0, grafo = camino).
- $B=\mathrm{diag}(b_k)$: $b_1=b_N=1$, $b_k=0$ interior (en 1D la integral de borde es evaluación puntual en $x=0$ y $x=1$).

Sistema explícito:
$$\tfrac h2 u_1' = \tfrac{u_2-u_1}{h} + u_1^{p_{11}}v_1^{p_{12}},\qquad h\,u_k' = \tfrac{u_{k+1}-2u_k+u_{k-1}}{h}\ (2\le k\le N-1),\qquad \tfrac h2 u_N' = \tfrac{u_{N-1}-u_N}{h} + u_N^{p_{11}}v_N^{p_{12}},$$
y lo mismo para $v$ con $u_k^{p_{21}}v_k^{p_{22}}$. Dividiendo por $m_k$: $u_1'=\frac{2}{h^2}(u_2-u_1)+\frac2h u_1^{p_{11}}v_1^{p_{12}}$, etc. Esto coincide con diferencias finitas centradas con nodo fantasma en el borde (ejemplo 3.3).

### 3.5 Discretización temporal / paso adaptativo / criterio de parada

**No hay ninguna en el paper.** Todo el análisis es sobre la EDO semidiscreta (1.3) con $t$ continuo. Tampoco se define criterio de parada numérico; sólo la definición teórica de blow-up discreto:

$T_h<\infty$ es tiempo de blow-up de (1.4) si
$$\lim_{t\nearrow T_h}\big(\|U(t)\|_\infty+\|V(t)\|_\infty\big)=\lim_{t\nearrow T_h}\Big(\max_k u_k(t)+\max_k v_k(t)\Big)=+\infty.$$
(En el PDF la segunda expresión aparece como $\max_k u_k(t)+\max_k u_k(t)$, errata evidente.)

### 3.6 Consistencia (Definición 2.1)

Sea $w$ solución regular de $w_t=\Delta w+g(x,t)$ en $\Omega\times(0,T)$, $\partial w/\partial\eta=f(x,t)$ en $\partial\Omega\times(0,T)$. El esquema es consistente si para $t\in(0,T-\tau)$
$$m_k w_t(x_k,t) = -\sum_{j=1}^N a_{kj}w(x_j,t) + m_k g(x_k,t) + b_k f(x_k,t) + \rho_{k,h}(t), \tag{2.1}$$
y existe $\rho:\mathbb R_+\to\mathbb R_+$ con $\max_k |\rho_{k,h}(t)|/m_k\le\rho(h)$ para todo $t\in(0,T-\tau)$, $\rho(h)\to0$ cuando $h\to0$. $\rho$ = módulo de consistencia.

---

## 4. Teoremas principales

- **Teorema 1.1 (convergencia).** Si $(u,v)$ es solución regular de (1.1)–(1.2), $(U,V)$ la de (1.4) y el método es consistente (Def. 2.1), existe $C$ independiente de $h$ con
$$\max_{1\le k\le N}\ \sup_{t\in[0,T-\tau]} |u(x_k,t)-u_k(t)|+|v(x_k,t)-v_k(t)| \le C\rho(h).$$
($0<\tau<T$ fijo.)

- **Teorema 1.2 (existencia de blow-up discreto).** Toda solución $(U,V)$ de (1.3) explota en tiempo finito **si y sólo si** se cumple alguna de: $p_{11}>1$, $p_{22}>1$, o $(p_{11}-1)(p_{22}-1)<p_{12}p_{21}$. (Mismas condiciones que en el continuo [16]; también las del Lema 3.1 para la EDO $x'=x^{p_{11}}y^{p_{12}}$, $y'=x^{p_{21}}y^{p_{22}}$.)

- **Teorema 1.3 (convergencia de tiempos de blow-up).** Sea $(u,v)$ solución de (1.1)–(1.2) con tiempo de blow-up $T$ y datos con $\Delta u_0,\Delta v_0\ge\kappa>0$; $(U,V)$ la solución numérica. Si el esquema es consistente y vale **una** de: $p_{11}>1$; $p_{22}>1$; o $p_{21},p_{12}>1$, entonces $\lim_{h\to0}T_h=T$.
  - La hipótesis $\Delta u_0,\Delta v_0\ge\kappa>0$ da monotonía $u_t,v_t>0$; también para $(U,V)$ (Lema 4.1).
  - En la prueba (caso $p_{11}>1$): $\delta(T_h-t)\le\int_{\max_k u_k(t)}^{\infty}x^{-p_{11}}\,dx$ (4.1).

- **Teorema 1.4 (necesidad).** Si $(U,V)$ solución de (1.4) con $U$ explotando en $T_h<\infty$ y $V$ acotada hasta $T_h$, entonces $p_{11}>1$ y $p_{21}<p_{11}-1$.

- **Teorema 1.5 (suficiencia).** Si $p_{11}>1$ y $p_{21}<p_{11}-1$, para todo dato $V_0$ de (1.4) existe un dato $U_0$ (en la prueba: $U_0$ suficientemente grande) tal que $U$ explota en tiempo finito $T_h$ y $V$ queda acotada hasta ese tiempo.

- **Teorema 1.6 (el esquema reproduce la no simultaneidad).** Si $u_0,v_0$ cumplen $\Delta u_0,\Delta v_0\ge\kappa>0$ y $u$ explota en $T$ con $v$ acotada hasta $T$, entonces, si el esquema es consistente, $U$ explota y $V$ queda acotada para todo $h$ suficientemente pequeño.

**Discrepancia en el abstract:** el abstract dice "$p_{11}>1$ and $p_{21}<2(p_{11}-1)$", pero los Teoremas 1.4 y 1.5 (y el resultado continuo [15] citado en la introducción y las pruebas) dicen $p_{21}<p_{11}-1$. La condición correcta según los teoremas y las pruebas es **$p_{21}<p_{11}-1$**.

**Tasas de blow-up (no enunciadas como teorema; aparecen en las pruebas, con $h$ fijo y constantes dependientes de $h$ salvo indicación):**
- Cota inferior (5.2), si $V$ acotada y $k$ nodo de borde que explota: $u_k(t)\ge C(T_h-t)^{-1/(p_{11}-1)}$.
- Cota superior (pruebas de 1.5 y 1.6): $\max_k u_k(t)\le C(T_h-t)^{-1/(p_{11}-1)}$; en la prueba del Teo. 1.6 (bajo Lema 4.1) $C$ es **independiente de $h$**.
- Es decir, en el régimen no simultáneo $\|U(t)\|_\infty\sim(T_h-t)^{-1/(p_{11}-1)}$.

**Conjuntos de blow-up:** el paper **no** trata conjuntos de blow-up (sólo menciona en la prueba del Lema 3.3 que el máximo se alcanza en un nodo de borde).

**Lemas auxiliares útiles:**
- Lema 2.1 (comparación): super/subsoluciones ordenadas en $t=0$ quedan ordenadas.
- Lema 3.2: la única solución no negativa de $0=-AZ+cBZ^\beta$ ($\beta,c>0$) es $Z\equiv0$.
- Lema 3.3: soluciones positivas de $MW'=-AW+cBW^\beta$ con $\beta>1$, $c>0$ explotan en tiempo finito. Funcional de Lyapunov $\Phi(W)=\frac12\langle A^{1/2}W,A^{1/2}W\rangle-\frac{c}{\beta+1}\langle BW^\beta,W\rangle$.
- Lema 3.4: si $p_{11}\le1$, $p_{22}\le1$, $(p_{11}-1)(p_{22}-1)<p_{12}p_{21}$, con $\alpha=(p_{12}-p_{22}+1)/(p_{21}-p_{11}+1)>0$: si $\alpha\ge1$, $Cu_k\ge v_k^\alpha$; si $\alpha<1$, $Cv_k\ge u_k^{1/\alpha}$ ($C$ independiente de $h$).
- Lema 4.1: si $p_{11}>1$, $u_k'(0)\ge\delta u_k^{p_{11}}(0)$ y $v_k'(0)\ge0$ para todo $k$, entonces $u_k'(t)\ge\delta u_k^{p_{11}}(t)$ y $v_k'(t)\ge0$ para $t<T_h$.
- Lema 4.2: si $p_{11},p_{22}\le1$, $p_{12},p_{21}>1$, $u_k'(0)\ge\delta v_k^{p_{12}}(0)$, $v_k'(0)\ge\delta u_k^{p_{21}}(0)$, entonces se mantienen para $t<T_h$.

---

## 5. Experimentos numéricos / figuras / tablas

**Ninguno.** El PDF no contiene experimentos numéricos, figuras ni tablas. No hay valores reportados que reproducir.

**No especificado (todo):** valores de $p_{ij}$ usados en cálculos, $N$ o $h$, dominio concreto, datos iniciales, integrador temporal, paso de tiempo, tolerancias, tiempo final, criterio de parada, valores de $T_h$.

---

## 6. Notas de implementación (propuesta propia, no del paper)

Si se quiere ilustrar numéricamente los resultados (no hay nada que "reproducir" del paper), lo natural es:

1. **Implementar (1.4) en 1D** con las matrices de 3.4 (vectorizado: $U'=M^{-1}(-AU+B\,U^{p_{11}}V^{p_{12}})$). Comprobar (H4): `A.sum(axis=1)==0`.
2. **Integrador:** EDO rígida ($\sim 1/h^2$) que explota; usar `scipy.integrate.solve_ivp` (BDF/Radau, `rtol` pequeño) con un evento que pare cuando $\max(\|U\|_\infty,\|V\|_\infty)$ supere un umbral grande (p.ej. $10^{8}$–$10^{12}$), o bien un esquema con paso adaptativo tipo $\Delta t_n=\tau\,\min(1,\|U^n\|_\infty^{-(p_{11}-1)})$ (heurística habitual en la literatura de estos autores, **no del paper**). Ver que las potencias se evalúan sólo con valores positivos (el principio del máximo garantiza positividad si los datos son positivos; con integradores explícitos grandes puede romperse).
3. **Verificaciones sugeridas:**
   - Teorema 1.2: con $p_{11}\le1$, $p_{22}\le1$, $(p_{11}-1)(p_{22}-1)\ge p_{12}p_{21}$ no debe haber blow-up; si alguna condición se cumple, sí.
   - Teoremas 1.4/1.5: con $p_{11}>1$, $p_{21}<p_{11}-1$, datos $U_0$ grandes ⇒ $\max_k v_k$ acotado mientras $\max_k u_k\to\infty$; con $p_{21}\ge p_{11}-1$ ambos deben explotar.
   - Tasa: en el régimen no simultáneo, $\log\|U\|_\infty$ vs. $\log(T_h-t)$ con pendiente $-1/(p_{11}-1)$; requiere estimar $T_h$ (p.ej. extrapolando $\|U\|_\infty^{-(p_{11}-1)}$, que debe ser ~lineal en $t$ cerca de $T_h$).
   - Teorema 1.3: $T_h\to T$ al refinar $h$, con datos que cumplan $\Delta u_0,\Delta v_0\ge\kappa>0$ (p.ej. $u_0$ convexa en 1D; ojo: deben ser compatibles con la condición de borde para la teoría, aunque numéricamente no es imprescindible). Para la monotonía discreta del Lema 4.1 conviene que $U'(0)\ge\delta U(0)^{p_{11}}$, $V'(0)\ge0$ en la malla.
4. **Erratas del PDF a tener en cuenta:**
   - Abstract: $p_{21}<2(p_{11}-1)$ vs. teoremas $p_{21}<p_{11}-1$ (usar la de los teoremas).
   - Definición de blow-up discreto: "$\max_k u_k+\max_k u_k$" debe ser "$\max_k u_k+\max_k v_k$".
   - Definición 2.2 (supersolución): la segunda desigualdad aparece como $B(\overline V^{p_{21}}\overline V^{p_{22}})$; debe ser $B(\overline U^{p_{21}}\overline V^{p_{22}})$ (ídem para subsolución).
   - En la prueba del Teo. 1.4 aparece "$1\ge k\ge N$" por $1\le k\le N$.
