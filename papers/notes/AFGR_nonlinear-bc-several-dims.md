# Notas: Acosta – Fernández Bonder – Groisman – Rossi, *Numerical approximation of a parabolic problem with a nonlinear boundary condition in several space dimensions*

> **Aviso principal.** Este artículo es **puramente teórico**: no contiene ningún experimento numérico, ninguna figura, ninguna tabla ni ningún resultado computado (se revisaron las 17 páginas del PDF; tampoco aparecen las palabras "figure"/"table" en el texto). No hay, por tanto, código MATLAB "perdido" que reproducir *de este artículo en sí*. Lo que sí contiene es un esquema semidiscreto general y teoremas cuantitativos (tasas, conjuntos de explosión, convergencia de tiempos) que se pueden **verificar numéricamente**. La sección 5 lo deja explícito y la sección 6 propone qué verificar, separando claramente lo que dice el artículo de lo que se sugiere aquí.

---

## 1. Referencia bibliográfica

- **Autores:** G. Acosta, J. Fernández Bonder, P. Groisman, J.D. Rossi.
- **Título (tal como aparece):** *Numerical approximation of a parabolic problem with a nonlinear boundary condition in several space dimensions.*
- **Afiliación:** Departamento de Matemática, F.C.E y N., UBA, (1428) Buenos Aires, Argentina. E-mails: gacosta, jfbonder, pgroisma, jrossi @dm.uba.ar.
- **Revista, año, volumen, páginas:** **No figuran en el PDF** (es un preprint compilado con pdfTeX; metadatos: fecha de creación 29-jul-2001, 17 páginas A4). *(Fuera del PDF, y sin verificar aquí: el trabajo se publicó, según recuerdo, en Discrete Contin. Dyn. Syst. Ser. B, vol. 2 (2002); confirmar antes de citar.)*
- **MSC 2000:** 65M20, 65M12, 35K57, 35B40.
- **Palabras clave:** blow-up, parabolic equations, semidiscretization in space, asymptotic behavior.
- **Financiación:** Universidad de Buenos Aires TX47, ANPCyT PICT No. 03-00000-00137, CONICET PIP0660/98, Fundación Antorchas.
- Antecedente directo en 1D: [DER] R.G. Durán, J.I. Etcheverry, J.D. Rossi, *Numerical approximation of a parabolic problem with a nonlinear boundary condition*, Discr. Cont. Dyn. Sys. 4(3) (1998) 497–506.

---

## 2. Problema continuo

Ecuación del calor con flujo no lineal en la frontera, ec. (1.1):
$$
\begin{cases}
u_t = \Delta u, & \text{en } \Omega\times(0,T),\\[2pt]
\dfrac{\partial u}{\partial \eta} = u^p, & \text{en } \partial\Omega\times(0,T),\\[2pt]
u = u_0 > 0, & \text{en } \Omega\times\{t=0\}.
\end{cases}
$$

- $\Omega\subset\mathbb{R}^n$ acotado y de frontera suave; $\eta$ = normal exterior.
- $u_0$ positivo y suficientemente regular para tener solución suave. Para la convergencia de tiempos de explosión (Thm 1.3, 1.6) se pide además $u_0$ **compatible** y $\Delta u_0 \ge \alpha > 0$ (esto implica $u_t>0$).
- $p>0$ en general; hay explosión en tiempo finito para todo dato positivo **si y sólo si $p>1$** ([RR], [W]).
- Hechos conocidos del continuo citados:
  - Conjunto de explosión localizado en la frontera: $B(u)\subset\partial\Omega$ ([HY], [RR]); para todo $\Omega'\subset\subset\Omega$ existe $K=K(d(\Omega',\partial\Omega))$ con $u\le K$ en $\Omega'\times[0,T)$.
  - Tasa continua ([HY]), para $1<p<\frac{n}{n-2}$:
    $$c\,(T-t)^{-\frac{1}{2(p-1)}} \le \|u(\cdot,t)\|_{L^\infty(\Omega)} \le C\,(T-t)^{-\frac{1}{2(p-1)}}.$$
  - Existe un ejemplo con explosión en un solo punto ([Hu]).

---

## 3. Esquema numérico

### 3.1 Sistema semidiscreto general (ec. (1.2)–(1.3))

Para cada $h>0$ pequeño hay nodos $\{x_1,\dots,x_N\}$ y $U(t)=(u_1(t),\dots,u_N(t))$, $u_k(t)\approx u(x_k,t)$, solución de
$$
M U'(t) = -A U(t) + B U^p(t), \qquad U(0)=U_0, \tag{1.2}
$$
donde $U^p$ es componente a componente. Explícitamente:
$$
m_k u_k'(t) = -\sum_{j=1}^N a_{kj} u_j(t) + b_k u_k^p(t),\qquad u_k(0)=u_{0,k},\qquad 1\le k\le N. \tag{1.3}
$$

**Hipótesis sobre las matrices (copiadas del artículo):**
- $M=\mathrm{diag}(m_k)$ y $B=\mathrm{diag}(b_k)$, "matrices diagonales con entradas positivas $m_k$ y $b_k$". *Observación:* en la práctica (y en el propio Lema 3.2, que habla de "$k$ tal que $b_k\neq 0$") $b_k=0$ en los nodos interiores; lo correcto es $m_k>0$, $b_k\ge 0$, con $b_k>0$ exactamente en los nodos de frontera.
- $A$ simétrica, "no negativa", con coeficientes no positivos fuera de la diagonal ($a_{ij}\le 0$ si $i\ne j$), $a_{ii}>0$, y $\sum_{j=1}^N a_{ij}\le 0$ (así está escrito en el PDF). *Observación:* para la matriz de rigidez de Neumann las sumas por filas son exactamente $0$ (las constantes están en el núcleo); con eso todas las hipótesis se cumplen.
- $U_0$ dato inicial; en el ejemplo de elementos finitos $U_0=u_0^I$ (interpolante nodal), con $U_0>0$.

### 3.2 Ejemplo 1: elementos finitos lineales con *mass lumping* (ec. (1.4))

- Triangulación **regular y aguda** de $\Omega$ ([Ci]); $h$ = norma (diámetro máximo) de la malla; $V_h$ = funciones lineales a trozos continuas en $H^1(\Omega)$.
- $u_h:[0,T_h)\to V_h$ verifica, para todo $v\in V_h$,
$$
\int_\Omega \big((u_h)_t\, v\big)^I = -\int_\Omega \nabla u_h\cdot\nabla v + \int_{\partial\Omega}\big((u_h)^p\, v\big)^I, \tag{1.4}
$$
  donde $(\cdot)^I$ es la interpolación de Lagrange lineal en los nodos.
- Con $\varphi_k$ la base nodal ("hat functions") esto da (1.2) con
  - $m_k = \int_\Omega \varphi_k$ (masa concentrada; en 2D, $m_k=\tfrac13\sum_{T\ni x_k}|T|$),
  - $a_{kj}=\int_\Omega\nabla\varphi_k\cdot\nabla\varphi_j$ (rigidez, **todos** los nodos incluidos los de frontera: condición de Neumann natural),
  - $b_k = \int_{\partial\Omega}\varphi_k$ (masa de frontera concentrada; en 2D, $b_k=\tfrac12\sum$ longitudes de las aristas de frontera que contienen a $x_k$; $b_k=0$ si $x_k$ es interior).
  - Las fórmulas explícitas de $m_k$ y $b_k$ en 2D son el cálculo estándar; el artículo sólo dice "$M$ es la matriz de masa concentrada, $A$ la de rigidez y $B$ corresponde a la condición de borde obtenida con lumping".
- La aguda garantiza $a_{ij}\le0$ para $i\ne j$ ([CR]: principio del máximo discreto).
- El artículo **no** discute la aproximación de $\Omega$ suave por un dominio poligonal $\Omega_h$.

### 3.3 Ejemplo 2: bilineales en el cuadrado
"Se verifica fácilmente" que elementos finitos bilineales sobre $\Omega=(0,1)\times(0,1)$ con elementos cuadrados conducen a un sistema de la forma (1.2). No se dan más detalles (presumiblemente también con lumping).

### 3.4 Ejemplo 3: diferencias finitas en el cubo
Si $\Omega=(0,1)^n$, un método de diferencias finitas semidiscreto lleva a un sistema (1.2). No se especifica el tratamiento de la condición de Neumann (nodos fantasma, etc.).

### 3.5 Consistencia (Definición 2.1)
Sea $w$ solución regular de $w_t=\Delta w+g(x,t)$ en $\Omega\times(0,T)$, $\partial w/\partial\eta=f(x,t)$ en $\partial\Omega\times(0,T)$. El esquema es **consistente** si para todo $t\in(0,T-\tau)$
$$
m_k w_t(x_k,t) = -\sum_{j=1}^N a_{kj}w(x_j,t) + m_k g(x_k,t) + b_k f(x_k,t) + \rho_{k,h}(t), \tag{2.1}
$$
y existe $\rho:\mathbb{R}_+\to\mathbb{R}_+$ con $\max_k |\rho_{k,h}(t)|/m_k \le \rho(h)$, $\rho(h)\to0$ cuando $h\to0$ ("módulo de consistencia"). Sólo se exige consistencia del problema lineal. (En 1D, [ALM2] prueba consistencia si $u\in C^{4,1}$; ver también [MM].)

### 3.6 Discretización temporal, paso adaptativo, criterios de parada
**No hay.** El artículo trata sólo la semidiscretización en espacio (sistema de EDOs). No se propone ningún esquema totalmente discreto, ni regla de paso adaptativo, ni criterio de parada. Al final de §1 los autores señalan que la diferencia en la tasa de explosión "sugiere que se necesita un método adaptativo para reproducir la misma tasa" y lo dejan "para trabajo futuro" (adaptatividad en espacio, no en tiempo).

### 3.7 Otras definiciones
- $T_h=\sup\{t:\ u_h(s)\text{ está definida para } s\in[0,t]\}$. Explosión en tiempo finito: $\lim_{t\nearrow T_h}\|U(t)\|_\infty=\lim_{t\nearrow T_h}\max_k u_k(t)=+\infty$.
- **Conjunto de explosión** $B(u_h)$ = nodos $x_k$ con $\lim_{t\nearrow T_h}u_k(t)=+\infty$.
- $F$ = conjunto de nodos $x_j$ con $u_j(t)\sim(T_h-t)^{-\frac1{p-1}}$ (equivalentemente, en variables (6.1), $y_j(s)\not\to0$). $F\neq\emptyset$ por Thm 1.4 y $F\subset B(u_h)$.
- **Definición 1.1 (distancia en grafo):** grafo con vértices en los nodos; dos nodos distintos están conectados sii $a_{ij}\neq0$. Distancia usual de grafo ([Ha]); $d(k)$ = distancia del nodo $x_k$ al conjunto $F$.
- Super/subsolución (Def. 2.2): $V$ es supersolución (resp. subsolución) de (1.2) si $MV'\ge(\le)\,-AV+BV^p$ componente a componente.
- Funcional de Lyapunov (Lema 3.1): $\Phi_h(U)=\tfrac12\langle A^{1/2}U,A^{1/2}U\rangle-\tfrac1{p+1}\langle BU^p,U\rangle$, con $\frac{d}{dt}\Phi_h(U(t))=-\langle MU',U'\rangle$.

---

## 4. Resultados principales

- **Lema 2.1 (comparación).** Si $\overline U$ supersolución, $\underline U$ subsolución y $\underline U(0)\le\overline U(0)$, entonces $\underline U(t)\le\overline U(t)$.
- **Teorema 1.1 (convergencia).** Si $u$ es solución regular de (1.1), $u_h$ la de (1.3) y el método es consistente (Def. 2.1), existe $C$ tal que
  $$\max_k\max_{0\le t\le T-\tau}|u(x_k,t)-u_k(t)|\le C\rho(h).$$
  (Para $0<\tau<T$ fijo; en la prueba se usa también $\max_k|u_0(x_k)-u_{0,k}|\le K\rho(h)$.)
- **Lema 3.1.** Si $U_0>0$, $U$ es no acotada (usa Lyapunov + principio del mínimo $\min_j u_j(t)\ge\min_j u_{0,j}>0$; el único punto fijo es $U\equiv0$).
- **Lema 3.2.** Si $\max_j u_j(t)>\max_j u_{0,j}$, entonces $\max_j u_j(t)=u_k(t)$ con $b_k\ne0$ (el máximo se alcanza en un nodo de frontera).
- **Teorema 1.2.** Las soluciones positivas de (1.3) explotan en tiempo finito **si y sólo si $p>1$**.
- **Lema 4.1.** Si $u_k'(0)\ge\delta u_k^p(0)$ para todo $k$, entonces $u_k'(t)\ge\delta u_k^p(t)$ para todo $k$ y $t<T_h$.
- **Teorema 1.3 (convergencia de tiempos de explosión).** Sea $u_0$ compatible con $\Delta u_0\ge\alpha>0$ y método consistente. Si $T$, $T_h$ son los tiempos de explosión de $u$ y $u_h$:
  $$\lim_{h\to0}T_h=T.$$
  Estimación clave de la prueba (con $\delta$ independiente de $h$):
  $$\delta(T_h-t)\le\int_{\max_k u_k(t)}^{+\infty}\frac{dx}{x^p}=\frac{(\max_k u_k(t))^{1-p}}{p-1}. \tag{4.2}$$
- **Teorema 1.4 (tasa de explosión numérica).** Si $p>1$ y $u_h$ explota en $T_h$:
  $$c\,(T_h-t)^{-\frac1{p-1}}\le\max_j u_j(t)\le C\,(T_h-t)^{-\frac1{p-1}},$$
  con $c,C>0$ **posiblemente dependientes de $h$**. La tasa numérica $(T_h-t)^{-1/(p-1)}$ **difiere** de la continua $(T-t)^{-1/(2(p-1))}$. (Primera vez observado en la literatura, según los autores.) En la prueba: si $u_k=\max_j u_j$ cerca de $T_h$, $u_k'\ge\frac{b_k}{2m_k}u_k^p$, y para $w=\sum_k u_k$, $w'\le Cw^p$.
- **Cambio de variables (6.1)** (inspirado en [GK], [HV1], [HV2]):
  $$y_k(s)=(T_h-t)^{\frac1{p-1}}u_k(t),\qquad T_h-t=e^{-s},$$
  que lleva a
  $$m_k y_k'(s)=-e^{-s}\sum_{j=1}^N a_{kj}y_j(s)-\frac{m_k}{p-1}y_k(s)+b_k y_k^p(s),\qquad y_k(-\ln T_h)=T_h^{\frac1{p-1}}u_0(x_k) \tag{6.2}$$
  (el PDF escribe "$1\le k\le N+1$", errata evidente por $N$). Las $y_k$ están uniformemente acotadas. Se define $\Gamma_k=\sup_s\big|\sum_j a_{kj}y_j(s)\big|$.
- **Lemas 6.1–6.3.** Cada $y_k$ verifica, cuando $s\to+\infty$, o bien $y_k(s)\to0$, o bien $y_k(s)\to l_k$, donde $l_k$ es la única raíz positiva de $g_k(x)=-\frac{m_k}{p-1}x+b_kx^p$, es decir
  $$l_k=\Big(\frac{m_k}{(p-1)\,b_k}\Big)^{\frac1{p-1}}$$
  (la fórmula cerrada es despeje inmediato, no está escrita en el artículo). Por tanto $k\in F$ requiere $b_k>0$: **$F$ está formado por nodos de frontera**, y para $k\in F$, $u_k(t)\approx l_k(T_h-t)^{-1/(p-1)}$.
  - Lema 6.1: si $b_ky_k^p(s_0)-\frac{m_k}{p-1}y_k(s_0)<-\Gamma_ke^{-s_0}$ para algún $s_0$, entonces $y_k\to0$.
  - Lema 6.2: para todo $s$, $b_ky_k^p(s)-\frac{m_k}{p-1}y_k(s)\le\Gamma_ke^{-s}$.
- **Teorema 1.5 (propagación del blow-up).** Sea $p>1$ y $K\in\mathbb{N}_0$ con
  $$\frac{K+2}{K+1}<p\le\frac{K+1}{K}\qquad(\text{si } p>2,\ K=0).$$
  Entonces $U$ explota exactamente en los nodos a distancia $\le K$ de $F$:
  $$u_k(t)\to+\infty\iff d(k)\le K.$$
  Además, si $d(k)\le K$:
  $$u_k(t)\sim(T_h-t)^{-\frac1{p-1}+d(k)}\quad\text{si } p\ne\tfrac{K+1}{K},$$
  y si $p=\frac{K+1}{K}$ y $d(k)=K$: $u_k(t)\sim\ln(T_h-t)$ (en valor absoluto; en la prueba aparece $-C\ln(T_h-t)$).
  Si $d(k)>d(j)$ entonces $u_k(t)/u_j(t)\to0$. El número de nodos de explosión fuera de $F$ depende de $p$ pero **no de $h$**.
  - Caso $d(k)=1$ detallado en la prueba:
    $$y_k(s)\sim\begin{cases}Ce^{-\frac1{p-1}s}& p>2\\ Cse^{-\frac1{p-1}s}& p=2\\ Ce^{-s}& p<2\end{cases}\qquad
    u_k(t)\sim\begin{cases}C\ (\text{acotada})& p>2\\ -C\ln(T_h-t)& p=2\\ C(T_h-t)^{\frac{p-2}{p-1}}& p<2.\end{cases}$$
  - Tabla de $K$ (derivada de la desigualdad, no está en el artículo):

    | rango de $p$ | $K$ | nodos que explotan |
    |---|---|---|
    | $p>2$ | 0 | sólo $F$ |
    | $3/2<p\le2$ | 1 | $d\le1$ (log si $p=2$, $d=1$) |
    | $4/3<p\le3/2$ | 2 | $d\le2$ (log si $p=3/2$, $d=2$) |
    | $5/4<p\le4/3$ | 3 | $d\le3$ (log si $p=4/3$, $d=3$) |
    | $\frac{K+2}{K+1}<p\le\frac{K+1}{K}$ | $K$ | $d\le K$ |

- **Teorema 1.6 (localización del conjunto de explosión).** Si el método es consistente y $\Delta u_0\ge\alpha>0$, dado $\varepsilon>0$ existe $h_0$ tal que para todo $0<h\le h_0$
  $$B(u_h)\subset B(u)+N_\varepsilon,\qquad N_\varepsilon=\{x\in\mathbb{R}^n:|x|<\varepsilon\}.$$
  En particular, como $B(u)\subset\partial\Omega$, $B(u_h)$ está en un entorno pequeño de $\partial\Omega$; y en el ejemplo de explosión en un punto de [Hu], $B(u_h)$ se concentra en ese punto cuando $h\to0$. En la prueba: $B(u_h)\subset F+\overline{N_{Kh}}\subset A+\overline{N_{Kh}}$, $A=B(u)+N_{\varepsilon/2}$.
- Los Teoremas 1.2, 1.4, 1.5 valen para **cualquier** sistema de la forma (1.3) con las hipótesis de §3.1, venga o no de una semidiscretización de (1.1).

---

## 5. Experimentos numéricos / figuras / tablas

**Ninguno.** El artículo no contiene experimentos numéricos, figuras ni tablas. No hay valores de $p$, $h$, datos iniciales, pasos de tiempo ni tolerancias reportados.

### No especificado (todo lo relativo a computación)
- Dominio concreto, malla concreta, valores de $h$.
- Valores de $p$.
- Dato inicial $u_0$.
- Integrador temporal, paso de tiempo, regla adaptativa, tolerancias, criterio de parada / estimación de $T_h$.
- Tratamiento de la frontera curva ($\Omega$ vs $\Omega_h$) y detalles del esquema de diferencias finitas en $(0,1)^n$ y de los bilineales.
- Valores de las constantes $c$, $C$, $\delta$, $\rho(h)$.

---

## 6. Notas de implementación (sugerencias propias, no del artículo)

Si se quiere ilustrar numéricamente el artículo, lo natural es verificar los teoremas:

1. **Montaje del sistema (2D, P1, lumping).** Ensamblar $A$ (rigidez completa, sin eliminar nodos de frontera), $m_k=\frac13\sum_{T\ni k}|T|$, $b_k=\frac12\sum_{e\ni k,\,e\subset\partial\Omega_h}|e|$. Comprobar: $A$ simétrica, $a_{ij}\le0$ fuera de la diagonal (malla **aguda** —o al menos de Delaunay en 2D, que da $a_{ij}\le0$; el artículo pide aguda), sumas por filas $\approx0$, $\sum_k m_k=|\Omega_h|$, $\sum_k b_k=|\partial\Omega_h|$. Ejemplo sencillo: $\Omega=(0,1)^2$ con malla uniforme de triángulos rectángulos (no estrictamente aguda pero da $a_{ij}\le0$: es el Laplaciano de 5 puntos) o el disco unidad con malla aguda.
2. **Rigidez.** El sistema es rígido ($\|A\|/\min m_k\sim h^{-2}$) y además explota. Usar un integrador implícito/rígido (p.ej. `scipy.integrate.solve_ivp` con `BDF`/`Radau` y Jacobiano disperso $-M^{-1}(A-pB\,\mathrm{diag}(U^{p-1}))$) o un esquema semi-implícito con paso adaptado al tamaño de la solución, p.ej. $\Delta t_n=\tau\min\{1,\|U^n\|_\infty^{1-p}\}$ (regla habitual en la literatura de este grupo; **no está en este artículo**). Parar cuando $\|U\|_\infty$ supere un umbral grande (p.ej. $10^{8}$–$10^{12}$).
3. **Estimar $T_h$.** Por (4.2)/Thm 1.4, para $k\in F$, $u_k^{1-p}\approx(p-1)\frac{b_k}{m_k}(T_h-t)$ cerca de $T_h$: extrapolar linealmente $\|U(t)\|_\infty^{1-p}$ a cero da una buena estimación de $T_h$.
4. **Tasa (Thm 1.4).** Verificar que la pendiente de $\log\|U\|_\infty$ frente a $\log(T_h-t)$ tiende a $-\frac1{p-1}$ (no a $-\frac1{2(p-1)}$). Equivalente: $y_k(s)=(T_h-t)^{1/(p-1)}u_k\to l_k=(m_k/((p-1)b_k))^{1/(p-1)}$ para $k\in F$. Notar que $m_k\sim h^n$, $b_k\sim h^{n-1}$, luego $l_k\sim h^{1/(p-1)}\to0$: la constante $c,C$ del Thm 1.4 depende de $h$, y la tasa discreta sólo se ve en una ventana $T_h-t\lesssim h^2$ (ahí el sistema "se desacopla" y $m_ku_k'\approx b_ku_k^p$); para $T_h-t\gg h^2$ se debería ver la tasa continua. Esto es una deducción, no un enunciado del artículo.
5. **Propagación (Thm 1.5).** Identificar $F$ (nodos con $y_k\to l_k$), calcular $d(k)$ en el grafo de $A$ (BFS sobre el patrón de no ceros fuera de la diagonal) y comprobar que explotan exactamente los nodos con $d(k)\le K(p)$ y que $u_k\sim(T_h-t)^{-1/(p-1)+d(k)}$. Con $p$ cerca de 1 (p.ej. $p=1.3$, $K=3$) la propagación es visible; para $p>2$ sólo explotan los nodos de $F$. Cuidado: las tasas de los nodos con $d\ge1$ son mucho más lentas, y para verlas hay que acercarse muchísimo a $T_h$ (precisión de máquina en $T_h-t$: conviene integrar en la variable $s=-\ln(T_h-t)$ usando (6.2) si se conoce $T_h$, o reparametrizar el tiempo).
6. **Convergencia de $T_h$ (Thm 1.3).** Con $\Delta u_0\ge\alpha>0$ (y $u_0$ compatible: $\partial u_0/\partial\eta=u_0^p$ en $\partial\Omega$) refinar $h$ y comprobar $T_h\to T$. El artículo no da tasa de convergencia de $T_h$.
7. **Localización (Thm 1.6).** Comprobar que $B(u_h)$ está contenido en una banda de anchura $\approx Kh$ alrededor de los nodos de $F$, todos en la frontera.
8. **Monotonía.** Con $\Delta u_0\ge\alpha$, $u_k'(t)\ge\delta u_k^p(t)>0$ (Lema 4.1): las soluciones deben ser crecientes en tiempo; útil como test. Positividad: $\min_ju_j(t)\ge\min_ju_{0,j}$.
9. **Escritura de $U^p$.** Usar $|U|^{p}$ o recortar a $U\ge0$ para evitar NaN con $p$ no entero si un integrador produce valores negativos transitorios.
