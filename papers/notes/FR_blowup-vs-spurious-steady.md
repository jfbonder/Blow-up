# Fernández Bonder & Rossi — *Blow-up vs. spurious steady solutions*

Notas para reimplementar (Python/NumPy/SciPy) todo lo computable del artículo sin volver a abrir el PDF
(`papers/FR_blowup-vs-spurious-steady.pdf`, 6 páginas).

---

## 1. Referencia bibliográfica

- **Autores:** Julián Fernández Bonder y Julio D. Rossi (Departamento de Matemática, FCEyN, UBA, Buenos Aires).
- **Título:** *Blow-up vs. spurious steady solutions*.
- **Revista / año / volumen / páginas:** **el PDF no lo indica.** Es una versión preprint (TeX compilado el 31-10-2001 según los metadatos del PDF). No figura revista, volumen ni páginas; no se completa aquí para no adivinar.
- MSC 1991: 35K55, 35B40, 65M20. Palabras clave: blow-up, semidiscretization, spurious solutions.
- Financiación: UBA grant TX047, ANPCyT PICT 03-00000-00137; Rossi también por Fundación Antorchas.
- Referencias citadas en el paper: [1] Budd–Huang–Russell (SIAM J. Sci. Comput. 17(2), 1996, 305–327); [2] Chipot–Fila–Quittner (Acta Math. Univ. Comenianae LX, 1 (1991), 35–103); [3] Durán–Etcheverry–Rossi (Disc. Cont. Dyn. Sys. 4(3), 1998, 497–506); [4] Henry, LNM 840, Springer 1981; [5] López Gómez–Márquez–Wolanski (Rev. UMA 38 (1993) 196–209); [6] Pao, Plenum 1992.

---

## 2. Problema continuo

Ecuación del calor 1D con absorción en el interior y flujo no lineal (reacción) en el borde $x=1$:

$$
\begin{cases}
u_t = u_{xx} - \lambda u^p & \text{en } (0,1)\times[0,T),\\
u_x(1,t) = u(1,t)^q & \text{en } [0,T),\\
u_x(0,t) = 0 & \text{en } [0,T),\\
u(x,0) = u_0(x) \ge 0 & \text{en } [0,1],
\end{cases}\qquad (1.1)
$$

con parámetros $p>1$, $q>1$, $\lambda>0$.

- Existencia/regularidad: [5], [2] para datos con condición de compatibilidad; en general solución en $H^1$ por aproximación ([2]).
- Competencia entre reacción en el borde y absorción: hay blow-up **si y sólo si** $p<2q-1$, o $p=2q-1$ con $\lambda<q$ ([5],[2]).

**Teorema 1.1** ([2], Teoremas 4.1, 4.2, 4.7, y [5]).
1. Si $p<2q-1$, o $p=2q-1$ con $\lambda<q$: si $u_0>v$, donde $v$ es cualquier solución estacionaria maximal, entonces $u$ explota en tiempo finito.
2. Si $p>2q-1$, o $p=2q-1$ con $\lambda\ge q$: toda solución positiva es global.

**Funcional de Lyapunov continuo** (Sección 2):
$$
\Phi(u) = \int_0^1 \frac{(u_x)^2}{2} + \lambda\int_0^1 \frac{u^{p+1}}{p+1} - \frac{u^{q+1}(1,t)}{q+1}.
$$
Si $\Phi(u_0)<0$ entonces $u$ explota en tiempo finito ([2], Teorema 4.5) — esto se usa en el caso $p\le q$.

---

## 3. Esquema numérico

### 3.1 Origen
Elementos finitos lineales ($P_1$) en $x$ sobre malla uniforme, $t$ continuo, con **mass lumping** (coincide con diferencias finitas de segundo orden). Forma matricial:
$$
\begin{cases}
MU' = -AU + u_N^q\, e_N - \lambda M U^p,\\
U(0) = u_0^I,
\end{cases}\qquad (1.2)
$$
con $U=(u_1,\dots,u_N)$, $e_N=(0,\dots,0,1)$, $M$ matriz de masa (lumped, diagonal), $A$ matriz de rigidez, $u_0^I$ interpolante de Lagrange de $u_0$; $U^p$ es componente a componente.

Explícitamente (deducido de (1.2)–(1.3), consistente con ellas): nodos $x_i$, $i=1,\dots,N$, partición uniforme de $[0,1]$, $x_1=0$, $x_N=1$, de modo que $h=1/(N-1)$ (el paper no escribe $h=1/(N-1)$ explícitamente, sólo "partición uniforme de $[0,1]$" con nodos $1..N$):
$$
M = h\,\mathrm{diag}\!\left(\tfrac12,1,\dots,1,\tfrac12\right),\qquad
A = \frac1h\begin{pmatrix}
1&-1&&&\\
-1&2&-1&&\\
&\ddots&\ddots&\ddots&\\
&&-1&2&-1\\
&&&-1&1
\end{pmatrix}.
$$
$A$ es simétrica semidefinida positiva (núcleo = constantes).

### 3.2 Sistema de EDOs entrada por entrada (ecuación (1.3) del paper)
$$
\begin{cases}
u_1' = \dfrac{2}{h^2}(u_2-u_1) - \lambda u_1^p,\\[2mm]
u_k' = \dfrac{1}{h^2}(u_{k+1}-2u_k+u_{k-1}) - \lambda u_k^p, & 2\le k\le N-1,\\[2mm]
u_N' = \dfrac{2}{h^2}(u_{N-1}-u_N) - \lambda u_N^p + \dfrac{2}{h}u_N^q,\\[2mm]
u_i(0) = u_0(x_i), & 1\le i\le N.
\end{cases}\qquad (1.3)
$$
- Neumann homogéneo en $x=0$: entra vía el factor $2/h^2$ (nodo frontera con masa $h/2$, equivalente a nodo fantasma $u_0=u_2$).
- Flujo no lineal en $x=1$: término $\frac{2}{h}u_N^q$ en la última ecuación (= $u_N^q/m_{NN}$ con $m_{NN}=h/2$).
- Absorción $-\lambda u_k^p$ en **todas** las ecuaciones, incluidos los nodos frontera (gracias al lumping la absorción es puntual).

### 3.3 Tiempo y criterios de parada
El paper **no** discretiza en tiempo ni da criterios de parada: trabaja sólo con la semidiscretización (1.3) como sistema de EDOs. El tiempo de blow-up semidiscreto $T_h$ se define como el tiempo en que $\max_k |u_k(t)|\to\infty$.

### 3.4 Convergencia
**Teorema 1.2.** Si $u\in C^{2,1}([0,1]\times[0,T_1])$ es solución de (1.1) y $u_h$ su aproximación semidiscreta, existe $C=C(T_1,u)$ tal que, para $h$ suficientemente pequeño,
$$\|u-u_h\|_{L^\infty([0,1]\times[0,T_1])}\le C h^{3/2}.$$
(Adaptación del Teorema 2 de [3].)

### 3.5 Funcional de Lyapunov discreto
$$
\Phi_h(U) = \tfrac12\langle A^{1/2}U;A^{1/2}U\rangle + \frac{\lambda}{p+1}\langle MU^p;U\rangle - \frac{U_N^{q+1}}{q+1}
= \tfrac12 U^TAU + \frac{\lambda}{p+1}\sum_i m_{ii}u_i^{p+1} - \frac{u_N^{q+1}}{q+1}.
$$
Es funcional de Lyapunov para (1.2) (decrece a lo largo de trayectorias). Se afirma $\Phi_h(u_0^I)\to\Phi(u_0)$ cuando $h\to0$.

---

## 4. Teoremas principales

### Teorema 1.3 (blow-up semidiscreto, Sección 2)
Sea $U$ solución positiva de (1.3).
1. Si $p\le q$ y el dato inicial es suficientemente grande, $U$ tiene tiempo de blow-up finito.
2. Si $p>q$, $U$ es global.

**Consecuencia clave:** para $q<p<2q-1$ (o $p=2q-1$, $\lambda<q$) el problema continuo tiene soluciones que explotan, pero **todas** las soluciones de (1.3) son globales, para todo $h$.

*Prueba de (1), versión Lyapunov:* si $W$ es estacionaria de (1.2), es decir
$$0=-AW-\lambda MW^p+w_N^q e_N,\qquad (2.1)$$
multiplicando por $W$ y usando $1<p\le q$ el paper obtiene (cálculo tal como está impreso)
$$
0=-\tfrac12\langle A^{1/2}W;A^{1/2}W\rangle-\lambda\sum_{i=1}^N m_{ii}\frac{w_i^{p+1}}{2}+\frac{w_N^{q+1}}{2}
\;\ge\; -\tfrac12\langle A^{1/2}W;A^{1/2}W\rangle-\lambda\sum_{i=1}^N m_{ii}\frac{w_i^{p+1}}{p+1}+\frac{w_N^{q+1}}{q+1}=-\Phi_h(W),
$$
luego toda estacionaria positiva tiene $\Phi_h(W)\ge0$; como toda solución global converge a una estacionaria ([4]), si $\Phi_h(U_0)<0$ hay blow-up. Como $\Phi_h(u_0^I)\to\Phi(u_0)$: si $\Phi(u_0)<0$, $u$ y $u_h$ explotan para todo $h$ pequeño (el tamaño del dato necesario **no** depende de $h$).

*Prueba alternativa (Remark):* si $u_N^{q-1}(0)>2/h$, entonces, para $h$ pequeño,
$$u_N'\ge -\frac{2u_N}{h^2}-\lambda u_N^p+\frac{2}{h}u_N^q\ge u_N^q,$$
así $u_N$ explota, y
$$T_h-t\le\int_{u_N(t)}^\infty\frac{dx}{x^q}=\frac{u_N(t)^{1-q}}{q-1}<\infty.\qquad(2.2)$$
(Esta versión sugiere, erróneamente, que el tamaño del dato depende de $h$.)

*Prueba de (2), $p>q$:* por contradicción; si $\max_k|u_k|\to\infty$, se toma $t_0$ y $j$ con $\max_k\sup_{[0,t_0]}u_k=u_j(t_0)>M$, así $u_j'(t_0)\ge0$; si $j<N$ las ecuaciones fuerzan $u_{j+1}=u_{j-1}=u_j$ (U constante); si $j=N$,
$$u_N'(t_0)=\frac{2(u_{N-1}-u_N)}{h^2}-\lambda u_N^p+\frac2h u_N^q<0$$
para $u_N$ grande (porque $p>q$; el texto impreso dice "as $p\ge q$"), contradicción. Nota: el umbral $M$ depende de $h$ ($\lambda M^{p-q}>2/h$).

### Sección 3: soluciones estacionarias espurias (se asume $q<p<2q-1$)
Estacionarias de (1.3):
$$
\begin{cases}
0=\dfrac{2}{h^2}(w_2-w_1)-\lambda w_1^p,\\[1mm]
0=\dfrac{1}{h^2}(w_{k+1}-2w_k+w_{k-1})-\lambda w_k^p, & 2\le k\le N-1,\\[1mm]
0=\dfrac{2}{h^2}(w_{N-1}-w_N)-\lambda w_N^p+\dfrac{2}{h}w_N^q.
\end{cases}\qquad(3.1)
$$
**Reducción a una ecuación escalar (shooting desde $x=0$):**

**Atención:** el paper escribe literalmente
$$w_2 = w_1+\lambda h^2 w_1^p\equiv F_2(w_1),$$
pero de la primera ecuación de (3.1) se despeja $w_2=w_1+\tfrac{\lambda h^2}{2}w_1^p$. Es una errata del paper (o inconsistencia con el factor 2 de (1.3)); para que $W$ sea realmente solución de (3.1)/(1.3) hay que usar $F_2(w_1)=w_1+\frac{\lambda h^2}{2}w_1^p$. No afecta a los argumentos cualitativos.

Para $k\ge3$ (de la ecuación interior en el nodo $k-1$):
$$
w_k=2w_{k-1}-w_{k-2}+\lambda h^2 w_{k-1}^p = 2F_{k-1}(w_1)-F_{k-2}(w_1)+\lambda h^2(F_{k-1}(w_1))^p\equiv F_k(w_1),
$$
con $F_1(w_1)=w_1$. En particular $w_3=2F_2(w_1)-w_1+\lambda h^2(F_2(w_1))^p$.
Propiedades: cada $F_k$ es creciente en $w_1$, las diferencias $w_k-w_{k-1}$ son crecientes en $w_1$, y $w_1<w_2<\dots<w_N$: **toda solución de (3.1) es creciente**. $F_N$ es continua creciente con rango $[0,\infty)$.

La última ecuación da la condición escalar
$$
0=2\,\frac{F_{N-1}-F_N}{h^2}-\lambda (F_N)^p+\frac{2F_N^q}{h}\equiv G(w_1).\qquad(3.2)
$$
Soluciones positivas de (3.1) $\leftrightarrow$ ceros positivos $w_1$ de $G$ (con $w_k=F_k(w_1)$).

**Existencia de la espuria.** Sea $x$ con $F_N(x)=a\,h^{-1/(p-q)}$. Usando $F_{N-1}-F_N\ge -F_N$:
$$
G(x)\ge-\frac{2a}{h^{2+\frac1{p-q}}}+\frac{2a^q}{h^{1+\frac q{p-q}}}-\lambda\frac{a^p}{h^{\frac p{p-q}}}\ge0
$$
si $a<(2/\lambda)^{1/(p-q)}$ y $h$ pequeño (aquí se usa $p<2q-1$: $1+\frac{q}{p-q}=\frac{p}{p-q}>2+\frac1{p-q}\iff p<2q-1$). Sea $y$ con $F_N(y)=b\,h^{-1/(p-q)}$, $b\ge(2/\lambda)^{1/(p-q)}$; usando $F_{N-1}-F_N\le0$:
$$
G(y)\le \frac{2b^q}{h^{1+\frac{q}{p-q}}}-\lambda\frac{b^p}{h^{\frac p{p-q}}}\le0.
$$
Por continuidad existe un cero de $G$, es decir una solución de (3.1), con
$$
w_N=F_N(w_1)=\frac{c}{h^{1/(p-q)}},\qquad \Big(\frac2\lambda\Big)^{\frac1{p-q}}-\varepsilon<c<\Big(\frac2\lambda\Big)^{\frac1{p-q}}\quad\text{si } h\le h(\varepsilon).
$$
Es decir, **fórmula asintótica explícita:** $w_N\approx\left(\dfrac{2}{\lambda h}\right)^{1/(p-q)}$ (balance $\lambda w_N^p\approx\frac2h w_N^q$), que tiende a $\infty$ cuando $h\to0$. No se da fórmula cerrada para $w_k$, $k<N$ (sólo la recursión $F_k$).

**Estabilidad (atractividad).** Linealización de (1.2) en $W$:
$$Z'=(-M^{-1}A+B)Z,$$
$B$ diagonal con
$$
b_{ii}=-p\lambda w_i^{p-1}\ (1\le i\le N-1),\qquad b_{NN}=-p\lambda w_N^{p-1}+\frac{2q\,w_N^{q-1}}{h}.
$$
Si se toma $\varepsilon$ con $(2/\lambda)^{1/(p-q)}-\varepsilon>(2q/(\lambda p))^{1/(p-q)}$ (posible pues $q<p$), entonces $w_N>\left(\frac{2q}{p\lambda h}\right)^{1/(p-q)}$, luego $b_{NN}<0$, $B$ es definida negativa y $-M^{-1}A+B$ también; todos los autovalores tienen parte real negativa $\Rightarrow$ $W$ es localmente asintóticamente estable.

### Teorema 1.4 (enunciado)
Si $q<p<2q-1$, existe una solución estacionaria espuria $W=(w_1,\dots,w_N)$ de (1.3), atractiva, con $|w_N|\sim h^{-1/(p-q)}$. Así, si el dato inicial es suficientemente grande, la solución es global y converge, cuando $t\to\infty$, a una estacionaria espuria con $w_N$ de orden $h^{-1/(p-q)}$.

Observaciones de fidelidad: la prueba sólo demuestra existencia + estabilidad lineal (atractividad local); la afirmación "datos grandes convergen a la espuria" no se prueba en detalle (se apoya en que toda solución global converge a una estacionaria, [4], y en el Teorema 1.3(2)). No se discute unicidad de la espuria ni otras estacionarias (p.ej. las que aproximan las del continuo).

### Resumen de regímenes
| Rango | Continuo (1.1) | Semidiscreto (1.3) |
|---|---|---|
| $p\le q$ | blow-up (datos grandes) | blow-up (datos grandes, umbral indep. de $h$) |
| $q<p<2q-1$ | blow-up (datos $>$ estacionaria maximal) | **todas globales**; aparece estacionaria espuria atractiva con $w_N\sim(2/(\lambda h))^{1/(p-q)}$ |
| $p=2q-1$, $\lambda<q$ | blow-up | todas globales ($p>q$) |
| $p>2q-1$ o ($p=2q-1$, $\lambda\ge q$) | todas globales | todas globales |

---

## 5. Experimentos numéricos / figuras / tablas

**El artículo no contiene ningún experimento numérico, figura ni tabla.** Es puramente teórico (6 páginas, la última son referencias). No hay valores concretos de $p,q,\lambda,N,h$ ni de datos iniciales.

### PROPUESTA (no está en el paper) de cálculos que ilustran los teoremas
Valores sugeridos: $q=2$, $p=2.5$ (cumple $q<p<2q-1=3$), $\lambda=1$; $N\in\{11,21,41,81,161\}$ ($h=1/(N-1)$). Para contraste: $p=1.5\le q$ y $p=4>2q-1$.

- **P1. Estacionaria espuria vs $h$.** Para cada $h$, hallar el cero de $G$ (3.2) por bisección en $w_1$ (preferiblemente en escala logarítmica), o Newton sobre (3.1) con semilla $w_N^{(0)}=(2/(\lambda h))^{1/(p-q)}$. Tabla: $h$, $w_1$, $w_N$, $c_h=w_N h^{1/(p-q)}$, y comparación con $(2/\lambda)^{1/(p-q)}$ (debería $c_h\to(2/\lambda)^{1/(p-q)}$ por debajo). Gráfico log-log de $w_N$ vs $h$ con pendiente $-1/(p-q)$. Perfil $w_k$ vs $x_k$ (creciente).
- **P2. Estabilidad.** Autovalores de $J=-M^{-1}A+B$ en la espuria; comprobar $\max\mathrm{Re}\,\lambda_i<0$ y $w_N>(2q/(p\lambda h))^{1/(p-q)}$.
- **P3. Dinámica $q<p<2q-1$.** Integrar (1.3) con dato grande (p.ej. $u_0\equiv K$ o $u_0(x)=K x^2$+cte) y ver convergencia a la espuria: $u_N(t)\to w_N$; $\Phi_h(U(t))$ decreciente. Comparar con el continuo (aprox. con $h$ muy fino hasta antes de $T$): la solución discreta crece hasta $\sim h^{-1/(p-q)}$ y se satura.
- **P4. Caso $p\le q$.** Blow-up semidiscreto: estimar $T_h$ para varios $h$ con dato fijo $\Phi(u_0)<0$; verificar la cota (2.2) cuando $u_N^{q-1}>2/h$.
- **P5. Caso $p>2q-1$.** Todas globales, convergencia a estacionaria acotada (independiente de $h$), contraste con P3.
- **P6. Convergencia (Teorema 1.2).** En intervalo $[0,T_1]$ antes del blow-up, error $L^\infty$ vs solución de referencia fina; orden observado ($\ge 3/2$; típicamente 2 para datos regulares).

---

## 6. Notas de implementación

- **RHS vectorizado** (con $u$ en array de longitud $N$, $h=1/(N-1)$):
  ```
  f = empty(N)
  f[0]    = 2/h**2*(u[1]-u[0])
  f[1:-1] = (u[2:]-2*u[1:-1]+u[:-2])/h**2
  f[-1]   = 2/h**2*(u[-2]-u[-1]) + 2/h*u[-1]**q
  f -= lam*u**p
  ```
  Usar `np.abs(u)**p` o recortar a $u\ge0$ si hay pequeñas negatividades numéricas (exponentes no enteros).
- **Rigidez:** autovalores de $M^{-1}A$ hasta $\sim4/h^2$; usar `solve_ivp(method='BDF' o 'Radau')` con Jacobiano analítico tridiagonal disperso: $J=-M^{-1}A+\mathrm{diag}(-p\lambda u^{p-1})+\frac{2q}{h}u_N^{q-1}e_Ne_N^T$.
- **Blow-up:** evento de parada `u[-1] > U_max` (p.ej. $10^{8}$–$10^{12}$) con `terminal=True`; o reescalar el tiempo (p.ej. $d\tau=u_N^{q-1}dt$) para aproximar $T_h$. La cota (2.2) da $T_h-t\le u_N^{1-q}/(q-1)$ una vez $u_N^{q-1}>2/h$.
- **Espectro real:** $J=-M^{-1}A+B$ es semejante a la simétrica $M^{-1/2}(-A+MB)M^{-1/2}$ (pues $M,B$ diagonales), así que se pueden usar `scipy.linalg.eigh` sobre esa forma simétrica; autovalores reales.
- **Shooting de $G$:** la recursión $F_k$ crece muy rápido; para $w_1$ grande puede desbordar (`inf`) — tratar desbordamiento como $G<0$ (el término $-\lambda F_N^p$ domina). $G$ sufre cancelación ($2(F_{N-1}-F_N)/h^2$ vs términos grandes); calcular $F_{N-1}-F_N$ acumulando las diferencias $d_k=w_k-w_{k-1}$ ($d_2=\frac{\lambda h^2}{2}w_1^p$, $d_k=d_{k-1}+\lambda h^2 w_{k-1}^p$) en lugar de restar. Usar el $F_2$ corregido ($\frac{\lambda h^2}{2}$, ver errata en §4). Pueden existir varios ceros de $G$ (p.ej. una estacionaria "acotada" con $w_N=O(1)$ y la espuria); localizar el de $w_N\approx(2/(\lambda h))^{1/(p-q)}$ escaneando $G$ en una malla logarítmica de $w_1$. Refinar con `brentq` y luego Newton sobre el sistema completo (3.1) con `scipy.optimize.root` para precisión.
- **Lyapunov:** $\Phi_h(U)=\tfrac12U^TAU+\frac{\lambda}{p+1}\sum m_{ii}u_i^{p+1}-\frac{u_N^{q+1}}{q+1}$ con $m_{11}=m_{NN}=h/2$, $m_{ii}=h$. Útil como chequeo (debe ser no creciente).
- **Datos iniciales:** $u_i(0)=u_0(x_i)$ (interpolación nodal). Para P4 elegir $u_0$ con $\Phi(u_0)<0$ (p.ej. constante $K$ grande: $\Phi=\lambda K^{p+1}/(p+1)-K^{q+1}/(q+1)<0$ si $p<q$ y $K$ grande).
- No hay discretización temporal en el paper: cualquier integrador es elección propia; documentar tolerancias (`rtol=1e-8`, `atol` pequeño) y comprobar independencia de ellas.
