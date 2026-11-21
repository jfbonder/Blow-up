# Notas: Dávila, Fernández Bonder, Groisman, Rossi, Sued — *Numerical analysis of stochastic differential equations with explosions*

> **Referencia publicada:** J. Dávila, J. Fernández Bonder, P. Groisman, J.D. Rossi and M. Sued. Numerical analysis of stochastic differential equations with explosions. *Stochastic Anal. Appl.*, 23 (2005), no. 4, 809–825.

Fuente: `papers/DFGRS_sde-explosions.pdf` (15 páginas, convertido de `DFGRS_sde-explosions.ps`, archivo DVI `estoca3-12.dvi`).
Objetivo de estas notas: poder reimplementar en Python (NumPy/SciPy) todos los cálculos numéricos del paper sin volver al PDF.

---

## 1. Referencia bibliográfica

- **Autores:** Juan Dávila, Julián Fernández Bonder, Pablo Groisman, Julio D. Rossi, Mariela Sued.
- **Título:** *Numerical analysis of stochastic differential equations with explosions*.
- **Revista / volumen / páginas:** **no figuran en el PDF**. Es un preprint; el PostScript tiene `%%CreationDate: Tue Dec 14 17:32:02 2004` (TeX output 2004.12.14). No se indica revista, año de publicación ni páginas (no se adivinan aquí).
- **MSC 2000:** 65C30, 65L20, 60H10. Palabras clave: Explosion, stochastic differential equations, numerical approximations.
- **Financiación:** UBA TX066, ANPCyT PICT 03-05009, Fundación Antorchas 13900-5, CONICET; JFB también ANPCyT PICT 03-10608.
- **Afiliaciones:** J. Dávila (Depto. Ingeniería Matemática, U. de Chile, CMM); J. Fernández Bonder y J.D. Rossi (Depto. Matemática, FCEyN, UBA); P. Groisman y M. Sued (Instituto de Cálculo, FCEyN, UBA).
- Referencia clave para el paso adaptativo: [1] G. Acosta, R. Durán, J.D. Rossi, *An adaptive time step procedure for a parabolic problem with blow-up*, Computing 68(4):343–373, 2002. Convergencia EM estándar: [5] Kloeden–Platen (1992); [3] Higham–Mao–Stuart, SIAM J. Numer. Anal. 40 (2002).

---

## 2. La EDE

$$(P)\qquad dx = b(x)\,dt + \sigma(x)\,dW,\qquad x(0)=z\in\mathbb{R}_{>0},$$

- $b,\sigma$ funciones suaves **positivas** ($C^1$, o localmente Lipschitz basta); $W$ proceso de Wiener unidimensional en $(\Omega,\mathcal F,\mathbb P)$ con filtración $\{\mathcal F_t\}_{t\ge0}$ con condiciones usuales.
- **Test de Feller** (heurística del intro): si $b(s)\sim s^p$, $\sigma(s)\sim s^q$ ($s\to\infty$), las soluciones explotan con probabilidad uno si $p>2q\vee 1$. Notación: $f\sim g$ significa que existen $0<c<C$ con $cg(s)\le f(s)\le Cg(s)$ para $s$ grande; $a\vee b=\max$, $a\wedge b=\min$.
- **Hipótesis (A)** (usada en Thms 1.3, 1.4): existen constantes positivas $\kappa_1,\kappa_2$ tales que
  $$\kappa_1\le\sigma^2(s)\le\kappa_2\,b(s),\qquad b \text{ no decreciente},\qquad \int_0^\infty\frac{ds}{b(s)}<\infty.$$
  (Remark 1.1: bajo (A) las soluciones de $(P)$ explotan con probabilidad uno, por Feller; (A) es más fuerte que lo que pide Feller. La EDO $\bar x'=b(\bar x)$, $\bar x(0)=z>0$, explota en tiempo finito sii $\int_z^\infty ds/b(s)<\infty$.)
- **Sección 2 (ecuación continua, resultados conocidos):** función de escala $s(z)=0$, $s'(\xi)=\exp\!\big[-\int_0^\xi 2b(t)\sigma(t)^{-2}\,dt\big]$. Con $y=s(x)$: $dy=\tilde\sigma(y)\,dW$, $\tilde\sigma=(s'\sigma)\circ s^{-1}$ (2.1). $x$ explota en tiempo finito sii $\ell:=s(+\infty)<\infty$. Cambio de tiempo: $A(t)=\int_0^t\tilde\sigma(B(u))^{-2}du$, $\gamma=A^{-1}$, $y(t)=B(\gamma(t))$ solución débil. Con $T_\ell=\inf\{t>0:B(t)=\ell\}$,
  $$T=A(T_\ell)=\int_0^{T_\ell}\tilde\sigma^{-2}(B(u))\,du\qquad(2.2)$$
  es el tiempo de explosión. $R(t):=\ell-B(T_\ell-t)$ es un BES(3) y $x(T-\varepsilon)=s^{-1}(\ell-R(T_\ell-\gamma(T-\varepsilon)))$. (No se usa numéricamente en el paper.)

**Parámetros concretos de los experimentos (Sección 6):**
$$b(\xi)=|\xi|^{1.1}+0.1,\qquad \sigma(\xi)=\sqrt{|\xi|^{1.1}+0.1},\qquad z=10.$$
Es decir $\sigma^2=b$, $p=1.1$, $q=0.55$ (caso límite $p=2q$ del test heurístico, pero (A) se cumple en $\xi\ge0$ con $\kappa_1=0.1$, $\kappa_2=1$; $\int_0^\infty d\xi/b<\infty$ porque $1.1>1$). Nota: $b$ es no decreciente sólo en $\xi\ge0$ (por el $|\xi|$); el paper no comenta esto.

---

## 3. Esquema numérico (exacto)

Parámetro $h>0$. Euler–Maruyama con paso adaptativo:
$$(P_h)\qquad X_{k+1}=X_k+\tau_k\,b(X_k)+\sigma(X_k)\,\Delta W_k,\qquad X_0=x(0)=z,$$
$$\Delta W_k=W_{t_{k+1}}-W_{t_k},\qquad \tau_k=t_{k+1}-t_k=\frac{h}{b(X_k)}.$$

- Por tanto $\tau_k b(X_k)=h$: **el incremento de deriva es exactamente $h$ en cada paso**, $X_{k+1}=X_k+h+\sigma(X_k)\Delta W_k$, y $X_k=z+hk+Y_{k-1}$ con $Y_k=\sum_{j=1}^k\sigma(X_j)\Delta W_j$ (ec. 4.2; ojo con los índices, ver §6).
- **Incrementos brownianos:** $\Delta W_k=\sqrt{\tau_k}\,Z_k$, $Z_k\sim N(0,1)$ i.i.d., independiente de $\mathcal F_{t_k}$ (Lemma 4.1: como $\tau_k$ es $\mathcal F_{t_k}$-medible y $t_k$ tiempo de parada, $(W(t_k+\tau_k)-W(t_k))/\sqrt{\tau_k}$ es normal estándar). En el paper se generaron con `randn` de MATLAB. Entonces $\sigma(X_k)\Delta W_k=\sqrt h\,\frac{\sigma(X_k)}{\sqrt{b(X_k)}}Z_k$; en los experimentos ($\sigma^2=b$) esto es simplemente $\sqrt h\,Z_k$:
  $$X_{k+1}=X_k+h+\sqrt h\,Z_k\quad(\text{caso de la Sección 6}),\qquad t_{k+1}=t_k+\frac{h}{|X_k|^{1.1}+0.1}.$$
- **Interpolante:** $X(t)$ cualquier interpolante de $X(t_k)=X_k$ (constante a trozos $X(t)\equiv X_k$ en $[t_k,t_{k+1})$, o lineal).
- **Tiempo de explosión numérico:** $T_h:=\sum_{k\ge1}\tau_k$ (así escrito en el paper). Un camino de $(P_h)$ explota en tiempo finito si $X(t,\omega)\to+\infty$ cuando $t\to T_h$ y $T_h(\omega)<\infty$. Con esta elección de $\tau_k$, $T_h<\infty$ implica $X(t)\nearrow+\infty$.
- **Tiempos de parada usados en la práctica:** $R^M:=\inf\{t:|x(t)|\ge M\}$ (continuo), $R_h^M:=\inf\{t:|X(t)|\ge M\}$ (numérico). En la práctica el algoritmo **se detiene cuando la solución calculada alcanza $M$**, y $R_h^M$ (el $t_k$ en ese momento) es la aproximación de $T$ (Thm 1.4, Secc. 5: "$R_h^M$ can be easily computed").
- **Umbral reportado:** $M=10^5$ (Figura 1). Para Figura 3 no se dice explícitamente (ver §5).
- Lemma 3.1: los $t_k$ son tiempos de parada y cada $\tau_k$ es $\mathcal F_{t_k}$-medible ("el caso $k=0$ vale pues $t_0$ es la constante $h/b(z)$" — literal del paper; parece una errata por $\tau_0$).
- Thm 1.1 / Prop 3.2 / Cor 1.2 valen para cualesquiera $\tau_k$ $\mathcal F_{t_k}$-medibles con $\tau_k\le Ch$.

---

## 4. Resultados principales

**Theorem 1.1 (convergencia del esquema).** Sea $x(\cdot)$ solución de $(P)$ y $X(\cdot)$ su aproximación EM $(P_h)$. Fijar $\bar\tau>0$ y $M>0$. Sean $\tau:=\bar\tau\wedge R^M$, $\tau_h:=\bar\tau\wedge R_h^{2M}$, con $R^M=\inf\{t:|x(t)|\ge M\}$, $R_h^M=\inf\{t:|X(t)|\ge M\}$. Entonces
$$\lim_{h\to0}\mathbb E\Big[\sup_{0\le t\le\tau\wedge\tau_h}|x(t)-X(t)|^2\Big]=0.$$
Remark 3.1: de hecho $\mathbb E[\sup_{0\le t\le\tau\wedge\tau_h}|x-X|^2]\le Ch$ (vía [5], truncando $b,\sigma$ fuera de $|x|\le2M$).

**Corollary 1.2.** Mismas hipótesis: para todo $\varepsilon>0$ y $0\le\alpha<\tfrac12$,
$$\mathbb P\Big(\sup_{0<t<R^M}|x(t)-X(t)|>\varepsilon h^\alpha\Big)\to0\quad(h\to0).$$

**Proposition 3.2.** Para $M>0$: (1) $\mathbb P(R^M\ge R_h^{2M})\to0$; (2) $\mathbb P(R_h^M\ge R^{2M})\to0$, cuando $h\to0$.

**Lemma 4.2.** $Y_k=\sum_{j=1}^k\sigma(X_j)\Delta W_j$ cumple $Y_k/k\to0$ c.s. (prueba: $\mathbb E[Y_k^4]\le3(\kappa_2hk)^2$, Chebyshev + Borel–Cantelli).

**Definition 4.3.** $f$ tiene variación regular en infinito si existe $p>0$ con $\lim_{s\to\infty}f(\alpha s)/f(s)=\alpha^p$ para todo $\alpha>0$.

**Theorem 1.3 (resultado principal).** Bajo (A):
1. Para todo dato $z>0$, $X(\cdot)$ explota en tiempo finito con probabilidad uno.
2. $\displaystyle\lim_{k\to\infty}\frac{X(t_k)}{hk}=1$ c.s. Además, para todo $\alpha>1$ existe $k_0=k_0(\omega)$ tal que para $k\ge k_0$
   $$\sum_{j=k}^\infty\frac{h}{b(\alpha hj)}\le T_h-t_k\le\sum_{j=k}^\infty\frac{h}{b(\alpha^{-1}hj)}.$$
3. Si $b$ tiene variación regular en infinito,
   $$\lim_{k\to\infty}\frac{T_h-t_k}{\int_{X_k}^\infty\frac{ds}{b(s)}}=1\quad\text{c.s.}$$
4. Para todo $h>0$, $h/b(z)\le T_h<+\infty$ c.s., y para todo $L>0$, $\mathbb P(T_h>L)>0$.

**Remark 1.2.** Si $b(s)\sim s^p$, la tasa de (3) es $X(t_k)(T_h-t_k)^{1/(p-1)}\to\big(\tfrac1{p-1}\big)^{1/(p-1)}$ ($t_k\to T_h$), como la EDO determinista $d\bar x=\bar x^p\,dt$.

**Theorem 1.4.** Bajo (A), para todo $\varepsilon>0$,
$$\lim_{M\to+\infty}\lim_{h\to0}\mathbb P(|R_h^M-T|>\varepsilon)=0.$$
(Base para aproximar el tiempo de explosión $T$ de $(P)$ por $R_h^M$.) No se dan tasas de convergencia para los tiempos de explosión.

---

## 5. Experimentos numéricos (Sección 6)

Común a todos: $b(\xi)=|\xi|^{1.1}+0.1$, $\sigma(\xi)=\sqrt{|\xi|^{1.1}+0.1}$, $z=10$; incrementos con `randn` (MATLAB). Esquema $(P_h)$ con $\tau_k=h/b(X_k)$. No hay tablas en el paper; sólo 3 figuras (la 1 con dos recuadros ampliados 1.1 y 1.2).

### Figura 1 — "Three sample paths with explosions"
- Qué: 3 trayectorias $X(t)$ vs $t$ (colores verde, azul, rojo). Se detiene el algoritmo cuando la solución calculada alcanza $M=10^5$.
- Ejes: $t$ de ≈1.75 a 5 (ticks 2, 2.5, …, 5); $X(t)$ de 0 a $9\times10^4$ (etiqueta "x $10^4$"). Las curvas, leídas de la figura, terminan visualmente cerca de $X\approx8\times10^4$ en $t\approx4.4$ (verde), $\approx4.6$ (azul), $\approx4.9$ (rojo). (Que terminen en $\approx8\times10^4$ y no en $10^5$ es lo que se ve; no se explica.)
- Recuadros: "Figure 1.1" = zoom de la esquina inferior izquierda (aprox. $t\in[1.75,1.93]$, $X\in[0,\,0.45\times10^4]$, las tres trayectorias); "Figure 1.2" = zoom de la trayectoria azul aprox. $t\in[4.0,4.2]$, $X\in[1.3,2]\times10^4$. Los zooms (p. 13) no tienen ejes; muestran la rugosidad browniana de las trayectorias.
- **No especificado:** valor de $h$, semilla, interpolación usada para graficar, por qué el eje $t$ empieza en ≈1.75.

### Figura 2 — "$X_k/hk\to1$ a.s."
- Qué: el cociente $X_k/(hk)$ vs $k$ para varias trayectorias (se ven 3: azul, roja, verde), con línea discontinua horizontal en 1. Ilustra Thm 1.3(2).
- Ejes: $k$ de 0 a $4\times10^5$ (único rótulo "4 x$10^5$" en el extremo derecho); eje vertical sólo marca "1". Al inicio los cocientes son grandes (pico ≫1, pues $X_k/(hk)\approx z/(hk)+1$), y para $k\gtrsim10^5$ quedan pegados a 1 con desviaciones de pocos %.
- **No especificado:** $h$, $M$/criterio de parada, número exacto de trayectorias, semilla. (Inferencia, no del paper: si se para en $M=10^5$ y $X_k\approx hk$, llegar a $k\approx4\times10^5$ sugiere $h\approx0.25$; alternativamente el eje puede estar truncado.)

### Figura 3 — "The kernel density estimator of $R_h^M$ for different values of $h$"
- Qué: estimación por núcleo de la densidad de $R_h^M$ para tres valores de $h$; ilustra Thm 1.4 ($R_h^M\to T$ en probabilidad).
- **1000 trayectorias por estimador.**
- Valores de $h$ según el texto: $h=1$, $h=0.5$, $h=0.1$. El texto dice que el mayor paso fue $\tau_1\simeq0.08$, $0.04$, $0.008$ respectivamente (coherente: $\tau=h/b(10)$, $b(10)=10^{1.1}+0.1\approx12.689$, da $0.0788,\,0.0394,\,0.00788$).
  - **Inconsistencia:** la leyenda de la figura dice `h=1` (azul sólido), `h=0.5` (verde discontinuo `--`), `h=0.01` (rojo `-.`). El texto y los $\tau_1$ apoyan $h=0.1$.
- Ejes: abscisa de 3.2 a 5.2 (ticks cada 0.2), ordenada de 0 a 2 (ticks cada 0.2). Las tres densidades: soporte aprox. 3.4–4.9, unimodales, pico ≈1.85–1.9 en $t\approx3.95$–4.0, algo de cola a la derecha (hombro en ≈4.3–4.5 en la curva $h=1$). Las curvas casi coinciden; la de $h$ menor está levemente a la izquierda.
- **No especificado:** $M$ (el texto sólo da $M=10^5$ para la Figura 1), núcleo y ancho de banda (presumiblemente `ksdensity` de MATLAB, no se dice), semilla, si se usan los mismos números aleatorios para los tres $h$.

### Comprobación propia (no del paper; simulación rápida en NumPy, semilla 0, `default_rng`)
Para orientar la reimplementación:

| $M$ | $h$ | caminos | media $R_h^M$ | desv. est. | pasos medios |
|---|---|---|---|---|---|
| $10^5$ | 1 | 1000 | 4.86 | 0.24 | ≈$10^5$ |
| $10^5$ | 0.5 | 1000 | 4.84 | 0.24 | ≈$2\times10^5$ |
| $10^5$ | 0.1 | 300 | 4.81 | 0.23 | ≈$10^6$ |
| $10^4$ | 1 | 1000 | 4.04 | 0.24 | ≈$10^4$ |
| $10^4$ | 0.5 | 1000 | 4.02 | 0.24 | ≈$2\times10^4$ |
| $10^4$ | 0.1 | 1000 | 4.00 | 0.25 | ≈$10^5$ |

- Con $M=10^5$ los tiempos de parada (≈4.4–5.5) son coherentes con la Figura 1. **La Figura 3 (pico en ≈4.0, soporte 3.4–4.9) se reproduce con $M=10^4$, no con $10^5$** — hipótesis de reconstrucción, el paper no lo dice. Con $M=10^4$ y $h=1$ hay ≈$10^4$ pasos; con $M=10^5$, $h=0.25$ hay ≈$4\times10^5$ pasos, compatible con el eje de la Figura 2.
- La cola determinista $\int_M^\infty ds/b(s)\approx 10\,M^{-0.1}$ vale ≈3.98 para $M=10^4$ y ≈3.16 para $M=10^5$: $R_h^M$ está lejos de $T$ para estos $M$ (la convergencia en $M$ es lentísima porque $p=1.1$). Para la EDO determinista desde $z=10$: $\int_{10}^\infty ds/b\approx7.94$.

---

## 6. Notas de implementación

- **Paso:** `tau = h / b(X)`; `X += h + sigma(X)*sqrt(tau)*randn()` (con $\sigma^2=b$, `X += h + sqrt(h)*randn()`); `t += tau`; parar cuando `abs(X) >= M`; devolver `t` como $R_h^M$. Se puede usar $t$ antes o después del último paso (el $\inf$ sobre la interpolante constante a trozos da el $t_{k}$ en que $X_k\ge M$ por primera vez).
- **Índices:** el paper es inconsistente: $T_h=\sum_{k\ge1}\tau_k$, $X_0=z$, "el mayor paso $\tau_1\simeq h/b(z)$", Thm 1.3(4) "$T_h\ge\tau_1=h/b(z)$", Lemma 3.1 "$t_0$ es la constante $h/b(z)$". Interpretación práctica: $t_0=0$, el primer paso es $h/b(z)$, $T_h=$ suma de todos los pasos.
- "Mayor paso $=\tau_1$" es aproximado: $\tau_k$ es máximo cuando $X_k$ es mínimo, y $X_k$ puede bajar de $z$ al principio.
- $X$ podría volverse negativa en teoría (el $|\xi|$ en $b,\sigma$ mantiene todo bien definido); con $z=10$, $h\le1$ es prácticamente imposible.
- **Monte Carlo:** vectorizar sobre trayectorias con máscara de activas (cada camino necesita ≈$(M-z)/h$ pasos, porque la deriva avanza exactamente $h$ por paso). $h=0.1$, $M=10^5$, 1000 caminos ⇒ ≈$10^9$ actualizaciones: costoso; con $M=10^4$ es trivial.
- **KDE:** usar `scipy.stats.gaussian_kde` (núcleo gaussiano) como sustituto de `ksdensity`; ancho de banda no especificado en el paper.
- **Semillas:** el paper no da semillas (MATLAB `randn`, generador de la época); los resultados sólo son reproducibles estadísticamente.
- Para Figura 2 guardar todo el vector $X_k$ de pocas trayectorias y graficar $X_k/(hk)$, $k\ge1$.
- Para Figura 1 graficar $(t_k,X_k)$ de 3 trayectorias hasta $M=10^5$.

## 7. No especificado (resumen)
- Revista/volumen/páginas (preprint, dic. 2004).
- $h$ de Figuras 1 y 2; número de caminos de Figura 2 (se ven 3).
- $M$ de Figuras 2 y 3 (sólo $M=10^5$ explícito para Figura 1).
- Tipo de núcleo y ancho de banda del KDE.
- Semillas / estado del generador; si se comparten números aleatorios entre distintos $h$.
- Interpolación usada en las gráficas.
- Conflicto $h=0.1$ (texto) vs $h=0.01$ (leyenda de la Figura 3).
