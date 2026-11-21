# Notas: Fernández Bonder & Groisman — *Time–space white noise eliminates global solutions in reaction diffusion equations*

Objetivo de estas notas: poder reimplementar (Python/NumPy/SciPy) todos los cálculos numéricos del paper sin volver a abrir el PDF. Lo que dice el paper se marca como tal; lo que es interpretación o sugerencia propia se marca **[interpretación]** o **[sugerencia]**.

---

## 1. Referencia bibliográfica

- **Autores:** Julián Fernández Bonder y Pablo Groisman (Departamento de Matemática, FCEyN, Universidad de Buenos Aires; ambos miembros de CONICET).
- **Título:** *Time–space white noise eliminates global solutions in reaction diffusion equations*. (Título corrido de las páginas: "Reaction diffusion equations with additive noise".)
- **Revista, año, volumen, páginas:** **No figuran en el PDF.** El archivo es un preprint (15 páginas, generado con TeX el 2008-08-27). No se indica revista ni volumen; no se adivina aquí.
- Financiación citada: UBA grant TX066, ANPCyT PICT No. 03-05009, CONICET.
- MSC 2000: 60H15, 35R60, 35B60. Palabras clave: Explosion, Stochastic Partial Differential Equations, Reaction–Diffusion Equations.
- Nota: la sección "Organization of the paper" menciona "Section 5" para simulaciones; efectivamente §4 = aproximaciones numéricas (teoría), §5 = experimentos numéricos.

## 2. El problema (SPDE)

$$u_t = u_{xx} + f(u) + \sigma \dot W(x,t), \qquad x\in(0,1),\ t>0 \tag{1.1}$$

- $W$: lámina browniana 2-dimensional (space-time white noise $\dot W$ en $\mathbb R_+\times[0,1]$); ruido **aditivo**.
- $\sigma>0$ parámetro (intensidad del ruido).
- Condiciones de borde: **Dirichlet homogéneas**, $u(0,t)=u(1,t)=0$.
- Dato inicial: $u_0\in C_0([0,1])$ (en los teoremas de explosión: $u_0\ge 0$).
- Hipótesis sobre $f$ (§3): $f$ no negativa, convexa (por tanto localmente Lipschitz) y $\int^\infty \frac{1}{f}<\infty$. Caso modelo del abstract: $f(u)=(u_+)^p$, $p>1$.
- **En todos los experimentos (§5):** $f(u)=(u_+)^2$, $\sigma=6.36$, $u_0\equiv 0$.

### Formulación débil (2.1)
Para $\varphi\in C^2((0,1))\cap C_0([0,1])$:
$$\int_0^1 u(x,t)\varphi\,dx-\int_0^1 u_0\varphi\,dx=\int_0^t\!\!\int_0^1 u\,\varphi_{xx}\,dx\,ds+\int_0^t\!\!\int_0^1 f(u)\varphi\,dx\,ds+\sigma\int_0^t\!\!\int_0^1\varphi(x)\,dW(x,s).$$
Formulación integral equivalente con el núcleo del calor $G_t(x,y)$ de $(0,1)$:
$u(x,t)-\int_0^1 G_t(x,y)u_0(y)dy=\int_0^t\!\int_0^1 G_{t-s}(x,y)f(u(y,s))dyds+\sigma\int_0^t\!\int_0^1 G_{t-s}(x,y)dW(y,s)$.

Existencia local para $f$ localmente Lipschitz: truncar $f_n(x)=f(-n)\mathbf 1_{(-\infty,-n]}+f(x)\mathbf 1_{(-n,n)}+f(n)\mathbf 1_{[n,\infty)}$, $u^n$ solución global, $T_n$ = primer tiempo en que $\|u^n(\cdot,t)\|_\infty=n$, $T:=\lim T_n$ (tiempo de explosión). "Blow-up en tiempo finito" significa $\mathbb P(T<\infty)>0$; si $T(\omega)<\infty$ entonces $\lim_{t\nearrow T}\|u(\cdot,t,\omega)\|_\infty=\infty$.

### Solución estacionaria determinista
Para $\sigma=0$, existe una solución estacionaria positiva $v$: $v_{xx}=-f(v)$ en $(0,1)$, $v(0)=v(1)=0$. El paper **solo dice** que, para $f(v)=(v_+)^2$, es "de tamaño 12"; **no da fórmula**.

**[interpretación/cálculo propio]** Con $f(v)=v^2$: integral de energía $\tfrac12 v'^2+\tfrac13 v^3=\tfrac13 M^3$, $M=\max v=v(1/2)$. Imponer $\int_0^M \frac{dv}{\sqrt{\frac23(M^3-v^3)}}=\frac12$ da
$$M = 6\,I^2,\qquad I=\int_0^1\frac{ds}{\sqrt{1-s^3}}=\tfrac13 B\!\left(\tfrac13,\tfrac12\right)\approx 1.402182,$$
$$M\approx 11.7967\ (\approx 12,\ \text{coincide con el paper}).$$
Forma implícita: $x(v)=\sqrt{\tfrac{3}{2}}\int_0^{v}\frac{dw}{\sqrt{M^3-w^3}}$ para $x\in[0,1/2]$, simétrica respecto de $1/2$. Numéricamente: resolver el BVP con `scipy.integrate.solve_bvp` o shooting.

Contexto determinista ($\sigma=0$, ref. [4] Cortázar–Elgueta): para cada $u_0$ existe $\lambda^*$ tal que con dato $\lambda u_0$: $\lambda<\lambda^*$ ⇒ $u\to0$ uniformemente; $\lambda>\lambda^*$ ⇒ blow-up en tiempo finito; $\lambda=\lambda^*$ ⇒ converge a la única estacionaria positiva. Por comparación, si $u_0\le v$ la solución determinista es global.

## 3. Esquema numérico (lo que dice el paper)

### 3.1 Semidiscretización en espacio (§4, ec. (4.1))
Diferencias finitas de segundo orden en malla uniforme de paso $h=1/n$, $x_i=i/n$:
$$du_i=\frac{1}{h^2}(u_{i+1}-2u_i+u_{i-1})\,dt+f(u_i)\,dt+\frac{\sigma}{\sqrt h}\,dw_i,\qquad 2\le i\le n-1, \tag{4.1}$$
con $u_1(t)=u_n(t)=0$, $u_i(0)=u_0(ih)$, $1\le i\le n$.
- §4: "Los movimientos brownianos $w_i$ se obtienen por integración espacial de la lámina browniana en el intervalo $[ih,(i+1)h)$."
- §5: mismo sistema pero "en el intervalo $[(i-\tfrac12)h,(i+\tfrac12)h)$".
- Forma vectorial: $dU=(-AU+f(U))dt+\frac{\sigma}{\sqrt h}dW$, $U(0)=U^0$, $-A$ laplaciano discreto, $f$ componente a componente, $(U^0)_i=u_0(ih)$.
- $u^n(t,\cdot)$ se extiende a $[0,1]$ por interpolación lineal en $x$.

**Inconsistencias de indexación del paper (anotar):** con $h=1/n$ y nodos $x_i=ih$, $i=1..n$, imponer $u_1=u_n=0$ pondría el borde izquierdo en $x=h$. En §5: "$h=0.02$ (i.e.: $n=50$ nodes)". Las figuras muestran perfiles en $[0,1]$ con valor 0 en ambos extremos y ~50 segmentos.
**[sugerencia]** Usar $x_k=kh$, $k=0,\dots,N$, $N=1/h=50$, $u_0=u_N=0$, $N-1=49$ incógnitas interiores.

**Escala del ruido — ambigüedad:** El paper escribe $\frac{\sigma}{\sqrt h}dw_i$. Si $w_i$ fuese literalmente la integral de la lámina sobre una celda de longitud $h$, tendría varianza $h\,t$ y el término valdría $\sigma\,d\beta_i$ ($\beta_i$ BM estándar), lo cual NO es la discretización consistente del ruido blanco. En la prueba del Teorema 4.1 se usa que $\sum_i\sqrt h\,\phi_i^n\frac{\sigma}{\sqrt h}dw_i=\sigma\,dB$ con $B$ estándar y $\|\phi^n\|_2=1$, lo cual exige que los $w_i$ sean **BM estándar independientes**. **[interpretación]** Por tanto $w_i = h^{-1/2}\,W(\text{celda}_i\times[0,t])$ son BM estándar independientes y el incremento de ruido por paso es
$$\frac{\sigma}{\sqrt h}\,\Delta w_i=\sigma\sqrt{\frac{\tau_j}{h}}\;\xi_i^j,\qquad \xi_i^j\sim N(0,1)\ \text{i.i.d.},$$
equivalente a discretizar $\dot W$ en la celda espacio-tiempo como $\xi/\sqrt{h\,\tau_j}$ (escala $\sqrt{1/(h\,dt)}$). El paper **no** escribe explícitamente la escala $\sqrt{1/(h\,dt)}$; ésta es la lectura coherente con Gyöngy [10] y con la prueba del Teo. 4.1.

### 3.2 Discretización temporal (§5): Euler explícito adaptativo
"Usamos un procedimiento adaptativo similar al de [5] (Dávila–Fernández Bonder–Rossi–Groisman–Sued, caso unidimensional), reemplazando el valor de la solución (número real) por la norma $L^1$ de $u^j$, como se hace en [9] (Groisman, caso determinista)". Esquema totalmente discreto:
$$u_i^{j+1}-u_i^j=\frac{\tau_j}{h^2}(u_{i+1}^j-2u_i^j+u_{i-1}^j)+\tau_j f(u_i^j)+\frac{\sigma}{\sqrt h}\big(w_i(t^{j+1})-w_i(t^j)\big),$$
con $u_1^j=u_n^j=0$ para todo $j\ge1$, $u_i^0=u_0(ih)$, $1\le i\le n$, y
$$t_0=0,\qquad \tau_j=\frac{\tau}{\sum_i h\,u_i^j},\qquad t_{j+1}-t_j=\tau_j,$$
$\tau$ = parámetro de discretización temporal; los $w_i$ son los del esquema semidiscreto.
- (Para comparación: en [5] la regla es $\tau_k=h/b(X_k)$, $b$ = drift.)
- El paper subraya: "la adaptatividad en tiempo es esencial, ya que un paso fijo produce aproximaciones globalmente definidas".
- Adaptatividad en espacio: no usada; para $\sigma=0$ se sabe que no hace falta para capturar el tiempo máximo de existencia, pero sí para la forma de la solución cerca de la singularidad (refs [3,7,8,9]).
- **Criterio de explosión:** "detenemos la simulación cuando el máximo de la solución alcanza el valor $10^{13}$" (experimento estadístico, Fig. 4). Tiempo de explosión numérico = $t_j$ en ese momento.
- El paper afirma que **no está probada** la convergencia del esquema totalmente discreto ni de los tiempos de explosión ("tema de un paper futuro").

### 3.3 No especificado (esquema)
- Valor de $\tau$.
- Qué hacer cuando $\sum_i h u_i^j\le 0$ o $\approx0$ (ocurre en $t=0$ con $u_0\equiv0$ ⇒ división por cero; y la solución toma valores negativos, ver Fig. 1, $t=50$, donde el mínimo ≈ −3.7). El texto dice "norma $L^1$" pero la fórmula escribe $\sum_i h u_i^j$ sin valor absoluto.
- Restricción de estabilidad del Euler explícito ($\tau_j\le h^2/2=2\times10^{-4}$ para $h=0.02$).
- Semilla / generador aleatorio; tiempo final máximo para las corridas de Fig. 4 (si alguna no explotó).
- Ancho de banda / núcleo del estimador de densidad de Fig. 4.
- Software: el usuario indica MATLAB (las figuras tienen estilo MATLAB); el paper no lo dice.

## 4. Resultados teóricos

**Teorema 3.2.** Sea $f$ no negativa, convexa, con $\int^\infty 1/f<\infty$. Entonces, para todo dato inicial $u_0\ge0$, la solución $u$ de (1.1) explota en tiempo finito (aleatorio) $T$ con $\mathbb P^{u_0}(T<\infty)=1$. (En §3 se enuncia para todo $u_0\in C_0([0,1])$.)

Idea de prueba (relevante para numérica: reducción a una EDE escalar):
- $\phi(x)=\frac{\pi}{2}\sin(\pi x)$: primera autofunción del Laplaciano Dirichlet en $(0,1)$ normalizada en $L^1$ ($\int\phi=1$), autovalor $\lambda_1=\pi^2$.
- $\Phi(t):=\int_0^1\phi(x)u(x,t)dx$, $z_0=\Phi(0)$.
- Jensen (convexidad): $\int\phi f(u)\ge f(\Phi)$.
- $B(t):=\frac{\sqrt8}{\pi}\int_0^t\!\int_0^1\phi\,dW$ es BM estándar (pues $\|\phi\|_{L^2}^2=\pi^2/8$).
- Desigualdad estocástica: $d\Phi\ge(-\lambda_1\Phi+f(\Phi))dt+\frac{\pi}{\sqrt8}\sigma\,dB$.
- Comparación con $dz=(-\lambda_1 z+f(z))dt+\sigma dB$, $z(0)=z_0$: $e=\Phi-z$ cumple $de\ge(-\lambda_1e+\frac{f(\Phi)-f(z)}{\Phi-z}e)dt$, $e(0)=0$ ⇒ $\Phi\ge z$. (Nota: en la desigualdad de $\Phi$ el coeficiente del ruido es $\frac{\pi}{\sqrt8}\sigma$, en la EDE de $z$ se escribe $\sigma$; el paper no comenta la diferencia — irrelevante para el resultado cualitativo.)

**Lema 3.1.** La solución de $dz=(-\lambda_1z+f(z))dt+\sigma dB$, $z(0)=0$ (3.1), explota en tiempo finito con probabilidad uno. Prueba: test de Feller (Karatzas–Shreve cap. 5). Función de escala $p(x)=\int_0^x\exp\!\big(-\frac{2}{\sigma^2}\int_0^s b(\xi)d\xi\big)ds$, $b(\xi)=-\lambda_1\xi+f(\xi)$; $p(-\infty)=-\infty$, $p(+\infty)<\infty$ ⇒ $\mathbb P(\lim_{t\nearrow S}z=+\infty)=1$; y $v(x)=2\int_0^x\frac{p(x)-p(y)}{\sigma^2p'(y)}dy$ [en el PDF aparece $\sigma^2 p(y)$ en el denominador; en Karatzas–Shreve es $p'(y)$] con $v(+\infty)<\infty$ (comportamiento dado por $1/f$) ⇒ $\mathbb P(S<\infty)=1$.

**Teorema 4.1 (explosión de las aproximaciones finito-dimensionales).** Misma $f$. Para todo $U^0\ge0$ la solución $U$ de (4.1) explota en tiempo finito aleatorio $T^n$ con $\mathbb P^{U^0}(T^n<\infty)=1$. Prueba: $A$ simétrica definida positiva, autovalores $0<\lambda_1^n\le\dots\le\lambda_n^n$, $\phi^n$ autovector de $\lambda_1^n$ con $\|\phi^n\|_2=1$, $\Phi^n(t)=\sum_i\sqrt h\,\phi_i^n u_i(t)$; $d\Phi^n\ge(-\lambda_1^n\Phi^n+f(\Phi^n))dt+\sigma dB$ y se aplica el Lema 3.1.
**[sugerencia]** Para verificar numéricamente: $\lambda_1^n=\frac{4}{h^2}\sin^2(\pi h/2)$, $\phi^n_i\propto\sin(\pi x_i)$.

**Teorema 4.2 (Gyöngy [10], Teo. 3.1).** $f$ globalmente Lipschitz, $u_0\in C^3([0,1])$. (1) Para todo $p\ge1$, $T>0$ existe $K=K(p,T)$ con $\sup_{0\le t\le T}\sup_{x}\mathbb E|u^n(t,x)-u(t,x)|^{2p}\le K/n^p$. (2) $u^n\to u$ uniformemente en $[0,T]\times[0,1]$ c.s.

**Teorema 4.3 (convergencia local para $f$ localmente Lipschitz).** $f$ no negativa convexa con $\int 1/f<\infty$. Sean $R_M=\inf\{t>0:\|u(\cdot,t)\|_\infty\ge M\}$, $R_M^n=\inf\{t>0:\|u^n(\cdot,t)\|_\infty\ge M\}$. (1) $\sup_{t\le T}\sup_x\mathbb E\big(|u^n-u|^{2p}\mathbf 1_{\{t\le R_M\wedge R_M^n\}}\big)\le K/n^p$. (2) Para todo $M$, $\|u^n-u\|_{L^\infty([0,T\wedge R_M]\times[0,1])}\to0$ c.s. Prueba: truncar $f_M$ ($=f(s)$ si $|s|\le M$, $f(M)$ si $s\ge M$, $f(-M)$ si $s\le -M$), aplicar Gyöngy y unicidad hasta $R_M\wedge R_M^n$; $\liminf R_M^n\ge R_M$.
Observación del paper: los tiempos de explosión de $u$ y $u^n$ difieren en general, así que no se espera un resultado mejor.

**Remark 4.2:** la discretización temporal de (4.1) puede hacerse como en [5].

### Heurística de metaestabilidad (§1 y §5)
- Para $\sigma$ pequeño y dato pequeño: la solución permanece en la cuenca de atracción de $0$ del problema determinista; explota cuando el ruido la empuja por encima de la estacionaria ($\approx12$) "más el orden del ruido $\sigma$".
- Probabilidad de tal evento en un intervalo fijo: $p_\sigma\sim\exp(-1/\sigma^2)$; por lo tanto $P(T_\sigma>e^{1/2\sigma^2})\sim\exp(\exp(-1/2\sigma^2))$ (tal como está escrito en el paper). Conclusión: para $\sigma$ pequeño las explosiones no se observan numéricamente.
- Conjetura (no probada): $\lim_{\sigma\to0}\frac{T_\sigma}{\mathbb E(T_\sigma)}=Z$, $Z$ exponencial de media 1.

## 5. Experimentos numéricos (§5)

Parámetros comunes: $f(u)=(u_+)^2$, $\sigma=6.36$, $u_0\equiv0$, $h=0.02$ ($n=50$ nodos), Dirichlet homogéneo, Euler explícito con paso adaptativo $\tau_j=\tau/\sum_i h u_i^j$.

### Exp. 0 — Barrido $\sigma\le5$ (sin figura)
"Corrimos el código con $\sigma\le5$ hasta $t=1000$ y no observamos explosiones sino un comportamiento metaestable." No se indican qué valores de $\sigma$, cuántas corridas, ni $\tau$.

### Figura 1 — Perfiles de una trayectoria a distintos tiempos
"Profiles of a sample solution at different times." Malla 3×3 de gráficos $u(x,t)$ vs $x\in[0,1]$ en 9 instantes. **Tabla 1** ("The maximum of the solution at different times"):

| Snapshot | Tiempo | $\|u(\cdot,t,\omega)\|_\infty$ |
|---|---|---|
| 1 | 1.0000 | 5.6159 |
| 2 | 50.0000 | 3.3863 |
| 3 | 72.0202 | 15.5104 |
| 4 | 72.4202 | 18.2885 |
| 5 | 72.4802 | 38.5848 |
| 6 | 72.5002 | 82.8705 |
| 7 | 72.5012 | 203.0799 |
| 8 | 72.5068 | $2.2695\times10^3$ |
| 9 | 72.5076 | $1.8128\times10^{12}$ |

Lectura visual de los ejes (aprox.): $t=1$: $y\in[-2,6]$, perfil muy rugoso; $t=50$: $y\in[-4,4]$, oscila alrededor de 0 (mín ≈ −3.7, máx ≈ 3.4 cerca de $x\approx0.96$); $t=72.0202$: $y\in[-2,16]$, forma de joroba rugosa con pico en $x\approx0.58$; $t=72.4202$: $[0,20]$; $t=72.4802$: $[0,40]$, pico en $x\approx0.6$; $t=72.5002$: $[0,90]$, joroba suave centrada ≈0.52; $t=72.5012$: $[0,250]$, pico ≈0.5; $t=72.5068$: $[0,2500]$, pico estrecho ≈0.5; $t=72.5076$: escala $\times10^{12}$, $[0,2]$, pico de un solo nodo en $x=0.5$ (resto ≈ 0). Es una sola trayectoria; el tiempo de explosión de esa trayectoria ≈ 72.5076.

### Figura 2 — Evolución de $\|u(\cdot,t)\|_\infty$
"The evolution of the maximum of a sample solution with initial data $u_0\equiv0$." Eje $t\in[0,80]$, eje $y\in[0,90]$. Oscila entre ~0 y ~10 hasta $t\approx72$, luego sube abruptamente; el último valor graficado ≈ 83 (coincide con snapshot 6, $t=72.5002$), es decir la curva está cortada antes de la explosión. Presumiblemente la misma trayectoria que la Fig. 1 (no se dice explícitamente).

### Figura 3 — Superficie $u(x,t)$
"The graph of a sample solution with initial datum $u_0\equiv0$." Superficie 3-D (estilo `surf` de MATLAB) con ejes $x\in[0,1]$, $t\in[0,80]$, $u\in[-20,40]$ (máx visible ≈ 45). Ruido rugoso de amplitud ~±10 sobre $t\in[0,\sim70]$ y una joroba al final. Trayectoria muestral (posiblemente la misma; no especificado), truncada en $u\approx40$–$45$. Resolución temporal de la gráfica no especificada (se ven ~100 líneas en $t$, es decir, se submuestrea).

### Figura 4 — Estadística del tiempo de explosión
"The kernel density estimator of the explosion time for $\sigma=6.36$ and the corresponding box-plot."
- **832 simulaciones** independientes con $\sigma=6.36$, $u_0\equiv0$.
- Criterio de parada / explosión: $\max_i u_i\ge 10^{13}$.
- **Media muestral = 46.8834; desviación estándar muestral = 43.8857.** (media ≈ d.e. ⇒ distribución cercana a exponencial, consistente con metaestabilidad.)
- Panel izquierdo: KDE, eje $t\in[0,300]$, densidad máx ≈ 0.0175 en $t\approx5$, decrecimiento monótono tipo exponencial (≈0.0065 en $t=50$, ≈0.002 en $t=100$, ≈0.0003 en $t=200$). Estilo MATLAB `ksdensity`.
- Panel derecho: box-plot (MATLAB `boxplot`), lectura visual: mediana ≈ 33, Q1 ≈ 15, Q3 ≈ 66, bigote superior ≈ 145–150, muchos outliers entre ~150 y ~230, y dos ≈ 320–330.

### No especificado (experimentos)
- $\tau$ (parámetro del paso adaptativo) y tratamiento de $\sum h u_i^j\le0$.
- Semillas; si las Figs. 1–3 son la misma trayectoria.
- Ancho de banda/núcleo del KDE, convención de bigotes del box-plot (MATLAB por defecto: 1.5·IQR).
- Valores concretos de $\sigma$ en el barrido $\sigma\le5$.
- Por qué $\sigma=6.36$ precisamente.

## 6. Notas de implementación [sugerencias]

1. **Malla:** $h=0.02$, $x_k=kh$, $k=0..50$; vector de 49 incógnitas interiores; bordes fijos en 0. Laplaciano: `(u[2:]-2*u[1:-1]+u[:-2])/h**2` sobre el vector con bordes.
2. **Reacción:** `np.maximum(u,0)**2`.
3. **Ruido:** por paso, `sigma*np.sqrt(tau_j/h)*rng.standard_normal(49)` (equivale a $\frac{\sigma}{\sqrt h}\Delta w_i$ con $w_i$ BM estándar; escala $\sqrt{1/(h\,\tau_j)}$ para $\dot W$). Con la lectura literal alternativa (w_i = integral de celda, var $h\,t$) el ruido sería $\sigma\sqrt{\tau_j}\,\xi$, ~7 veces menor para $h=0.02$; comprobar cuál reproduce la amplitud de la Fig. 1 ($|u|\sim$ 2–6 en $t=1$, $\pm4$ en $t=50$); el paper no da más datos para decidir, así que hay que validarlo empíricamente contra esas amplitudes.
4. **Paso adaptativo:** $\tau_j=\tau/\sum_i h u_i^j$, necesita salvaguarda. Opción: $\tau_j=\min\big(\tau_{\max},\ \tau/\max(\|u^j\|_{L^1},\epsilon)\big)$ con $\|u\|_{L^1}=h\sum|u_i|$ y $\tau_{\max}\le h^2/2=2\times10^{-4}$ (estabilidad del Euler explícito para la difusión). Nótese que en $t\approx72.5$ el paso debe bajar muchos órdenes; la explosión de $\|u\|_\infty$ de 83 a $1.8\times10^{12}$ ocurre en $\Delta t\approx0.0074$.
5. **Detección de blow-up:** parar cuando `u.max() >= 1e13`; registrar $t_j$. Poner además un $t_{\max}$ (p. ej. 1000, como en el barrido $\sigma\le5$) para las trayectorias que no exploten.
6. **Experimentos a reproducir:** (a) una trayectoria con snapshots en $t=1,50$ y cerca de la explosión + $\|u\|_\infty(t)$ + superficie; (b) 832 trayectorias ⇒ media (≈46.9), d.e. (≈43.9), KDE (`scipy.stats.gaussian_kde`) en $[0,300]$, box-plot; (c) $\sigma\le5$ hasta $t=1000$ sin explosión (metaestable); (d) opcional: test de exponencialidad de $T_\sigma/\bar T_\sigma$ (QQ-plot o KS contra Exp(1)).
7. **Verificaciones auxiliares:** estacionaria $v$ con máximo ≈ 11.80; EDE escalar $dz=(-\pi^2z+z_+^2)dt+\sigma dB$ (Lema 3.1) con el esquema adaptativo de [5] ($\tau_k=h/b(X_k)$) para comparar tiempos de explosión con el proyectado $\Phi(t)=\int\phi u$ (que debe satisfacer $\Phi\ge z$ en el continuo).
8. La exactitud cuantitativa de los tiempos (media 46.88) depende de $\tau$, $h$ y la salvaguarda, que el paper no da; esperar sólo concordancia cualitativa/orden de magnitud.
