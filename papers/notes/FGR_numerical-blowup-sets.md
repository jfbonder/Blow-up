# Notas: Fernández Bonder, Groisman, Rossi — *On numerical blow-up sets*

> Objetivo de estas notas: poder reimplementar (Python/NumPy/SciPy) todo cálculo numérico del artículo sin volver a abrir el PDF.
> Fuente: `papers/FGR_numerical-blowup-sets.pdf` (7 páginas, generado con TeX el 2001-10-31).

---

## 1. Referencia bibliográfica

- **Autores:** Julián Fernández Bonder, Pablo Groisman, Julio D. Rossi
  (Departamento de Matemática, FCEyN, UBA, (1428) Buenos Aires, Argentina; e-mails `jfbonder@dm.uba.ar`, `pgroisma@dm.uba.ar`, `jrossi@dm.uba.ar`).
- **Título:** *On numerical blow-up sets.*
- **Revista, año, volumen, páginas:** **No figuran en el PDF.** El archivo es una versión preprint (sin cabecera de revista; metadatos de creación TeX: 31/10/2001). No se indica revista, volumen ni páginas; no se adivinan aquí.
- MSC 1991: 35K55, 35B40, 65M20. Palabras clave: numerical approximations, nonlinear boundary conditions.
- Financiación: Universidad de Buenos Aires (TX47, TX48), ANPCyT PICT No. 03-00000-00137; J.D. Rossi también Fundación Antorchas.

---

## 2. Problema continuo

Ecuación del calor 1D con flujo no lineal en $x=1$ (ec. (1) del artículo):

$$
\begin{aligned}
u_t(x,t) &= u_{xx}(x,t), && (x,t)\in(0,1)\times[0,T),\\
u_x(1,t) &= f(u(1,t)), && t\in[0,T),\\
u_x(0,t) &= 0, && t\in[0,T),\\
u(x,0) &= \varphi(x), && x\in[0,1].
\end{aligned}
$$

- $f$ y $\varphi$ positivas y suficientemente regulares (existencia y unicidad). Si $\varphi$ cumple una condición de compatibilidad la solución es suave (refs. [2],[10]). La condición de compatibilidad no se escribe explícitamente; la natural es $\varphi'(0)=0$, $\varphi'(1)=f(\varphi(1))$.
- Se consideran soluciones **positivas**.

Resultados conocidos para el problema continuo (citados, no probados aquí):

- Si $f$ convexa y
  $$\int^{+\infty}\frac{ds}{f(s)f'(s)}<+\infty \qquad (2)$$
  hay blow-up (ref. [13], Walter).
- Si además $f$ creciente y
  $$\int^{+\infty}\frac{ds}{f(s)}<+\infty \qquad (3)$$
  el conjunto de blow-up es $B(u)=\{1\}$ (*single point blow-up*).
- Si $f$ verifica (2) pero no (3), $B(u)$ puede ser $[0,1]$ (*global blow-up*) o $[l,1]$ con $0<l<1$ (*regional blow-up*). Ejemplo: $f(s)=s\log^p(s)$: global para $1/2<p<1$, regional para $p=1$.
- Para $f(s)=s(\ln s)^p$: el continuo explota sii $p>1/2$; $B(u)=\{1\}$ si $p>1$; intervalo acotado si $p=1$; todo $[0,1]$ si $1/2<p<1$.

Parámetros: los ejemplos del artículo usan $f(s)=s^p$, $f(s)=s(\ln s)^p$, $f(s)=e^s$ (ver §4). Ningún valor concreto de $p$, $N$, $\varphi$ se fija en el artículo.

---

## 3. Esquema numérico

### 3.1 Malla y espacio

- $x_i = i/N$, $0\le i\le N$; $h=1/N$; $I_i=[x_i,x_{i+1}]$.
- $V_h$ = funciones continuas afines a trozos en cada $I_i$, base nodal $\psi_i$ con $\psi_i(x_j)=\delta_{ij}$.
- $u_h(x,t)=\sum_{i=0}^N u_i(t)\psi_i(x)$. (4)

### 3.2 Semidiscretización (Galerkin P1, sólo en espacio)

Sustituyendo en la formulación débil: $MU'=-AU+f(U)$ (5), con $U=(u_0,\dots,u_N)$, $M$ matriz de masa, $A$ matriz de rigidez, y $f(U):=f(u_N)\,e_N$, $e_N=(0,\dots,0,1)$ (el término de frontera sólo entra en la última ecuación).

Con **mass lumping**, $M=\mathrm{diag}(m_0,\dots,m_N)$, $m_i=h$ para $1\le i\le N-1$, $m_0=m_N=h/2$. El sistema resultante (ec. (6)) es, entrada por entrada:

$$
\begin{aligned}
u_0'(t) &= \frac{2}{h^2}\bigl(-u_0(t)+u_1(t)\bigr),\\
u_i'(t) &= \frac{1}{h^2}\bigl(u_{i+1}(t)-2u_i(t)+u_{i-1}(t)\bigr), \qquad 1\le i\le N-1,\\
u_N'(t) &= \frac{2}{h^2}\bigl(-u_N(t)+u_{N-1}(t)\bigr)+\frac{2}{h}\,f(u_N(t)).
\end{aligned}
$$

Equivalentemente $u_N' = \frac{2}{h^2}\bigl(-u_N+u_{N-1}+h f(u_N)\bigr)$ (forma usada en el Lema 1). Coincide con el esquema clásico de diferencias finitas de segundo orden (nodo fantasma en los extremos). Datos iniciales: $u_i(0)=\varphi(x_i)$ (implícito: no se escribe, pero es la interpolación natural).

En forma matricial ($N+1$ incógnitas):
$U' = -M^{-1}AU + M^{-1}f(u_N)e_N$, donde $A=\frac1h\,\mathrm{tridiag}(-1,2,-1)$ con $A_{00}=A_{NN}=1/h$.

### 3.3 Discretización temporal / paso adaptativo / criterio de parada

**El artículo no tiene discretización temporal**: es un estudio puramente semidiscreto (ODE en $t$ continuo). No hay paso adaptativo ni criterio de parada en el artículo. (Ver §6 para propuestas.)

### 3.4 Hechos citados sobre el esquema

- $u_h\to u$ uniformemente si $u$ es suave (ref. [5], Durán–Etcheverry–Rossi).
- En [5] también: las soluciones de (6) explotan **sii** $f$ verifica (3). Por tanto hay $f$ para las que el continuo explota y el discreto no: p. ej. $f(s)=s\ln(s)^p$ con $1/2<p\le 1$.
- Ref. [1] (Acosta, Fernández Bonder, Groisman, Rossi) trató $f(s)=s^p$: los conjuntos de blow-up numéricos aproximan los continuos cuando $h\to0$. Ref. [8]: análogo en el caso semilineal.

### 3.5 Definiciones

- $T_h$: tiempo de blow-up de $U$ (solución de (6)).
- $B_h(U)$ / $B(u_h)$: conjunto de blow-up numérico = nodos $x_i$ con $u_i(t)\to\infty$ cuando $t\nearrow T_h$ (interpretado como el intervalo que generan, p. ej. $[1-Kh,1]$).

---

## 4. Resultados principales

Notación: $\displaystyle \mathcal F(z)=\int_z^{+\infty}\frac{dy}{f(y)}$ (finito por (3)).

**Teorema 1.** Sea $f$ creciente verificando (3). Entonces el número de nodos de blow-up de las soluciones de (6) depende sólo de $f$ y es independiente de $h$. Es decir, o el blow-up numérico es global, o está localizado en un entorno pequeño de $\{x=1\}$ para $h$ suficientemente pequeño. Mecanismo de propagación: sea
$$
G_0(t)=\mathcal F^{-1}(T_h-t),\qquad G_{k+1}(t)=\int_0^t G_k(s)\,ds .
$$
Entonces $u_{N-k}$ explota **sii** $G_k$ explota, y en tal caso
$$
u_{N-k}(t)\sim\int_0^t u_{N-k+1}(s)\,ds .
$$
(En la prueba las integrales son $\int_{t_0}^t$, con $t_0<T_h$ dado por el Lema 2; "$\sim$" significa acotado arriba y abajo por constantes que pueden depender de $h$: $C_1\int_{t_0}^t u_{N-k+1}+C \le u_{N-k}\le C_2\int_{t_0}^t u_{N-k+1}+C$.)

**Teorema 2** (monotonía). Bajo las hipótesis del Teorema 1, el número de nodos de blow-up es monótono decreciente en $f$: si $f_1(s)\ge f_2(s)$ entonces $B_h(U_1)\subset B_h(U_2)$. (Prueba: $f_1\ge f_2\Rightarrow \mathcal F_1\le\mathcal F_2$.)

**Teorema 3** (tasas estrictamente monótonas). Para nodos $x_i$ que explotan,
$$
\lim_{t\nearrow T_h}\frac{u_i(t)}{u_{i+1}(t)}=0 .
$$
(Prueba: $u_N/u_{N-1}\sim u_N/\int_{t_0}^t u_N \to \lim u_N'/u_N=\lim C f(u_N)/u_N=+\infty$ por (9), e inducción.)

**Lemas técnicos.**

- **Lema 1 (principio del máximo).** $h>0$ fijo; si $U$ cumple las desigualdades $\le$ en (6) [esto es, (7)], entonces $\max_k u_k(t)\le\max\{\max_k u_k(0);\ \sup_{0<\tau<t}u_N(\tau)\}$.
- **Lema 2.** Si $U$ explota, existe $t_0<T_h$ con $\max_{0\le i\le N}u_i(t)=u_N(t)$ para $t_0<t<T_h$ (8). Usa que (3) implica $\lim_{s\to\infty}f(s)/s=+\infty$ (9) y la cota (10): $u_N'\ge-\frac{2}{h^2}u_N+\frac2h f(u_N)\ge\frac2h f(u_N)\bigl(1-\frac{u_N}{hf(u_N)}\bigr)$.
- **Lema 3.** Existe $C$ (que puede depender de $h$) tal que $\max_k u_k(t)=u_N(t)\sim\mathcal F^{-1}(C(T_h-t))$. Más precisamente $\mathcal F^{-1}(C(T_h-t))\le u_N(t)\le c_1\mathcal F^{-1}(C(T_h-t))$ (cota inferior de $u_N'\le\frac2h f(u_N)$ cerca de $T_h$).
- Paso clave para $u_{N-1}$: $(e^{2t/h^2}u_{N-1})'\ge \frac{e^{2t/h^2}}{h^2}u_N$ y $u_{N-1}'\le\frac{2}{h^2}(u_N-u_{N-1})$.

### Ejemplos (con número explícito de nodos y tasas)

**(I) $f(s)=s^p$.** Las soluciones numéricas explotan sii $p>1$ (igual que el continuo). Tasas:
$$
u_{N-k}(t)\sim (T_h-t)^{-1/(p-1)+k},\qquad 0\le k<\frac1{p-1};
$$
si $1/(p-1)=k$ es entero, entonces $u_{N-k}\sim-\ln(T_h-t)$. El artículo dice: "el conjunto de blow-up está compuesto por exactamente $K$ nodos, $K=[1/(p-1)]$" ($[x]$ = parte entera) y
$$
B(u_h)=[1-Kh,\,1]=B(u)+[-Kh,0].
$$
**Observación (inconsistencia interna, señalada aquí, no en el artículo):** $[1-Kh,1]$ contiene los $K+1$ nodos $x_{N-K},\dots,x_N$, y la fórmula de tasas también da $K+1$ nodos que explotan ($k=0,\dots,K$: si $1/(p-1)\notin\mathbb Z$ todos con tasa potencial; si $1/(p-1)=K\in\mathbb Z$, $K$ potenciales y el último, $u_{N-K}$, logarítmico). Por ejemplo $p=2$: $u_N\sim(T_h-t)^{-1}$, $u_{N-1}\sim-\ln(T_h-t)$, $u_{N-2}$ acotado. Lo coherente con $G_k$ es: **explotan los nodos $x_{N-k}$, $0\le k\le[1/(p-1)]$**. Comprobación: $\mathcal F(z)=z^{1-p}/(p-1)$, $\mathcal F^{-1}(w)=((p-1)w)^{-1/(p-1)}$, $G_k\sim(T_h-t)^{-1/(p-1)+k}$ mientras el exponente sea $<0$, logarítmico tras integrar exponente $-1$, acotado después. Requiere $N>K$ ("$h$ pequeño").

**(II) $f(s)=s(\ln s)^p$.** Las soluciones numéricas explotan sii $p>1$ (distinto del continuo, $p>1/2$). Tasa:
$$
\max_i u_i(t)\sim\exp\!\Bigl(\frac{1}{(T_h-t)^{1/(p-1)}}\Bigr).
$$
Por (I) y el Teorema 2, el conjunto de blow-up es **todo $[0,1]$** (blow-up global numérico), mientras que el continuo con $p>1$ tiene $B(u)=\{1\}$. Los autores lo presentan como el primer ejemplo de tal fenómeno. (Comprobación: $\mathcal F(z)=(\ln z)^{1-p}/(p-1)$, $\mathcal F^{-1}(w)=\exp(((p-1)w)^{-1/(p-1)})$; todas las $G_k$ explotan.) Nota: $f>0$ sólo para $s>1$; hay que tomar $\varphi>1$.

**(III) $f(s)=e^s$.** Blow-up en un solo punto, $B(u_h)=\{1\}$; tasa $u_N(t)\sim-\ln(T_h-t)$. Conjunto y tasa coinciden con los del continuo. ($\mathcal F(z)=e^{-z}$, $\mathcal F^{-1}(w)=-\ln w$, $G_1$ acotada.)

Mensaje global: el blow-up **regional** (intervalo $[l,1]$ con $l$ fijo independiente de $h$) nunca se reproduce con este esquema: o $B$ se concentra en $O(h)$ de la frontera (número fijo de nodos), o es todo $[0,1]$.

---

## 5. Experimentos numéricos, figuras y tablas

**El artículo no contiene ningún experimento numérico, figura ni tabla.** Es puramente teórico (7 páginas: introducción, enunciados, ejemplos, lemas, pruebas, referencias). No hay números reportados que copiar.

### No especificado
- Valores de $p$, $N$/$h$, dato inicial $\varphi$, método de integración temporal, paso de tiempo, tolerancias, criterio de parada, valor de $T_h$: nada de esto aparece.

### Propuesta de cálculos ilustrativos (PROPUESTA, no está en el artículo)
1. **P1 — $f=s^p$, conteo de nodos e independencia de $h$.** $p\in\{3,\ 2,\ 1.5,\ 1.3,\ 1.25\}$ ($1/(p-1)=0.5,1,2,3.33,4$; nodos esperados $K+1=1,2,3,4,5$ según §4). $N\in\{10,20,40,80\}$. Verificar que el número de nodos que explotan no cambia con $N$ y que $B(u_h)$ se encoge como $[1-Kh,1]$.
2. **P2 — tasas en $f=s^p$.** Pendiente de $\log u_{N-k}$ vs $\log(T_h-t)$ $\to -1/(p-1)+k$; para $1/(p-1)$ entero (p. ej. $p=2$, $p=1.5$) comprobar $u_{N-K}/(-\ln(T_h-t))\to$ const.
3. **P3 — Teorema 3.** Graficar $u_i(t)/u_{i+1}(t)$ para nodos que explotan: $\to0$.
4. **P4 — $f=s(\ln s)^p$, $p=2$ (y p. ej. $p=1.5$).** Mostrar que todos los nodos explotan (perfil plano relativo: $u_0(t)\to\infty$), en contraste con el continuo de un solo punto; $\ln\ln u_N$ vs $\ln(T_h-t)$ con pendiente $-1/(p-1)$. Con $p\in(1/2,1]$ comprobar que el discreto no explota.
5. **P5 — $f=e^s$.** Sólo $u_N$ explota; $u_N+\ln(T_h-t)$ acotado.
6. **P6 — Teorema 2.** Comparar $f_1=s^2\ge f_2=s^{1.5}$ (en $s\ge1$): $B_h(U_1)\subset B_h(U_2)$.

---

## 6. Notas de implementación

- **Sistema de ODEs:** implementar (6) vectorizado; $N+1$ incógnitas; el término no lineal sólo en la ecuación $N$ con factor $2/h$. Rígido (autovalores $\sim -4/h^2$): usar `scipy.integrate.solve_ivp` con `method="Radau"`/`"BDF"` o `LSODA` con jacobiano tridiagonal (más $\frac2h f'(u_N)$ en la entrada $(N,N)$), `rtol`≈1e-10, `atol` pequeño.
- **Dato inicial (propuesta):** positivo; p. ej. constante $\varphi\equiv c$ ($c>1$ para $s(\ln s)^p$; con $c$ y $f$ crecientes la solución es creciente en $t$ y $x$). Alternativa compatible: $\varphi(x)=c+\frac{f(c)}{2}x^2$ aproximadamente. El resultado (número de nodos) no depende de $\varphi$ según el Teorema 1.
- **Blow-up / parada (propuesta):** integrar hasta que $\max_i u_i=u_N$ supere un umbral grande (p. ej. $10^{8}$–$10^{12}$, o $e^{s}$ evaluable: $u_N\lesssim 700$ en float64 para $f=e^s$) mediante un evento de `solve_ivp`; alternativamente cambio de variable o paso adaptativo tipo $\tau_j=\tau\min\{1,1/f(u_N)\}$ / $\tau_j\propto 1/u_N^{p-1}$ si se usa Euler (no proviene de este artículo).
- **Estimación de $T_h$:** con la tasa de $u_N$ conocida (Lema 3), ajustar $u_N\approx A\,\mathcal F^{-1}(C(T_h-t))$; o para $s^p$ extrapolar $u_N^{-(p-1)}$ lineal en $t$ hasta cero. Para $e^s$: $e^{-u_N}$ lineal en $t$.
- **Detección de nodos que explotan:** como el umbral finito no es infinito, decidir con tendencias: un nodo explota si $u_i(t)\to\infty$ al acercarse a $T_h$ (p. ej. crecimiento sin cota al reducir $T_h-t$ en varias décadas); para el nodo logarítmico la señal es débil ($-\ln(T_h-t)$), por lo que conviene aproximarse mucho a $T_h$ (necesita $T_h-t$ muy pequeño ⇒ precisión en $t$; trabajar con $\tau=T_h-t$ o usar `np.longdouble` si hace falta). Los nodos acotados tienden a un límite finito.
- **Precisión del cálculo de $u_{N-k}$:** las diferencias $u_{i+1}-2u_i+u_{i-1}$ con $u_N$ enorme pierden precisión relativa en nodos lejanos; vigilar cancelación.
- El parámetro $h$ entra en las constantes $C$ (dependen de $h$) pero no en el número de nodos.
