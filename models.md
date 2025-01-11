**1. Baseline Modell: Persistence method = ARIMA(0,1,0)**

$$
P_t - P_{t-1} = \beta_0 + \epsilon_t
$$

wobei:
- $P_t$ die Leistung zum Zeitpunkt $t$ ist,
- $P_{t-1}$ die Leistung zum Zeitpunkt $t-1$ ist,
- $\beta_0$ eine Konstante ist, die den Trend repräsentiert,
- $\epsilon_t$ ein zufälliger Fehlerterm (weißer Rauschen) zum Zeitpunkt \( t \) ist, der einer Normalverteilung folgt:
$$ \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$
- Der Trend wird als konstant über die Zeit angenommen mit einem Wert:
$$
\beta_0 \approx 1.31 \text{ MW (installierte Kapazität von 50Hertz)}
$$

Evaluation: Pinball Loss Function

Um das Modell probabilistisch zu bewerten, verwenden wir die Pinball Loss Function, die häufig in Quantilregressionsmodellen verwendet wird. Die Pinball Loss Function für ein vorhergesagtes Quantil $\hat{Q}_\tau$ bei Quantilniveau $\tau$ wird wie folgt definiert:

$$
\mathcal{L}_\tau(y, \hat{Q}_\tau) = 
\begin{cases} 
\tau \cdot (y - \hat{Q}_\tau) & \text{wenn } y \geq \hat{Q}_\tau \\
(1-\tau) \cdot (\hat{Q}_\tau - y) & \text{wenn } y < \hat{Q}_\tau
\end{cases}
$$

wobei:
- $y$ der tatsächliche Wert der Leistung (Zielvariable) ist,
- $\hat{Q}_\tau$ das vorhergesagte Quantil der Leistung zum Quantilniveau $\tau$ ist,
-$\tau$ das Quantilniveau ist (z.B. $tau = 0.5$ für den Median, $tau = 0.95$ für das 95. Perzentil usw.).
---------------------------------

**2. Standard Modell**
Gegeben sei das Ziel $Y$ (elektrische Leistung in MW) und die Eingaben $X$ (elektrische Leistung der vorherigen Tage und Windgeschwindigkeit):

$$
Y | X \sim \Gamma(\mu_X, \sigma_X^2)
$$

Das Modell schätzt die Verteilungsparameter $\mu_X$ und $\sigma_X^2$:
$$\mu(X_p) = \text{TreeOutput}_\mu(X_p), \sigma^2(X_p) = \text{TreeOutput}_{\sigma^2}(X_p)$$

Die Gamma-Verteilung mit Formparameter $k$ und Skalenparameter $\theta$:

$$
f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}
$$
Beziehungen zwischen $\mu$ und $\sigma^2$:

$$
\mu = k\theta, \quad \sigma^2 = k\theta^2
$$

Alternativ:

$$
\theta = \frac{\sigma^2}{\mu}, \quad k = \frac{\mu^2}{\sigma^2}
$$



