[Previous: 10. Time Series Analysis](10-time-series-analysis.md) | [Home](../README.md) | [Next: 12. Stock Price Prediction using Sequence Models (LSTM)](12-stock-price-prediction-using-sequence-models-lst.md)

---

## 11. Weather Forecasting using ARIMA and SARIMA

### 11.1 AR and MA Models

- **AR(p):** aaj ki value = pichli p values ka linear combination `Y(t) = c + a1 Y(t-1) + ... + ap Y(t-p) + error`
- **MA(q):** aaj ki value = pichle q errors ka combination `Y(t) = c + error(t) + b1 error(t-1) + ... + bq error(t-q)`

### 11.2 ARIMA(p, d, q)

<p align="center"><img src="../images/15-arima-sarima-workflow.svg" alt="ARIMA / SARIMA modelling workflow · 2 decisions" width="800"><br><sub><i>ARIMA / SARIMA modelling workflow · 2 decisions</i></sub></p>

| Param | Meaning | Kaise choose karein |
| --- | --- | --- |
| p | AR order | PACF cut-off |
| d | Differencing count | ADF test se (usually 0, 1, 2) |
| q | MA order | ACF cut-off |

### 11.3 SARIMA(p,d,q)(P,D,Q,m)

ARIMA + **seasonal** component. `m` = season length (monthly data m=12, daily with weekly pattern m=7). SARIMAX = SARIMA + exogenous variables (e.g. holiday, humidity).

| ARIMA | SARIMA |
| --- | --- |
| Non-seasonal data | Seasonal data (weather, sales) |
| 3 parameters | 7 parameters |

### 11.4 Implementation

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
res = model.fit(disp=False)
print(res.summary())                      # AIC, coefficients

forecast = res.get_forecast(steps=len(test))
pred = forecast.predicted_mean
ci = forecast.conf_int()                  # confidence interval
print('RMSE:', np.sqrt(mean_squared_error(test, pred)))
```

**Auto selection:** `pmdarima.auto_arima(train, seasonal=True, m=12)` ya AIC/BIC grid search (**lower AIC = better**).

**Workflow**

```
Plot -> Decompose -> ADF test -> Differencing -> ACF/PACF -> Pick (p,d,q)
 -> Fit -> Residual check (white noise?) -> Forecast -> RMSE/MAE/MAPE
```

**Metrics:** MAE, RMSE, MAPE.

---

[Previous: 10. Time Series Analysis](10-time-series-analysis.md) | [Home](../README.md) | [Next: 12. Stock Price Prediction using Sequence Models (LSTM)](12-stock-price-prediction-using-sequence-models-lst.md)
