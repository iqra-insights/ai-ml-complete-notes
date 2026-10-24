[Previous: 9. Chatbot](09-chatbot.md) | [Home](../README.md) | [Next: 11. Weather Forecasting using ARIMA and SARIMA](11-weather-forecasting-using-arima-and-sarima.md)

---

## 10. Time Series Analysis

Time series = time ke order mein collected data (daily sales, temperature, stock price). **Order matter karta hai**, isliye random train-test split nahi karte.

### 10.1 Time Series Data in ML

**Components**

- **Trend:** lambe time ka up/down direction
- **Seasonality:** fixed period par repeat pattern (har December sales up)
- **Cyclic:** irregular lambe cycles (economic cycles)
- **Noise / Residual:** random variation

**Applications:** sales forecasting, demand planning, weather, stock, energy load, anomaly detection.

### 10.2 Working with Time Series in Python

```python
import pandas as pd
df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
df = df.asfreq('D')                          # fixed frequency
df['value'] = df['value'].interpolate()      # missing fill
monthly = df['value'].resample('M').mean()   # resample
df['roll_7'] = df['value'].rolling(7).mean() # rolling mean
df['lag_1'] = df['value'].shift(1)           # lag feature
train, test = df[:'2024-06'], df['2024-07':] # time-based split
```

### 10.3 Stationarity

Stationary = mean, variance aur autocorrelation time ke saath **constant**. Zyada tar classical models (ARIMA) ko stationary data chahiye.

**Check karne ke tareeke**

- Visual: rolling mean/std plot
- **ADF test:** H0 = non-stationary. **p < 0.05 means stationary.**
- **KPSS test:** H0 = stationary (ulta)

**Stationary banane ke tareeke:** Differencing, Log / sqrt transform (variance ke liye), Seasonal differencing, Detrending.

```python
from statsmodels.tsa.stattools import adfuller
print(adfuller(series.dropna())[1])   # p-value
diff = series.diff().dropna()          # 1st-order differencing
```

### 10.4 Time Series Decomposition

```
Additive:        Y = Trend + Seasonality + Residual   (seasonal swing constant)
Multiplicative:  Y = Trend x Seasonality x Residual   (swing level ke saath badhe)
```

```python
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(series, model='additive', period=12).plot()
```

Advanced: STL decomposition (robust, flexible).

### 10.5 Pattern Recognition: ACF and PACF

- **ACF:** series ka apne lags se total correlation
- **PACF:** lag ka direct correlation (beech ke lags hata kar)

```
AR(p): PACF p lag ke baad cut off,  ACF slowly decay
MA(q): ACF  q lag ke baad cut off,  PACF slowly decay
```

**Baaki patterns:** autocorrelation, seasonality spikes, white noise (koi pattern nahi), random walk.

---

[Previous: 9. Chatbot](09-chatbot.md) | [Home](../README.md) | [Next: 11. Weather Forecasting using ARIMA and SARIMA](11-weather-forecasting-using-arima-and-sarima.md)
