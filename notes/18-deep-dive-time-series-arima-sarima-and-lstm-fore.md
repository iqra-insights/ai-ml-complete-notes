[Previous: 17. Deep Dive: RNN, LSTM, GRU, Fake News and Chatbot](17-deep-dive-rnn-lstm-gru-fake-news-and-chatbot.md) | [Home](../README.md) | [Next: 19. Revision Plan and Mini-Project Checklist](19-revision-plan-and-mini-project-checklist.md)

---

## 18. Deep Dive: Time Series, ARIMA/SARIMA and LSTM Forecasting

### 18.1 Time series data handling

| Task | Pandas |
| --- | --- |
| Parse dates + index | `pd.read_csv(..., parse_dates=['date'], index_col='date')` |
| Fixed frequency | `df.asfreq('D')` |
| Missing values | `interpolate()`, `ffill()`; **future values se fill mat karo** (leakage) |
| Resample | `df.resample('W').sum()` / `.mean()` |
| Rolling stats | `df['x'].rolling(7).mean()` |
| Expanding stats | `df['x'].expanding().mean()` |
| Lag features | `df['x'].shift(1)`, `shift(7)` |
| Date features | `df.index.month`, `.dayofweek`, `.is_month_end` |

**Validation:** random split nahi. Use `TimeSeriesSplit` ya **walk-forward validation** (train upto t, predict t+1, window aage badhao).

```python
from sklearn.model_selection import TimeSeriesSplit
for tr, te in TimeSeriesSplit(n_splits=5).split(X):
    model.fit(X.iloc[tr], y.iloc[tr]); preds = model.predict(X.iloc[te])
```

**Metrics:**

```math
MAPE=\frac{100}{n}\sum\left|\frac{y-\hat y}{y}\right|,\quad RMSE=\sqrt{\frac1n\sum(y-\hat y)^2}
```

MAPE tab fail hota hai jab actual value 0 ke paas ho. **Hamesha naive baseline se compare karo** (kal ki value = aaj ki prediction).

### 18.2 Stationarity tests: how to read

| Test | H0 | Stationary kab maanein |
| --- | --- | --- |
| ADF | Unit root hai (non-stationary) | p < 0.05 |
| KPSS | Series stationary hai | p > 0.05 |

Agar ADF aur KPSS ek doosre se alag nikalein, to series trend-stationary ya difference-stationary ho sakti hai. Differencing try karo aur dobara test karo.

```python
series_diff = series.diff().dropna()                 # trend hataane ke liye
series_seas = series.diff(12).dropna()               # monthly seasonality ke liye
series_log  = np.log(series).diff().dropna()         # variance + trend
```

**Over-differencing se bacho:** agar ACF lag 1 par bahut negative (< -0.5) ho to zyada differencing hui hai.

### 18.3 Reading ACF and PACF

| Pattern | Suggests |
| --- | --- |
| PACF lag p ke baad cut off, ACF slowly decays | AR(p) |
| ACF lag q ke baad cut off, PACF slowly decays | MA(q) |
| Dono slowly decay | ARMA (dono) |
| ACF slowly decay, bahut lambi | Non-stationary, difference karo |
| Spikes har m lag par | Seasonality, period m |

**Special cases:** ARIMA(0,1,0) = random walk, ARIMA(1,0,0) = AR(1), ARIMA(0,1,1) = simple exponential smoothing.

### 18.4 SARIMA parameter selection

Example: monthly data, trend + yearly seasonality. Start: `order=(1,1,1)`, `seasonal_order=(1,1,1,12)`.

| Part | Meaning |
| --- | --- |
| (p, d, q) | Non-seasonal AR, differencing, MA |
| (P, D, Q) | Seasonal versions (lag m, 2m, 3m ke saath) |
| m | Season length: monthly 12, daily-with-weekly 7, hourly-with-daily 24 |

**Model selection:** AIC/BIC grid search ya `pmdarima.auto_arima`. Lower AIC = behtar fit, par overfit se bacho: test set par RMSE dekho.

### 18.5 Residual diagnostics

Achha model ke residuals **white noise** hone chahiye: mean 0, constant variance, koi autocorrelation nahi.

```python
res.plot_diagnostics(figsize=(10, 7))                # residual, histogram, Q-Q, ACF
from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(res.resid, lags=[10], return_df=True))   # p > 0.05 = white noise
```

### 18.6 Beyond ARIMA

| Model | Kab |
| --- | --- |
| Holt-Winters (ETS) | Simple trend + seasonality, kam data |
| Prophet | Holidays, missing data, business series, easy |
| XGBoost with lag features | Kai external features hon |
| LSTM / GRU | Bahut data, non-linear patterns, multivariate |
| Temporal Fusion Transformer | Advanced, multi-horizon |

### 18.7 LSTM forecasting: details that matter

- **Scaling:** scaler **sirf train** par fit karo, warna future ki information leak hoti hai.
- **Window size:** 30 / 60 / 90 try karo, validation error se choose karo.
- **Features:** sirf Close nahi, Volume, moving averages, RSI jaise indicators add karke multivariate banao (input shape `(samples, steps, features)`).
- **Multi-step forecast strategies:**

| Strategy | How | Trade-off |
| --- | --- | --- |
| Recursive | Ek step predict, usse input mein daalo | Error accumulate hota hai |
| Direct | Har horizon ke liye alag model | Zyada models |
| Multi-output | Ek model se N steps ek saath (`Dense(N)`) | Simple, achha balance |
| Seq2Seq | Encoder-decoder LSTM | Flexible, complex |

- **Compare karo:** naive baseline aur ARIMA se. Agar LSTM naive se behtar nahi hai to wo useful nahi.
- **Reproducibility:** `tf.random.set_seed(42)`, `np.random.seed(42)`.

### 18.8 Weather forecasting project steps (ARIMA/SARIMA)

1. Data load (date index), missing fill, monthly/daily resample.
2. Plot + seasonal decomposition (period 12 ya 365/7).
3. ADF test, differencing, ACF/PACF.
4. SARIMA fit (auto\_arima se start), AIC compare.
5. Residual diagnostics.
6. Forecast next N periods with confidence interval plot.
7. RMSE/MAE vs naive seasonal baseline (pichle saal ka same month).

---

[Previous: 17. Deep Dive: RNN, LSTM, GRU, Fake News and Chatbot](17-deep-dive-rnn-lstm-gru-fake-news-and-chatbot.md) | [Home](../README.md) | [Next: 19. Revision Plan and Mini-Project Checklist](19-revision-plan-and-mini-project-checklist.md)
