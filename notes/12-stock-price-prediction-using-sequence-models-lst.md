[Previous: 11. Weather Forecasting using ARIMA and SARIMA](11-weather-forecasting-using-arima-and-sarima.md) | [Home](../README.md) | [Next: 13. Quick Revision Cheat Sheet](13-quick-revision-cheat-sheet.md)

---

## 12. Stock Price Prediction using Sequence Models (LSTM)

### 12.1 Pipeline

```
Historical prices -> Scale (MinMax) -> Windows (60 days -> next day) -> LSTM -> Predict -> Inverse scale
```

### 12.2 Preparing Data for LSTM

LSTM input shape: **(samples, time\_steps, features)**.

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices.reshape(-1, 1))   # fit sirf train par ideally

def make_windows(data, n=60):
    X, y = [], []
    for i in range(n, len(data)):
        X.append(data[i-n:i, 0]); y.append(data[i, 0])
    return np.array(X)[..., None], np.array(y)

X, y = make_windows(scaled)
split = int(0.8 * len(X))                      # time-based split, shuffle nahi
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
```

### 12.3 Forecasting using LSTM

```python
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(50),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
pred = scaler.inverse_transform(model.predict(X_test))
```

**Evaluate:** RMSE / MAE / MAPE + actual vs predicted plot.

### 12.4 Future Forecast

Recursive forecasting: last window se next predict karo, usse window mein add karo, repeat.

```python
window = X_test[-1].copy(); future = []
for _ in range(30):
    nxt = model.predict(window[None, ...], verbose=0)[0, 0]
    future.append(nxt)
    window = np.vstack([window[1:], [[nxt]]])
future = scaler.inverse_transform(np.array(future).reshape(-1, 1))
```

**Important honesty point:** stock prices noisy aur market news se drive hote hain. Model sirf past pattern seekhta hai, isliye long-term forecast error badhta jata hai. Ise **learning/portfolio project** samjho, trading advice nahi. Compare with naive baseline (kal ki price = aaj ki price).

---

[Previous: 11. Weather Forecasting using ARIMA and SARIMA](11-weather-forecasting-using-arima-and-sarima.md) | [Home](../README.md) | [Next: 13. Quick Revision Cheat Sheet](13-quick-revision-cheat-sheet.md)
