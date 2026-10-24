# Reference snippets from notes/12-stock-price-prediction-using-sequence-models-lst.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
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

# ---- Snippet 2 ----
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

# ---- Snippet 3 ----
window = X_test[-1].copy(); future = []
for _ in range(30):
    nxt = model.predict(window[None, ...], verbose=0)[0, 0]
    future.append(nxt)
    window = np.vstack([window[1:], [[nxt]]])
future = scaler.inverse_transform(np.array(future).reshape(-1, 1))

