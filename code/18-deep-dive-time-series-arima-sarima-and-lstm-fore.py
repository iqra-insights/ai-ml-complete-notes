# Reference snippets from notes/18-deep-dive-time-series-arima-sarima-and-lstm-fore.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
from sklearn.model_selection import TimeSeriesSplit
for tr, te in TimeSeriesSplit(n_splits=5).split(X):
    model.fit(X.iloc[tr], y.iloc[tr]); preds = model.predict(X.iloc[te])

# ---- Snippet 2 ----
series_diff = series.diff().dropna()                 # trend hataane ke liye
series_seas = series.diff(12).dropna()               # monthly seasonality ke liye
series_log  = np.log(series).diff().dropna()         # variance + trend

# ---- Snippet 3 ----
res.plot_diagnostics(figsize=(10, 7))                # residual, histogram, Q-Q, ACF
from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(res.resid, lags=[10], return_df=True))   # p > 0.05 = white noise

