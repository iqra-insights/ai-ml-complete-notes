# Reference snippets from notes/11-weather-forecasting-using-arima-and-sarima.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
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

