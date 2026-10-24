# Reference snippets from notes/10-time-series-analysis.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
import pandas as pd
df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
df = df.asfreq('D')                          # fixed frequency
df['value'] = df['value'].interpolate()      # missing fill
monthly = df['value'].resample('M').mean()   # resample
df['roll_7'] = df['value'].rolling(7).mean() # rolling mean
df['lag_1'] = df['value'].shift(1)           # lag feature
train, test = df[:'2024-06'], df['2024-07':] # time-based split

# ---- Snippet 2 ----
from statsmodels.tsa.stattools import adfuller
print(adfuller(series.dropna())[1])   # p-value
diff = series.diff().dropna()          # 1st-order differencing

# ---- Snippet 3 ----
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(series, model='additive', period=12).plot()

