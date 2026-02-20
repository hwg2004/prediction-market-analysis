import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

SYMBOL = "META"
START_DATE = "2022-01-01"
END_DATE = "2023-01-01"
client = StockHistoricalDataClient(API_KEY, API_SECRET)

request_params = StockBarsRequest(
    symbol_or_symbols=[SYMBOL],
    timeframe=TimeFrame.Day,
    start=START_DATE,
    end=END_DATE,
)

bars = client.get_stock_bars(request_params).df
df = bars.df.reset_index()

df = df[df['symbol'] == SYMBOL]

#features
df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(5).std()
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_20"] = df["close"].rolling(20).mean()

df["target"] = df["close"].shift(-1)
df = df.dropna()

features = [
    "close", 
    "volume",
    "return", 
    "volatility", 
    "ma_5", 
    "ma_10", 
    "ma_20"
]

x = df[features]
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10, 
    random_state=42
)
model.fit(x_train, y_train)

preds = model.predict(x_test)
mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: {mae}")

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Price')
plt.plot(preds, label='Predicted Price')
plt.title('META Price Prediction')  
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

latest_data = x.iloc[-1:].values
prediction = model.predict(latest_data)[0]
print(f"Predicted closing price for {SYMBOL}: ${prediction}")

