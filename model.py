import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

print("Starting model training...")

# Download stock data
data = yf.download("AAPL", start="2018-01-01", progress=False)

print("Data downloaded")

# Create target column
data["Next_Close"] = data["Close"].shift(-1)
data.dropna(inplace=True)

X = data[["Close"]]
y = data["Next_Close"]

print("Training model...")

model = LinearRegression()
model.fit(X, y)

with open("stock_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and stock_model.pkl created")
