# ============================================================
# PHASE 1 — Data Collection & Inspection
# Stock: Apple Inc. (AAPL)
# ============================================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Download AAPL historical data (10 years) ---
print("Downloading AAPL data...")
ticker = "AAPL"
df = yf.download(ticker, start="2014-01-01", end="2024-01-01", auto_adjust=True)

# --- 2. Basic inspection ---
print("\n--- Shape (rows, columns) ---")
print(df.shape)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Last 5 rows ---")
print(df.tail())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Basic statistics ---")
print(df.describe())

# --- 3. Keep only the columns we need ---
df = df[["Open", "High", "Low", "Close", "Volume"]]
df.index = pd.to_datetime(df.index)

# --- 4. Save to CSV for use in later phases ---
os.makedirs("data", exist_ok=True)
df.to_csv("data/aapl_raw.csv")
print("\nData saved to data/aapl_raw.csv")

# --- 5. Plot the closing price ---
plt.figure(figsize=(14, 5))
plt.plot(df.index, df["Close"], color="#1a7abf", linewidth=1.2)
plt.title("AAPL Closing Price — 2014 to 2024", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/aapl_price_chart.png", dpi=150)
plt.show()
print("Chart saved to data/aapl_price_chart.png")