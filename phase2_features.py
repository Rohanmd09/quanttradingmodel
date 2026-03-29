# ============================================================
# PHASE 2 — Feature Engineering
# Building signals from raw AAPL price data
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- 1. Load the cleaned data from Phase 1 ---
df = pd.read_csv("data/aapl_raw.csv", header=[0, 1], index_col=0)
df.columns = ["Close", "High", "Low", "Open", "Volume"]
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Convert all price columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Loaded {len(df)} rows of data.")

# --- 2. Daily Return ---
# How much did the price move (%) each day?
df["daily_return"] = df["Close"].pct_change()

# --- 3. Moving Averages ---
# Short-term trend (10 days) vs long-term trend (50 days)
df["ma_10"] = df["Close"].rolling(window=10).mean()
df["ma_50"] = df["Close"].rolling(window=50).mean()

# MA spread: positive = short trend above long trend (bullish signal)
df["ma_spread"] = (df["ma_10"] - df["ma_50"]) / df["ma_50"]

# --- 4. Volatility ---
# Rolling 10-day standard deviation of daily returns
df["volatility_10"] = df["daily_return"].rolling(window=10).std()

# --- 5. RSI (Relative Strength Index) ---
# Measures if a stock is overbought (>70) or oversold (<30)
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["rsi_14"] = compute_rsi(df["Close"])

# --- 6. Volume Change ---
# Is trading volume spiking? Spikes often signal big moves
df["volume_change"] = df["Volume"].pct_change()

# --- 7. Price Position ---
# Where is today's close within the last 20 days' range? (0=bottom, 1=top)
df["price_position"] = (
    (df["Close"] - df["Close"].rolling(20).min()) /
    (df["Close"].rolling(20).max() - df["Close"].rolling(20).min())
)

# --- 8. Target Label ---
# 1 = price went UP next day, 0 = price went DOWN or flat
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# --- 9. Drop rows with NaN (from rolling windows) ---
df_clean = df.dropna()
print(f"Rows after dropping NaN: {len(df_clean)}")
print(f"Rows removed by rolling windows: {len(df) - len(df_clean)}")

# --- 10. Class balance check ---
up_days = df_clean["target"].sum()
down_days = len(df_clean) - up_days
print(f"\nTarget label balance:")
print(f"  Up days   (1): {up_days}  ({100*up_days/len(df_clean):.1f}%)")
print(f"  Down days (0): {down_days}  ({100*down_days/len(df_clean):.1f}%)")

# --- 11. Feature summary ---
features = ["daily_return", "ma_10", "ma_50", "ma_spread",
            "volatility_10", "rsi_14", "volume_change", "price_position"]

print(f"\nFeature summary:")
print(df_clean[features].describe().round(4))

# --- 12. Save enriched dataset ---
df_clean.to_csv("data/aapl_features.csv")
print("\nSaved to data/aapl_features.csv")

# --- 13. Visualise the features ---
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

# Plot 1: Closing price with MAs
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_clean.index, df_clean["Close"], color="#1a7abf", linewidth=1, label="Close")
ax1.plot(df_clean.index, df_clean["ma_10"], color="#f5a623", linewidth=1.2, label="MA 10")
ax1.plot(df_clean.index, df_clean["ma_50"], color="#e94f37", linewidth=1.2, label="MA 50")
ax1.set_title("AAPL Close Price with Moving Averages")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Daily Return
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_clean.index, df_clean["daily_return"], color="#444", linewidth=0.6)
ax2.axhline(0, color="red", linewidth=0.8, linestyle="--")
ax2.set_title("Daily Return")
ax2.grid(True, alpha=0.3)

# Plot 3: Volatility
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df_clean.index, df_clean["volatility_10"], color="#9b59b6", linewidth=0.8)
ax3.set_title("10-Day Rolling Volatility")
ax3.grid(True, alpha=0.3)

# Plot 4: RSI
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(df_clean.index, df_clean["rsi_14"], color="#27ae60", linewidth=0.8)
ax4.axhline(70, color="red", linewidth=0.8, linestyle="--", label="Overbought (70)")
ax4.axhline(30, color="blue", linewidth=0.8, linestyle="--", label="Oversold (30)")
ax4.set_title("RSI (14-day)")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: MA Spread
ax5 = fig.add_subplot(gs[2, 1])
ax5.bar(df_clean.index, df_clean["ma_spread"],
        color=["#27ae60" if v > 0 else "#e94f37" for v in df_clean["ma_spread"]],
        width=1, alpha=0.7)
ax5.axhline(0, color="black", linewidth=0.8)
ax5.set_title("MA Spread (MA10 vs MA50)")
ax5.grid(True, alpha=0.3)

# Plot 6: Price Position
ax6 = fig.add_subplot(gs[3, 0])
ax6.plot(df_clean.index, df_clean["price_position"], color="#e67e22", linewidth=0.8)
ax6.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
ax6.set_title("Price Position (20-day range)")
ax6.grid(True, alpha=0.3)

# Plot 7: Volume Change
ax7 = fig.add_subplot(gs[3, 1])
ax7.plot(df_clean.index, df_clean["volume_change"], color="#1abc9c", linewidth=0.6)
ax7.axhline(0, color="red", linewidth=0.8, linestyle="--")
ax7.set_title("Volume Change (%)")
ax7.grid(True, alpha=0.3)

plt.suptitle("AAPL — Engineered Features (2014–2024)", fontsize=15, y=1.01)
plt.savefig("data/aapl_features_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature chart saved to data/aapl_features_chart.png")