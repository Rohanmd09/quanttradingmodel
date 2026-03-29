# ============================================================
# PHASE 4 — Strategy Construction
# Turning model predictions into trading signals
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# --- 1. Load predictions from Phase 3 ---
df = pd.read_csv("data/aapl_predictions.csv", header=[0, 1], index_col=0)
df.columns = ["Close", "High", "Low", "Open", "Volume",
              "daily_return", "ma_10", "ma_50", "ma_spread",
              "volatility_10", "rsi_14", "volume_change",
              "price_position", "target",
              "lr_pred", "rf_pred", "lr_proba", "rf_proba"]
df.index = pd.to_datetime(df.index)
df = df.sort_index()

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

print(f"Loaded {len(df)} rows of test data.")
print(f"Period: {df.index[0].date()} → {df.index[-1].date()}")

# ============================================================
# STRATEGY RULES (using Random Forest — the stronger model)
# ============================================================
# Signal = 1  → BUY  (hold position, expect price to rise)
# Signal = 0  → FLAT (exit position, sit in cash)
#
# Position sizing: We invest a fixed fraction of capital
# Stop-loss: If daily loss exceeds 2%, we exit that day
# ============================================================

POSITION_SIZE  = 0.95   # invest 95% of available capital when signal=1
STOP_LOSS      = -0.02  # exit if daily return drops below -2%
CONFIDENCE_THR = 0.55   # only trade when model is at least 55% confident

# --- 2. Build the signal ---
# Use RF probability for confidence filtering
df["signal"] = 0
df.loc[df["rf_proba"] >= CONFIDENCE_THR, "signal"] = 1

print(f"\nSignal distribution:")
print(f"  BUY  signals (1): {df['signal'].sum()}  ({100*df['signal'].mean():.1f}%)")
print(f"  FLAT signals (0): {(df['signal']==0).sum()}  ({100*(df['signal']==0).mean():.1f}%)")

# --- 3. Apply stop-loss rule ---
# If signal says BUY but daily return < -2%, we exit (override to 0)
stop_loss_triggered = (df["signal"] == 1) & (df["daily_return"] < STOP_LOSS)
df.loc[stop_loss_triggered, "signal"] = 0
print(f"  Stop-loss overrides: {stop_loss_triggered.sum()} days")

# --- 4. Calculate strategy returns ---
# Strategy return on day t = signal(t) * actual_return(t) * position_size
df["strategy_return"] = df["signal"] * df["daily_return"] * POSITION_SIZE

# Buy-and-hold return (baseline): fully invested every day
df["buyhold_return"] = df["daily_return"]

# --- 5. Cumulative returns ---
df["strategy_cumulative"] = (1 + df["strategy_return"]).cumprod()
df["buyhold_cumulative"]  = (1 + df["buyhold_return"]).cumprod()

# --- 6. Performance metrics ---
trading_days = len(df)
years = trading_days / 252   # ~252 trading days per year

# Total return
strat_total  = df["strategy_cumulative"].iloc[-1] - 1
bh_total     = df["buyhold_cumulative"].iloc[-1] - 1

# Annualised return
strat_annual = (1 + strat_total) ** (1/years) - 1
bh_annual    = (1 + bh_total)   ** (1/years) - 1

# Volatility (annualised)
strat_vol = df["strategy_return"].std() * np.sqrt(252)
bh_vol    = df["buyhold_return"].std()  * np.sqrt(252)

# Sharpe Ratio (assuming risk-free rate = 4% for 2022-2023)
RISK_FREE = 0.04
strat_sharpe = (strat_annual - RISK_FREE) / strat_vol if strat_vol > 0 else 0
bh_sharpe    = (bh_annual    - RISK_FREE) / bh_vol    if bh_vol    > 0 else 0

# Maximum Drawdown
def max_drawdown(cumulative_series):
    rolling_max = cumulative_series.cummax()
    drawdown    = (cumulative_series - rolling_max) / rolling_max
    return drawdown.min()

strat_mdd = max_drawdown(df["strategy_cumulative"])
bh_mdd    = max_drawdown(df["buyhold_cumulative"])

# Win rate (days where strategy made money)
winning_days = (df[df["signal"] == 1]["strategy_return"] > 0).sum()
total_traded = (df["signal"] == 1).sum()
win_rate     = winning_days / total_traded if total_traded > 0 else 0

# --- 7. Print results ---
print(f"\n{'='*50}")
print(f"  STRATEGY PERFORMANCE SUMMARY")
print(f"{'='*50}")
print(f"  {'Metric':<25} {'Strategy':>12} {'Buy & Hold':>12}")
print(f"  {'-'*50}")
print(f"  {'Total Return':<25} {strat_total*100:>11.1f}% {bh_total*100:>11.1f}%")
print(f"  {'Annualised Return':<25} {strat_annual*100:>11.1f}% {bh_annual*100:>11.1f}%")
print(f"  {'Annualised Volatility':<25} {strat_vol*100:>11.1f}% {bh_vol*100:>11.1f}%")
print(f"  {'Sharpe Ratio':<25} {strat_sharpe:>12.3f} {bh_sharpe:>12.3f}")
print(f"  {'Max Drawdown':<25} {strat_mdd*100:>11.1f}% {bh_mdd*100:>11.1f}%")
print(f"  {'Win Rate (traded days)':<25} {win_rate*100:>11.1f}%          —")
print(f"  {'Days Traded':<25} {total_traded:>12} {trading_days:>12}")
print(f"{'='*50}")

# --- 8. Save strategy data ---
df.to_csv("data/aapl_strategy.csv")
print(f"\nStrategy data saved to data/aapl_strategy.csv")

# --- 9. Visualisations ---
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.4)

# Plot 1: Cumulative returns comparison
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df["strategy_cumulative"],
         color="#27ae60", linewidth=2, label="RF Strategy")
ax1.plot(df.index, df["buyhold_cumulative"],
         color="#1a7abf", linewidth=2, label="Buy & Hold")
ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax1.fill_between(df.index, df["strategy_cumulative"], 1,
                 where=df["strategy_cumulative"] >= 1,
                 alpha=0.08, color="#27ae60")
ax1.fill_between(df.index, df["strategy_cumulative"], 1,
                 where=df["strategy_cumulative"] < 1,
                 alpha=0.08, color="#e94f37")
ax1.set_title("Cumulative Returns: Strategy vs Buy & Hold (2022–2023)")
ax1.set_ylabel("Portfolio Value (starting at 1.0)")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Drawdown chart
ax2 = fig.add_subplot(gs[1, 0])
strat_roll_max  = df["strategy_cumulative"].cummax()
strat_dd_series = (df["strategy_cumulative"] - strat_roll_max) / strat_roll_max
bh_roll_max     = df["buyhold_cumulative"].cummax()
bh_dd_series    = (df["buyhold_cumulative"] - bh_roll_max) / bh_roll_max
ax2.fill_between(df.index, strat_dd_series * 100, 0,
                 color="#e94f37", alpha=0.5, label="Strategy")
ax2.fill_between(df.index, bh_dd_series * 100, 0,
                 color="#1a7abf", alpha=0.3, label="Buy & Hold")
ax2.set_title("Drawdown Over Time")
ax2.set_ylabel("Drawdown (%)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Daily strategy returns distribution
ax3 = fig.add_subplot(gs[1, 1])
traded = df[df["signal"] == 1]["strategy_return"]
ax3.hist(traded * 100, bins=40, color="#27ae60", alpha=0.75, edgecolor="white")
ax3.axvline(0, color="red", linewidth=1, linestyle="--")
ax3.axvline(traded.mean() * 100, color="orange", linewidth=1.5,
            linestyle="--", label=f"Mean: {traded.mean()*100:.2f}%")
ax3.set_title("Distribution of Daily Returns (Traded Days)")
ax3.set_xlabel("Daily Return (%)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Signal activity over time
ax4 = fig.add_subplot(gs[2, 0])
ax4.fill_between(df.index, df["signal"], 0,
                 color="#1a7abf", alpha=0.4, label="In market (BUY)")
ax4.plot(df.index, df["signal"], color="#1a7abf", linewidth=0.5)
ax4.set_title("Signal Activity (1=In Market, 0=Cash)")
ax4.set_yticks([0, 1])
ax4.set_yticklabels(["Cash", "In Market"])
ax4.grid(True, alpha=0.3)

# Plot 5: Monthly returns heatmap-style bar
ax5 = fig.add_subplot(gs[2, 1])
df["month"] = df.index.to_period("M")
monthly = df.groupby("month")["strategy_return"].sum() * 100
colors_m = ["#27ae60" if v >= 0 else "#e94f37" for v in monthly]
monthly.plot(kind="bar", ax=ax5, color=colors_m, alpha=0.85, width=0.8)
ax5.axhline(0, color="black", linewidth=0.8)
ax5.set_title("Monthly Strategy Returns (%)")
ax5.set_xlabel("")
ax5.tick_params(axis="x", labelsize=7, rotation=45)
ax5.grid(True, alpha=0.3, axis="y")

plt.suptitle("AAPL — Strategy Construction & Performance", fontsize=15, y=1.01)
plt.savefig("data/aapl_strategy_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to data/aapl_strategy_chart.png")
print("\nPhase 4 complete. Ready for Phase 5 — Backtesting & Evaluation!")