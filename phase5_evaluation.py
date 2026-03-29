# ============================================================
# PHASE 5 & 6 — Backtesting + Full Evaluation Report
# Complete portfolio-ready summary of the entire project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# --- 1. Load all data ---
# Load strategy CSV — flatten multi-level columns automatically
df = pd.read_csv("data/aapl_strategy.csv", index_col=0, header=0)
df.index = pd.to_datetime(df.index)

# Flatten multi-level column names if present (e.g. "('Close', 'AAPL')" → "Close")
def flatten_col(c):
    if isinstance(c, tuple):
        return c[0]
    c = str(c)
    if c.startswith("("):
        import ast
        try:
            return ast.literal_eval(c)[0]
        except:
            pass
    return c

df.columns = [flatten_col(c) for c in df.columns]

# Deduplicate column names (keep first occurrence)
df = df.loc[:, ~df.columns.duplicated()]
df = df.sort_index()

# Print columns so we can verify
print("Columns loaded:", list(df.columns))

for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    except:
        pass
df = df.dropna(subset=["Close", "strategy_return", "buyhold_return"])

print(f"Loaded {len(df)} rows.")
print(f"Backtest period: {df.index[0].date()} → {df.index[-1].date()}")

# --- 2. Recalculate cumulative returns cleanly ---
df["strategy_cumulative"] = (1 + df["strategy_return"]).cumprod()
df["buyhold_cumulative"]  = (1 + df["buyhold_return"]).cumprod()

# --- 3. Full metrics calculation ---
trading_days = len(df)
years        = trading_days / 252
RISK_FREE    = 0.04

def calc_metrics(returns, cumulative, label):
    total      = cumulative.iloc[-1] - 1
    annual     = (1 + total) ** (1 / years) - 1
    vol        = returns.std() * np.sqrt(252)
    sharpe     = (annual - RISK_FREE) / vol if vol > 0 else 0
    roll_max   = cumulative.cummax()
    drawdown   = (cumulative - roll_max) / roll_max
    mdd        = drawdown.min()
    pos_days   = (returns > 0).sum()
    neg_days   = (returns < 0).sum()
    win_rate   = pos_days / (pos_days + neg_days) if (pos_days + neg_days) > 0 else 0
    best_day   = returns.max()
    worst_day  = returns.min()
    avg_return = returns[returns != 0].mean()
    return {
        "label"      : label,
        "total"      : total,
        "annual"     : annual,
        "vol"        : vol,
        "sharpe"     : sharpe,
        "mdd"        : mdd,
        "win_rate"   : win_rate,
        "best_day"   : best_day,
        "worst_day"  : worst_day,
        "avg_return" : avg_return,
        "drawdown"   : drawdown,
    }

strat_m = calc_metrics(df["strategy_return"], df["strategy_cumulative"], "RF Strategy")
bh_m    = calc_metrics(df["buyhold_return"],  df["buyhold_cumulative"],  "Buy & Hold")

# --- 4. Rolling metrics ---
df["rolling_sharpe"] = (
    df["strategy_return"].rolling(60).mean() /
    df["strategy_return"].rolling(60).std()
) * np.sqrt(252)

df["rolling_vol_strat"] = df["strategy_return"].rolling(30).std() * np.sqrt(252)
df["rolling_vol_bh"]    = df["buyhold_return"].rolling(30).std()  * np.sqrt(252)

# --- 5. Monthly breakdown ---
df["month_period"] = df.index.to_period("M")
monthly_strat = df.groupby("month_period")["strategy_return"].sum() * 100
monthly_bh    = df.groupby("month_period")["buyhold_return"].sum()  * 100
monthly_diff  = monthly_strat - monthly_bh

# --- 6. Print full report ---
print(f"\n{'='*60}")
print(f"   QUANTITATIVE TRADING MODEL — FULL EVALUATION REPORT")
print(f"   Asset: AAPL  |  Period: 2022-01-13 to 2023-12-29")
print(f"   Model: Random Forest Classifier")
print(f"{'='*60}")
print(f"\n  {'Metric':<28} {'RF Strategy':>12} {'Buy & Hold':>12}")
print(f"  {'-'*54}")
print(f"  {'Total Return':<28} {strat_m['total']*100:>11.2f}% {bh_m['total']*100:>11.2f}%")
print(f"  {'Annualised Return':<28} {strat_m['annual']*100:>11.2f}% {bh_m['annual']*100:>11.2f}%")
print(f"  {'Annualised Volatility':<28} {strat_m['vol']*100:>11.2f}% {bh_m['vol']*100:>11.2f}%")
print(f"  {'Sharpe Ratio':<28} {strat_m['sharpe']:>12.3f} {bh_m['sharpe']:>12.3f}")
print(f"  {'Max Drawdown':<28} {strat_m['mdd']*100:>11.2f}% {bh_m['mdd']*100:>11.2f}%")
print(f"  {'Win Rate':<28} {strat_m['win_rate']*100:>11.2f}% {bh_m['win_rate']*100:>11.2f}%")
print(f"  {'Best Single Day':<28} {strat_m['best_day']*100:>11.2f}% {bh_m['best_day']*100:>11.2f}%")
print(f"  {'Worst Single Day':<28} {strat_m['worst_day']*100:>11.2f}% {bh_m['worst_day']*100:>11.2f}%")
print(f"  {'Avg Return (active days)':<28} {strat_m['avg_return']*100:>11.3f}% {bh_m['avg_return']*100:>11.3f}%")
print(f"  {'Days in Market':<28} {int((df['signal']==1).sum()):>12} {trading_days:>12}")
print(f"{'='*60}")

# --- 7. Risk-adjusted verdict ---
print(f"\n  RISK-ADJUSTED VERDICT:")
if strat_m["vol"] < bh_m["vol"]:
    vol_reduction = (1 - strat_m["vol"] / bh_m["vol"]) * 100
    print(f"  ✓ Strategy reduced volatility by {vol_reduction:.1f}%")
if abs(strat_m["mdd"]) < abs(bh_m["mdd"]):
    dd_reduction = (1 - abs(strat_m["mdd"]) / abs(bh_m["mdd"])) * 100
    print(f"  ✓ Strategy reduced max drawdown by {dd_reduction:.1f}%")
if strat_m["win_rate"] > 0.5:
    print(f"  ✓ Strategy win rate above 50%")
else:
    print(f"  ✗ Strategy win rate below 50% — model struggles in bear markets")
print(f"\n  CONCLUSION: The strategy prioritises capital preservation over")
print(f"  raw returns. In a bear market (2022), selective trading with a")
print(f"  stop-loss reduced drawdown significantly at the cost of upside.")

# ============================================================
# MASTER VISUALISATION — 6 panel portfolio-ready chart
# ============================================================
fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)

# --- Panel 1: Cumulative returns ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df["strategy_cumulative"] * 100 - 100,
         color="#27ae60", linewidth=2.2, label="RF Strategy", zorder=3)
ax1.plot(df.index, df["buyhold_cumulative"] * 100 - 100,
         color="#1a7abf", linewidth=2.2, label="Buy & Hold", zorder=2)
ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax1.fill_between(df.index,
                 df["strategy_cumulative"] * 100 - 100,
                 df["buyhold_cumulative"]  * 100 - 100,
                 where=df["strategy_cumulative"] >= df["buyhold_cumulative"],
                 alpha=0.12, color="#27ae60", label="Strategy outperforms")
ax1.fill_between(df.index,
                 df["strategy_cumulative"] * 100 - 100,
                 df["buyhold_cumulative"]  * 100 - 100,
                 where=df["strategy_cumulative"] < df["buyhold_cumulative"],
                 alpha=0.10, color="#e94f37", label="Strategy underperforms")
# Annotate final values
ax1.annotate(f"Strategy: {strat_m['total']*100:.1f}%",
             xy=(df.index[-1], strat_m['total']*100),
             xytext=(-110, 10), textcoords="offset points",
             fontsize=10, color="#27ae60", fontweight="bold")
ax1.annotate(f"Buy & Hold: {bh_m['total']*100:.1f}%",
             xy=(df.index[-1], bh_m['total']*100),
             xytext=(-120, -18), textcoords="offset points",
             fontsize=10, color="#1a7abf", fontweight="bold")
ax1.set_title("Cumulative Return: RF Strategy vs Buy & Hold", fontsize=13)
ax1.set_ylabel("Total Return (%)")
ax1.legend(fontsize=9, loc="lower left")
ax1.grid(True, alpha=0.3)

# --- Panel 2: Drawdown ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(df.index, strat_m["drawdown"] * 100, 0,
                 color="#e94f37", alpha=0.6, label=f"Strategy (max {strat_m['mdd']*100:.1f}%)")
ax2.fill_between(df.index, bh_m["drawdown"] * 100, 0,
                 color="#1a7abf", alpha=0.25, label=f"Buy & Hold (max {bh_m['mdd']*100:.1f}%)")
ax2.set_title("Drawdown Comparison", fontsize=11)
ax2.set_ylabel("Drawdown (%)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Rolling volatility ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df.index, df["rolling_vol_strat"] * 100,
         color="#27ae60", linewidth=1.5, label="Strategy (30d)")
ax3.plot(df.index, df["rolling_vol_bh"] * 100,
         color="#1a7abf", linewidth=1.5, label="Buy & Hold (30d)", alpha=0.7)
ax3.set_title("Rolling 30-Day Volatility (Annualised)", fontsize=11)
ax3.set_ylabel("Volatility (%)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- Panel 4: Monthly returns comparison ---
ax4 = fig.add_subplot(gs[2, 0])
x     = np.arange(len(monthly_strat))
width = 0.38
ax4.bar(x - width/2, monthly_strat.values, width=width,
        color=["#27ae60" if v >= 0 else "#e94f37" for v in monthly_strat],
        alpha=0.85, label="Strategy")
ax4.bar(x + width/2, monthly_bh.values, width=width,
        color=["#1a7abf" if v >= 0 else "#c0392b" for v in monthly_bh],
        alpha=0.55, label="Buy & Hold")
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_title("Monthly Returns: Strategy vs Buy & Hold (%)", fontsize=11)
ax4.set_xticks(x[::3])
ax4.set_xticklabels([str(p) for p in monthly_strat.index[::3]],
                     rotation=45, fontsize=7)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="y")

# --- Panel 5: Summary scorecard ---
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis("off")
scorecard_data = [
    ["Metric",             "Strategy",                      "Buy & Hold"],
    ["Total Return",       f"{strat_m['total']*100:.1f}%",  f"{bh_m['total']*100:.1f}%"],
    ["Annual Return",      f"{strat_m['annual']*100:.1f}%", f"{bh_m['annual']*100:.1f}%"],
    ["Volatility",         f"{strat_m['vol']*100:.1f}%",    f"{bh_m['vol']*100:.1f}%"],
    ["Sharpe Ratio",       f"{strat_m['sharpe']:.3f}",      f"{bh_m['sharpe']:.3f}"],
    ["Max Drawdown",       f"{strat_m['mdd']*100:.1f}%",    f"{bh_m['mdd']*100:.1f}%"],
    ["Win Rate",           f"{strat_m['win_rate']*100:.1f}%", f"{bh_m['win_rate']*100:.1f}%"],
    ["Days in Market",     f"{int((df['signal']==1).sum())}", f"{trading_days}"],
]
table = ax5.table(cellText=scorecard_data[1:],
                  colLabels=scorecard_data[0],
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.1, 1.9)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f8f9fa")
    else:
        cell.set_facecolor("white")
ax5.set_title("Performance Scorecard", fontsize=11, pad=12)

plt.suptitle("AAPL Quantitative Trading Model — Full Backtest Report\n"
             "Random Forest Strategy vs Buy & Hold | Jan 2022 – Dec 2023",
             fontsize=14, y=1.01)
plt.savefig("data/aapl_full_report.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nMaster chart saved to data/aapl_full_report.png")
print("\n" + "="*60)
print("  ALL PHASES COMPLETE!")
print("  Your project files:")
print("    data/aapl_raw.csv             — Phase 1 raw data")
print("    data/aapl_features.csv        — Phase 2 engineered features")
print("    data/aapl_predictions.csv     — Phase 3 model predictions")
print("    data/aapl_strategy.csv        — Phase 4 strategy signals")
print("    data/aapl_price_chart.png     — Phase 1 chart")
print("    data/aapl_features_chart.png  — Phase 2 chart")
print("    data/aapl_model_results.png   — Phase 3 chart")
print("    data/aapl_strategy_chart.png  — Phase 4 chart")
print("    data/aapl_full_report.png     — Phase 5 master chart")
print("="*60)