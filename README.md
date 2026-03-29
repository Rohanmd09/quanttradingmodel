# 📈 AI-Driven Quantitative Trading Model

A machine learning-based trading system that predicts next-day stock direction and demonstrates how **risk management and strategy design impact real-world performance more than raw prediction accuracy**.

---

## 🚀 Overview

This project implements an end-to-end quantitative trading pipeline using Apple Inc. (AAPL) historical data.

It combines:

* Machine learning (Logistic Regression, Random Forest)
* Feature engineering (trend, momentum, volatility, volume)
* Strategy design (confidence threshold, stop-loss)
* Backtesting and risk evaluation

The goal is not to “beat the market,” but to **understand how predictive signals translate into actual portfolio outcomes under realistic constraints**.

---

## 🎯 Key Results

* **Model Accuracy:** 56.5% (out-of-sample)
* **Strategy Return:** -2.3%
* **Buy & Hold Return:** +11.4%

### 📊 Risk Metrics

* **Volatility Reduced:** -60.1%
* **Max Drawdown Reduced:** -55.0%
* **Worst Daily Loss:** -1.83% vs -7.83%

> 💡 **Key Insight:**
> A small predictive edge (~56%) does not guarantee profitability.
> Strategy design, risk control, and market regime have a far greater impact on outcomes.

---

## 🧠 Project Architecture

The system is built in 5 structured phases:

### 1. Data Collection

* Download historical AAPL data using `yfinance`

### 2. Feature Engineering

* Transform raw price data into signals:

  * Returns
  * Moving Average Spread
  * Volatility
  * RSI
  * Volume Change
  * Price Position

### 3. Model Training

* Logistic Regression (baseline)
* Random Forest (final model)

### 4. Strategy Construction

* Trade only when model confidence > 0.55
* Long-only strategy (no short-selling)
* Stop-loss at -2%

### 5. Backtesting & Evaluation

* Compare against buy-and-hold
* Evaluate risk-adjusted performance

---

## 📂 Project Structure

```
src/
 ├── phase1_data.py
 ├── phase2_features.py
 ├── phase3_models.py
 ├── phase4_strategy.py
 └── phase5_evaluation.py

data/
outputs/
report/
```

Each phase represents a step in the quantitative pipeline, making the project modular and easy to follow.

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Phases
Or run each phase manually:

```bash
python src/phase1_data.py
python src/phase2_features.py
python src/phase3_models.py
python src/phase4_strategy.py
python src/phase5_evaluation.py
```

---

## Key Learnings

* Even small predictive signals (~56%) can influence outcomes
* Risk management (stop-loss, exposure control) is critical
* Strategy design matters more than model complexity
* Volume-based signals were more important than expected
* Avoiding overfitting is more valuable than chasing accuracy

---

## Limitations

* No transaction costs or slippage
* Single asset (AAPL only)
* Limited feature set
* Short test period (2022–2023)

---

## Future Improvements

* Add multi-asset portfolio
* Implement XGBoost / LightGBM
* Introduce regime detection
* Include transaction cost modeling
* Expand feature set (macro, sentiment, options data)

---

## Full Report

A detailed breakdown of methodology, results, and reasoning is available in:

```
report/AAPL_Quant_Report_Final.pdf
```

---

## Author

Rohan
Quantitative Finance Project (2026)

---

## Final Thought

This project demonstrates that:

> **The hardest part of quantitative investing is not building a predictive model — it is designing a strategy that turns predictions into robust, real-world performance.**
