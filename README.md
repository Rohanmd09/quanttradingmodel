#  AI-Driven Quantitative Trading Model

A machine learning-based trading system that predicts next-day stock direction and evaluates performance under real-world constraints.

---

##  Overview

This project implements an end-to-end quantitative trading pipeline using Apple Inc. (AAPL) historical data.

The model applies supervised machine learning to generate predictive signals, which are then translated into a rule-based trading strategy and evaluated through backtesting.

---

##  Key Results

* **Model Accuracy:** 56.5% (out-of-sample)
* **Volatility Reduction:** -60.1%
* **Max Drawdown Reduction:** -55.0%
* **Strategy Return:** -2.3% vs +11.4% (Buy & Hold)

> Key Insight: Predictive accuracy alone is not sufficient — strategy design and risk management determine real-world performance.

---

##  Project Architecture

The project is built in 5 structured phases:

1. **Data Collection**

   * Download historical price data using `yfinance`

2. **Feature Engineering**

   * Create signals (returns, volatility, RSI, MA spread, etc.)

3. **Model Training**

   * Logistic Regression (baseline)
   * Random Forest (final model)

4. **Strategy Construction**

   * Convert predictions into trading signals
   * Apply confidence threshold and stop-loss

5. **Backtesting & Evaluation**

   * Compare strategy vs buy-and-hold
   * Analyze risk-adjusted performance

---

##  Project Structure

```
src/
 ├── phase1_data.py
 ├── phase2_features.py
 ├── phase3_models.py
 ├── phase4_strategy.py
 └── phase5_evaluation.py
```

Each phase represents a distinct step in the quantitative pipeline.

---

##  Installation

```bash
pip install -r requirements.txt
```

---

##  Running the Project

Run each phase sequentially:

```bash
python src/phase1_data.py
python src/phase2_features.py
python src/phase3_models.py
python src/phase4_strategy.py
python src/phase5_evaluation.py
```

---

##  Key Insights

* Small predictive edges (~56%) can meaningfully impact outcomes
* Risk management (stop-loss, exposure control) is critical
* Strategy design matters as much as model accuracy
* Volume-based signals were more influential than expected

---

##  Limitations

* No transaction costs or slippage
* Single asset (AAPL only)
* Limited feature set
* Short evaluation window

---

##  Report

Full detailed report available in.

---

##  Author

Rohan MD
Quantitative Finance Project (2026)

---

##  Future Improvements

* Add multi-asset portfolio
* Implement XGBoost / LightGBM
* Introduce regime detection
* Incorporate transaction costs

---

