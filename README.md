# LSTM-tradeSignalClassifier
Predicts profitable stock trade opportunities using LSTM on OHLCV data. Includes trade simulation, backtesting, and performance evaluation.


# LSTM-Based Stock Trade Signal Classifier

This project uses **Long Short-Term Memory (LSTM)** neural networks to classify whether a stock price sequence leads to a **profitable trade** ("Trade") or **non-profitable** ("No Trade"). It combines deep learning with backtesting logic to generate actionable signals based on past price movement.

---

## Problem Statement

Given historical stock data (OHLCV), the model learns patterns that predict whether the next few days offer a good opportunity to enter a trade. This helps in:
- Reducing noise from day-to-day volatility
- Improving the Sharpe Ratio by filtering bad trades
- Making systematic trading decisions using ML

---

##  Project Features

-  **Sequence Modeling** using LSTM to learn time-based price patterns  
-  **Trade Simulation Logic** using rolling windows, stop-loss, and take-profit  
-  **Binary Classifier**: `1 = Trade`, `0 = No Trade`  
-  **Backtesting**: Visualizes trades and compares model vs. baseline  
-  **Performance Metrics**: Win rate, Sharpe Ratio, drawdown

---

## 🗂 Project Structure

trade-signal-classifier/
├── data/
│ └── stock_data.csv # Raw OHLCV data (e.g., from yFinance)
├── src/
│ ├── features.py # Moving averages, volatility, price change
│ ├── train_lstm.py # LSTM model training and evaluation
│ ├── trade_labeler.py # Labels trade vs. no-trade windows
│ └── backtest.py # Backtesting logic with trade filters
├── requirements.txt
└── README.md
