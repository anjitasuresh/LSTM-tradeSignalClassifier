import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Load data
df = yf.download('AAPL', start='2020-01-01', end='2024-12-31')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

import matplotlib.pyplot as plt

def plot_price_with_sma(df, ticker):
    plt.figure(figsize=(14, 5))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['sma_20'], label='SMA 20', linestyle='--')
    plt.plot(df['sma_50'], label='SMA 50', linestyle='--')
    plt.title(f"{ticker} Closing Price with SMA")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_price_with_sma(stock_data['AAPL'], 'AAPL')


def plot_rsi(df, ticker):
    plt.figure(figsize=(12, 3))
    plt.plot(df['rsi_14'], label='RSI 14')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title(f"{ticker} RSI (Relative Strength Index)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rsi(stock_data['AAPL'], 'AAPL')

import seaborn as sns

def plot_label_distribution(df, ticker):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title(f"{ticker} - Label Distribution (Buy=1, Hold=0, Sell=-1)")
    plt.xlabel("Signal")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

plot_label_distribution(stock_data['AAPL'], 'AAPL')

import matplotlib.pyplot as plt

def plot_price_with_sma(df, ticker):
    plt.figure(figsize=(14, 5))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['sma_20'], label='SMA 20', linestyle='--')
    plt.plot(df['sma_50'], label='SMA 50', linestyle='--')
    plt.title(f"{ticker} Closing Price with SMA")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_price_with_sma(stock_data['TSLA'], 'TSLA')

def plot_rsi(df, ticker):
    plt.figure(figsize=(12, 3))
    plt.plot(df['rsi_14'], label='RSI 14')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title(f"{ticker} RSI (Relative Strength Index)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rsi(stock_data['TSLA'], 'TSLA')

import seaborn as sns

def plot_label_distribution(df, ticker):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title(f"{ticker} - Label Distribution (Buy=1, Hold=0, Sell=-1)")
    plt.xlabel("Signal")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

plot_label_distribution(stock_data['TSLA'], 'TSLA')

# 2. Add features
df['price_change'] = df['Close'].pct_change()
df['volatility_5'] = df['price_change'].rolling(window=5).std()
df['sma_20'] = df['Close'].rolling(window=20).mean()
df.dropna(inplace=True)

# 3. Scale features and target
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'price_change', 'volatility_5', 'sma_20']
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df[feature_cols])

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(df['Close'].shift(-1).dropna().values.reshape(-1, 1))

# Match y to X (remove last row from X to match y length)
X_scaled = X_scaled[:-1]

# 4. Convert to sequences
def create_sequences(X, y, window_size=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size=30)

# 5. Train/test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# 6. LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 7. Predict and unscale
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# 8. Plot
plt.figure(figsize=(12, 4))
plt.plot(y_test_unscaled[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.legend()
plt.title("LSTM: Actual vs Predicted Closing Price")
plt.grid(True)
plt.show()

# After model.predict():
y_pred_scaled = model.predict(X_test)

# Inverse transform to get actual prices
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
mae = mean_absolute_error(y_test_true, y_pred)
r2 = r2_score(y_test_true, y_pred)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# Get the last day's features from each sequence
last_day_features = X_test[:, -1, :]  # shape: (samples, features)

# Unscale the last day's features to get real prices
last_day_features_unscaled = scaler_X.inverse_transform(last_day_features)

# Extract the actual close price from each last day
close_today_unscaled = last_day_features_unscaled[:, feature_cols.index('Close')]

predicted_return = (y_pred.flatten() / close_today_unscaled) - 1

def signal_from_return(r):
    if r > 0.02:      # Only BUY if return > 2%
        return 1
    elif r < -0.02:   # Only SELL if return < -2%
        return -1
    else:
        return 0      # Otherwise HOLD
signals = np.array([signal_from_return(r) for r in predicted_return])

import matplotlib.pyplot as plt
import numpy as np

# Create signal DataFrame
signal_df = pd.DataFrame({
    'Date': df.index[-len(signals):],
    'Close': y_test_true.flatten(),
    'Predicted_Close': y_pred.flatten(),
    'Signal': signals
})

# Plot actual close price
plt.figure(figsize=(14, 6))
plt.plot(signal_df['Date'], signal_df['Close'], label='Actual Close', color='blue')

# Plot buy signals
buy_signals = signal_df[signal_df['Signal'] == 1]
plt.scatter(buy_signals['Date'], buy_signals['Close'], label='Buy', marker='^', color='green', s=100)

# Plot sell signals
sell_signals = signal_df[signal_df['Signal'] == -1]
plt.scatter(sell_signals['Date'], sell_signals['Close'], label='Sell', marker='v', color='red', s=100)

# Optional: plot predicted prices
plt.plot(signal_df['Date'], signal_df['Predicted_Close'], label='Predicted Close', color='orange', linestyle='--')

# Final touches
plt.legend()
plt.grid(True)
plt.title("LSTM Signals on Actual Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# Parameters
initial_capital = 100000
cooldown_days = 5
max_holding_days = 10
sl_pct = 0.02   # 2% Stop-Loss
tp_pct = 0.06   # 4% Take-Profit

# Initialization
cash = initial_capital
position = 0
entry_day = -1
entry_price = 0
cooldown = 0
equity_curve = []
trades = []

df_bt = signal_df.copy().reset_index(drop=True)

# Backtest loop
for i in range(len(df_bt)):
    price = df_bt.loc[i, 'Close']
    signal = df_bt.loc[i, 'Signal']

    # Check SL/TP only if holding a position
    if position > 0:
        # Stop-loss
        if price < entry_price * (1 - sl_pct):
            cash += position * price
            trades.append(('SL-SELL', df_bt.loc[i, 'Date'], price))
            print(f"🛑 STOP-LOSS SELL on {df_bt.loc[i, 'Date']} at ₹{price:.2f}")
            position = 0
            cooldown = cooldown_days
        # Take-profit
        elif price > entry_price * (1 + tp_pct):
            cash += position * price
            trades.append(('TP-SELL', df_bt.loc[i, 'Date'], price))
            print(f"🎯 TAKE-PROFIT SELL on {df_bt.loc[i, 'Date']} at ₹{price:.2f}")
            position = 0
            cooldown = cooldown_days

    # Auto-exit if holding too long
    if position > 0 and (i - entry_day) >= max_holding_days:
        cash += position * price
        trades.append(('AUTO-SELL', df_bt.loc[i, 'Date'], price))
        print(f"🚨 AUTO-SELL on {df_bt.loc[i, 'Date']} at ₹{price:.2f}")
        position = 0
        cooldown = cooldown_days

    # Buy logic
    if signal == 1 and position == 0 and cooldown == 0:
        position = cash // price
        cash -= position * price
        entry_day = i
        entry_price = price
        trades.append(('BUY', df_bt.loc[i, 'Date'], price))
        print(f"✅ BUY on {df_bt.loc[i, 'Date']} at ₹{price:.2f} (Shares: {position})")

    # Manual Sell logic (on LSTM signal)
    elif signal == -1 and position > 0:
        cash += position * price
        trades.append(('SELL', df_bt.loc[i, 'Date'], price))
        print(f"🔻 SIGNAL-SELL on {df_bt.loc[i, 'Date']} at ₹{price:.2f}")
        position = 0
        cooldown = cooldown_days

    # Cooldown logic
    if cooldown > 0:
        cooldown -= 1

    # Track equity
    equity_curve.append(cash + position * price)

df_bt['Equity'] = equity_curve

# Calculate total return
total_return = (df_bt['Equity'].iloc[-1] / initial_capital - 1) * 100

# Calculate CAGR
days = (df_bt['Date'].iloc[-1] - df_bt['Date'].iloc[0]).days
cagr = ((df_bt['Equity'].iloc[-1] / initial_capital) ** (365 / days) - 1) * 100

# Calculate max drawdown
rolling_max = df_bt['Equity'].cummax()
drawdown = df_bt['Equity'] / rolling_max - 1
max_drawdown = drawdown.min() * 100

# Print summary
print("\n📊 Performance Metrics:")
print(f"✅ Total Return: {total_return:.2f}%")
print(f"📈 CAGR: {cagr:.2f}%")
print(f"📉 Max Drawdown: {max_drawdown:.2f}%")

import numpy as np
import pandas as pd

# Recreate trade DataFrame (if not already)
trade_df = pd.DataFrame(trades, columns=['Type', 'Date', 'Price'])

# Pair up BUY with SELL/AUTO-SELL
profits = []
for i in range(1, len(trade_df), 2):
    if trade_df.iloc[i]['Type'] in ['SELL', 'AUTO-SELL']:
        entry = trade_df.iloc[i - 1]['Price']
        exit = trade_df.iloc[i]['Price']
        profit = exit - entry
        profits.append(profit)

# Convert to array
profits = np.array(profits)

# Separate wins and losses
wins = profits[profits > 0]
losses = profits[profits < 0]

# Calculate trade stats
avg_win = wins.mean() if len(wins) > 0 else 0
avg_loss = losses.mean() if len(losses) > 0 else 0
profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.inf

# Output
print("\n📊 Trade Performance Summary:")
print(f"✅ Average Winning Trade: ₹{avg_win:.2f}")
print(f"❌ Average Losing Trade: ₹{avg_loss:.2f}")
print(f"⚖️ Profit Factor: {profit_factor:.2f}")
print(f"📈 Total Closed Trades: {len(profits)}")

# Calculate daily returns from equity curve
df_bt['Daily_Return'] = df_bt['Equity'].pct_change().fillna(0)

# Sharpe Ratio (Assuming 0% risk-free rate)
sharpe_ratio = (df_bt['Daily_Return'].mean() / df_bt['Daily_Return'].std()) * np.sqrt(252)

# Win Rate
trade_df = pd.DataFrame(trades, columns=['Type', 'Date', 'Price'])
profits = []
for i in range(1, len(trade_df), 2):
    if trade_df.iloc[i]['Type'] in ['SELL', 'AUTO-SELL']:
        entry = trade_df.iloc[i - 1]['Price']
        exit = trade_df.iloc[i]['Price']
        profits.append(exit - entry)

win_rate = (np.array(profits) > 0).sum() / len(profits) * 100 if profits else 0

# Print
print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"✅ Win Rate: {win_rate:.2f}% ({len(profits)} closed trades)")

print("\n--- Final Summary ---")
print(f"💰 Final Cash: ₹{cash:.2f}")
print(f"📦 Final Position (shares): {position}")
print(f"📈 Final Portfolio Value: ₹{equity_curve[-1]:.2f}")
print(f"🔁 Total Trades Executed: {len(trades)}")

# Optional: display last few trades
import pandas as pd
pd.DataFrame(trades, columns=['Type', 'Date', 'Price']).tail(10)

import matplotlib.pyplot as plt

# Convert trade list to DataFrame
trade_df = pd.DataFrame(trades, columns=["Type", "Date", "Price"])

# Plot Close price
plt.figure(figsize=(14, 6))
plt.plot(df_bt['Date'], df_bt['Close'], label='Close Price', color='blue', linewidth=1.5)

# Plot each type of trade
for label, color, marker in [
    ('BUY', 'green', '^'),
    ('SELL', 'red', 'v'),
    ('SL-SELL', 'black', 'x'),
    ('TP-SELL', 'gold', '*'),
    ('AUTO-SELL', 'orange', 'P')
]:
    subset = trade_df[trade_df['Type'] == label]
    plt.scatter(subset['Date'], subset['Price'], label=label, color=color, marker=marker, s=120)

# Final touches
plt.title("Trade Entry and Exit Points (SL/TP/Signal)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
