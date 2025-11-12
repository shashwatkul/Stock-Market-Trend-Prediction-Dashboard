import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import date

# Define stock symbols (you can choose your favorites)
stocks = ['TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'RELIANCE.NS', 'ICICIBANK.NS']

# Download 5-year daily data
end_date = date.today()
data = yf.download(stocks, start='2015-01-01', end=end_date)['Close']

# Create folders if they don’t exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Save raw data
data.to_csv('data/raw/stocks_raw.csv')

# Calculate daily returns
returns = data.pct_change().dropna()

# Correlation matrix
corr = returns.corr()

# Save cleaned data
returns.to_csv('data/processed/stocks_cleaned.csv')
import matplotlib.dates as mdates

# --- Visualizations with exact dates ---
plt.figure(figsize=(12,6))
for stock in stocks:
    plt.plot(data.index, data[stock], label=stock)

plt.title('📈 Stock Price Trends (2015–2025)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Format X-axis to show proper dates
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # major ticks every year
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,7)))  # minor ticks every Jan/July
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # format as "Jan 2020"

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Stock Returns Correlation')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Choose one stock to predict (e.g., RELIANCE)
target_stock = 'RELIANCE.NS'
df = data[[target_stock]].copy()
df['Target'] = df[target_stock].shift(-1)

# =======================================================
# 🎯 FEATURE ENGINEERING (Enhanced)
# =======================================================

# Create features for target stock
df = data[[target_stock]].copy()
df.rename(columns={target_stock: "Close"}, inplace=True)

# Daily Return
df["Return"] = df["Close"].pct_change()

# Moving Averages
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()
df["MA14"] = df["Close"].rolling(window=14).mean()
df["MA30"] = df["Close"].rolling(window=30).mean()

# Rolling Volatility (30-day std deviation of returns)
df["Volatility"] = df["Return"].rolling(window=30).std()

# Target Variable → Next Day Close
df["Target"] = df["Close"].shift(-1)

# Drop NaN rows
df = df.dropna()

print("✅ Features created:", df.columns.tolist())

# =======================================================
# ✂️ TRAIN-TEST SPLIT
# =======================================================

# Use all engineered features
features = ["Close", "Return", "MA5", "MA10", "MA14", "MA30", "Volatility"]
X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

# =======================================================
# 🧠 MODEL TRAINING (Linear Regression)
# =======================================================

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n🔍 Model Evaluation for {target_stock}:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# =======================================================
# 📊 VISUALIZATION
# =======================================================

plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red', alpha=0.7)
plt.title(f'{target_stock} - Actual vs Predicted Closing Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# =======================================================
# 🔮 5-Day Iterative Forecast (Using All 7 Features)
# =======================================================
import numpy as np

last_data = df.iloc[-60:].copy()  # last 60 days to compute rolling features
future_predictions = []

for i in range(5):
    # Compute rolling features based on the latest data
    close = last_data["Close"].iloc[-1]  # last known close
    returns = last_data["Close"].pct_change()

    ma5 = last_data["Close"].tail(5).mean()
    ma10 = last_data["Close"].tail(10).mean()
    ma14 = last_data["Close"].tail(14).mean()
    ma30 = last_data["Close"].tail(30).mean()
    vol = returns.tail(30).std()

    # Prepare the next input feature vector
    X_future = np.array([[close, returns.iloc[-1], ma5, ma10, ma14, ma30, vol]])

    # Predict next day's closing price
    next_pred = model.predict(X_future)[0]
    future_predictions.append(next_pred)

    # Append predicted value to rolling window for next iteration
    new_row = {
        "Close": next_pred,
        "Return": (next_pred - close) / close,
        "MA5": ma5,
        "MA10": ma10,
        "MA14": ma14,
        "MA30": ma30,
        "Volatility": vol,
        "Target": np.nan
    }

    last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)

# ---- Display Forecast Results ----
print("\n📈 Next 5-Day Predicted Prices for", target_stock)
for i, p in enumerate(future_predictions, 1):
    print(f"Day +{i}: ₹{p:.2f}")

# ---- Plot Forecast ----
plt.figure(figsize=(10,6))
plt.plot(df["Close"].tail(60).values, label="Recent Actual Prices", color="blue")
plt.plot(range(60, 65), future_predictions, label="5-Day Forecast", color="red", marker="o")
plt.title(f"{target_stock} - Next 5 Day Price Forecast")
plt.xlabel("Days")
plt.ylabel("Price (INR)")
plt.legend()
plt.show()
