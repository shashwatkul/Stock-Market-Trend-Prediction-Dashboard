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
data = yf.download(stocks, start='2020-01-01', end=end_date)['Close']

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

# Visualizations
plt.figure(figsize=(10,6))
for stock in stocks:
    plt.plot(data[stock], label=stock)
plt.title('Stock Price Trends (2024)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
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

# Features — past prices and moving averages
df['MA5'] = df[target_stock].rolling(window=5).mean()
df['MA10'] = df[target_stock].rolling(window=10).mean()
df = df.dropna()

# Define X and y
X = df[['MA5', 'MA10']]
y = df['Target']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation for {target_stock}:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot predictions vs actual
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.title(f'{target_stock} - Actual vs Predicted Closing Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()


import numpy as np

# ---- 5-Day Iterative Forecast ----
last_data = df.iloc[-10:].copy()  # take last few rows to start prediction window
future_predictions = []

for i in range(5):
    ma5 = last_data[target_stock].tail(5).mean()
    ma10 = last_data[target_stock].tail(10).mean()

    X_future = np.array([[ma5, ma10]])
    next_pred = model.predict(X_future)[0]

    # Append predicted value to keep rolling window
    future_predictions.append(next_pred)
    new_row = {target_stock: next_pred, 'MA5': ma5, 'MA10': ma10, 'Target': np.nan}
    last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)

# ---- Display Forecast Results ----
print("\nNext 5-Day Predicted Prices for", target_stock)
for i, p in enumerate(future_predictions, 1):
    print(f"Day +{i}: {p:.2f}")

# ---- Plot Prediction Extension ----
plt.figure(figsize=(10,6))
plt.plot(df[target_stock].tail(60).values, label='Recent Actual Prices', color='blue')
plt.plot(range(60, 65), future_predictions, label='5-Day Forecast', color='red', marker='o')
plt.title(f'{target_stock} - Next 5 Day Price Forecast')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
