
# 📊 Stock Market Trend Prediction Dashboard

A data-driven dashboard for analyzing and predicting stock market trends using machine learning and interactive visualizations.  
This project helps investors, analysts, and enthusiasts explore historical data, visualize patterns, and forecast future movements.


## 🚀 Features


-Fetches 5 years of daily stock data using yfinance

-Calculates daily returns and correlation matrix

-Generates trend visualizations for major Indian stocks

-Trains a Linear Regression Model to predict next-day prices

-Performs 5-day future forecasting

-Saves processed data for reuse


## 🧠Tech Stack

**Data Handling:**	pandas, numpy

**Data Source:**	yfinance

**Visualization:**	matplotlib, seaborn

**Machine Learning:**	scikit-learn

**Utility:**	os, datetime
## Installation

1. Clone the Repository

```bash
git clone https://github.com/shashwatkul/Stock-Market-Trend-Prediction-Dashboard.git
cd Stock-Market-Trend-Prediction

```
2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate       # on Windows
source venv/bin/activate    # on macOS/Linux
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run the Script
```bash
python stock_analysis.py
```

    
## Deployment

To deploy this project run

```bash
  npm run deploy
```


## Usage/Examples

```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```

