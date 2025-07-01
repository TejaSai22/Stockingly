# Stockingly Pro

## 📈 Advanced Stock Analysis & Portfolio Dashboard

Stockingly Pro is a powerful, interactive Streamlit web app for stock market analysis, portfolio management, technical and fundamental research, options analytics, sentiment analysis, and more. It is designed for both beginners and advanced investors.

---

## 🚀 Features

- **Stock Data Analysis**: Fetches data from Polygon.io and Yahoo Finance APIs
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, CCI, and more
- **Advanced Charting**: Interactive candlestick, volume, overlays, and technical indicator charts (Plotly)
- **Fundamental Analysis**: Company financials, PE ratio, EPS, revenue, profit margin, analyst ratings, and more
- **Portfolio Tracker**: Add/view positions, P&L, allocation charts, risk metrics, CSV export
- **Stock Screener**: Filter stocks by price, volume, PE, dividend yield, sector, RSI, and more
- **Backtesting Engine**: Simulate trading strategies (SMA crossover, RSI mean reversion) with performance metrics
- **Options Analytics**: View options chains, visualize open interest, IV, and calculate Black-Scholes prices
- **Sentiment Analysis**: Analyze news and social sentiment (TextBlob, Reddit simulation, MarketAux API)
- **News Integration**: Latest headlines and stock-specific news (MarketAux, NewsAPI)
- **Currency Exchange**: Live rates, historical analysis, converter, and allocation heatmaps
- **Custom Styling**: Modern UI with custom CSS and themes
- **Session State**: Persistent portfolio, watchlist, and settings

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Stockingly
   ```
2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python -m venv stockingly_env
   # On Windows:
   .\stockingly_env\Scripts\activate
   # On Mac/Linux:
   source stockingly_env/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run main_clean2.py
   ```
2. **Open your browser** and go to the URL shown (usually http://localhost:8501)
3. **Enter your API keys** (Polygon.io, NewsAPI) in the sidebar for full functionality

---

## 🔑 API Keys
- **Polygon.io**: For real-time and historical stock data ([get a free API key](https://polygon.io/))
- **NewsAPI**: For additional news coverage ([get a free API key](https://newsapi.org/))
- **MarketAux**: Financial news (already integrated)

Enter your API keys in the sidebar for best results.

---

## 📚 Main Features Overview

- **Analysis Tab**: Stock selection, technical/fundamental analysis, advanced charts, risk metrics, trading signals
- **Portfolio Tab**: Add/view positions, allocation charts, performance tracking, risk metrics, CSV export
- **Screener Tab**: Filter stocks by multiple criteria, export results
- **Backtesting Tab**: Simulate trading strategies and compare to buy & hold
- **Options Tab**: View and analyze options chains
- **Sentiment Tab**: Analyze news and social sentiment for stocks
- **Currency Tab**: Currency converter, live rates, historical analysis
- **News Tab**: Latest headlines and stock-specific news

---

## ⚠️ Disclaimer
This app is for informational and educational purposes only and does not constitute financial advice. Please consult a professional before making investment decisions.

---

## 🧩 Dependencies
- streamlit
- pandas
- plotly
- yfinance
- numpy
- requests
- ta
- textblob
- scipy
- forex-python
- requests-cache
- (and others, see requirements.txt)

---

## 📬 Feedback & Contributions
Pull requests and suggestions are welcome! Please open an issue or PR to contribute.
