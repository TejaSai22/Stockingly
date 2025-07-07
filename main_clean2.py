import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from polygon import RESTClient
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional, Tuple
import json
import time
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
import requests_cache
from forex_python.converter import CurrencyRates, CurrencyCodes
import sqlite3
import io
import base64
from scipy import stats
import ta
from textblob import TextBlob
import re
import datetime
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Enhanced page configuration
st.set_page_config(
    page_title="Stockingly Pro - Advanced Stock Analysis", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .fundamental-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .strategy-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .news-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    
    .screener-result {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .options-chain {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    .backtest-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data classes for better structure
@dataclass
class StockData:
    symbol: str
    data: pd.DataFrame
    last_updated: datetime

@dataclass
class FundamentalData:
    symbol: str
    market_cap: float
    pe_ratio: float
    eps: float
    dividend_yield: float
    revenue: float
    profit_margin: float
    debt_to_equity: float
    roe: float
    analyst_rating: str
    target_price: float

@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: int
    equity_curve: pd.Series

@dataclass
class OptionsData:
    calls: pd.DataFrame
    puts: pd.DataFrame
    expiration_dates: List[str]
    underlying_price: float

# Initialize session state
def init_session_state():
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {}
    if 'portfolio_history' not in st.session_state:
        st.session_state.portfolio_history = []
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'theme' not in st.session_state:
        st.session_state.theme = 'plotly_dark'
    if 'client' not in st.session_state:
        st.session_state.client = None
    if 'cached_data' not in st.session_state:
        st.session_state.cached_data = {}
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = {}
    if 'screener_results' not in st.session_state:
        st.session_state.screener_results = []

init_session_state()

# Enhanced sidebar
def create_sidebar():
    st.sidebar.markdown("# 🚀 Stockingly Pro")
    st.sidebar.markdown("---")
    
    # API Configuration
    st.sidebar.subheader("🔑 API Configuration")
    polygon_api_key = st.sidebar.text_input("Polygon API Key", type="password")
    newsapi_key = st.sidebar.text_input("NewsAPI Key (Optional)", type="password")
    
    # Theme selector
    st.sidebar.subheader("🎨 Theme")
    theme = st.sidebar.selectbox("Chart Theme", 
                                ["plotly_dark", "plotly_white", "ggplot2", "seaborn"])
    st.session_state.theme = theme
    
    # Portfolio section
    st.sidebar.subheader("💼 Portfolio")
    if st.sidebar.button("View Portfolio"):
        st.session_state.show_portfolio = True
    
    # Quick portfolio stats
    if st.session_state.portfolio:
        total_positions = len(st.session_state.portfolio)
        st.sidebar.metric("Positions", total_positions)
    
    # Watchlist section
    st.sidebar.subheader("👁️ Watchlist")
    watchlist_symbol = st.sidebar.text_input("Add to Watchlist").upper()
    if st.sidebar.button("Add Symbol") and watchlist_symbol:
        if watchlist_symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(watchlist_symbol)
            st.sidebar.success(f"Added {watchlist_symbol} to watchlist")
    
    # Display watchlist
    if st.session_state.watchlist:
        st.sidebar.write("Current Watchlist:")
        for symbol in st.session_state.watchlist:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(symbol)
            if col2.button("❌", key=f"remove_{symbol}"):
                st.session_state.watchlist.remove(symbol)
                st.rerun()
    
    # Alerts section
    st.sidebar.subheader("🔔 Price Alerts")
    alert_symbol = st.sidebar.text_input("Symbol for Alert").upper()
    alert_price = st.sidebar.number_input("Alert Price", min_value=0.01, step=0.01)
    alert_type = st.sidebar.selectbox("Alert Type", ["Above", "Below"])
    
    if st.sidebar.button("Set Alert") and alert_symbol and alert_price:
        alert = {
            'symbol': alert_symbol,
            'price': alert_price,
            'type': alert_type,
            'created': datetime.now()
        }
        st.session_state.alerts.append(alert)
        st.sidebar.success(f"Alert set for {alert_symbol}")
    
    return polygon_api_key, newsapi_key

# Enhanced stock data fetching with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_enhanced_stock_data(symbol: str, start_date: datetime, end_date: datetime, api_key: str = None):
    """Fetch stock data with multiple sources and enhanced error handling"""
    
    # Try Polygon API first if available
    if api_key:
        try:
            client = RESTClient(api_key)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            aggs = []
            for a in client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan='day',
                from_=start_str,
                to=end_str,
                limit=50000
            ):
                aggs.append(a)
            
            if aggs:
                df = pd.DataFrame([{
                    'Date': pd.to_datetime(agg.timestamp, unit='ms'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume,
                    'VWAP': getattr(agg, 'vwap', None)
                } for agg in aggs])
                
                return df.sort_values('Date')
        except Exception as e:
            st.warning(f"Polygon API failed: {str(e)}. Falling back to Yahoo Finance.")
    
    # Fallback to Yahoo Finance
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None

# Fundamental Analysis Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_fundamental_data(symbol: str, polygon_api_key: str = None) -> Optional[dict]:
    """Fetch fundamental data for a stock from Polygon first, then Yahoo as fallback."""
    # Try Polygon API first if available
    if polygon_api_key:
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}?apiKey={polygon_api_key}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if "results" in data:
                    res = data["results"]
                    return {
                        "market_cap": res.get("market_cap", "N/A"),
                        "pe_ratio": res.get("weighted_shares_outstanding", "N/A"),  # Not always available
                        "eps": res.get("earnings_per_share", "N/A"),
                        "revenue": res.get("total_revenue", "N/A"),
                        "name": res.get("name", symbol),
                        "symbol": symbol
                    }
        except Exception as e:
            st.warning(f"Polygon API failed: {e}")
    # Fallback to Yahoo Finance
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "market_cap": info.get('marketCap', "N/A"),
            "pe_ratio": info.get('trailingPE', "N/A"),
            "eps": info.get('trailingEps', "N/A"),
            "revenue": info.get('totalRevenue', "N/A"),
            "name": info.get('shortName', symbol),
            "symbol": symbol
        }
    except Exception as e:
        st.error(f"Error fetching fundamental data: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_earnings_calendar(symbol: str):
    """Fetch earnings calendar and analyst estimates"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get earnings dates
        earnings_dates = ticker.calendar
        
        # Get analyst estimates
        estimates = ticker.analyst_price_target
        
        return {
            'earnings_dates': earnings_dates,
            'estimates': estimates,
            'recommendations': ticker.recommendations
        }
    except Exception as e:
        st.warning(f"Could not fetch earnings data: {str(e)}")
        return None

def create_fundamental_analysis_section(symbol: str, polygon_api_key: str = None):
    """Create comprehensive fundamental analysis section using hybrid data source."""
    st.subheader("\U0001F4C8 Fundamental Analysis")
    with st.spinner(f'Fetching fundamental data for {symbol}...'):
        fund_data = fetch_fundamental_data(symbol, polygon_api_key)
    if fund_data:
        st.write("#### Key Financial Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Market Cap", format_large_number(fund_data['market_cap']))
        with col2:
            st.metric("P/E Ratio", f"{fund_data['pe_ratio'] if str(fund_data['pe_ratio']).replace('.', '', 1).isdigit() else 'N/A'}")
        with col3:
            st.metric("EPS", f"${fund_data['eps'] if fund_data['eps'] != 'N/A' else 'N/A'}")
        with col4:
            st.metric("Revenue", format_large_number(fund_data['revenue']))
    else:
        st.error("Could not fetch fundamental data from Polygon or Yahoo Finance.")
    earnings_data = fetch_earnings_calendar(symbol)
    # Earnings calendar
    if earnings_data and earnings_data.get('earnings_dates') is not None:
        st.write("#### Upcoming Earnings")
        try:
            earnings_df = earnings_data['earnings_dates']
            if not earnings_df.empty:
                st.dataframe(earnings_df, use_container_width=True)
            else:
                st.info("No upcoming earnings data available")
        except:
            st.info("Earnings calendar not available")
    
    # Analyst recommendations
    if earnings_data and earnings_data.get('recommendations') is not None:
        st.write("#### Recent Analyst Recommendations")
        try:
            rec_df = earnings_data['recommendations'].tail(10)
            if not rec_df.empty:
                st.dataframe(rec_df, use_container_width=True)
        except:
            st.info("Analyst recommendations not available")

# Enhanced Stock Screener
def create_enhanced_screener():
    """Create enhanced stock screener with multiple criteria"""
    st.subheader("🔍 Enhanced Stock Screener")
    
    # Screener criteria
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Price & Volume**")
        min_price = st.number_input("Min Price ($)", min_value=0.0, value=1.0)
        max_price = st.number_input("Max Price ($)", min_value=0.0, value=1000.0)
        min_volume = st.number_input("Min Volume", min_value=0, value=100000)
        
    with col2:
        st.write("**Fundamental Metrics**")
        max_pe = st.number_input("Max P/E Ratio", min_value=0.0, value=50.0)
        min_dividend_yield = st.number_input("Min Dividend Yield (%)", min_value=0.0, value=0.0)
        market_cap_filter = st.selectbox("Market Cap", 
                                       ["Any", "Micro (<$300M)", "Small ($300M-$2B)", 
                                        "Mid ($2B-$10B)", "Large (>$10B)"])
        
    with col3:
        st.write("**Technical Indicators**")
        rsi_min = st.number_input("Min RSI", min_value=0, max_value=100, value=30)
        rsi_max = st.number_input("Max RSI", min_value=0, max_value=100, value=70)
        sector_filter = st.selectbox("Sector", 
                                   ["Any", "Technology", "Healthcare", "Financial Services",
                                    "Consumer Cyclical", "Communication Services", "Industrials",
                                    "Consumer Defensive", "Energy", "Utilities", "Real Estate",
                                    "Basic Materials"])
    
    # Predefined stock universe (in real app, this would be from a database)
    stock_universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ', 'ADBE',
        'CRM', 'ORCL', 'IBM', 'CSCO', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS',
        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY', 'MRK', 'GILD',
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'OKE', 'WMB', 'PSX', 'VLO',
        'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'TGT', 'LOW', 'SBUX', 'MCD'
    ]
    
    if st.button("🔍 Run Enhanced Screener", type="primary"):
        with st.spinner('Screening stocks...'):
            screener_results = []
            progress_bar = st.progress(0)
            
            for i, symbol in enumerate(stock_universe):
                try:
                    # Update progress
                    progress_bar.progress((i + 1) / len(stock_universe))
                    
                    # Fetch basic data
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="3mo")
                    
                    if hist.empty:
                        continue
                    
                    # Extract metrics
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1]
                    market_cap = info.get('marketCap', 0)
                    pe_ratio = info.get('trailingPE', 0)
                    dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                    sector = info.get('sector', 'Unknown')
                    
                    # Calculate RSI
                    rsi = ta.momentum.RSIIndicator(hist['Close']).rsi().iloc[-1]
                    
                    # Apply filters
                    if current_price < min_price or current_price > max_price:
                        continue
                    if volume < min_volume:
                        continue
                    if pe_ratio > max_pe and pe_ratio > 0:
                        continue
                    if dividend_yield < min_dividend_yield:
                        continue
                    if rsi < rsi_min or rsi > rsi_max:
                        continue
                    
                    # Market cap filter
                    if market_cap_filter != "Any":
                        if market_cap_filter == "Micro (<$300M)" and market_cap >= 300e6:
                            continue
                        elif market_cap_filter == "Small ($300M-$2B)" and (market_cap < 300e6 or market_cap >= 2e9):
                            continue
                        elif market_cap_filter == "Mid ($2B-$10B)" and (market_cap < 2e9 or market_cap >= 10e9):
                            continue
                        elif market_cap_filter == "Large (>$10B)" and market_cap < 10e9:
                            continue
                    
                    # Sector filter
                    if sector_filter != "Any" and sector != sector_filter:
                        continue
                    
                    # Calculate additional metrics
                    price_change_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                    price_change_1w = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100 if len(hist) >= 5 else 0
                    
                    screener_results.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'Volume': volume,
                        'Market Cap': market_cap,
                        'P/E': pe_ratio,
                        'Div Yield': dividend_yield,
                        'RSI': rsi,
                        'Sector': sector,
                        '1D Change': price_change_1d,
                        '1W Change': price_change_1w
                    })
                    
                except Exception as e:
                    continue
            
            progress_bar.empty()
            
            if screener_results:
                st.success(f"Found {len(screener_results)} stocks matching criteria")
                
                # Convert to DataFrame
                results_df = pd.DataFrame(screener_results)
                
                # Format columns
                results_df['Price'] = results_df['Price'].apply(lambda x: f"${x:.2f}")
                results_df['Volume'] = results_df['Volume'].apply(lambda x: f"{x:,.0f}")
                results_df['Market Cap'] = results_df['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.0f}M")
                results_df['P/E'] = results_df['P/E'].apply(lambda x: f"{x:.2f}" if x > 0 else "N/A")
                results_df['Div Yield'] = results_df['Div Yield'].apply(lambda x: f"{x:.2f}%")
                results_df['RSI'] = results_df['RSI'].apply(lambda x: f"{x:.1f}")
                results_df['1D Change'] = results_df['1D Change'].apply(lambda x: f"{x:+.2f}%")
                results_df['1W Change'] = results_df['1W Change'].apply(lambda x: f"{x:+.2f}%")
                
                # Display results
                st.dataframe(results_df, use_container_width=True)
                
                # Save results to session state
                st.session_state.screener_results = screener_results
                
                # Export option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("No stocks found matching the specified criteria. Try adjusting your filters.")

# Backtesting Engine
class SimpleBacktester:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def sma_crossover_strategy(self, short_window: int = 20, long_window: int = 50, initial_capital: float = 10000):
        """Simple Moving Average Crossover Strategy"""
        df = self.data.copy()
        
        # Calculate moving averages
        df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        df['Signal'] = 0
        df['Signal'][short_window:] = np.where(
            df['SMA_Short'][short_window:] > df['SMA_Long'][short_window:], 1, 0
        )
        df['Position'] = df['Signal'].diff()
        
        # Backtest
        capital = initial_capital
        shares = 0
        equity = [initial_capital]
        trades = []
        
        for i in range(len(df)):
            if df['Position'].iloc[i] == 1:  # Buy signal
                shares = capital / df['Close'].iloc[i]
                capital = 0
                trades.append({
                    'Date': df['Date'].iloc[i],
                    'Action': 'BUY',
                    'Price': df['Close'].iloc[i],
                    'Shares': shares
                })
            elif df['Position'].iloc[i] == -1 and shares > 0:  # Sell signal
                capital = shares * df['Close'].iloc[i]
                trades.append({
                    'Date': df['Date'].iloc[i],
                    'Action': 'SELL',
                    'Price': df['Close'].iloc[i],
                    'Shares': shares,
                    'P&L': capital - initial_capital
                })
                shares = 0
            
            # Calculate current equity
            current_equity = capital + (shares * df['Close'].iloc[i])
            equity.append(current_equity)
        
        # Final sell if still holding
        if shares > 0:
            capital = shares * df['Close'].iloc[-1]
            trades.append({
                'Date': df['Date'].iloc[-1],
                'Action': 'SELL',
                'Price': df['Close'].iloc[-1],
                'Shares': shares,
                'P&L': capital - initial_capital
            })
        
        # Calculate performance metrics
        equity_series = pd.Series(equity[1:], index=df['Date'])
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        profitable_trades = [t for t in trades if t.get('P&L', 0) > 0]
        win_rate = len(profitable_trades) / len([t for t in trades if 'P&L' in t]) if trades else 0
        
        return BacktestResult(
            strategy_name=f"SMA Crossover ({short_window}/{long_window})",
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trades=len(trades),
            equity_curve=equity_series
        )
    
    def rsi_mean_reversion_strategy(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70, initial_capital: float = 10000):
        """RSI Mean Reversion Strategy"""
        df = self.data.copy()
        
        # Calculate RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_period).rsi()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['RSI'] < oversold, 'Signal'] = 1  # Buy when oversold
        df.loc[df['RSI'] > overbought, 'Signal'] = -1  # Sell when overbought
        
        # Backtest logic (similar to SMA strategy)
        capital = initial_capital
        shares = 0
        equity = [initial_capital]
        trades = []
        position = 0  # 0: no position, 1: long
        
        for i in range(len(df)):
            if df['Signal'].iloc[i] == 1 and position == 0:  # Buy signal
                shares = capital / df['Close'].iloc[i]
                capital = 0
                position = 1
                trades.append({
                    'Date': df['Date'].iloc[i],
                    'Action': 'BUY',
                    'Price': df['Close'].iloc[i],
                    'Shares': shares
                })
            elif df['Signal'].iloc[i] == -1 and position == 1:  # Sell signal
                capital = shares * df['Close'].iloc[i]
                trades.append({
                    'Date': df['Date'].iloc[i],
                    'Action': 'SELL',
                    'Price': df['Close'].iloc[i],
                    'Shares': shares,
                    'P&L': capital - initial_capital
                })
                shares = 0
                position = 0
            
            # Calculate current equity
            current_equity = capital + (shares * df['Close'].iloc[i])
            equity.append(current_equity)
        
        # Calculate performance metrics (similar to SMA strategy)
        equity_series = pd.Series(equity[1:], index=df['Date'])
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(df)) - 1 if len(df) > 0 else 0
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        profitable_trades = [t for t in trades if t.get('P&L', 0) > 0]
        win_rate = len(profitable_trades) / len([t for t in trades if 'P&L' in t]) if trades else 0
        
        return BacktestResult(
            strategy_name=f"RSI Mean Reversion ({oversold}/{overbought})",
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trades=len(trades),
            equity_curve=equity_series
        )

def create_backtesting_section(symbol: str, df: pd.DataFrame):
    """Create backtesting and strategy simulation section"""
    st.subheader("🔄 Strategy Backtesting")
    
    # Strategy selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**SMA Crossover Strategy**")
        sma_short = st.number_input("Short SMA Period", min_value=5, max_value=50, value=20)
        sma_long = st.number_input("Long SMA Period", min_value=20, max_value=200, value=50)
        sma_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000)
        
        if st.button("📈 Backtest SMA Strategy"):
            with st.spinner('Running SMA backtest...'):
                backtester = SimpleBacktester(df)
                result = backtester.sma_crossover_strategy(sma_short, sma_long, sma_capital)
                st.session_state.backtest_results['SMA'] = result
    
    with col2:
        st.write("**RSI Mean Reversion Strategy**")
        rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14)
        rsi_oversold = st.number_input("Oversold Level", min_value=10, max_value=40, value=30)
        rsi_overbought = st.number_input("Overbought Level", min_value=60, max_value=90, value=70)
        rsi_capital = st.number_input("Initial Capital ($)", min_value=1000, value=10000, step=1000, key="rsi_capital")
        
        if st.button("📉 Backtest RSI Strategy"):
            with st.spinner('Running RSI backtest...'):
                backtester = SimpleBacktester(df)
                result = backtester.rsi_mean_reversion_strategy(rsi_period, rsi_oversold, rsi_overbought, rsi_capital)
                st.session_state.backtest_results['RSI'] = result
    
    # Display backtest results
    if st.session_state.backtest_results:
        st.write("#### Backtest Results")
        
        for strategy_name, result in st.session_state.backtest_results.items():
            with st.expander(f"📊 {result.strategy_name} Results"):
                # Performance metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Return", f"{result.total_return:.2%}")
                
                with col2:
                    st.metric("Annual Return", f"{result.annual_return:.2%}")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                
                with col4:
                    st.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
                
                with col5:
                    st.metric("Win Rate", f"{result.win_rate:.2%}")
                
                # Equity curve chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    mode='lines',
                    name='Strategy Equity',
                    line=dict(color='blue', width=2)
                ))
                
                # Add buy & hold comparison
                buy_hold_equity = (df['Close'] / df['Close'].iloc[0]) * result.equity_curve.iloc[0]
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=buy_hold_equity,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{result.strategy_name} - Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    template=st.session_state.theme,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Strategy summary
                st.markdown(f"""
                <div class="backtest-summary">
                    <h4>Strategy Summary</h4>
                    <p><strong>Total Trades:</strong> {result.trades}</p>
                    <p><strong>Final Portfolio Value:</strong> ${result.equity_curve.iloc[-1]:,.2f}</p>
                    <p><strong>Best Strategy:</strong> {'✅ Outperformed Buy & Hold' if result.total_return > (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1) else '❌ Underperformed Buy & Hold'}</p>
                </div>
                """, unsafe_allow_html=True)

# Sentiment Analysis Functions
def analyze_text_sentiment(text: str) -> Dict:
    """Analyze sentiment of text using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': abs(polarity)
        }
    except:
        return {
            'sentiment': 'neutral',
            'polarity': 0,
            'subjectivity': 0,
            'confidence': 0
        }

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_reddit_sentiment(symbol: str, limit: int = 100):
    """Fetch Reddit sentiment (simulated - would use Reddit API in production)"""
    # This is a placeholder - in production, you'd use the Reddit API
    # For now, we'll simulate some sentiment data
    
    sentiments = []
    for i in range(limit):
        # Simulate random sentiment data
        polarity = np.random.normal(0, 0.3)
        sentiment = 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
        
        sentiments.append({
            'source': 'Reddit',
            'sentiment': sentiment,
            'polarity': polarity,
            'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 24)),
            'text': f"Sample Reddit post about {symbol}"
        })
    
    return sentiments

def create_sentiment_analysis_section(symbol: str):
    """Create sentiment analysis section"""
    st.subheader("💭 Sentiment Analysis")
    
    # Sentiment sources
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**News Sentiment**")
        if st.button("📰 Analyze News Sentiment"):
            with st.spinner(f'Analyzing news sentiment for {symbol}...'):
                # Fetch news (reuse existing function
                articles = fetch_comprehensive_news(symbol, None, 20)
                
                if articles:
                    sentiments = []
                    for article in articles:
                        if article.get('description'):
                            sentiment_data = analyze_text_sentiment(article['description'])
                            sentiments.append({
                                'title': article['title'],
                                'sentiment': sentiment_data['sentiment'],
                                'polarity': sentiment_data['polarity'],
                                'source': article['source']
                            })
                    
                    if sentiments:
                        # Sentiment distribution
                        sentiment_counts = pd.Series([s['sentiment'] for s in sentiments]).value_counts()
                        
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title=f"News Sentiment Distribution for {symbol}",
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Average sentiment score
                        avg_polarity = np.mean([s['polarity'] for s in sentiments])
                        st.metric("Average Sentiment Score", f"{avg_polarity:.3f}")
                        
                        # Recent sentiment articles
                        st.write("#### Recent Sentiment Analysis")
                        for sentiment in sentiments[:5]:
                            sentiment_color = "sentiment-positive" if sentiment['sentiment'] == 'positive' else "sentiment-negative" if sentiment['sentiment'] == 'negative' else "sentiment-neutral"
                            st.markdown(f"""
                            <div class="news-card">
                                <div class="news-title">{sentiment['title']}</div>
                                <div class="{sentiment_color}">
                                    Sentiment: {sentiment['sentiment'].title()} ({sentiment['polarity']:.3f})
                                </div>
                                <div class="news-meta">Source: {sentiment['source']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No news articles found for sentiment analysis")
    
    with col2:
        st.write("**Social Media Sentiment**")
        if st.button("🐦 Analyze Social Sentiment"):
            with st.spinner(f'Analyzing social sentiment for {symbol}...'):
                # Simulate social media sentiment (would use Twitter API in production)
                social_sentiments = fetch_reddit_sentiment(symbol, 50)
                
                if social_sentiments:
                    # Sentiment over time
                    sentiment_df = pd.DataFrame(social_sentiments)
                    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                    
                    # Hourly sentiment aggregation
                    hourly_sentiment = sentiment_df.set_index('timestamp').resample('H')['polarity'].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hourly_sentiment.index,
                        y=hourly_sentiment.values,
                        mode='lines+markers',
                        name='Sentiment Score',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        title=f"Social Media Sentiment Trend for {symbol}",
                        xaxis_title="Time",
                        yaxis_title="Sentiment Score",
                        template=st.session_state.theme
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_sentiment = sentiment_df['polarity'].mean()
                        st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                    
                    with col2:
                        positive_pct = (sentiment_df['sentiment'] == 'positive').mean() * 100
                        st.metric("Positive %", f"{positive_pct:.1f}%")
                    
                    with col3:
                        negative_pct = (sentiment_df['sentiment'] == 'negative').mean() * 100
                        st.metric("Negative %", f"{negative_pct:.1f}%")

# Enhanced Portfolio Analytics
def create_enhanced_portfolio_section():
    """Create enhanced portfolio tracking with analytics"""
    st.subheader("💼 Enhanced Portfolio Analytics")
    
    # Add position form
    with st.expander("➕ Add New Position"):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            symbol = st.text_input("Symbol").upper()
        with col2:
            shares = st.number_input("Shares", min_value=0.0, step=0.1)
        with col3:
            avg_price = st.number_input("Average Price", min_value=0.01, step=0.01)
        with col4:
            purchase_date = st.date_input("Purchase Date", datetime.date.today())
        with col5:
            st.write("")  # Spacer
            if st.button("Add Position") and symbol and shares and avg_price:
                st.session_state.portfolio[symbol] = {
                    'shares': shares,
                    'avg_price': avg_price,
                    'purchase_date': purchase_date,
                    'date_added': datetime.now()
                }
                st.success(f"Added {symbol} to portfolio")
                st.rerun()
    
    # Portfolio overview
    if st.session_state.portfolio:
        portfolio_data = []
        total_value = 0
        total_cost = 0
        sector_allocation = {}
        
        for symbol, position in st.session_state.portfolio.items():
            try:
                # Get current data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                info = ticker.info
                
                if hist.empty:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                market_value = position['shares'] * current_price
                cost_basis = position['shares'] * position['avg_price']
                pnl = market_value - cost_basis
                pnl_pct = (pnl / cost_basis) * 100
                
                # Sector allocation
                sector = info.get('sector', 'Unknown')
                if sector in sector_allocation:
                    sector_allocation[sector] += market_value
                else:
                    sector_allocation[sector] = market_value
                
                portfolio_data.append({
                    'Symbol': symbol,
                    'Shares': position['shares'],
                    'Avg Price': position['avg_price'],
                    'Current Price': current_price,
                    'Market Value': market_value,
                    'Cost Basis': cost_basis,
                    'P&L': pnl,
                    'P&L %': pnl_pct,
                    'Weight': 0,  # Will calculate after total
                    'Sector': sector,
                    'Purchase Date': position.get('purchase_date', datetime.date.today())
                })
                
                total_value += market_value
                total_cost += cost_basis
                
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        if portfolio_data:
            # Calculate weights
            for item in portfolio_data:
                item['Weight'] = (item['Market Value'] / total_value) * 100
            
            # Portfolio summary
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            st.write("#### Portfolio Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            with col3:
                st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")
            with col4:
                st.metric("Positions", len(portfolio_data))
            with col5:
                # Calculate portfolio beta (simplified)
                portfolio_beta = 1.0  # Placeholder
                st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
            
            # Portfolio composition charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Asset allocation pie chart
                fig = px.pie(
                    values=[item['Market Value'] for item in portfolio_data],
                    names=[item['Symbol'] for item in portfolio_data],
                    title="Portfolio Allocation by Stock"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sector allocation pie chart
                if sector_allocation:
                    fig = px.pie(
                        values=list(sector_allocation.values()),
                        names=list(sector_allocation.keys()),
                        title="Portfolio Allocation by Sector"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed portfolio table
            st.write("#### Portfolio Holdings")
            df_portfolio = pd.DataFrame(portfolio_data)
            
            # Format the dataframe for display
            df_display = df_portfolio.copy()
            df_display['Avg Price'] = df_display['Avg Price'].apply(lambda x: f"${x:.2f}")
            df_display['Current Price'] = df_display['Current Price'].apply(lambda x: f"${x:.2f}")
            df_display['Market Value'] = df_display['Market Value'].apply(lambda x: f"${x:,.2f}")
            df_display['Cost Basis'] = df_display['Cost Basis'].apply(lambda x: f"${x:,.2f}")
            df_display['P&L'] = df_display['P&L'].apply(lambda x: f"${x:,.2f}")
            df_display['P&L %'] = df_display['P&L %'].apply(lambda x: f"{x:.2f}%")
            df_display['Weight'] = df_display['Weight'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(df_display, use_container_width=True)
            
            # Portfolio performance tracking
            st.write("#### Portfolio Performance")
            
            # Simulate historical portfolio value (in production, this would be stored)
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            portfolio_values = []
            
            for date in dates:
                # Simulate portfolio value changes
                daily_return = np.random.normal(0.001, 0.02)  # 0.1% average daily return with 2% volatility
                if not portfolio_values:
                    portfolio_values.append(total_cost)
                else:
                    portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
            
            # Portfolio performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template=st.session_state.theme
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            st.write("#### Portfolio Risk Metrics")
            
            # Calculate portfolio volatility (simplified)
            returns = pd.Series(portfolio_values).pct_change().dropna()
            portfolio_volatility = returns.std() * np.sqrt(252)
            portfolio_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Volatility", f"{portfolio_volatility:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
            with col3:
                # Value at Risk (95%)
                var_95 = np.percentile(returns, 5)
                st.metric("VaR (95%)", f"{var_95:.2%}")
            with col4:
                # Maximum drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown =  drawdown.min()
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            
            # Export portfolio
            st.write("#### Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export to CSV
                csv = df_portfolio.to_csv(index=False)
                st.download_button(
                    label="📥 Export Portfolio to CSV",
                    data=csv,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export performance report
                report = f"""
Portfolio Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Portfolio Summary:
- Total Value: ${total_value:,.2f}
- Total Cost: ${total_cost:,.2f}
- Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:.2f}%)
- Number of Positions: {len(portfolio_data)}

Risk Metrics:
- Portfolio Volatility: {portfolio_volatility:.2%}
- Sharpe Ratio: {portfolio_sharpe:.2f}
- VaR (95%): {var_95:.2%}
- Max Drawdown: {max_drawdown:.2%}

Holdings:
{df_portfolio.to_string(index=False)}
                """
                
                st.download_button(
                    label="📊 Export Performance Report",
                    data=report,
                    file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    else:
        st.info("No positions in portfolio. Add some positions to get started!")

# Keep all existing functions from the previous version
# (Technical indicators, charting, news, currency exchange, etc.)

# Enhanced technical indicators
def calculate_technical_indicators(df: pd.DataFrame):
    """Calculate comprehensive technical indicators using TA library"""
    
    # Basic indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Additional indicators
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
    
    return df

# Enhanced charting functions (keeping existing implementation)
def create_advanced_candlestick_chart(df: pd.DataFrame, symbol: str):
    """Create advanced candlestick chart with multiple indicators"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=[
            f"{symbol} - Price & Volume",
            "RSI",
            "MACD",
            "Stochastic Oscillator"
        ]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['BB_Upper'],
            name='BB Upper', line=dict(color='rgba(255,0,0,0.3)'),
            fill=None
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['BB_Lower'],
            name='BB Lower', line=dict(color='rgba(255,0,0,0.3)'),
            fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
        ), row=1, col=1)
    
    # Moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_20'],
            name='SMA 20', line=dict(color='blue', width=1)
        ), row=1, col=1)
    
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_50'],
            name='SMA 50', line=dict(color='orange', width=1)
        ), row=1, col=1)
    
    # Volume
    colors = ['green' if row['Close'] >= row['Open'] else 'red' 
              for _, row in df.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'],
        name='Volume', marker_color=colors,
        opacity=0.5, yaxis='y2'
    ), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'],
            name='RSI', line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD'],
            name='MACD', line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD_Signal'],
            name='Signal', line=dict(color='red')
        ), row=3, col=1)
        
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['MACD_Histogram'],
            name='Histogram', marker_color='gray'
        ), row=3, col=1)
    
    # Stochastic
    if 'Stoch_K' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Stoch_K'],
            name='%K', line=dict(color='blue')
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Stoch_D'],
            name='%D', line=dict(color='red')
        ), row=4, col=1)
        
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        template=st.session_state.theme,
        title_text=f"{symbol} - Advanced Technical Analysis"
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Stochastic", range=[0, 100], row=4, col=1)
    
    return fig

# Risk analysis functions (keeping existing implementation)
def calculate_risk_metrics(df: pd.DataFrame) -> Dict:
    """Calculate various risk metrics"""
    returns = df['Close'].pct_change().dropna()
    
    # Basic metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5)
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Beta (using SPY as market proxy)
    try:
        spy = yf.Ticker("SPY")
        spy_data = spy.history(start=df['Date'].min(), end=df['Date'].max())
        spy_returns = spy_data['Close'].pct_change().dropna()
        
        # Align dates
        common_dates = returns.index.intersection(spy_returns.index)
        if len(common_dates) > 30:
            stock_aligned = returns.loc[common_dates]
            spy_aligned = spy_returns.loc[common_dates]
            beta = np.cov(stock_aligned, spy_aligned)[0][1] / np.var(spy_aligned)
        else:
            beta = None
    except:
        beta = None
    
    return {
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'max_drawdown': max_drawdown,
        'beta': beta
    }

# News integration (keeping existing MarketAux implementation)
@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_marketaux_news(symbol: str = None, limit: int = 10):
    """Fetch news from MarketAux API"""
    try:
        # Your MarketAux API key
        api_key = "Xj5Lm49bVID1DqjmfLHTPNhQ7JyPR0bficSdfwAn"
        base_url = "https://api.marketaux.com/v1/news/all"
        
        params = {
            'api_token': api_key,
            'limit': limit,
            'language': 'en',
            'sort': 'published_desc'
        }
        
        # Add symbol filter if provided
        if symbol:
            params['symbols'] = symbol
            params['entity_types'] = 'equity'
        
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            articles = []
            
            for article in data.get('data', []):
                # Extract sentiment
                sentiment = 'neutral'
                sentiment_score = 0
                
                if 'sentiment' in article:
                    sentiment_score = article['sentiment']
                    if sentiment_score > 0.1:
                        sentiment = 'positive'
                    elif sentiment_score < -0.1:
                        sentiment = 'negative'
                
                # Extract entities (related stocks)
                entities = []
                if 'entities' in article:
                    entities = [entity.get('symbol', '') for entity in article['entities'] if entity.get('symbol')]
                
                articles.append({
                    'title': article.get('title', 'No Title'),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'publishedAt': article.get('published_at', ''),
                    'source': article.get('source', 'MarketAux'),
                    'urlToImage': article.get('image_url', ''),
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score,
                    'entities': entities,
                    'api_source': 'MarketAux'
                })
            
            return articles
            
    except Exception as e:
        st.error(f"MarketAux API error: {str(e)}")
        return []

def fetch_comprehensive_news(symbol: str = None, newsapi_key: str = None, limit: int = 10):
    """Fetch news from multiple sources with fallbacks"""
    all_articles = []
    
    # Try MarketAux first (best for financial news)
    if symbol:
        st.info("🔍 Fetching from MarketAux (Financial News API)...")
        marketaux_articles = fetch_marketaux_news(symbol, limit)
        if marketaux_articles:
            all_articles.extend(marketaux_articles)
            st.success(f"✅ Found {len(marketaux_articles)} articles from MarketAux")
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        if article['title'] not in seen_titles:
            seen_titles.add(article['title'])
            unique_articles.append(article)
    
    return unique_articles[:limit]

def format_sentiment_display(sentiment: str, score: float = 0):
    """Format sentiment with emoji and color"""
    if sentiment == 'positive':
        return f"😊 Positive ({score:.2f})" if score != 0 else "😊 Positive"
    elif sentiment == 'negative':
        return f"😟 Negative ({score:.2f})" if score != 0 else "😟 Negative"
    else:
        return f"😐 Neutral ({score:.2f})" if score != 0 else "😐 Neutral"

def format_published_date(date_str: str):
    """Format published date for display"""
    try:
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(date_str)
        
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    except:
        return date_str

def create_news_section():
    """Enhanced news section with MarketAux integration"""
    st.subheader("📰 Market News & Analysis")
    
    # News configuration
    with st.expander("🔧 News Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            newsapi_key = st.text_input(
                "NewsAPI Key (Optional)", 
                type="password",
                help="Get free API key from newsapi.org for additional news coverage"
            )
            st.caption("💡 MarketAux is already configured for financial news")
        
        with col2:
            news_symbol = st.text_input(
                "Stock Symbol for News", 
                value=st.session_state.get('selected_symbol', 'AAPL'),
                help="Enter stock symbol (e.g., AAPL, TSLA, MSFT)"
            ).upper()
    
    # Stock-specific news section
    st.markdown("### 📊 Stock-Specific News")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Latest news for **{news_symbol}**")
    with col2:
        article_limit = st.selectbox("Articles to show", [5, 10, 15, 20], index=1)
    
    if st.button("📰 Get Stock News", type="primary") or st.session_state.get('auto_stock_news', False):
        if news_symbol:
            with st.spinner(f'Fetching latest news for {news_symbol}...'):
                articles = fetch_comprehensive_news(news_symbol, newsapi_key, article_limit)
                
                if articles:
                    st.success(f"✅ Found {len(articles)} articles for {news_symbol}")
                    
                    # Display articles
                    for i, article in enumerate(articles, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="news-card">
                                <div class="news-title">📄 {i}. {article['title']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Article content
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                if article['description']:
                                    # Truncate long descriptions
                                    desc = article['description']
                                    if len(desc) > 300:
                                        desc = desc[:300] + "..."
                                    st.write(desc)
                                
                                # Metadata
                                meta_info = []
                                
                                # Source and time
                                if article['publishedAt']:
                                    time_ago = format_published_date(article['publishedAt'])
                                    meta_info.append(f"📅 {time_ago}")
                                
                                meta_info.append(f"📰 {article['source']}")
                                meta_info.append(f"🔗 {article['api_source']}")
                                
                                # Sentiment
                                if article.get('sentiment') != 'neutral' or article.get('sentiment_score', 0) != 0:
                                    sentiment_display = format_sentiment_display(
                                        article['sentiment'], 
                                        article.get('sentiment_score', 0)
                                    )
                                    meta_info.append(f"💭 {sentiment_display}")
                                
                                # Related entities
                                if article.get('entities') and len(article['entities']) > 1:
                                    entities = [e for e in article['entities'] if e != news_symbol]
                                    if entities:
                                        meta_info.append(f"🏢 Related: {', '.join(entities[:3])}")
                                
                                st.caption(" | ".join(meta_info))
                                
                                # Read more link
                                if article['url']:
                                    st.markdown(f"[🔗 Read Full Article]({article['url']})")
                            
                            with col2:
                                # Display image if available
                                if article.get('urlToImage'):
                                    try:
                                        st.image(article['urlToImage'], width=200, caption="Article Image")
                                    except:
                                        pass
                            
                            st.markdown("---")
                    
                    # Auto-refresh option
                    st.session_state.auto_stock_news = st.checkbox(
                        "🔄 Auto-refresh stock news", 
                        value=st.session_state.get('auto_stock_news', False),
                        help="Automatically refresh news when you change symbols"
                    )
                    
                else:
                    st.warning(f"No news articles found for {news_symbol}. Try a different symbol or check your API configuration.")
        else:
            st.error("Please enter a stock symbol")
    
    # Market overview section
    st.markdown("---")
    st.markdown("### 📈 General Market Headlines")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Latest market news and analysis")
    with col2:
        headline_limit = st.selectbox("Headlines to show", [5, 10, 15], index=1, key="headlines_limit")
    
    if st.button("📈 Get Market Headlines"):
        with st.spinner('Fetching market headlines...'):
            # Get general market news (no specific symbol)
            market_articles = fetch_marketaux_news(None, headline_limit)
            
            if market_articles:
                st.success(f"✅ Found {len(market_articles)} market headlines")
                
                # Display in a more compact format
                for i, article in enumerate(market_articles, 1):
                    with st.container():
                        # Title
                        st.markdown(f"**{i}. {article['title']}**")
                        
                        # Description
                        if article['description']:
                            desc = article['description']
                            if len(desc) > 200:
                                desc = desc[:200] + "..."
                            st.write(desc)
                        
                        # Metadata row
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            if article['publishedAt']:
                                time_ago = format_published_date(article['publishedAt'])
                                st.caption(f"📅 {time_ago}")
                        
                        with col2:
                            st.caption(f"📰 {article['source']}")
                            if article.get('sentiment') != 'neutral':
                                sentiment_display = format_sentiment_display(article['sentiment'])
                                st.caption(f"💭 {sentiment_display}")
                        
                        with col3:
                            if article['url']:
                                st.markdown(f"[Read →]({article['url']})")
                        
                        st.markdown("---")
            else:
                st.warning("Unable to fetch market headlines. Please try again later.")

# Currency exchange functions (keeping existing implementation)
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_currency_rates(base_currency='USD'):
    """Fetch current currency exchange rates"""
    try:
        # Using exchangerate-api.com (free tier)
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data['rates'], data['date']
        else:
            # Fallback to forex-python
            c = CurrencyRates()
            major_currencies = ['EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY', 'INR', 'KRW', 'MXN']
            rates = {}
            
            for currency in major_currencies:
                if currency != base_currency:
                    try:
                        if base_currency == 'USD':
                            rates[currency] = c.get_rate('USD', currency)
                        else:
                            rates[currency] = c.get_rate(base_currency, currency)
                    except:
                        rates[currency] = 1.0
            
            return rates, datetime.now().strftime('%Y-%m-%d')
            
    except Exception as e:
        st.error(f"Error fetching currency rates: {str(e)}")
        return {}, ""

def create_currency_exchange_section():
    """Create the currency exchange section"""
    st.subheader("💱 Currency Exchange")
    
    # Currency converter
    st.write("### Currency Converter")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Major currencies list
    major_currencies = {
        'USD': '🇺🇸 US Dollar',
        'EUR': '🇪🇺 Euro',
        'GBP': '🇬🇧 British Pound',
        'JPY': '🇯🇵 Japanese Yen',
        'AUD': '🇦🇺 Australian Dollar',
        'CAD': '🇨🇦 Canadian Dollar',
        'CHF': '🇨🇭 Swiss Franc',
        'CNY': '🇨🇳 Chinese Yuan',
        'INR': '🇮🇳 Indian Rupee',
        'KRW': '🇰🇷 South Korean Won',
        'MXN': '🇲🇽 Mexican Peso',
        'BRL': '🇧🇷 Brazilian Real',
        'RUB': '🇷🇺 Russian Ruble',
        'SGD': '🇸🇬 Singapore Dollar',
        'HKD': '🇭🇰 Hong Kong Dollar'
    }
    
    with col1:
        from_currency = st.selectbox("From Currency", 
                                   options=list(major_currencies.keys()),
                                   format_func=lambda x: major_currencies[x],
                                   index=0)
    
    with col2:
        to_currency = st.selectbox("To Currency", 
                                 options=list(major_currencies.keys()),
                                 format_func=lambda x: major_currencies[x],
                                 index=1)
    
    with col3:
        amount = st.number_input("Amount", min_value=0.01, value=1.00, step=0.01)
    
    with col4:
        st.write("")  # Spacer
        convert_button = st.button("💱 Convert", type="primary")
    
    # Fetch current rates and perform conversion
    if convert_button or st.session_state.get('auto_convert', False):
        with st.spinner('Fetching exchange rates...'):
            rates, last_updated = fetch_currency_rates(from_currency)
            
            if rates and to_currency in rates:
                if from_currency == to_currency:
                    converted_amount = amount
                    rate = 1.0
                else:
                    rate = rates[to_currency]
                    converted_amount = amount * rate
                
                # Display conversion result
                st.success(f"**{amount:,.2f} {from_currency} = {converted_amount:,.4f} {to_currency}**")
                st.info(f"Exchange Rate: 1 {from_currency} = {rate:.4f} {to_currency}")
                st.caption(f"Last updated: {last_updated}")
                
                # Store in session state for auto-update
                st.session_state.auto_convert = True
                st.session_state.last_conversion = {
                    'from': from_currency,
                    'to': to_currency,
                    'amount': amount,
                    'rate': rate,
                    'result': converted_amount
                }
            else:
                st.error("Unable to fetch exchange rate. Please try again.")

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stockingly Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    polygon_api_key, newsapi_key = create_sidebar()
    
    # Initialize API client
    if polygon_api_key and (st.session_state.client is None or 
                           getattr(st.session_state.client, '_api_key', None) != polygon_api_key):
        try:
            st.session_state.client = RESTClient(polygon_api_key)
            st.session_state.client._api_key = polygon_api_key
            st.sidebar.success("✅ Connected to Polygon API")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
            st.session_state.client = None
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Analysis", "📈 Prediction", "💼 Portfolio", "🔍 Screener", "🔄 Backtesting", 
        "💭 Sentiment", "💱 Currency", "📰 News"
    ])
    
    with tab1:
        # Enhanced stock selection
        st.subheader("Stock Analysis")
        
        # Popular stocks with enhanced categories
        popular_stocks = {
            'Mega Cap Tech': [
                {'name': 'Apple', 'ticker': 'AAPL'},
                {'name': 'Microsoft', 'ticker': 'MSFT'},
                {'name': 'Alphabet', 'ticker': 'GOOGL'},
                {'name': 'Amazon', 'ticker': 'AMZN'},
                {'name': 'Meta', 'ticker': 'META'},
                {'name': 'Tesla', 'ticker': 'TSLA'},
                {'name': 'NVIDIA', 'ticker': 'NVDA'}
            ],
            'AI & Semiconductors': [
                {'name': 'Advanced Micro Devices', 'ticker': 'AMD'},
                {'name': 'Intel', 'ticker': 'INTC'},
                {'name': 'Qualcomm', 'ticker': 'QCOM'},
                {'name': 'Broadcom', 'ticker': 'AVGO'}
            ],
            'Finance': [
                {'name': 'JPMorgan Chase', 'ticker': 'JPM'},
                {'name': 'Visa', 'ticker': 'V'},
                {'name': 'Mastercard', 'ticker': 'MA'},
                {'name': 'Bank of America', 'ticker': 'BAC'}
            ],
            'ETFs': [
                {'name': 'S&P 500', 'ticker': 'SPY'},
                {'name': 'NASDAQ 100', 'ticker': 'QQQ'},
                {'name': 'Russell 2000', 'ticker': 'IWM'},
                {'name': 'Total Market', 'ticker': 'VTI'}
            ]
        }
        
        # Stock selection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Quick select
            st.write("**Quick Select:**")
            for category, stocks in popular_stocks.items():
                with st.expander(f"{category}"):
                    cols = st.columns(4)
                    for i, stock in enumerate(stocks):
                        if cols[i % 4].button(f"{stock['name']}\n({stock['ticker']})", 
                                            key=f"btn_{category}_{stock['ticker']}"):
                            st.session_state.selected_symbol = stock['ticker']
        
        with col2:
            # Manual entry
            manual_symbol = st.text_input("Or enter symbol:", 
                                        value=st.session_state.get('selected_symbol', 'AAPL')).upper()
            st.session_state.selected_symbol = manual_symbol
            
            # Date range
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=365))
            with col_end:
                end_date = st.date_input("End Date", datetime.date.today())
        
        # Analysis button
        if st.button("🚀 Analyze Stock", type="primary"):
            symbol = st.session_state.selected_symbol
            
            with st.spinner(f'Analyzing {symbol}...'):
                # Fetch data
                df = fetch_enhanced_stock_data(symbol, start_date, end_date, polygon_api_key)
                
                if df is not None and len(df) > 0:
                    # Calculate indicators
                    df = calculate_technical_indicators(df)
                    
                    # Display key metrics
                    latest = df.iloc[-1]
                    prev_day = df.iloc[-2] if len(df) > 1 else latest
                    
                    price_change = latest['Close'] - prev_day['Close']
                    pct_change = (price_change / prev_day['Close']) * 100
                    
                    # Enhanced metrics display
                    st.subheader("📊 Key Metrics")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Current Price", f"${latest['Close']:.2f}", 
                                f"{price_change:.2f} ({pct_change:.2f}%)")
                    
                    with col2:
                        st.metric("Volume", f"{latest['Volume']:,}")
                    
                    with col3:
                        st.metric("52W High", f"${df['High'].max():.2f}")
                    
                    with col4:
                        st.metric("52W Low", f"${df['Low'].min():.2f}")
                    
                    with col5:
                        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not df['RSI'].isna().all() else 50
                        st.metric("RSI", f"{current_rsi:.1f}")
                    
                    # Advanced chart
                    st.subheader("📈 Advanced Technical Analysis")
                    fig = create_advanced_candlestick_chart(df, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fundamental analysis
                    create_fundamental_analysis_section(symbol, polygon_api_key)
                    
                    # Risk analysis
                    st.subheader("⚠️ Risk Analysis")
                    risk_metrics = calculate_risk_metrics(df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Volatility (Annual)", f"{risk_metrics['volatility']:.2%}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
                    
                    with col4:
                        if risk_metrics['beta']:
                            st.metric("Beta", f"{risk_metrics['beta']:.2f}")
                        else:
                            st.metric("Beta", "N/A")
                    
                    # Trading signals
                    st.subheader("🎯 Trading Signals")
                    
                    signals = []
                    
                    # RSI signals
                    if current_rsi > 70:
                        signals.append("🔴 RSI indicates overbought conditions")
                    elif current_rsi < 30:
                        signals.append("🟢 RSI indicates oversold conditions")
                    
                    # Moving average signals
                    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                        if latest['Close'] > df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
                            signals.append("🟢 Price above both 20-day and 50-day moving averages")
                        elif latest['Close'] < df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1]:
                            signals.append("🔴 Price below both 20-day and 50-day moving averages")
                    
                    # MACD signals
                    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                            signals.append("🟢 MACD above signal line")
                        else:
                            signals.append("🔴 MACD below signal line")
                    
                    for signal in signals:
                        st.write(signal)
                    
                    # Recent data table
                    st.subheader("📋 Recent Data")
                    display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if 'SMA_20' in df.columns:
                        display_cols.append('SMA_20')
                    if 'SMA_50' in df.columns:
                        display_cols.append('SMA_50')
                    if 'RSI' in df.columns:
                        display_cols.append('RSI')
                    
                    display_df = df[display_cols].tail(10)
                    st.dataframe(display_df.sort_values('Date', ascending=False), use_container_width=True)
                
                else:
                    st.error("Failed to fetch data. Please check the symbol and try again.")
    
    with tab2:
        st.subheader("Stock Price Prediction")
        st.write("Predict future stock prices using ARIMA, Prophet, or an Advanced ML model (Random Forest with technical indicators).")
        today = datetime.date.today()
        
        # Model selection (add ARIMAX)
        model_choice = st.selectbox("Select Prediction Model", ["ARIMAX", "Prophet", "Advanced ML"], index=1)
        
        # Symbol input (reuse logic from Analysis tab)
        symbol = st.text_input("Stock Symbol", value=st.session_state.get('selected_symbol', 'AAPL')).upper()
        st.session_state.selected_symbol = symbol
        
        # Start date only (end date is always today)
        start_date = st.date_input("Start Date", today - datetime.timedelta(days=365), max_value=today, key="pred_start")
        end_date = today
        st.caption(f"Note: The model will use historical data from the selected start date up to today ({today}) to predict the next N days into the future.")
        
        # Forecast horizon
        forecast_days = st.selectbox("Days to Predict Ahead", [7, 14, 30, 90], index=2)
        
        # Predict button
        if st.button("🔮 Predict Future Prices"):
            with st.spinner(f"Fetching data for {symbol}..."):
                df = fetch_enhanced_stock_data(symbol, start_date, end_date, polygon_api_key)
            st.write("### Debug: Raw DataFrame")
            st.write(df)
            # Always set index to 'Date' column if present
            if df is not None and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            if df is None or df.empty or 'Close' not in df.columns or len(df) < 30:
                st.error("Not enough usable data to make a prediction. Please select a different start date or symbol. (At least 30 days of data with 'Close' prices required)")
                st.stop()
            else:
                # Add technical indicators
                import ta
                df = df.copy()
                df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                df['MACD'] = ta.trend.macd(df['Close'])
                df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
                bb = ta.volatility.BollingerBands(df['Close'], window=20)
                df['BB_High'] = bb.bollinger_hband()
                df['BB_Low'] = bb.bollinger_lband()
                df['Volume'] = df['Volume'] if 'Volume' in df.columns else 0
                df = df.dropna()
                df = df.sort_index()
                # Robust check for last_date
                if len(df.index) == 0 or pd.isnull(df.index[-1]):
                    st.error("No valid dates in your data. Please check your data source.")
                    st.stop()
                last_date = df.index[-1]
                feature_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low']
                if model_choice == "ARIMAX":
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    import warnings
                    import numpy as np
                    warnings.filterwarnings("ignore")
                    y = df['Close']
                    exog_cols = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low']
                    exog = df[exog_cols]
                    model = SARIMAX(y, exog=exog, order=(5,1,0), enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    # For future exog, use the last row repeated
                    last_exog = exog.iloc[[-1]].values
                    exog_forecast = pd.DataFrame(np.repeat(last_exog, forecast_days, axis=0), columns=exog_cols)
                    forecast_obj = model_fit.get_forecast(steps=forecast_days, exog=exog_forecast)
                    forecast = forecast_obj.predicted_mean
                    conf_int = forecast_obj.conf_int(alpha=0.05)
                    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
                    prob_up = (forecast.values - y.iloc[-1] > 0).astype(int)
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Predicted Close': forecast.values,
                        'Lower Bound': conf_int.iloc[:, 0].values,
                        'Upper Bound': conf_int.iloc[:, 1].values,
                        'Prob_Up': prob_up
                    })
                    forecast_df.set_index('Date', inplace=True)
                    st.success(f"ARIMAX prediction for {symbol} ({forecast_days} days ahead):")
                    st.dataframe(forecast_df)
                    import plotly.graph_objs as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual'))
                    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted Close'], mode='lines+markers', name='Predicted'))
                    # Add prediction interval as shaded area
                    fig.add_traces([
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['Upper Bound'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['Lower Bound'],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(width=0),
                            showlegend=True,
                            name='Prediction Interval',
                            hoverinfo='skip'
                        )
                    ])
                    fig.update_layout(title=f"ARIMAX Forecast for {symbol}", xaxis_title="Date", yaxis_title="Price", template=st.session_state.theme)
                    st.plotly_chart(fig, use_container_width=True)
                elif model_choice == "Prophet":
                    try:
                        prophet_df = df.reset_index()
                        prophet_df = prophet_df.rename(columns={prophet_df.columns[0]: 'ds', 'Close': 'y'})
                        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                        m = Prophet(daily_seasonality=True)
                        m.fit(prophet_df[['ds', 'y']])
                        future = m.make_future_dataframe(periods=forecast_days, freq='D')
                        forecast = m.predict(future)
                        forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds').tail(forecast_days)
                        forecast_result['Prob_Up'] = (forecast_result['yhat'].diff().fillna(0) > 0).astype(int)
                        forecast_result.rename(columns={'yhat': 'Predicted Close', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}, inplace=True)
                        st.success(f"Prophet prediction for {symbol} ({forecast_days} days ahead):")
                        st.dataframe(forecast_result)
                        import plotly.graph_objs as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual'))
                        fig.add_trace(go.Scatter(x=forecast_result.index, y=forecast_result['Predicted Close'], mode='lines+markers', name='Predicted'))
                        # Add prediction interval as shaded area
                        fig.add_traces([
                            go.Scatter(
                                x=forecast_result.index,
                                y=forecast_result['Upper Bound'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            go.Scatter(
                                x=forecast_result.index,
                                y=forecast_result['Lower Bound'],
                                mode='lines',
                                fill='tonexty',
                                fillcolor='rgba(0,100,80,0.2)',
                                line=dict(width=0),
                                showlegend=True,
                                name='Prediction Interval',
                                hoverinfo='skip'
                            )
                        ])
                        fig.update_layout(title=f"Prophet Forecast for {symbol}", xaxis_title="Date", yaxis_title="Price", template=st.session_state.theme)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Prophet prediction failed: {e}")
                elif model_choice == "Advanced ML":
                    try:
                        from sklearn.ensemble import RandomForestRegressor
                        import numpy as np
                        # Use technical indicators, volume, and lagged closes as features
                        lagged_lags = [1,2,3]
                        df = add_lagged_features(df, col='Close', lags=lagged_lags)
                        feature_cols = ['SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Volume'] + [f'Close_lag{lag}' for lag in lagged_lags]
                        for col in feature_cols:
                            if col not in df.columns:
                                df[col] = 0
                        X_exog = df[feature_cols]
                        y = df['Close']
                        # Bootstrapped recursive forecast for prediction intervals
                        mean_preds, lower, upper = bootstrapped_recursive_forecast(df, feature_cols, forecast_days, n_bootstrap=30)
                        forecast_dates = [df.index[-1] + pd.Timedelta(days=i+1) for i in range(forecast_days)]
                        prob_up = (mean_preds - y.iloc[-1] > 0).astype(int)
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Predicted Close': mean_preds,
                            'Lower Bound': lower,
                            'Upper Bound': upper,
                            'Prob_Up': prob_up
                        })
                        forecast_df.set_index('Date', inplace=True)
                        st.success(f"Advanced ML (Random Forest, bootstrapped, lagged) prediction for {symbol} ({forecast_days} days ahead):")
                        st.dataframe(forecast_df)

                        import plotly.graph_objs as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual'))
                        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted Close'], mode='lines+markers', name='Predicted'))
                        # Add prediction interval as shaded area
                        fig.add_traces([
                            go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df['Upper Bound'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df['Lower Bound'],
                                mode='lines',
                                fill='tonexty',
                                fillcolor='rgba(0,100,80,0.2)',
                                line=dict(width=0),
                                showlegend=True,
                                name='Prediction Interval',
                                hoverinfo='skip'
                            )
                        ])
                        fig.update_layout(title=f"Advanced ML Forecast for {symbol}", xaxis_title="Date", yaxis_title="Price", template=st.session_state.theme)
                        st.plotly_chart(fig, use_container_width=True)

                        # Walk-forward validation button and logic
                        if st.button("Run Walk-Forward Validation (Backtest)"):
                            st.info("Running walk-forward validation (this may take a few seconds)...")
                            walk_df = walk_forward_validation(df, feature_cols, window_size=180)
                            st.write("### Walk-Forward Validation Results")
                            st.dataframe(walk_df)
                            # Plot actual vs predicted
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=walk_df.index, y=walk_df['Actual Close'], mode='lines', name='Actual'))
                            fig2.add_trace(go.Scatter(x=walk_df.index, y=walk_df['Predicted Close'], mode='lines', name='Predicted'))
                            fig2.update_layout(title="Walk-Forward Validation: Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", template=st.session_state.theme)
                            st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.error(f"Advanced ML prediction failed: {e}")
        else:
            st.caption("Select your options and click Predict to see results.")
    
    with tab3:
        create_enhanced_portfolio_section()
    
    with tab4:
        create_enhanced_screener()
    
    with tab5:
        # Backtesting section
        symbol = st.session_state.get('selected_symbol', 'AAPL')
        
        # Fetch data for backtesting
        if st.button("📊 Load Data for Backtesting"):
            with st.spinner(f'Loading data for {symbol}...'):
                start_date = datetime.date.today() - datetime.timedelta(days=730)  # 2 years of data
                end_date = datetime.date.today()
                df = fetch_enhanced_stock_data(symbol, start_date, end_date, polygon_api_key)
                
                if df is not None and len(df) > 0:
                    st.session_state.backtest_data = df
                    st.success(f"✅ Loaded {len(df)} days of data for {symbol}")
        
        if 'backtest_data' in st.session_state:
            create_backtesting_section(symbol, st.session_state.backtest_data)
        else:
            st.info("Load stock data first to run backtests")
    
    with tab6:
        symbol = st.session_state.get('selected_symbol', 'AAPL')
        create_sentiment_analysis_section(symbol)
    
    with tab7:
        create_currency_exchange_section()
    
    with tab8:  # News tab
        create_news_section()

def format_large_number(val):
    try:
        val = float(val)
        if val >= 1e12:
            return f"${val/1e12:.2f}T"
        elif val >= 1e9:
            return f"${val/1e9:.2f}B"
        elif val >= 1e6:
            return f"${val/1e6:.2f}M"
        else:
            return f"${val:,.0f}"
    except:
        return "N/A"

def walk_forward_validation(df, feature_cols, window_size=180):
    """Perform walk-forward validation for Random Forest."""
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    lagged_lags = [1,2,3]
    df = add_lagged_features(df, col='Close', lags=lagged_lags)
    feature_cols = feature_cols + [f'Close_lag{lag}' for lag in lagged_lags if f'Close_lag{lag}' not in feature_cols]
    df = df.copy().dropna()
    preds = []
    actuals = []
    pred_dates = []
    for i in range(window_size, len(df)-1):
        train = df.iloc[i-window_size:i]
        test = df.iloc[[i+1]]
        X_train = train[feature_cols]
        y_train = train['Close']
        X_test = test[feature_cols]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        preds.append(pred)
        actuals.append(test['Close'].values[0])
        pred_dates.append(test.index[0])
    return pd.DataFrame({'Date': pred_dates, 'Predicted Close': preds, 'Actual Close': actuals}).set_index('Date')

def bootstrapped_recursive_forecast(df, feature_cols, forecast_days, n_bootstrap=30):
    """Recursive forecast with bootstrapped Random Forests for prediction intervals."""
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    lagged_lags = [1,2,3]
    df = add_lagged_features(df, col='Close', lags=lagged_lags)
    feature_cols = feature_cols + [f'Close_lag{lag}' for lag in lagged_lags if f'Close_lag{lag}' not in feature_cols]
    preds_matrix = []
    for b in range(n_bootstrap):
        # Bootstrap sample
        df_boot = df.sample(frac=1, replace=True, random_state=42+b)
        X_exog = df_boot[feature_cols]
        y = df_boot['Close']
        model = RandomForestRegressor(n_estimators=100, random_state=42+b)
        model.fit(X_exog, y)
        df_forecast = df.copy()
        preds = []
        for i in range(forecast_days):
            last_features = df_forecast[feature_cols].iloc[[-1]]
            next_pred = model.predict(last_features)[0]
            preds.append(next_pred)
            new_row = df_forecast.iloc[-1].copy()
            new_row['Close'] = next_pred
            new_row.name = df_forecast.index[-1] + pd.Timedelta(days=1)
            df_forecast = df_forecast.append(new_row)
            df_forecast = calculate_technical_indicators(df_forecast)
            df_forecast = add_lagged_features(df_forecast, col='Close', lags=lagged_lags)
        preds_matrix.append(preds)
    preds_matrix = np.array(preds_matrix)  # shape: (n_bootstrap, forecast_days)
    mean_preds = preds_matrix.mean(axis=0)
    lower = np.percentile(preds_matrix, 2.5, axis=0)
    upper = np.percentile(preds_matrix, 97.5, axis=0)
    return mean_preds, lower, upper

def add_lagged_features(df, col='Close', lags=[1,2,3]):
    """Add lagged features for the specified column."""
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

if __name__ == "__main__":
    main()
