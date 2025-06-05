import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from polygon import RESTClient
import numpy as np

# Set page config for better layout
st.set_page_config(page_title="Stockingly - Stock Analysis", layout="wide", page_icon="📈")

# Sidebar for inputs
st.sidebar.title("🔑 API & Settings")
polygon_api_key = st.sidebar.text_input("Polygon API Key", type="password")
st.sidebar.markdown("---")

# Main app title
st.title("📈 Stockingly - Interactive Stock Analysis")

# Add a welcome message with expandable info
with st.expander("ℹ️ Welcome to Stockingly - Your Stock Analysis Companion", expanded=True):
    st.markdown("""
    Welcome to Stockingly! This app helps you analyze stock performance using various technical indicators. 
    Whether you're a beginner or an experienced investor, these tools can help you understand market trends.
    
    ### How to use:
    1. Select a stock from the Quick Select tab or enter a symbol manually
    2. Choose a date range for your analysis
    3. Click 'Fetch Data' to see the analysis
    4. Explore different tabs for various technical indicators
    
    Hover over the ℹ️ icons for more information about each section.
    """)
    
    st.markdown("---")

# Initialize session state for caching
if 'client' not in st.session_state:
    st.session_state.client = None

# Initialize Polygon client if API key is provided
if polygon_api_key and (st.session_state.client is None or getattr(st.session_state.client, '_api_key', None) != polygon_api_key):
    try:
        st.session_state.client = RESTClient(polygon_api_key)
        st.session_state.client._api_key = polygon_api_key  # Store API key as attribute
        st.sidebar.success("✅ Connected to Polygon API")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {str(e)}")
        st.session_state.client = None

# Popular stocks dictionary with unique tickers
popular_stocks = {
    'Tech': [
        {'name': 'Apple', 'ticker': 'AAPL'},
        {'name': 'Microsoft', 'ticker': 'MSFT'},
        {'name': 'Alphabet (Google)', 'ticker': 'GOOGL'},
        {'name': 'Amazon', 'ticker': 'AMZN'},
        {'name': 'Meta (Facebook)', 'ticker': 'META'},
        {'name': 'Tesla', 'ticker': 'TSLA'},
        {'name': 'NVIDIA', 'ticker': 'NVDA'}
    ],
    'Finance': [
        {'name': 'JPMorgan Chase', 'ticker': 'JPM'},
        {'name': 'Visa', 'ticker': 'V'},
        {'name': 'Bank of America', 'ticker': 'BAC'},
        {'name': 'Goldman Sachs', 'ticker': 'GS'}
    ],
    'Retail': [
        {'name': 'Walmart', 'ticker': 'WMT'},
        {'name': 'Target', 'ticker': 'TGT'},
        {'name': 'Costco', 'ticker': 'COST'},
        {'name': 'Home Depot', 'ticker': 'HD'}
    ],
    'ETFs': [
        {'name': 'S&P 500', 'ticker': 'SPY'},
        {'name': 'NASDAQ 100', 'ticker': 'QQQ'},
        {'name': 'Dow Jones', 'ticker': 'DIA'},
        {'name': 'Total Market', 'ticker': 'VTI'}
    ]
}

# Stock input and date range
st.subheader("Stock Selection")

# Create tabs for different categories
tab1, tab2 = st.tabs(["Quick Select", "Manual Entry"])

with tab1:
    st.write("Select a stock from popular options:")
    
    # Create columns for each category
    cols = st.columns(len(popular_stocks))
    
    for i, (category, stocks) in enumerate(popular_stocks.items()):
        with cols[i]:
            st.markdown(f"**{category}**")
            for stock in stocks:
                # Create a unique key using both category and ticker
                btn_key = f"btn_{category}_{stock['ticker']}"
                if st.button(f"{stock['name']} ({stock['ticker']})", key=btn_key):
                    st.session_state.selected_symbol = stock['ticker']
    
    # Display selected symbol with some spacing
    st.markdown("---")
    selected_symbol = st.session_state.get('selected_symbol', 'AAPL')
    st.markdown(f"### Selected: **{selected_symbol}**")
    
with tab2:
    selected_symbol = st.text_input("Or enter a stock symbol", 
                                   value=st.session_state.get('selected_symbol', 'AAPL'),
                                   key="manual_symbol").upper()
    st.session_state.selected_symbol = selected_symbol

# Date range selection
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Use the selected symbol
symbol = selected_symbol

# Add some space
st.markdown("---")

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    if not hasattr(st.session_state, 'client') or not st.session_state.client:
        st.error("API client not initialized. Please check your API key.")
        return None
        
    try:
        # Convert dates to string format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch data
        aggs = []
        try:
            for a in st.session_state.client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan='day',
                from_=start_str,
                to=end_str,
                limit=50000
            ):
                aggs.append(a)
        except Exception as e:
            st.error(f"Error fetching data from Polygon API: {str(e)}")
            st.info("Please check your API key and ensure it has the correct permissions.")
            return None
            
        if not aggs:
            st.error("No data found for the given symbol and date range.")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame([{
            'Date': pd.to_datetime(agg.timestamp, unit='ms'),
            'Open': agg.open,
            'High': agg.high,
            'Low': agg.low,
            'Close': agg.close,
            'Volume': agg.volume,
            'VWAP': agg.vwap
        } for agg in aggs])
        
        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        return df.sort_values('Date')
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to create candlestick chart
def create_candlestick_chart(df):
    """
    Creates an interactive candlestick chart with volume bars.
    
    Args:
        df (DataFrame): Stock data with OHLCV (Open, High, Low, Close, Volume)
        
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    with st.expander("📊 Understanding Price Charts"):
        st.markdown("""
        ### Price Charts 101
        - **Candlesticks** show the open, high, low, and close prices for each period
        - **Green** candles mean the price increased (close > open)
        - **Red** candles mean the price decreased (close < open)
        - The **wicks** show the price range during the period
        - **Volume bars** show trading activity - higher volume means more interest in the stock
        """)
    
    # Create subplots for price and volume
    fig = make_subplots(rows=2, cols=1, 
                      shared_xaxes=True, 
                      vertical_spacing=0.1,
                      row_heights=[0.7, 0.3],
                      subplot_titles=["Price with Moving Averages", "Volume"])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df['Date'],
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'],
                               name='Price'),
                 row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df['Date'], 
                           y=df['SMA_20'], 
                           name='20-day SMA',
                           line=dict(color='blue', width=1.5)),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Date'], 
                           y=df['SMA_50'], 
                           name='50-day SMA',
                           line=dict(color='orange', width=1.5)),
                 row=1, col=1)
    
    # Volume
    colors = ['green' if row['Open'] - row['Close'] >= 0 
             else 'red' for index, row in df.iterrows()]
    
    fig.add_trace(go.Bar(x=df['Date'], 
                        y=df['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.5),
                 row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price & Volume',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_dark'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# Main app logic
if st.session_state.client and symbol:
    if st.button("Fetch Data"):
        with st.spinner('Fetching stock data...'):
            df = fetch_stock_data(symbol, start_date, end_date)
            
            if df is not None:
                # Display metrics with explanations
                latest = df.iloc[-1]
                prev_day = df.iloc[-2] if len(df) > 1 else latest
                
                # Calculate price change
                price_change = latest['Close'] - prev_day['Close']
                pct_change = (price_change / prev_day['Close']) * 100
                
                # Calculate 52-week high/low if we have enough data
                week_52_high = df['High'].max()
                week_52_low = df['Low'].min()
                
                # Display metrics in columns with tooltips
                st.subheader("Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    with st.container():
                        st.metric("Current Price", f"${latest['Close']:.2f}", 
                                 f"{price_change:.2f} ({pct_change:.2f}%)",
                                 delta_color="normal" if price_change >= 0 else "inverse")
                        st.caption("Latest closing price with daily change")
                        
                with col2:
                    with st.container():
                        st.metric("Day's Range", f"${latest['Low']:.2f} - ${latest['High']:.2f}")
                        st.caption("Lowest and highest price today")
                        
                with col3:
                    with st.container():
                        st.metric("Volume", f"{latest['Volume']:,}")
                        st.caption("Number of shares traded today")
                        
                with col4:
                    with st.container():
                        st.metric("52-Week Range", f"${week_52_low:.2f} - ${week_52_high:.2f}")
                        st.caption("52-week high and low prices")
                
                st.markdown("---")
                
                # Initialize session state for checkboxes if they don't exist
                if 'show_bollinger' not in st.session_state:
                    st.session_state.show_bollinger = False
                if 'show_ichimoku' not in st.session_state:
                    st.session_state.show_ichimoku = False
                
                st.subheader("Price Analysis")
                
                # Create and display the chart
                fig = create_candlestick_chart(df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display recent data
                st.subheader("Recent Data")
                st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50']].tail(10).sort_index(ascending=False).style.format({
                    'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 
                    'Close': '{:.2f}', 'SMA_20': '{:.2f}', 'SMA_50': '{:.2f}',
                    'Volume': '{:,.0f}'
                }), use_container_width=True)
                
                # Technical Analysis Section
                st.subheader("Technical Analysis")
                
                # Create tabs for different indicators
                tab_rsi, tab_macd, tab_learn = st.tabs(["RSI", "MACD", "Learn More"])
                
                with tab_rsi:
                    st.markdown("### Relative Strength Index (RSI)")
                    st.markdown("""
                    RSI measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
                    - **Above 70**: Potentially overbought (might be a good time to sell)
                    - **Below 30**: Potentially oversold (might be a good time to buy)
                    - **Between 30-70**: Generally considered neutral
                    """)
                    
                    # Calculate RSI
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Create RSI chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=rsi, name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.1, 
                                     annotation_text="Overbought", annotation_position="top left")
                    fig_rsi.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.1,
                                     annotation_text="Oversold", annotation_position="bottom left")
                    fig_rsi.add_hline(y=50, line_dash="dash", line_color="grey")
                    fig_rsi.update_layout(height=400, showlegend=False,
                                        xaxis_title="Date", yaxis_title="RSI Value",
                                        margin=dict(t=40, b=40, l=40, r=40))
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                with tab_macd:
                    st.markdown("### Moving Average Convergence Divergence (MACD)")
                    st.markdown("""
                    MACD shows the relationship between two moving averages of a security's price.
                    - **MACD Line (Blue)**: 12-day EMA - 26-day EMA
                    - **Signal Line (Orange)**: 9-day EMA of MACD Line
                    - **Histogram**: Difference between MACD and Signal Line
                    
                    **Trading Signals**:
                    - When MACD crosses above Signal: Potential buy signal
                    - When MACD crosses below Signal: Potential sell signal
                    """)
                    
                    # Calculate MACD
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    macd = exp1 - exp2
                    signal = macd.ewm(span=9, adjust=False).mean()
                    
                    # Create MACD chart
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Bar(x=df['Date'], y=macd-signal, name='Histogram',
                                            marker_color=['green' if val >= 0 else 'red' for val in (macd-signal)]))
                    fig_macd.add_trace(go.Scatter(x=df['Date'], y=macd, name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=df['Date'], y=signal, name='Signal', line=dict(color='orange')))
                    fig_macd.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                    fig_macd.update_layout(height=500, showlegend=True,
                                         xaxis_title="Date", yaxis_title="MACD",
                                         margin=dict(t=40, b=40, l=40, r=40))
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                with tab_learn:
                    st.markdown("### Understanding Technical Analysis")
                    st.markdown("""
                    #### Why Use Technical Analysis?
                    Technical analysis helps investors identify potential trading opportunities by analyzing statistical trends 
                    gathered from trading activity, such as price movement and volume.
                    
                    #### Key Concepts:
                    - **Trends**: The general direction in which a security's price is moving
                    - **Support & Resistance**: Price levels where a stock tends to stop and reverse
                    - **Volume**: The number of shares traded, indicating the strength of a price move
                    - **Moving Averages**: Help smooth out price data to identify trends
                    
                    #### Risk Management:
                    - Never rely on a single indicator
                    - Consider using stop-loss orders
                    - Be aware of market conditions and news
                    - Diversify your investments
                    
                    Remember, technical analysis is not foolproof and should be used in conjunction with other 
                    forms of analysis and risk management techniques.
                    """)
                
                # Removed duplicate chart display
                
                # Add some basic analysis
                st.subheader("Market Analysis")
                
                # Price trend analysis
                price_trend = ""
                if len(df) >= 50:
                    short_term = df['Close'].iloc[-20:].mean()
                    long_term = df['Close'].iloc[-50:].mean()
                    if short_term > long_term:
                        price_trend = "📈 The stock is in an **uptrend** (20-day SMA > 50-day SMA)."
                    else:
                        price_trend = "📉 The stock is in a **downtrend** (20-day SMA < 50-day SMA)."
                
                # Volume analysis
                volume_analysis = ""
                avg_volume = df['Volume'].mean()
                if latest['Volume'] > avg_volume * 1.5:
                    volume_analysis = "🔊 **High volume** detected in the latest trading session, indicating strong interest."
                
                # RSI analysis
                rsi_analysis = ""
                current_rsi = rsi.iloc[-1]
                if current_rsi > 70:
                    rsi_analysis = "⚠️ **Overbought** (RSI > 70) - The stock may be overvalued in the short term."
                elif current_rsi < 30:
                    rsi_analysis = "🛒 **Oversold** (RSI < 30) - The stock may be undervalued in the short term."
                
                # Display analysis
                analysis = []
                if price_trend:
                    analysis.append(f"- {price_trend}")
                if volume_analysis:
                    analysis.append(f"- {volume_analysis}")
                if rsi_analysis:
                    analysis.append(f"- {rsi_analysis}")
                
                if analysis:
                    st.markdown("\n".join(analysis))
                else:
                    st.info("Not enough data for detailed analysis. Try selecting a longer time period.")
                
                # Add some educational content
                with st.expander("ℹ️ How to interpret these charts"):
                    st.markdown("""
                    - **Candlestick Chart**: Shows the open, high, low, and close prices for each day.
                    - **Moving Averages**: Help identify trends. When the shorter-term (20-day) SMA is above the longer-term (50-day) SMA, it may indicate an uptrend.
                    - **Volume Bars**: Show trading volume. Green bars indicate price increases, red bars indicate decreases.
                    - **RSI (Relative Strength Index)**: Measures momentum. Values above 70 suggest overbought conditions, below 30 suggest oversold.
                    - **MACD (Moving Average Convergence Divergence)**: A trend-following momentum indicator.
                    """)
else:
    # Show a message if no API key is provided
    st.warning("Please enter a valid Polygon API Key in the sidebar to begin.")
    
    # Add some sample visualizations to demonstrate what the app can do
    st.info("💡 Once you enter your API key, you'll be able to:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Interactive Charts
        - Candlestick charts
        - Volume analysis
        - Moving averages
        """)
        
    with col2:
        st.markdown("""
        ### 📈 Technical Analysis
        - RSI (Relative Strength Index)
        - MACD
        - Trend analysis
        """)
        
    with col3:
        st.markdown("""
        ### 🔍 Market Insights
        - Price trends
        - Volume analysis
        - Support/Resistance levels
        """)
