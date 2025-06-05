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

# Initialize session state for caching
if 'client' not in st.session_state:
    st.session_state.client = None

# Initialize Polygon client if API key is provided
if polygon_api_key and (st.session_state.client is None or st.session_state.client.api_key != polygon_api_key):
    try:
        st.session_state.client = RESTClient(polygon_api_key)
        st.sidebar.success("✅ Connected to Polygon API")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {str(e)}")

# Stock input and date range
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    symbol = st.text_input("Stock Symbol", "AAPL").upper()
with col2:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
with col3:
    end_date = st.date_input("End Date", datetime.now())

# Add some space
st.markdown("---")

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    try:
        # Convert dates to string format
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch data
        aggs = []
        for a in st.session_state.client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan='day',
            from_=start_str,
            to=end_str,
            limit=50000
        ):
            aggs.append(a)
            
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
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True, 
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])
    
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
                # Display metrics
                latest = df.iloc[-1]
                prev_day = df.iloc[-2] if len(df) > 1 else latest
                
                # Calculate price change
                price_change = latest['Close'] - prev_day['Close']
                pct_change = (price_change / prev_day['Close']) * 100
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${latest['Close']:.2f}", 
                             f"{price_change:.2f} ({pct_change:.2f}%)",
                             delta_color="normal" if price_change >= 0 else "inverse")
                with col2:
                    st.metric("Day's Range", f"${latest['Low']:.2f} - ${latest['High']:.2f}")
                with col3:
                    st.metric("Volume", f"{latest['Volume']:,}")
                with col4:
                    st.metric("VWAP", f"${latest['VWAP']:.2f}")
                
                # Display the chart
                st.plotly_chart(create_candlestick_chart(df), use_container_width=True)
                
                # Display recent data
                st.subheader("Recent Data")
                st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50']].tail(10).sort_index(ascending=False).style.format({
                    'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 
                    'Close': '{:.2f}', 'SMA_20': '{:.2f}', 'SMA_50': '{:.2f}',
                    'Volume': '{:,.0f}'
                }), use_container_width=True)
                
                # Add some technical analysis
                st.subheader("Technical Indicators")
                
                # Calculate RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Create RSI chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df['Date'], y=rsi, name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="Relative Strength Index (RSI)", height=400)
                
                # Calculate MACD
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                
                # Create MACD chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Bar(x=df['Date'], y=macd-signal, name='Histogram'))
                fig_macd.add_trace(go.Scatter(x=df['Date'], y=macd, name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df['Date'], y=signal, name='Signal', line=dict(color='orange')))
                fig_macd.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                fig_macd.update_layout(title="MACD (12, 26, 9)", height=400)
                
                # Display technical indicators side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_rsi, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_macd, use_container_width=True)
                
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
                    
# Show a message if no API key is provided
if not st.session_state.client:
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
        
# Fetch and display historical data as a line chart
if col3.button("Get Historical"):
    if not polygon_api_key or not symbol:
        st.error("Please provide both the API Key and stock symbol.")
    else:
        try:
            # Define the date range
            from_date = "2024-01-01"
            to_date = "2024-04-29"

            # Fetch historical aggregated data
            data_request = client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=from_date,
                to=to_date
            )

            # Convert data to DataFrame
            chart_data = pd.DataFrame([{
                "timestamp": agg.timestamp,
                "close": agg.close
            } for agg in data_request])

            # Convert timestamp to datetime
            chart_data['date'] = pd.to_datetime(chart_data['timestamp'], unit='ms')

            # Plot the line chart
            st.line_chart(chart_data.set_index('date')['close'])

        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
