import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import ta
import time
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ==================== DATA FETCHING ====================

_data_cache = {}
ALPHAVANTAGE_API_KEY = 'MQT91HLMWHASIDR2'  # Replace with your key from https://www.alphavantage.co/support/#api-key

def fetch_from_alphavantage(ticker):
    """Fetch data from Alpha Vantage (Backup source)"""
    try:
        print(f"🔄 Trying Alpha Vantage API for {ticker}...")
        # Remove exchange suffixes for Alpha Vantage
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={clean_ticker}&outputsize=full&apikey={ALPHAVANTAGE_API_KEY}'
        response = requests.get(url, timeout=15)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df['Volume'] = df['Volume'].astype(int)
            
            # Get last 6 months
            six_months_ago = datetime.now() - timedelta(days=180)
            df = df[df.index >= six_months_ago]
            
            if len(df) > 10:
                print(f"✓ Data fetched from Alpha Vantage ({len(df)} days)")
                return df, {'longName': clean_ticker, 'symbol': ticker}
        elif 'Note' in data:
            print(f"⚠️ Alpha Vantage: {data['Note'][:80]}")
        elif 'Information' in data:
            print(f"⚠️ Alpha Vantage: {data['Information'][:80]}")
        else:
            print(f"⚠️ Alpha Vantage: No data returned for {clean_ticker}")
        return None, None
    except Exception as e:
        print(f"⚠️ Alpha Vantage error: {str(e)[:100]}")
        return None, None

def fetch_from_yahoo_simple(ticker):
    """Direct Yahoo Finance API (Backup method)"""
    try:
        print(f"🔄 Trying direct Yahoo Finance CSV download for {ticker}...")
        
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (180 * 24 * 60 * 60)  # 6 months ago
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        
        if response.status_code == 200 and 'Date,Open' in response.text:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Handle potential 'Adj Close' column
            if 'Adj Close' in df.columns:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            if len(df) > 10:
                print(f"✓ Data fetched from Yahoo Finance CSV ({len(df)} days)")
                return df, {'longName': ticker, 'symbol': ticker}
        else:
            print(f"⚠️ Yahoo CSV: HTTP {response.status_code}")
        return None, None
    except Exception as e:
        print(f"⚠️ Direct Yahoo error: {str(e)[:100]}")
        return None, None

def fetch_from_yfinance(ticker, period='6mo'):
    """Fetch data from yfinance"""
    try:
        print(f"🔄 Trying yfinance for {ticker}...")
        time.sleep(3)  # Increased delay
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval='1d', timeout=20, raise_errors=False)
        
        if not data.empty and len(data) > 10:
            info = {}
            try:
                info = stock.info
            except:
                info = {'longName': ticker, 'symbol': ticker}
            print(f"✓ Data fetched from yfinance ({len(data)} days)")
            return data, info
        return None, None
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower():
            print(f"⚠️ Yahoo Finance rate limit - trying alternatives...")
        else:
            print(f"⚠️ yfinance error: {error_msg[:100]}")
        return None, None

def fetch_stock_data(ticker, period='6mo'):
    """Fetch stock data with multiple fallback sources"""
    cache_key = f"{ticker}_{period}"
    
    # Check cache first
    if cache_key in _data_cache:
        cache_time, cached_data, cached_info = _data_cache[cache_key]
        if time.time() - cache_time < 600:  # 10 minute cache
            print(f"📦 Using cached data for {ticker}")
            return cached_data, cached_info
    
    data, info = None, None
    
    # Method 1: Try yfinance (Primary)
    data, info = fetch_from_yfinance(ticker, period)
    
    # Method 2: Try direct Yahoo Finance API
    if data is None or data.empty:
        time.sleep(2)
        data, info = fetch_from_yahoo_simple(ticker)
    
    # Method 3: Try Alpha Vantage (Backup)
    if data is None or data.empty:
        time.sleep(2)
        data, info = fetch_from_alphavantage(ticker)
    
    # Cache successful result
    if data is not None and not data.empty:
        _data_cache[cache_key] = (time.time(), data, info)
        return data, info
    
    return None, None

def create_technical_features(df):
    """Add technical indicators"""
    data = df.copy()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Weekly_Return'] = data['Close'].pct_change(5)
    data['Monthly_Return'] = data['Close'].pct_change(20)
    data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['Bollinger_Upper'] = bollinger.bollinger_hband()
    data['Bollinger_Lower'] = bollinger.bollinger_lband()
    data['Bollinger_Middle'] = bollinger.bollinger_mavg()
    
    return data

def analyze_stock(ticker_symbol):
    """Analyze stock and return processed data"""
    tried_tickers = []
    data = None
    info = None
    ticker_to_use = None
    
    # Try with .NS suffix for Indian stocks
    if not (ticker_symbol.endswith('.NS') or ticker_symbol.endswith('.BO')):
        us_stocks = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD']
        
        if ticker_symbol.upper() not in us_stocks:
            ticker_ns = f"{ticker_symbol}.NS"
            data_ns, info_ns = fetch_stock_data(ticker_ns)
            tried_tickers.append(ticker_ns)
            
            if data_ns is not None and not data_ns.empty:
                data = data_ns
                info = info_ns
                ticker_to_use = ticker_ns
            else:
                ticker_bo = f"{ticker_symbol}.BO"
                data_bo, info_bo = fetch_stock_data(ticker_bo)
                tried_tickers.append(ticker_bo)
                
                if data_bo is not None and not data_bo.empty:
                    data = data_bo
                    info = info_bo
                    ticker_to_use = ticker_bo
        
        if data is None or data.empty:
            data_raw, info_raw = fetch_stock_data(ticker_symbol)
            tried_tickers.append(ticker_symbol)
            if data_raw is not None and not data_raw.empty:
                data = data_raw
                info = info_raw
                ticker_to_use = ticker_symbol
    else:
        data, info = fetch_stock_data(ticker_symbol)
        tried_tickers.append(ticker_symbol)
        if data is not None and not data.empty:
            ticker_to_use = ticker_symbol
    
    if data is None or data.empty or len(data) < 20:
        return None
    
    data_with_features = create_technical_features(data)
    
    return {
        'ticker': ticker_to_use,
        'data': data_with_features,
        'info': info
    }

# ==================== PLOTLY VISUALIZATION ====================

def create_plotly_dashboard(result):
    """Create interactive Plotly dashboard with modern design"""
    df = result['data']
    info = result['info']
    ticker = result['ticker']
    
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    
    # Color scheme
    colors = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#10b981',
        'danger': '#ef4444',
        'warning': '#f59e0b',
        'info': '#3b82f6',
        'dark': '#1f2937',
        'light': '#f9fafb',
        'grid': '#e5e7eb'
    }
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Price chart with Bollinger Bands (Candlestick)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color=colors['success'],
        decreasing_line_color=colors['danger'],
        increasing_fillcolor=colors['success'],
        decreasing_fillcolor=colors['danger']
    ), row=1, col=1)
    
    # Moving averages with gradient effect
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'],
        name='MA20',
        line=dict(color=colors['info'], width=2),
        opacity=0.8
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA50'],
        name='MA50',
        line=dict(color=colors['warning'], width=2),
        opacity=0.8
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Bollinger_Upper'],
        name='BB Upper',
        line=dict(color=colors['grid'], width=1, dash='dot'),
        showlegend=False,
        opacity=0.5
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Bollinger_Lower'],
        name='BB Lower',
        line=dict(color=colors['grid'], width=1, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.05)',
        showlegend=False,
        opacity=0.5
    ), row=1, col=1)
    
    # Volume bars with gradient colors
    volume_colors = [colors['success'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                     else colors['danger'] for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume',
        marker=dict(
            color=volume_colors,
            line=dict(width=0)
        ),
        showlegend=False,
        opacity=0.7
    ), row=2, col=1)
    
    # RSI with zones
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        name='RSI',
        line=dict(color=colors['primary'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ), row=3, col=1)
    
    # RSI zones
    fig.add_hrect(y0=70, y1=100, fillcolor=colors['danger'], opacity=0.1, 
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor=colors['success'], opacity=0.1, 
                  line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=colors['danger'], 
                  opacity=0.5, line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=colors['success'], 
                  opacity=0.5, line_width=1, row=3, col=1)
    
    # MACD with histogram
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        name='MACD',
        line=dict(color=colors['primary'], width=2)
    ), row=4, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'],
        name='Signal',
        line=dict(color=colors['danger'], width=2)
    ), row=4, col=1)
    
    histogram_colors = [colors['success'] if val > 0 else colors['danger'] 
                        for val in df['MACD_Histogram']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['MACD_Histogram'],
        name='Histogram',
        marker=dict(color=histogram_colors, line=dict(width=0)),
        showlegend=False,
        opacity=0.6
    ), row=4, col=1)
    
    # Update layout with modern styling
    fig.update_layout(
        height=900,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', 
                  size=12, color=colors['dark']),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=colors['grid'],
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=70, r=30, t=40, b=40),
        xaxis4=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            showline=True,
            linewidth=1,
            linecolor=colors['grid']
        )
    )
    
    # Update all axes
    for i in range(1, 5):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            showline=True,
            linewidth=1,
            linecolor=colors['grid'],
            row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            showline=True,
            linewidth=1,
            linecolor=colors['grid'],
            row=i, col=1
        )
    
    return fig

def get_signal_badge(rsi, macd, macd_signal, price, ma50):
    """Generate trading signals with colors"""
    signals = []
    
    if rsi > 70:
        signals.append(('Overbought', 'danger', '⚠️'))
    elif rsi < 30:
        signals.append(('Oversold', 'success', '✓'))
    else:
        signals.append(('Neutral RSI', 'secondary', '●'))
    
    if macd > macd_signal:
        signals.append(('Bullish MACD', 'success', '↑'))
    else:
        signals.append(('Bearish MACD', 'danger', '↓'))
    
    if price > ma50:
        signals.append(('Above MA50', 'success', '↑'))
    else:
        signals.append(('Below MA50', 'danger', '↓'))
    
    return signals

# ==================== DASH APP ====================

# Custom CSS
custom_css = '''
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f5f7fa;
    margin: 0;
    padding: 0;
}

.main-container {
    background: white;
    border-radius: 0;
    box-shadow: none;
    margin: 0;
    padding: 0;
    overflow: hidden;
}

.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 50px 30px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    position: relative;
    z-index: 1;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1rem;
    opacity: 0.9;
    font-weight: 400;
    position: relative;
    z-index: 1;
    margin-top: 8px;
}

.search-section {
    padding: 30px;
    background: white;
    border-bottom: 1px solid #e5e7eb;
}

.stat-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #e5e7eb;
    transition: all 0.2s ease;
    height: 100%;
}

.stat-card:hover {
    border-color: #667eea;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
}

.stat-label {
    font-size: 0.75rem;
    color: #6b7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    line-height: 1;
}

.signal-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 16px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px;
}

.chart-container {
    padding: 30px;
    background: white;
}

.info-section {
    padding: 30px;
    background: #f9fafb;
    border-top: 1px solid #e5e7eb;
}

.custom-input {
    border-radius: 6px !important;
    border: 1px solid #d1d5db !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}

.custom-input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    outline: none !important;
}

.custom-button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.custom-button:hover {
    opacity: 0.9;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

.footer {
    text-align: center;
    padding: 25px;
    color: #6b7280;
    font-size: 0.85rem;
    background: white;
    border-top: 1px solid #e5e7eb;
}

/* Clean up default Dash/Bootstrap spacing */
.container-fluid {
    padding: 0 !important;
}

.row {
    margin: 0 !important;
}
'''

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

app.title = "Hand&Brain - Stock Analytics"

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
''' + custom_css + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    
    dbc.Container([
        # Hero Section
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Img(src='/assets/logo.png', style={
                        'height': '80px',
                        'marginBottom': '20px',
                        'filter': 'brightness(0) invert(1)'
                    })
                ], width=12, className="text-center")
            ]),
            html.H1("Hand&Brain", className="hero-title"),
            html.P("Advanced Technical Analysis & Real-time Market Intelligence", 
                   className="hero-subtitle")
        ], className="hero-section"),
        
        # Search Section
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.Input(
                            id='ticker-input',
                            placeholder='Enter stock symbol (e.g., RELIANCE, AAPL, TSLA)',
                            type='text',
                            className='custom-input',
                            style={'borderRight': 'none'}
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-search me-2"), "Analyze"],
                            id='analyze-button',
                            className='custom-button',
                            n_clicks=0
                        )
                    ], className="mb-3"),
                    html.Div([
                        dbc.Badge("Indian Stocks", color="light", text_color="dark", className="me-2"),
                        html.Small("RELIANCE • TCS • INFY • HDFCBANK", className="text-muted me-3"),
                        dbc.Badge("US Stocks", color="light", text_color="dark", className="me-2"),
                        html.Small("AAPL • TSLA • MSFT • GOOGL", className="text-muted")
                    ])
                ], width=12)
            ])
        ], className="search-section"),
        
        # Loading and Error Messages
        dcc.Loading(
            id="loading",
            type="circle",
            color="#667eea",
            children=[
                html.Div(id='error-message'),
                
                # Stats Cards Section
                html.Div(id='stats-section'),
                
                # Chart Section
                html.Div(id='chart-section'),
                
                # Signals Section
                html.Div(id='signals-section'),
            ]
        ),
        
        # Footer
        html.Div([
            html.P([
                "⚠️ Disclaimer: This platform is for informational and educational purposes only. ",
                html.Strong("Not financial advice."),
                " Always consult with a qualified financial advisor before making investment decisions."
            ], className="mb-2"),
            html.P([
                "Powered by ",
                html.Strong("Hand&Brain"),
                " • Real-time data via Yahoo Finance & Alpha Vantage"
            ], className="mb-0", style={'fontSize': '0.8rem', 'opacity': '0.7'})
        ], className="footer")
        
    ], fluid=True, className="main-container", style={'maxWidth': '100%', 'margin': '0'})
], style={'background': '#f5f7fa', 'minHeight': '100vh', 'padding': '0'})

@app.callback(
    [Output('stats-section', 'children'),
     Output('chart-section', 'children'),
     Output('signals-section', 'children'),
     Output('error-message', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('ticker-input', 'value')]
)
def update_dashboard(n_clicks, ticker):
    if n_clicks == 0 or not ticker:
        return None, None, None, None
    
    ticker = ticker.strip().upper()
    result = analyze_stock(ticker)
    
    if result is None:
        error = dbc.Alert([
            html.I(className="fas fa-exclamation-circle me-2"),
            "Unable to fetch data for this ticker. Please verify the symbol and try again."
        ], color="danger", className="m-4")
        return None, None, None, error
    
    # Extract data
    df = result['data']
    info = result['info']
    ticker = result['ticker']
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    
    current_price = df['Close'].iloc[-1]
    daily_change = df['Daily_Return'].iloc[-1] * 100
    weekly_change = df['Weekly_Return'].iloc[-1] * 100 if not pd.isna(df['Weekly_Return'].iloc[-1]) else 0
    monthly_change = df['Monthly_Return'].iloc[-1] * 100 if not pd.isna(df['Monthly_Return'].iloc[-1]) else 0
    rsi = df['RSI'].iloc[-1]
    volume = df['Volume'].iloc[-1]
    
    # Stats Cards
    stats = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Current Price", className="stat-label"),
                        html.H2(f"{currency}{current_price:.2f}", className="stat-value", 
                               style={'color': '#667eea'})
                    ], className="stat-card")
                ], width=12, lg=3, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("Daily Change", className="stat-label"),
                        html.H2(f"{daily_change:+.2f}%", className="stat-value",
                               style={'color': '#10b981' if daily_change > 0 else '#ef4444'})
                    ], className="stat-card")
                ], width=12, lg=3, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("RSI (14)", className="stat-label"),
                        html.H2(f"{rsi:.1f}", className="stat-value",
                               style={'color': '#ef4444' if rsi > 70 else '#10b981' if rsi < 30 else '#667eea'})
                    ], className="stat-card")
                ], width=12, lg=3, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("Volume", className="stat-label"),
                        html.H2(f"{volume/1e6:.1f}M", className="stat-value",
                               style={'color': '#667eea'})
                    ], className="stat-card")
                ], width=12, lg=3, className="mb-3"),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Weekly Change", className="stat-label"),
                        html.H4(f"{weekly_change:+.2f}%", 
                               style={'color': '#10b981' if weekly_change > 0 else '#ef4444', 'margin': '0', 'fontSize': '1.25rem'})
                    ], className="stat-card")
                ], width=12, lg=4, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("Monthly Change", className="stat-label"),
                        html.H4(f"{monthly_change:+.2f}%",
                               style={'color': '#10b981' if monthly_change > 0 else '#ef4444', 'margin': '0', 'fontSize': '1.25rem'})
                    ], className="stat-card")
                ], width=12, lg=4, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("Market Cap", className="stat-label"),
                        html.H4(f"{currency}{info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A",
                               style={'color': '#667eea', 'margin': '0', 'fontSize': '1.25rem'})
                    ], className="stat-card")
                ], width=12, lg=4, className="mb-3"),
            ])
        ], fluid=True)
    ], style={'padding': '30px', 'background': '#f9fafb'})
    
    # Chart
    fig = create_plotly_dashboard(result)
    chart = html.Div([
        dcc.Graph(
            figure=fig,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            }
        )
    ], className="chart-container")
    
    # Trading Signals
    signals = get_signal_badge(
        rsi, 
        df['MACD'].iloc[-1], 
        df['MACD_Signal'].iloc[-1],
        current_price,
        df['MA50'].iloc[-1]
    )
    
    signal_badges = []
    for signal_text, signal_color, icon in signals:
        signal_badges.append(
            dbc.Badge([
                html.Span(icon, style={'marginRight': '8px'}),
                signal_text
            ], color=signal_color, className="signal-badge")
        )
    
    signals_section = html.Div([
        html.H4("📊 Technical Signals", className="mb-3", style={'color': '#1f2937'}),
        html.Div(signal_badges, className="d-flex flex-wrap")
    ], className="info-section")
    
    return stats, chart, signals_section, None

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Hand&Brain Analytics Dashboard...")
    print("="*60)
    print("\n📊 Open your browser and go to: http://127.0.0.1:8050")
    print("\n💡 Tip: Get free Alpha Vantage API key for backup data source")
    print("   Visit: https://www.alphavantage.co/support/#api-key")
    print("   Update ALPHAVANTAGE_API_KEY variable in the code")
    print("="*60)
    app.run(debug=True, host='127.0.0.1', port=8050)