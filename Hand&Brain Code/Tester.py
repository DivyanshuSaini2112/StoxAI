import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import ta
import time
import requests
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pickle
import os
from pathlib import Path
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

CACHE_DIR = Path("./stock_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY = 3600  # 1 hour

# Professional Color Scheme
COLORS = {
    'primary': '#0066FF',
    'secondary': '#00D9FF',
    'success': '#00E676',
    'danger': '#FF1744',
    'warning': '#FFD600',
    'dark': '#0A1929',
    'dark_secondary': '#1A2332',
    'card_bg': '#132F4C',
    'text': '#E7EBF0',
    'text_secondary': '#B2BAC2',
    'border': '#1E3A5F',
    'grid': '#1E3A5F',
}

# ==================== CACHING SYSTEM ====================

class CacheManager:
    """Enhanced caching with compression"""
    
    @staticmethod
    def get_cache_path(ticker, data_type='price'):
        return CACHE_DIR / f"{ticker}_{data_type}.pkl"
    
    @staticmethod
    def save_to_cache(ticker, data, info, data_type='price'):
        try:
            cache_path = CacheManager.get_cache_path(ticker, data_type)
            cache_data = {
                'timestamp': time.time(),
                'data': data,
                'info': info
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            return True
        except Exception as e:
            return False
    
    @staticmethod
    def load_from_cache(ticker, data_type='price'):
        try:
            cache_path = CacheManager.get_cache_path(ticker, data_type)
            
            if not cache_path.exists():
                return None, None
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if time.time() - cache_data['timestamp'] > CACHE_EXPIRY:
                return None, None
            
            return cache_data['data'], cache_data['info']
        except Exception as e:
            return None, None

# ==================== DATA FETCHING ====================

def fetch_from_yfinance(ticker, period='6mo', max_retries=3):
    """Fetch data from yfinance with retry logic"""
    try:
        # Check cache first
        cached_data, cached_info = CacheManager.load_from_cache(ticker, 'yfinance')
        if cached_data is not None:
            print(f"✓ Loaded {ticker} from cache")
            return cached_data, cached_info
        
        print(f"📡 Fetching {ticker} from yfinance...")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = min(2 ** attempt, 5)  # Cap at 5 seconds
                    print(f"   ⏳ Retry {attempt + 1}/{max_retries} (waiting {wait_time}s)...")
                    time.sleep(wait_time)
                else:
                    time.sleep(0.5)  # Small initial delay
                
                # Try to fetch data
                stock = yf.Ticker(ticker)
                
                # Method 1: Try with period parameter
                data = stock.history(period=period, interval='1d', auto_adjust=True, actions=False)
                
                # If that fails, try with explicit date range
                if data.empty or len(data) < 10:
                    print(f"   ⚠️ Period method failed, trying date range...")
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=180)
                    data = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=True, actions=False)
                
                if not data.empty and len(data) > 10:
                    print(f"   ✅ Successfully fetched {len(data)} days of data")
                    
                    # Ensure required columns exist
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in data.columns for col in required_cols):
                        info = {'longName': ticker, 'symbol': ticker, 'currency': 'USD'}
                        
                        # Try to get additional info (but don't fail if it doesn't work)
                        try:
                            stock_info = stock.info
                            if stock_info and isinstance(stock_info, dict):
                                info.update(stock_info)
                        except Exception as e:
                            print(f"   ⚠️ Could not fetch info (non-critical): {str(e)[:50]}")
                        
                        CacheManager.save_to_cache(ticker, data, info, 'yfinance')
                        return data, info
                    else:
                        print(f"   ⚠️ Missing required columns: {data.columns.tolist()}")
                else:
                    print(f"   ⚠️ Insufficient data: {len(data) if not data.empty else 0} days")
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ Error on attempt {attempt + 1}: {error_msg[:80]}")
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"   ❌ All retry attempts exhausted")
        
        return None, None
        
    except Exception as e:
        print(f"❌ Fatal error in fetch_from_yfinance: {str(e)[:100]}")
        return None, None

def fetch_stock_data(ticker, period='6mo'):
    """Master fetch function with intelligent routing"""
    ticker_upper = ticker.strip().upper()
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing: {ticker_upper}")
    print(f"{'='*60}")
    
    # Auto-add .NS for Indian stocks
    if not (ticker_upper.endswith('.NS') or ticker_upper.endswith('.BO')):
        us_stocks = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'META', 
                     'NFLX', 'NVDA', 'AMD', 'INTC', 'IBM', 'ORCL', 'CSCO', 'BA',
                     'DIS', 'V', 'MA', 'JPM', 'BAC', 'WMT', 'PG', 'JNJ', 'UNH']
        if ticker_upper not in us_stocks:
            print(f"💡 Assuming Indian stock, adding .NS suffix")
            ticker_upper = f"{ticker_upper}.NS"
    
    # Try primary ticker
    data, info = fetch_from_yfinance(ticker_upper, period)
    
    # If .NS failed, try .BO (BSE)
    if (data is None or data.empty) and ticker_upper.endswith('.NS'):
        ticker_bo = ticker_upper.replace('.NS', '.BO')
        print(f"💡 NSE failed, trying BSE: {ticker_bo}")
        data, info = fetch_from_yfinance(ticker_bo, period)
        if data is not None and not data.empty:
            ticker_upper = ticker_bo
    
    # If Indian stock failed, try without suffix (maybe it's US stock)
    if (data is None or data.empty) and (ticker_upper.endswith('.NS') or ticker_upper.endswith('.BO')):
        ticker_clean = ticker_upper.replace('.NS', '').replace('.BO', '')
        print(f"💡 Trying without suffix: {ticker_clean}")
        data, info = fetch_from_yfinance(ticker_clean, period)
        if data is not None and not data.empty:
            ticker_upper = ticker_clean
    
    if data is None or data.empty:
        print(f"❌ Failed to fetch data for {ticker}")
        print(f"{'='*60}\n")
    else:
        print(f"✅ Successfully loaded {len(data)} days for {ticker_upper}")
        print(f"{'='*60}\n")
    
    return data, info, ticker_upper

def fetch_stock_news(ticker):
    """Fetch news with multiple sources"""
    try:
        cache_path = CACHE_DIR / f"{ticker}_news.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if time.time() - cached['timestamp'] < CACHE_EXPIRY:
                    return cached.get('news', [])
            except:
                pass
        
        news = []
        
        try:
            time.sleep(1)
            stock = yf.Ticker(ticker)
            news_data = stock.news
            
            if news_data and isinstance(news_data, list):
                for item in news_data[:6]:
                    try:
                        title = item.get('title', '')
                        if title and len(title) > 5:
                            news.append({
                                'title': title[:120],
                                'source': item.get('publisher', 'Unknown'),
                                'link': item.get('link', '#'),
                                'timestamp': item.get('providerPublishTime', int(time.time()))
                            })
                    except:
                        continue
        except:
            pass
        
        if len(news) == 0:
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            news = [
                {
                    'title': f'Market Analysis for {clean_ticker}',
                    'source': 'Financial News',
                    'link': f'https://www.google.com/search?q={clean_ticker}+stock+news',
                    'timestamp': int(time.time())
                }
            ]
        
        if len(news) > 0:
            try:
                cache_data = {'timestamp': time.time(), 'news': news}
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            except:
                pass
        
        return news
        
    except Exception as e:
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        return [{
            'title': f'News temporarily unavailable',
            'source': 'System',
            'link': f'https://finance.yahoo.com/quote/{ticker}',
            'timestamp': int(time.time())
        }]

# ==================== ADVANCED TECHNICAL ANALYSIS ====================

def create_advanced_features(df):
    """Add comprehensive technical indicators"""
    data = df.copy()
    
    # Moving Averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # Returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Weekly_Return'] = data['Close'].pct_change(5)
    data['Monthly_Return'] = data['Close'].pct_change(20)
    
    # Volatility Metrics
    data['Volatility_10d'] = data['Daily_Return'].rolling(window=10).std()
    data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
    data['Volatility_50d'] = data['Daily_Return'].rolling(window=50).std()
    
    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['RSI_Smooth'] = data['RSI'].rolling(window=3).mean()
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
    data['Bollinger_Upper'] = bollinger.bollinger_hband()
    data['Bollinger_Lower'] = bollinger.bollinger_lband()
    data['Bollinger_Middle'] = bollinger.bollinger_mavg()
    data['Bollinger_Width'] = (data['Bollinger_Upper'] - data['Bollinger_Lower']) / data['Bollinger_Middle']
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    
    # Average True Range
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    
    # On Balance Volume
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    
    # Commodity Channel Index
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    
    # Money Flow Index
    data['MFI'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
    
    # Support and Resistance
    data['Support'] = data['Low'].rolling(window=20).min()
    data['Resistance'] = data['High'].rolling(window=20).max()
    
    return data

def calculate_risk_metrics(df):
    """Calculate advanced risk metrics"""
    returns = df['Daily_Return'].dropna()
    
    metrics = {}
    
    # Sharpe Ratio (annualized, assuming risk-free rate = 0)
    metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min() * 100
    
    # Value at Risk (95%)
    metrics['var_95'] = np.percentile(returns, 5) * 100
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    metrics['sortino_ratio'] = (returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
    
    # Win Rate
    metrics['win_rate'] = (len(returns[returns > 0]) / len(returns)) * 100 if len(returns) > 0 else 0
    
    return metrics

def generate_trading_signals(df):
    """Generate comprehensive trading signals"""
    signals = []
    latest = df.iloc[-1]
    
    # RSI Signals
    if latest['RSI'] > 70:
        signals.append(('Overbought (RSI > 70)', 'danger', '⚠️', 'Bearish'))
    elif latest['RSI'] < 30:
        signals.append(('Oversold (RSI < 30)', 'success', '✓', 'Bullish'))
    else:
        signals.append(('Neutral RSI', 'warning', '●', 'Neutral'))
    
    # MACD Signals
    if latest['MACD'] > latest['MACD_Signal']:
        signals.append(('Bullish MACD Cross', 'success', '↑', 'Bullish'))
    else:
        signals.append(('Bearish MACD Cross', 'danger', '↓', 'Bearish'))
    
    # Moving Average Signals
    if latest['Close'] > latest['MA50']:
        signals.append(('Above MA50', 'success', '↑', 'Bullish'))
    else:
        signals.append(('Below MA50', 'danger', '↓', 'Bearish'))
    
    if latest['MA20'] > latest['MA50']:
        signals.append(('Golden Cross (MA20 > MA50)', 'success', '★', 'Bullish'))
    elif latest['MA20'] < latest['MA50']:
        signals.append(('Death Cross (MA20 < MA50)', 'danger', '★', 'Bearish'))
    
    # Bollinger Bands
    if latest['Close'] > latest['Bollinger_Upper']:
        signals.append(('Above Upper Bollinger', 'danger', '⚠️', 'Bearish'))
    elif latest['Close'] < latest['Bollinger_Lower']:
        signals.append(('Below Lower Bollinger', 'success', '✓', 'Bullish'))
    
    # Stochastic
    if latest['Stoch_K'] > 80:
        signals.append(('Stochastic Overbought', 'danger', '⚠️', 'Bearish'))
    elif latest['Stoch_K'] < 20:
        signals.append(('Stochastic Oversold', 'success', '✓', 'Bullish'))
    
    # Volume Analysis
    avg_volume = df['Volume'].tail(20).mean()
    if latest['Volume'] > avg_volume * 1.5:
        signals.append(('High Volume Alert', 'info', '📊', 'Attention'))
    
    return signals

# ==================== ADVANCED VISUALIZATION ====================

def create_professional_dashboard(result):
    """Create professional multi-chart dashboard"""
    df = result['data']
    info = result['info']
    ticker = result['ticker']
    
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    
    fig = make_subplots(
        rows=5, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.03,
        horizontal_spacing=0.05,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
        subplot_titles=(
            "Price Action & Technical Indicators", "Volume Analysis",
            "RSI (14) & Stochastic Oscillator", "Money Flow Index",
            "MACD Indicator", "Bollinger Bands Width",
            "CCI & ATR", "Support & Resistance"
        ),
        specs=[[{"secondary_y": False, "colspan": 2}, None],
               [{"secondary_y": False, "colspan": 2}, None],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Main Price Chart with Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color=COLORS['success'],
        decreasing_line_color=COLORS['danger'],
        increasing_fillcolor=COLORS['success'],
        decreasing_fillcolor=COLORS['danger']
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20',
                            line=dict(color=COLORS['primary'], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50',
                            line=dict(color=COLORS['secondary'], width=1.5)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Upper'], name='BB Upper',
                            line=dict(color=COLORS['text_secondary'], width=1, dash='dot'),
                            showlegend=False, opacity=0.3), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Lower'], name='BB Lower',
                            line=dict(color=COLORS['text_secondary'], width=1, dash='dot'),
                            fill='tonexty', fillcolor='rgba(0, 102, 255, 0.05)',
                            showlegend=False, opacity=0.3), row=1, col=1)
    
    # Volume
    volume_colors = [COLORS['success'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                     else COLORS['danger'] for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                        marker=dict(color=volume_colors), showlegend=False, opacity=0.6),
                 row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color=COLORS['primary'], width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS['danger'], 
                  opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS['success'], 
                  opacity=0.5, row=3, col=1)
    
    # Stochastic
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='Stoch %K',
                            line=dict(color=COLORS['secondary'], width=1.5)), row=3, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='Stoch %D',
                            line=dict(color=COLORS['warning'], width=1.5)), row=3, col=2)
    
    # MFI
    fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], name='MFI',
                            line=dict(color=COLORS['primary'], width=2),
                            fill='tozeroy', fillcolor='rgba(0, 102, 255, 0.1)'),
                 row=4, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color=COLORS['primary'], width=2)), row=4, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                            line=dict(color=COLORS['danger'], width=2)), row=4, col=2)
    
    histogram_colors = [COLORS['success'] if val > 0 else COLORS['danger'] 
                        for val in df['MACD_Histogram']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram',
                        marker=dict(color=histogram_colors), showlegend=False, opacity=0.5),
                 row=4, col=2)
    
    # CCI
    fig.add_trace(go.Scatter(x=df.index, y=df['CCI'], name='CCI',
                            line=dict(color=COLORS['primary'], width=2)), row=5, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color=COLORS['danger'], 
                  opacity=0.5, row=5, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color=COLORS['success'], 
                  opacity=0.5, row=5, col=1)
    
    # Support & Resistance
    fig.add_trace(go.Scatter(x=df.index, y=df['Support'], name='Support',
                            line=dict(color=COLORS['success'], width=2, dash='dash')),
                 row=5, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'], name='Resistance',
                            line=dict(color=COLORS['danger'], width=2, dash='dash')),
                 row=5, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price',
                            line=dict(color=COLORS['primary'], width=1.5)), row=5, col=2)
    
    # Layout
    fig.update_layout(
        height=1400,
        hovermode='x unified',
        plot_bgcolor=COLORS['dark'],
        paper_bgcolor=COLORS['dark'],
        font=dict(family='Inter, sans-serif', size=11, color=COLORS['text']),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor=COLORS['card_bg'],
            bordercolor=COLORS['border'],
            borderwidth=1,
            font=dict(size=9)
        ),
        margin=dict(l=60, r=30, t=80, b=40),
        xaxis_rangeslider_visible=False
    )
    
    # Update all axes
    for i in range(1, 6):
        for j in range(1, 3):
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=COLORS['grid'],
                showline=True,
                linewidth=1,
                linecolor=COLORS['border'],
                row=i, col=j
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=COLORS['grid'],
                showline=True,
                linewidth=1,
                linecolor=COLORS['border'],
                row=i, col=j
            )
    
    return fig

def analyze_stock(ticker_symbol):
    """Main analysis function"""
    data, info, final_ticker = fetch_stock_data(ticker_symbol)
    
    if data is None or data.empty or len(data) < 20:
        return None
    
    data_with_features = create_advanced_features(data)
    risk_metrics = calculate_risk_metrics(data_with_features)
    
    return {
        'ticker': final_ticker,
        'data': data_with_features,
        'info': info,
        'risk_metrics': risk_metrics
    }

# ==================== PROFESSIONAL UI ====================

app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "ProStock Analytics - Professional Trading Dashboard"

app.layout = html.Div([
    dbc.Container([
        # Professional Navbar
        html.Nav([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("📊", style={'fontSize': '2rem'}),
                            html.Div([
                                html.Div("ProStock Analytics", style={
                                    'fontSize': '1.8rem',
                                    'fontWeight': '800',
                                    'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]})',
                                    'WebkitBackgroundClip': 'text',
                                    'WebkitTextFillColor': 'transparent',
                                    'marginBottom': '4px'
                                }),
                                html.Div("Professional Trading Intelligence", style={
                                    'fontSize': '0.75rem',
                                    'color': COLORS['text_secondary'],
                                    'fontWeight': '400',
                                    'letterSpacing': '1px',
                                    'textTransform': 'uppercase'
                                })
                            ])
                        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px'})
                    ], width=12, md=6),
                    dbc.Col([
                        html.Div([
                            dbc.Badge("Real-time Data", color="primary", className="me-2"),
                            dbc.Badge("Advanced Analytics", color="info", className="me-2"),
                            dbc.Badge("AI Insights", color="success")
                        ], style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'center', 
                                 'flexWrap': 'wrap', 'gap': '8px'})
                    ], width=12, md=6, className="d-none d-md-block")
                ], align="center")
            ], fluid=True)
        ], style={
            'background': f'linear-gradient(135deg, {COLORS["dark_secondary"]} 0%, {COLORS["card_bg"]} 100%)',
            'backdropFilter': 'blur(10px)',
            'borderBottom': f'1px solid {COLORS["border"]}',
            'padding': '1.5rem 0',
            'position': 'sticky',
            'top': '0',
            'zIndex': '1000',
            'boxShadow': '0 4px 20px rgba(0,0,0,0.3)'
        }),
        
        # Search Section
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText(html.I(className="fas fa-search"), 
                                         style={'background': COLORS['dark_secondary'], 
                                               'border': f"2px solid {COLORS['border']}", 
                                               'color': COLORS['primary']}),
                        dbc.Input(
                            id='ticker-input',
                            placeholder='Enter ticker symbol (e.g., ITC, RELIANCE, AAPL, TSLA, NVDA)',
                            type='text',
                            style={
                                'background': f'{COLORS["dark_secondary"]} !important',
                                'border': f'2px solid {COLORS["border"]} !important',
                                'color': f'{COLORS["text"]} !important',
                                'borderRadius': '12px !important',
                                'padding': '1rem 1.5rem !important',
                                'fontSize': '1rem !important'
                            },
                            debounce=True
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-chart-line me-2"), "Analyze"],
                            id='analyze-button',
                            style={
                                'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]}) !important',
                                'border': 'none !important',
                                'borderRadius': '12px !important',
                                'padding': '1rem 2.5rem !important',
                                'fontWeight': '700 !important',
                                'fontSize': '1rem !important',
                                'boxShadow': '0 4px 15px rgba(0, 102, 255, 0.3) !important'
                            },
                            n_clicks=0
                        )
                    ]),
                    html.Div([
                        html.Small("Quick Access: ", style={'color': COLORS['text_secondary'], 'marginRight': '8px'}),
                        dbc.Button("ITC", size="sm", color="secondary", outline=True, className="me-1", id="quick-itc", n_clicks=0),
                        dbc.Button("RELIANCE", size="sm", color="secondary", outline=True, className="me-1", id="quick-reliance", n_clicks=0),
                        dbc.Button("TCS", size="sm", color="secondary", outline=True, className="me-1", id="quick-tcs", n_clicks=0),
                        dbc.Button("AAPL", size="sm", color="secondary", outline=True, className="me-1", id="quick-aapl", n_clicks=0),
                        dbc.Button("TSLA", size="sm", color="secondary", outline=True, className="me-1", id="quick-tsla", n_clicks=0),
                        dbc.Button("NVDA", size="sm", color="secondary", outline=True, className="me-1", id="quick-nvda", n_clicks=0),
                    ], style={'marginTop': '1rem'})
                ], width=12)
            ])
        ], style={
            'padding': '2rem',
            'background': COLORS['card_bg'],
            'borderBottom': f'1px solid {COLORS["border"]}'
        }),
        
        # Loading Indicator
        dcc.Loading(
            id="loading-main",
            type="cube",
            color=COLORS['primary'],
            children=[
                html.Div(id='error-message'),
                html.Div(id='stats-grid'),
                html.Div(id='risk-metrics-section'),
                html.Div(id='chart-section'),
                html.Div(id='signals-section'),
                html.Div(id='news-section'),
            ]
        ),
        
        # Footer
        html.Div([
            html.Hr(style={'borderColor': COLORS['border'], 'margin': '2rem 0'}),
            html.P([
                html.I(className="fas fa-exclamation-triangle me-2"),
                html.Strong("Disclaimer: "),
                "This platform provides data for informational purposes only. ",
                "Not financial advice. Always conduct your own research and consult with financial advisors."
            ], style={'marginBottom': '1rem', 'fontSize': '0.85rem', 'color': COLORS['text_secondary']}),
            html.P([
                "© 2024 ProStock Analytics • Powered by yfinance & Advanced Technical Analysis"
            ], style={'fontSize': '0.8rem', 'opacity': '0.7', 'marginBottom': '0', 'color': COLORS['text_secondary']})
        ], style={
            'background': COLORS['dark_secondary'],
            'padding': '2rem',
            'textAlign': 'center',
            'color': COLORS['text_secondary'],
            'borderTop': f'1px solid {COLORS["border"]}'
        })
        
    ], fluid=True, style={'background': COLORS['dark'], 'minHeight': '100vh', 'padding': '0'})
], style={'background': COLORS['dark'], 'minHeight': '100vh'})

# Callbacks
@app.callback(
    Output('ticker-input', 'value'),
    [Input('quick-itc', 'n_clicks'),
     Input('quick-reliance', 'n_clicks'),
     Input('quick-tcs', 'n_clicks'),
     Input('quick-aapl', 'n_clicks'),
     Input('quick-tsla', 'n_clicks'),
     Input('quick-nvda', 'n_clicks')],
    prevent_initial_call=True
)
def quick_select(itc, reliance, tcs, aapl, tsla, nvda):
    """Handle quick select buttons"""
    button_id = ctx.triggered_id
    if button_id == 'quick-itc':
        return 'ITC'
    elif button_id == 'quick-reliance':
        return 'RELIANCE'
    elif button_id == 'quick-tcs':
        return 'TCS'
    elif button_id == 'quick-aapl':
        return 'AAPL'
    elif button_id == 'quick-tsla':
        return 'TSLA'
    elif button_id == 'quick-nvda':
        return 'NVDA'
    return None

@app.callback(
    [Output('stats-grid', 'children'),
     Output('risk-metrics-section', 'children'),
     Output('chart-section', 'children'),
     Output('signals-section', 'children'),
     Output('news-section', 'children'),
     Output('error-message', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('ticker-input', 'value')],
    prevent_initial_call=True
)
def update_dashboard(n_clicks, ticker):
    """Main dashboard update callback"""
    if not ticker:
        return None, None, None, None, None, None
    
    ticker = ticker.strip().upper()
    result = analyze_stock(ticker)
    
    if result is None:
        error = html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-circle me-3", style={'fontSize': '1.5rem'}),
                html.Div([
                    html.H5("Unable to fetch data", className="mb-2"),
                    html.P(f"Could not retrieve data for '{ticker}'. Please verify the symbol and try again.", 
                          className="mb-0")
                ])
            ], color="danger", className="m-4 d-flex align-items-center", 
            style={'background': 'rgba(255, 23, 68, 0.1)', 'border': f"1px solid {COLORS['danger']}"})
        ])
        return None, None, None, None, None, error
    
    df = result['data']
    info = result['info']
    ticker = result['ticker']
    risk_metrics = result['risk_metrics']
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    
    # Current metrics
    current_price = df['Close'].iloc[-1]
    daily_change = df['Daily_Return'].iloc[-1] * 100
    daily_change_abs = df['Close'].iloc[-1] - df['Close'].iloc[-2]
    weekly_change = df['Weekly_Return'].iloc[-1] * 100 if not pd.isna(df['Weekly_Return'].iloc[-1]) else 0
    monthly_change = df['Monthly_Return'].iloc[-1] * 100 if not pd.isna(df['Monthly_Return'].iloc[-1]) else 0
    rsi = df['RSI'].iloc[-1]
    volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume'].tail(20).mean()
    volatility = df['Volatility_20d'].iloc[-1] * 100 if not pd.isna(df['Volatility_20d'].iloc[-1]) else 0
    
    # Stats Grid
    stats_grid = html.Div([
        dbc.Container([
            html.H3([
                html.Span("📊 ", style={'marginRight': '12px'}),
                "Market Overview"
            ], style={
                'fontSize': '1.5rem',
                'fontWeight': '700',
                'color': COLORS['text'],
                'marginBottom': '2rem',
                'display': 'flex',
                'alignItems': 'center'
            }),
            dbc.Row([
                # Current Price
                dbc.Col([
                    html.Div([
                        html.Div("CURRENT PRICE", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.H2(f"{currency}{current_price:.2f}", style={
                            'fontSize': '2rem',
                            'fontWeight': '800',
                            'margin': '0',
                            'lineHeight': '1',
                            'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]})',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent'
                        }),
                        html.Div([
                            html.Span(f"{daily_change:+.2f}%", 
                                    style={'color': COLORS['success'] if daily_change > 0 else COLORS['danger'],
                                          'fontSize': '0.9rem', 'fontWeight': '600'}),
                            html.Span(f" ({currency}{daily_change_abs:+.2f})", 
                                    style={'color': COLORS['text_secondary'], 'fontSize': '0.8rem', 'marginLeft': '8px'})
                        ], style={'fontSize': '0.85rem', 'fontWeight': '600', 'marginTop': '0.5rem'})
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '16px',
                        'padding': '1.5rem',
                        'position': 'relative',
                        'overflow': 'hidden'
                    })
                ], width=12, md=6, lg=3, className="mb-3"),
                
                # Volume
                dbc.Col([
                    html.Div([
                        html.Div("VOLUME", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.H2(f"{volume/1e6:.2f}M", style={
                            'fontSize': '2rem',
                            'fontWeight': '800',
                            'margin': '0',
                            'lineHeight': '1',
                            'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]})',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent'
                        }),
                        html.Div([
                            html.Span("Avg: ", style={'color': COLORS['text_secondary'], 'fontSize': '0.8rem'}),
                            html.Span(f"{avg_volume/1e6:.2f}M", 
                                    style={'color': COLORS['primary'], 'fontSize': '0.85rem', 'fontWeight': '600'})
                        ], style={'fontSize': '0.85rem', 'fontWeight': '600', 'marginTop': '0.5rem'})
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '16px',
                        'padding': '1.5rem',
                        'position': 'relative',
                        'overflow': 'hidden'
                    })
                ], width=12, md=6, lg=3, className="mb-3"),
                
                # RSI
                dbc.Col([
                    html.Div([
                        html.Div("RSI (14)", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.H2(f"{rsi:.1f}", style={
                            'fontSize': '2rem',
                            'fontWeight': '800',
                            'margin': '0',
                            'lineHeight': '1',
                            'color': COLORS['danger'] if rsi > 70 else COLORS['success'] if rsi < 30 else COLORS['primary']
                        }),
                        html.Div([
                            html.Span("Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral",
                                    style={'color': COLORS['danger'] if rsi > 70 else COLORS['success'] if rsi < 30 else COLORS['text_secondary'],
                                          'fontSize': '0.8rem', 'fontWeight': '600'})
                        ], style={'fontSize': '0.85rem', 'fontWeight': '600', 'marginTop': '0.5rem'})
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '16px',
                        'padding': '1.5rem',
                        'position': 'relative',
                        'overflow': 'hidden'
                    })
                ], width=12, md=6, lg=3, className="mb-3"),
                
                # Volatility
                dbc.Col([
                    html.Div([
                        html.Div("VOLATILITY (20D)", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.H2(f"{volatility:.2f}%", style={
                            'fontSize': '2rem',
                            'fontWeight': '800',
                            'margin': '0',
                            'lineHeight': '1',
                            'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]})',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent'
                        }),
                        html.Div([
                            html.Span("Daily Std Dev", 
                                    style={'color': COLORS['text_secondary'], 'fontSize': '0.8rem'})
                        ], style={'fontSize': '0.85rem', 'fontWeight': '600', 'marginTop': '0.5rem'})
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '16px',
                        'padding': '1.5rem',
                        'position': 'relative',
                        'overflow': 'hidden'
                    })
                ], width=12, md=6, lg=3, className="mb-3"),
            ]),
            
            # Additional Metrics Row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("WEEKLY CHANGE", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.H3(f"{weekly_change:+.2f}%", 
                               style={'color': COLORS['success'] if weekly_change > 0 else COLORS['danger'],
                                     'margin': '0', 'fontSize': '1.5rem', 'fontWeight': '700'})
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '12px',
                        'padding': '1.25rem',
                        'textAlign': 'center'
                    })
                ], width=12, md=4, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("MONTHLY CHANGE", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.H3(f"{monthly_change:+.2f}%",
                               style={'color': COLORS['success'] if monthly_change > 0 else COLORS['danger'],
                                     'margin': '0', 'fontSize': '1.5rem', 'fontWeight': '700'})
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '12px',
                        'padding': '1.25rem',
                        'textAlign': 'center'
                    })
                ], width=12, md=4, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("52-WEEK RANGE", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.H3(f"{currency}{df['Low'].tail(252).min():.2f} - {currency}{df['High'].tail(252).max():.2f}",
                               style={'color': COLORS['primary'], 'margin': '0', 'fontSize': '1.2rem', 'fontWeight': '700'})
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '12px',
                        'padding': '1.25rem',
                        'textAlign': 'center'
                    })
                ], width=12, md=4, className="mb-3"),
            ])
        ], fluid=True)
    ], style={
        'padding': '2rem',
        'background': COLORS['dark'],
        'borderBottom': f'1px solid {COLORS["border"]}'
    })
    
    # Risk Metrics Section
    risk_section = html.Div([
        dbc.Container([
            html.H3([
                html.Span("⚖️ ", style={'marginRight': '12px'}),
                "Risk Analytics"
            ], style={
                'fontSize': '1.5rem',
                'fontWeight': '700',
                'color': COLORS['text'],
                'marginBottom': '2rem',
                'display': 'flex',
                'alignItems': 'center'
            }),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Sharpe Ratio", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.Div(f"{risk_metrics['sharpe_ratio']:.2f}", style={
                            'fontSize': '1.75rem',
                            'fontWeight': '700',
                            'margin': '0.5rem 0',
                            'color': COLORS['success'] if risk_metrics['sharpe_ratio'] > 1 else COLORS['warning']
                        })
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '12px',
                        'padding': '1.25rem',
                        'textAlign': 'center'
                    })
                ], width=12, md=6, lg=3, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("Max Drawdown", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.Div(f"{risk_metrics['max_drawdown']:.2f}%", style={
                            'fontSize': '1.75rem',
                            'fontWeight': '700',
                            'margin': '0.5rem 0',
                            'color': COLORS['danger']
                        })
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '12px',
                        'padding': '1.25rem',
                        'textAlign': 'center'
                    })
                ], width=12, md=6, lg=3, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("VaR (95%)", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.Div(f"{risk_metrics['var_95']:.2f}%", style={
                            'fontSize': '1.75rem',
                            'fontWeight': '700',
                            'margin': '0.5rem 0',
                            'color': COLORS['danger']
                        })
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '12px',
                        'padding': '1.25rem',
                        'textAlign': 'center'
                    })
                ], width=12, md=6, lg=3, className="mb-3"),
                
                dbc.Col([
                    html.Div([
                        html.Div("Win Rate", style={
                            'fontSize': '0.7rem',
                            'color': COLORS['text_secondary'],
                            'fontWeight': '600',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px',
                            'marginBottom': '0.5rem'
                        }),
                        html.Div(f"{risk_metrics['win_rate']:.1f}%", style={
                            'fontSize': '1.75rem',
                            'fontWeight': '700',
                            'margin': '0.5rem 0',
                            'color': COLORS['success'] if risk_metrics['win_rate'] > 50 else COLORS['warning']
                        })
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '12px',
                        'padding': '1.25rem',
                        'textAlign': 'center'
                    })
                ], width=12, md=6, lg=3, className="mb-3"),
            ])
        ], fluid=True)
    ], style={
        'padding': '2rem',
        'background': COLORS['dark'],
        'borderBottom': f'1px solid {COLORS["border"]}'
    })
    
    # Chart Section
    fig = create_professional_dashboard(result)
    chart = html.Div([
        dbc.Container([
            html.H3([
                html.Span("📈 ", style={'marginRight': '12px'}),
                "Technical Analysis"
            ], style={
                'fontSize': '1.5rem',
                'fontWeight': '700',
                'color': COLORS['text'],
                'marginBottom': '1.5rem',
                'display': 'flex',
                'alignItems': 'center'
            }),
            dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                },
                style={'borderRadius': '12px', 'overflow': 'hidden'}
            )
        ], fluid=True)
    ], style={
        'padding': '2rem',
        'background': COLORS['dark'],
        'borderBottom': f'1px solid {COLORS["border"]}'
    })
    
    # Trading Signals
    signals = generate_trading_signals(df)
    
    signal_elements = []
    for signal_text, signal_type, icon, sentiment in signals:
        if sentiment == 'Bullish':
            bg_color = 'rgba(0, 230, 118, 0.15)'
            text_color = COLORS['success']
            border_color = COLORS['success']
        elif sentiment == 'Bearish':
            bg_color = 'rgba(255, 23, 68, 0.15)'
            text_color = COLORS['danger']
            border_color = COLORS['danger']
        else:
            bg_color = 'rgba(255, 214, 0, 0.15)'
            text_color = COLORS['warning']
            border_color = COLORS['warning']
        
        signal_elements.append(
            html.Div([
                html.Span(icon, style={'fontSize': '1.2rem'}),
                html.Span(signal_text)
            ], style={
                'display': 'inline-flex',
                'alignItems': 'center',
                'padding': '0.75rem 1.25rem',
                'borderRadius': '10px',
                'fontSize': '0.85rem',
                'fontWeight': '600',
                'margin': '0.5rem',
                'gap': '8px',
                'backdropFilter': 'blur(10px)',
                'border': f'1px solid {border_color}',
                'background': bg_color,
                'color': text_color
            })
        )
    
    signals_section = html.Div([
        dbc.Container([
            html.H3([
                html.Span("🎯 ", style={'marginRight': '12px'}),
                "Trading Signals"
            ], style={
                'fontSize': '1.5rem',
                'fontWeight': '700',
                'color': COLORS['text'],
                'marginBottom': '1.5rem',
                'display': 'flex',
                'alignItems': 'center'
            }),
            html.Div(signal_elements, style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px'})
        ], fluid=True)
    ], style={
        'padding': '2rem',
        'background': COLORS['dark'],
        'borderBottom': f'1px solid {COLORS["border"]}'
    })
    
    # News Section
    news_items = fetch_stock_news(ticker)
    news_cards = []
    
    for item in news_items[:6]:
        title = item.get('title', '')
        source = item.get('source', 'Unknown')
        link = item.get('link', '#')
        
        if title and len(title) > 5:
            news_cards.append(
                html.A([
                    html.Div([
                        html.Div(title, style={
                            'fontSize': '0.95rem',
                            'fontWeight': '600',
                            'color': COLORS['text'],
                            'marginBottom': '0.5rem',
                            'lineHeight': '1.5'
                        }),
                        html.Div([
                            html.I(className="fas fa-newspaper me-2"),
                            html.Span(source),
                            html.Span(" • ", style={'margin': '0 8px'}),
                            html.Span("Recent", style={'color': COLORS['primary']})
                        ], style={
                            'fontSize': '0.75rem',
                            'color': COLORS['text_secondary'],
                            'display': 'flex',
                            'alignItems': 'center',
                            'gap': '8px'
                        })
                    ], style={
                        'background': COLORS['card_bg'],
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '12px',
                        'padding': '1.25rem',
                        'marginBottom': '1rem',
                        'transition': 'all 0.3s ease',
                        'cursor': 'pointer'
                    })
                ], href=link, target="_blank", style={'textDecoration': 'none', 'color': 'inherit'})
            )
    
    news_section = html.Div([
        dbc.Container([
            html.H3([
                html.Span("📰 ", style={'marginRight': '12px'}),
                "Market News"
            ], style={
                'fontSize': '1.5rem',
                'fontWeight': '700',
                'color': COLORS['text'],
                'marginBottom': '1.5rem',
                'display': 'flex',
                'alignItems': 'center'
            }),
            html.Div(news_cards if news_cards else [
                html.Div([
                    html.I(className="fas fa-info-circle me-2"),
                    "News temporarily unavailable. ",
                    html.A("Search on Google", 
                          href=f"https://www.google.com/search?q={ticker.replace('.NS', '').replace('.BO', '')}+stock+news",
                          target="_blank", style={'color': COLORS['primary']})
                ], style={
                    'background': COLORS['card_bg'],
                    'borderLeft': f'4px solid {COLORS["primary"]}',
                    'borderRadius': '8px',
                    'padding': '1rem 1.5rem',
                    'margin': '1rem 0',
                    'color': COLORS['text']
                })
            ])
        ], fluid=True)
    ], style={
        'padding': '2rem',
        'background': COLORS['dark'],
        'borderBottom': f'1px solid {COLORS["border"]}'
    })
    
    return stats_grid, risk_section, chart, signals_section, news_section, None

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ProStock Analytics - Professional Trading Dashboard")
    print("="*70)
    print("\n✨ FEATURES:")
    print("   • Advanced Technical Indicators (RSI, MACD, Bollinger, Stochastic, MFI, CCI)")
    print("   • Risk Analytics (Sharpe, Sortino, VaR, Max Drawdown)")
    print("   • Real-time Market Data")
    print("   • Professional Dark Theme UI")
    print("   • Smart Caching System")
    print("   • Multi-source News Integration")
    print("\n📊 SUPPORTED MARKETS:")
    print("   • Indian Stocks (NSE/BSE)")
    print("   • US Stocks (NYSE/NASDAQ)")
    print("\n🌐 Dashboard URL:")
    print("   http://127.0.0.1:8050")
    print("\n🧪 TESTING CONNECTION:")
    
    # Quick connection test
    try:
        print("   Testing yfinance connection with AAPL...")
        test_stock = yf.Ticker("AAPL")
        test_data = test_stock.history(period="5d")
        if not test_data.empty:
            print("   ✅ Connection successful!")
        else:
            print("   ⚠️ Connection working but no data returned")
    except Exception as e:
        print(f"   ⚠️ Connection test failed: {str(e)[:60]}")
        print("   💡 You may experience issues fetching real-time data")
    
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)