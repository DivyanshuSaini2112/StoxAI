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

def fetch_from_yfinance(ticker, period='6mo'):
    """Fetch data from yfinance"""
    try:
        print(f"Fetching {ticker} from yfinance...")
        time.sleep(1)
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval='1d', timeout=15)
        
        if not data.empty:
            info = {}
            try:
                info = stock.info
            except:
                info = {'longName': ticker, 'symbol': ticker}
            return data, info
        return None, None
    except Exception as e:
        print(f"yfinance error: {e}")
        return None, None

def fetch_stock_data(ticker, period='6mo'):
    """Fetch stock data with caching"""
    cache_key = f"{ticker}_{period}"
    
    if cache_key in _data_cache:
        cache_time, cached_data, cached_info = _data_cache[cache_key]
        if time.time() - cache_time < 600:
            print(f"Using cached data for {ticker}")
            return cached_data, cached_info
    
    data, info = fetch_from_yfinance(ticker, period)
    
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
    data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['Bollinger_Upper'] = bollinger.bollinger_hband()
    data['Bollinger_Lower'] = bollinger.bollinger_lband()
    
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
    """Create interactive Plotly dashboard"""
    df = result['data']
    info = result['info']
    ticker = result['ticker']
    
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    company_name = info.get('longName', ticker)
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=('Price Chart with Bollinger Bands', 'Volume', 'RSI', 'MACD')
    )
    
    # Price chart with Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        name='Close Price',
        line=dict(color='#2E86DE', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 222, 0.1)'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'],
        name='MA20',
        line=dict(color='#FF6B6B', width=1.5, dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA50'],
        name='MA50',
        line=dict(color='#4ECDC4', width=1.5, dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Bollinger_Upper'],
        name='BB Upper',
        line=dict(color='rgba(128,128,128,0.3)', width=1),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Bollinger_Lower'],
        name='BB Lower',
        line=dict(color='rgba(128,128,128,0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)',
        showlegend=False
    ), row=1, col=1)
    
    # Volume bars
    colors = ['#26DE81' if df['Close'].iloc[i] > df['Close'].iloc[i-1] else '#FC5C65' 
              for i in range(1, len(df))]
    colors.insert(0, '#26DE81')
    
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        name='RSI',
        line=dict(color='#A55EEA', width=2)
    ), row=3, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        name='MACD',
        line=dict(color='#2E86DE', width=2)
    ), row=4, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'],
        name='Signal',
        line=dict(color='#FC5C65', width=2)
    ), row=4, col=1)
    
    colors_hist = ['#26DE81' if val > 0 else '#FC5C65' for val in df['MACD_Histogram']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['MACD_Histogram'],
        name='Histogram',
        marker_color=colors_hist,
        showlegend=False
    ), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{company_name} ({ticker})<br><sub>Last Updated: {df.index[-1].strftime('%d %b %Y')}</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=24, color='#2C3E50')
        ),
        height=900,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=40, t=100, b=40)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    
    return fig

# ==================== DASH APP ====================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("📈 Stock Analysis Dashboard", className="text-center mb-4", 
                       style={'color': '#2C3E50', 'fontWeight': '700'}),
                html.P("Real-time stock analysis with technical indicators", 
                      className="text-center text-muted mb-4")
            ])
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Enter Stock Symbol:", className="fw-bold mb-2"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='ticker-input',
                            placeholder='e.g., RELIANCE, AAPL, TSLA',
                            type='text',
                            className='form-control-lg'
                        ),
                        dbc.Button(
                            '🔍 Analyze',
                            id='analyze-button',
                            color='primary',
                            className='px-4',
                            n_clicks=0
                        )
                    ]),
                    html.Small("For Indian stocks: RELIANCE, TCS, INFY | US stocks: AAPL, TSLA, MSFT", 
                              className="text-muted mt-2 d-block")
                ])
            ], className="shadow-sm mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="circle",
                children=[
                    html.Div(id='error-message'),
                    html.Div(id='stats-cards'),
                    dcc.Graph(id='stock-chart', config={'displayModeBar': True})
                ]
            )
        ])
    ])
], fluid=True, style={'backgroundColor': '#F8F9FA', 'minHeight': '100vh', 'padding': '20px'})

@app.callback(
    [Output('stock-chart', 'figure'),
     Output('stats-cards', 'children'),
     Output('error-message', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('ticker-input', 'value')]
)
def update_dashboard(n_clicks, ticker):
    if n_clicks == 0 or not ticker:
        # Default empty state
        fig = go.Figure()
        fig.update_layout(
            title="Enter a stock ticker to begin analysis",
            template='plotly_white',
            height=600
        )
        return fig, None, None
    
    ticker = ticker.strip().upper()
    result = analyze_stock(ticker)
    
    if result is None:
        fig = go.Figure()
        fig.update_layout(
            title="Unable to fetch data",
            template='plotly_white',
            height=600
        )
        error = dbc.Alert(
            "❌ Could not fetch data for this ticker. Please try another symbol.",
            color="danger",
            className="mb-3"
        )
        return fig, None, error
    
    # Create chart
    fig = create_plotly_dashboard(result)
    
    # Create stats cards
    df = result['data']
    info = result['info']
    ticker = result['ticker']
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    
    current_price = df['Close'].iloc[-1]
    daily_change = df['Daily_Return'].iloc[-1] * 100
    rsi = df['RSI'].iloc[-1]
    
    stats = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Current Price", className="text-muted mb-2"),
                    html.H3(f"{currency}{current_price:.2f}", className="mb-0 fw-bold")
                ])
            ], className="shadow-sm text-center")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Daily Change", className="text-muted mb-2"),
                    html.H3(
                        f"{daily_change:+.2f}%",
                        className="mb-0 fw-bold",
                        style={'color': '#26DE81' if daily_change > 0 else '#FC5C65'}
                    )
                ])
            ], className="shadow-sm text-center")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("RSI", className="text-muted mb-2"),
                    html.H3(f"{rsi:.1f}", className="mb-0 fw-bold",
                           style={'color': '#FC5C65' if rsi > 70 else '#26DE81' if rsi < 30 else '#2C3E50'})
                ])
            ], className="shadow-sm text-center")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Volume", className="text-muted mb-2"),
                    html.H3(f"{df['Volume'].iloc[-1]/1e6:.1f}M", className="mb-0 fw-bold")
                ])
            ], className="shadow-sm text-center")
        ], width=3),
    ], className="mb-4")
    
    return fig, stats, None

if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Stock Analysis Web Dashboard...")
    print("="*60)
    print("\n📊 Open your browser and go to: http://127.0.0.1:8050")
    print("="*60)
    app.run(debug=True, host='127.0.0.1', port=8050)