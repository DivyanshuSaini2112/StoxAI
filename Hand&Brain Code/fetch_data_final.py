import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import datetime
import ta
from matplotlib.ticker import FuncFormatter
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import time
import requests
from datetime import datetime, timedelta

# Global cache
_data_cache = {}
_last_request_time = 0
MIN_REQUEST_INTERVAL = 1

def setup_fonts():
    """Set up font handling for better emoji support"""
    try:
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        preferred_fonts = ['Noto Sans', 'Segoe UI Emoji', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
        
        for font in preferred_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.family'] = font
                return
        matplotlib.rcParams['font.family'] = 'sans-serif'
    except Exception as e:
        matplotlib.rcParams['font.family'] = 'sans-serif'

def fetch_from_alphavantage(ticker, api_key='demo'):
    """
    Fetch data from Alpha Vantage (Free tier: 25 requests/day)
    Get free API key from: https://www.alphavantage.co/support/#api-key
    """
    try:
        print(f"🔄 Trying Alpha Vantage API...")
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}'
        response = requests.get(url, timeout=10)
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
            
            if len(df) > 0:
                print(f"✓ Data fetched from Alpha Vantage")
                return df, {'longName': ticker, 'symbol': ticker}
        return None, None
    except Exception as e:
        print(f"⚠️  Alpha Vantage error: {e}")
        return None, None

def fetch_from_yahoo_simple(ticker):
    """Simplified Yahoo Finance fetch without yfinance library"""
    try:
        print(f"🔄 Trying direct Yahoo Finance API...")
        
        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - (180 * 24 * 60 * 60)  # 6 months ago
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_time}&period2={end_time}&interval=1d&events=history"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            if len(df) > 0:
                print(f"✓ Data fetched from Yahoo Finance (direct)")
                return df, {'longName': ticker, 'symbol': ticker}
        return None, None
    except Exception as e:
        print(f"⚠️  Direct Yahoo error: {e}")
        return None, None

def fetch_from_yfinance(ticker, period='6mo'):
    """Try standard yfinance with increased timeout"""
    try:
        print(f"🔄 Trying yfinance library...")
        time.sleep(3)  # Wait before attempting
        
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval='1d', timeout=15)
        
        if not data.empty:
            info = {}
            try:
                info = stock.info
            except:
                info = {'longName': ticker, 'symbol': ticker}
            
            print(f"✓ Data fetched from yfinance")
            return data, info
        return None, None
    except Exception as e:
        print(f"⚠️  yfinance error: {e}")
        return None, None

def fetch_stock_data(ticker, period='6mo', use_alphavantage=False, alphavantage_key='demo'):
    """
    Fetch data with multiple fallback sources
    Priority: yfinance -> direct Yahoo -> Alpha Vantage
    """
    cache_key = f"{ticker}_{period}"
    
    # Check cache
    if cache_key in _data_cache:
        cache_time, cached_data, cached_info = _data_cache[cache_key]
        if time.time() - cache_time < 600:  # 10 minute cache
            print(f"📦 Using cached data for {ticker}")
            return cached_data, cached_info
    
    data, info = None, None
    
    # Method 1: Try yfinance (most reliable when not rate-limited)
    data, info = fetch_from_yfinance(ticker, period)
    
    # Method 2: Try direct Yahoo Finance API
    if data is None or data.empty:
        time.sleep(2)
        data, info = fetch_from_yahoo_simple(ticker)
    
    # Method 3: Try Alpha Vantage (requires API key)
    if (data is None or data.empty) and use_alphavantage:
        time.sleep(2)
        data, info = fetch_from_alphavantage(ticker, alphavantage_key)
    
    # Cache successful result
    if data is not None and not data.empty:
        _data_cache[cache_key] = (time.time(), data, info)
        return data, info
    
    return None, None

def create_technical_features(df):
    """Add technical indicators to the dataframe"""
    data = df.copy()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
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
    data['Bollinger_Middle'] = bollinger.bollinger_mavg()
    data['Bollinger_Upper'] = bollinger.bollinger_hband()
    data['Bollinger_Lower'] = bollinger.bollinger_lband()
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['Support1'] = (2 * data['Pivot']) - data['High']
    data['Resistance1'] = (2 * data['Pivot']) - data['Low']
    return data

def format_currency(value, currency_symbol):
    """Format currency values with appropriate scaling"""
    if value is None or np.isnan(value):
        return "N/A"
    if value >= 1e9:
        return f"{currency_symbol}{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{currency_symbol}{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{currency_symbol}{value/1e3:.2f}K"
    else:
        return f"{currency_symbol}{value:.2f}"

def format_percentage(value):
    """Format percentage values"""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value*100:.2f}%"

def millions_formatter(x, pos):
    """Format y-axis in millions for the volume chart"""
    return f'{int(x/1e6)}M'

def add_panel_styling(ax, title):
    """Add styling to panel charts for a more modern look"""
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.patch.set_facecolor('#f8f9fa')
    return ax

def create_fancy_table(ax, data, title, cmap=None):
    """Create a visually appealing table"""
    ax.axis('tight')
    ax.axis('off')
    ax.text(0.5, 1.05, title, fontsize=10, fontweight='bold', 
            ha='center', va='bottom', transform=ax.transAxes)
    table = ax.table(cellText=data[1:], colLabels=data[0], 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            if j == 0:
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#ecf0f1')
            else:
                if cmap is not None and len(data[i]) > 1:
                    try:
                        value = data[i][j]
                        if '%' in value and '↑' in value:
                            cell.set_facecolor('#e6f7e9')
                        elif '%' in value and '↓' in value:
                            cell.set_facecolor('#fae9e8')
                    except:
                        pass
        cell.set_edgecolor('#d4d4d4')
    return table

def create_signal_panel(fig, signals, pos):
    """Create an attractive technical signals panel"""
    signal_box = fig.add_axes(pos)
    signal_box.axis('off')
    
    patch = FancyBboxPatch((0, 0), 1, 1, 
                         boxstyle=matplotlib.patches.BoxStyle("Round", pad=0.3),
                         facecolor='#f8f9fa', edgecolor='#dee2e6', 
                         alpha=0.95, transform=signal_box.transAxes)
    signal_box.add_patch(patch)
    
    signal_box.text(0.5, 0.95, "TECHNICAL SIGNALS", 
                  ha="center", va="top", fontsize=12, fontweight='bold',
                  transform=signal_box.transAxes)
    
    for i, signal in enumerate(signals):
        y_pos = 0.85 - (i * 0.15)
        if "Overbought" in signal:
            icon, color = "! ", 'darkorange'
        elif "Oversold" in signal:
            icon, color = "✓ ", 'green'
        elif "Bullish" in signal:
            icon, color = "^ ", 'green'
        elif "Bearish" in signal:
            icon, color = "v ", 'crimson'
        elif ">" in signal:
            icon, color = "> ", 'green'
        elif "<" in signal:
            icon, color = "< ", 'crimson'
        else:
            icon, color = "= ", 'darkblue'
        
        signal_box.text(0.1, y_pos, icon, fontsize=12, ha="center", va="center", 
                      transform=signal_box.transAxes)
        signal_box.text(0.25, y_pos, signal, fontsize=10, ha="left", va="center", 
                      color=color, weight='medium', transform=signal_box.transAxes)
    return signal_box

def visualize_stock_data(df, info, ticker):
    """Create a comprehensive visualization of stock data"""
    setup_fonts()
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = '#f8f9fa'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#e6e6e6'
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.edgecolor'] = '#e6e6e6'
    plt.rcParams['axes.linewidth'] = 1.5
    
    colors = {
        'price': '#1f77b4', 'price_fill': '#c6dcef',
        'ma20': '#ff7f0e', 'ma50': '#2ca02c',
        'volume': '#3498db', 'volume_up': '#2ecc71', 'volume_down': '#e74c3c',
        'rsi': '#9b59b6', 'rsi_overbought': '#e74c3c', 'rsi_oversold': '#2ecc71',
        'macd': '#2980b9', 'signal': '#e74c3c', 
        'histogram_up': '#2ecc71', 'histogram_down': '#e74c3c',
        'bollinger': '#7f7f7f'
    }
    
    fig = plt.figure(figsize=(18, 12))
    plt.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.06, hspace=0.35, wspace=0.3)
    
    gs = GridSpec(7, 6, figure=fig)
    
    currency = "₹" if ticker.endswith('.NS') or ticker.endswith('.BO') else "$"
    company_name = info.get('longName', ticker)
    
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(0.5, 0.7, f"{company_name} ({ticker})", 
                  fontsize=18, fontweight='bold', ha='center', va='center')
    current_price = df['Close'].iloc[-1]
    last_date = df.index[-1].strftime('%d %b %Y')
    price_text = f"Current Price: {currency}{current_price:.2f} | Last Updated: {last_date}"
    title_ax.text(0.5, 0.3, price_text, fontsize=10, ha='center', va='center')
    
    ax_price = fig.add_subplot(gs[1:3, :4])
    ax_price.plot(df.index, df['Close'], color=colors['price'], linewidth=2, label='Close Price')
    ax_price.fill_between(df.index, df['Close'].min()*0.95, df['Close'], 
                         alpha=0.1, color=colors['price_fill'])
    ax_price.plot(df.index, df['MA20'], color=colors['ma20'], linestyle='--', 
                  linewidth=1.5, label='20-Day MA')
    ax_price.plot(df.index, df['MA50'], color=colors['ma50'], linestyle='--', 
                  linewidth=1.5, label='50-Day MA')
    ax_price.plot(df.index, df['Bollinger_Upper'], color=colors['bollinger'], 
                  linestyle='--', alpha=0.5, linewidth=1)
    ax_price.plot(df.index, df['Bollinger_Lower'], color=colors['bollinger'], 
                  linestyle='--', alpha=0.5, linewidth=1)
    ax_price.fill_between(df.index, df['Bollinger_Upper'], df['Bollinger_Lower'], 
                         color=colors['bollinger'], alpha=0.1)
    add_panel_styling(ax_price, "Price Chart")
    ax_price.set_ylabel('Price', fontsize=10)
    ax_price.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.7)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    ax_volume = fig.add_subplot(gs[3, :4], sharex=ax_price)
    volume_bars = ax_volume.bar(df.index, df['Volume'], color=colors['volume'], 
                              alpha=0.7, width=0.8)
    for i, bar in enumerate(volume_bars):
        if i > 0 and df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            bar.set_color(colors['volume_up'])
        else:
            bar.set_color(colors['volume_down'])
    add_panel_styling(ax_volume, "Volume")
    ax_volume.set_ylabel('Volume', fontsize=10)
    ax_volume.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax_volume.tick_params(axis='x', labelbottom=False)
    
    ax_rsi = fig.add_subplot(gs[4, :2], sharex=ax_price)
    ax_rsi.plot(df.index, df['RSI'], color=colors['rsi'], linewidth=1.5)
    ax_rsi.axhline(70, color=colors['rsi_overbought'], linestyle='--', alpha=0.5)
    ax_rsi.axhline(30, color=colors['rsi_oversold'], linestyle='--', alpha=0.5)
    ax_rsi.fill_between(df.index, df['RSI'], 70, 
                      where=(df['RSI'] >= 70), color=colors['rsi_overbought'], alpha=0.3)
    ax_rsi.fill_between(df.index, df['RSI'], 30, 
                      where=(df['RSI'] <= 30), color=colors['rsi_oversold'], alpha=0.3)
    add_panel_styling(ax_rsi, "Relative Strength Index")
    ax_rsi.set_ylabel('RSI', fontsize=10)
    ax_rsi.set_ylim(0, 100)
    
    ax_macd = fig.add_subplot(gs[4, 2:4], sharex=ax_price)
    ax_macd.plot(df.index, df['MACD'], color=colors['macd'], linewidth=1.5, label='MACD')
    ax_macd.plot(df.index, df['MACD_Signal'], color=colors['signal'], 
                 linewidth=1.5, label='Signal')
    for i in range(len(df.index)):
        if i < len(df.index) - 1:
            if df['MACD_Histogram'].iloc[i] >= 0:
                ax_macd.bar(df.index[i], df['MACD_Histogram'].iloc[i], 
                          color=colors['histogram_up'], alpha=0.5, width=0.8)
            else:
                ax_macd.bar(df.index[i], df['MACD_Histogram'].iloc[i], 
                          color=colors['histogram_down'], alpha=0.5, width=0.8)
    add_panel_styling(ax_macd, "MACD")
    ax_macd.set_ylabel('MACD', fontsize=10)
    ax_macd.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.7)
    
    pe_ratio = info.get('trailingPE', None)
    pb_ratio = info.get('priceToBook', None)
    
    price_metrics = [
        ["Price Metrics", "Value"],
        ["Open", f"{currency}{df['Open'].iloc[-1]:.2f}"],
        ["High", f"{currency}{df['High'].iloc[-1]:.2f}"],
        ["Low", f"{currency}{df['Low'].iloc[-1]:.2f}"],
        ["Close", f"{currency}{df['Close'].iloc[-1]:.2f}"],
    ]
    
    volume_metrics = [
        ["Volume Metrics", "Value"],
        ["Volume", f"{df['Volume'].iloc[-1]:,.0f}"],
        ["Avg Volume", f"{info.get('averageVolume', 0):,.0f}"],
    ]
    
    valuation_metrics = [
        ["Valuation", "Value"],
        ["Market Cap", format_currency(info.get('marketCap'), currency)],
        ["P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio and not np.isnan(pe_ratio) else 'N/A'],
        ["P/B Ratio", f"{pb_ratio:.2f}" if pb_ratio and not np.isnan(pb_ratio) else 'N/A'],
    ]
    
    performance_metrics = [
        ["Performance", "Value"],
        ["1D Change", format_percentage(df['Daily_Return'].iloc[-1])],
        ["1W Change", format_percentage(df['Weekly_Return'].iloc[-1])],
        ["1M Change", format_percentage(df['Monthly_Return'].iloc[-1])],
        ["Volatility", format_percentage(df['Volatility_20d'].iloc[-1])],
    ]
    
    price_table_ax = fig.add_subplot(gs[1, 4:])
    create_fancy_table(price_table_ax, price_metrics, "Price Summary")
    
    volume_table_ax = fig.add_subplot(gs[2, 4:])
    create_fancy_table(volume_table_ax, volume_metrics, "Volume Summary")
    
    valuation_table_ax = fig.add_subplot(gs[3, 4:])
    create_fancy_table(valuation_table_ax, valuation_metrics, "Valuation Summary")
    
    performance_table_ax = fig.add_subplot(gs[4, 4:])
    create_fancy_table(performance_table_ax, performance_metrics, 
                      "Performance Summary", cmap='RdYlGn')
    
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    
    signals = []
    if rsi > 70:
        signals.append(f"RSI: Overbought ({rsi:.1f})")
    elif rsi < 30:
        signals.append(f"RSI: Oversold ({rsi:.1f})")
    else:
        signals.append(f"RSI: Neutral ({rsi:.1f})")
    if macd > macd_signal:
        signals.append("MACD: Bullish Signal")
    else:
        signals.append("MACD: Bearish Signal")
    if df['Close'].iloc[-1] > df['MA50'].iloc[-1]:
        signals.append("Price > 50-Day MA")
    else:
        signals.append("Price < 50-Day MA")
    if df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
        signals.append("20-Day MA > 50-Day MA")
    else:
        signals.append("20-Day MA < 50-Day MA")
    if df['Close'].iloc[-1] > df['Bollinger_Upper'].iloc[-1]:
        signals.append("Price > Upper Bollinger")
    elif df['Close'].iloc[-1] < df['Bollinger_Lower'].iloc[-1]:
        signals.append("Price < Lower Bollinger")
    else:
        signals.append("Price within Bollinger Bands")
    
    create_signal_panel(fig, signals, [0.07, 0.05, 0.86, 0.18])
    
    footer_ax = fig.add_axes([0, 0, 1, 0.02])
    footer_ax.axis('off')
    footer_ax.text(0.5, 0.5, 
                   "Disclaimer: This analysis is for informational purposes only. Not financial advice.",
                   ha="center", va="center", fontsize=7, style='italic', alpha=0.7)
    
    return fig

def analyze_stock(ticker_symbol, use_alphavantage=False, alphavantage_key='demo'):
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker_symbol}...")
    print(f"{'='*60}")
    
    tried_tickers = []
    data = None
    info = None
    ticker_to_use = None
    
    # Try with .NS suffix for Indian stocks
    if not (ticker_symbol.endswith('.NS') or ticker_symbol.endswith('.BO')):
        us_stocks = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD']
        
        if ticker_symbol.upper() not in us_stocks:
            ticker_ns = f"{ticker_symbol}.NS"
            data_ns, info_ns = fetch_stock_data(ticker_ns, use_alphavantage=use_alphavantage, alphavantage_key=alphavantage_key)
            tried_tickers.append(ticker_ns)
            
            if data_ns is not None and not data_ns.empty:
                data = data_ns
                info = info_ns
                ticker_to_use = ticker_ns
                print(f"✓ Found on NSE: {ticker_ns}")
            else:
                ticker_bo = f"{ticker_symbol}.BO"
                data_bo, info_bo = fetch_stock_data(ticker_bo, use_alphavantage=use_alphavantage, alphavantage_key=alphavantage_key)
                tried_tickers.append(ticker_bo)
                
                if data_bo is not None and not data_bo.empty:
                    data = data_bo
                    info = info_bo
                    ticker_to_use = ticker_bo
                    print(f"✓ Found on BSE: {ticker_bo}")
        
        if data is None or data.empty:
            data_raw, info_raw = fetch_stock_data(ticker_symbol, use_alphavantage=use_alphavantage, alphavantage_key=alphavantage_key)
            tried_tickers.append(ticker_symbol)
            if data_raw is not None and not data_raw.empty:
                data = data_raw
                info = info_raw
                ticker_to_use = ticker_symbol
                print(f"✓ Found: {ticker_symbol}")
    else:
        data, info = fetch_stock_data(ticker_symbol, use_alphavantage=use_alphavantage, alphavantage_key=alphavantage_key)
        tried_tickers.append(ticker_symbol)
        if data is not None and not data.empty:
            ticker_to_use = ticker_symbol
            print(f"✓ Found: {ticker_symbol}")
    
    if data is None or data.empty:
        print(f"\n❌ Could not fetch data for any of these tickers: {', '.join(tried_tickers)}")
        print("\n💡 Solutions:")
        print("   1. Yahoo Finance is rate-limiting you. Wait 10-15 minutes")
        print("   2. Use Alpha Vantage API (free 25 requests/day)")
        print("      - Get API key: https://www.alphavantage.co/support/#api-key")
        print("      - Run with: analyze_stock('DLF', use_alphavantage=True, alphavantage_key='YOUR_KEY')")
        print("   3. Try a different ticker to test if the issue is ticker-specific")
        return None
    
    if len(data) < 20:
        print(f"❌ Not enough historical data for {ticker_to_use}.")
        return None
    
    print(f"✓ Successfully fetched {len(data)} days of data for {ticker_to_use}")
    print("📊 Calculating technical indicators...")
    data_with_features = create_technical_features(data)
    print("✓ Technical indicators complete")
    
    currency = "₹" if '.NS' in ticker_to_use or '.BO' in ticker_to_use else "$"
    current_price = data['Close'].iloc[-1]
    company_name = info.get('longName', ticker_to_use)
    
    print(f"\n{'='*60}")
    print(f"{company_name} ({ticker_to_use})")
    print(f"{'='*60}")
    print(f"💰 Current Price: {currency}{current_price:.2f}")
    print(f"📈 Day Range: {currency}{data['Low'].iloc[-1]:.2f} - {currency}{data['High'].iloc[-1]:.2f}")
    
    if info:
        if 'marketCap' in info and info['marketCap']:
            print(f"🏢 Market Cap: {format_currency(info['marketCap'], currency)}")
        if 'sector' in info and info['sector']:
            print(f"🏭 Sector: {info['sector']}")
        if 'industry' in info and info['industry']:
            print(f"🏬 Industry: {info['industry']}")
        if 'trailingPE' in info and info['trailingPE']:
            print(f"📊 P/E Ratio: {info['trailingPE']:.2f}")
        if 'priceToBook' in info and info['priceToBook']:
            print(f"📚 P/B Ratio: {info['priceToBook']:.2f}")
    
    print("\n🎨 Creating visualization dashboard...")
    fig = visualize_stock_data(data_with_features, info, ticker_to_use)
    print("✓ Dashboard completed")
    
    result = {
        'ticker': ticker_to_use,
        'company_name': company_name,
        'current_price': current_price,
        'data': data_with_features,
        'info': info,
        'figure': fig
    }
    
    return result

def clear_cache():
    """Clear the data cache"""
    global _data_cache
    _data_cache.clear()
    print("✓ Cache cleared")

if __name__ == "__main__":
    print("="*60)
    print("      📈 STOCK ANALYSIS TOOL 📊")
    print("="*60)
    print("\n🔑 Optional: Get free Alpha Vantage API key for backup data source")
    print("   Visit: https://www.alphavantage.co/support/#api-key")
    print("="*60)
    print("\nCommands:")
    print("  - Enter ticker symbol to analyze (e.g., DLF, RELIANCE, AAPL)")
    print("  - Type 'clear' to clear cache")
    print("  - Type 'quit' to exit")
    print("="*60)
    
    # Ask if user has Alpha Vantage key (optional)
    use_av = input("\n➤ Do you have an Alpha Vantage API key? (y/n, default: n): ").strip().lower()
    av_key = 'demo'
    use_alphavantage = False
    
    if use_av == 'y':
        av_key = input("➤ Enter your Alpha Vantage API key: ").strip()
        if av_key:
            use_alphavantage = True
            print("✓ Alpha Vantage enabled as backup data source")
    
    while True:
        try:
            ticker_symbol = input("\n➤ Enter stock ticker symbol: ").strip().upper()
            
            if ticker_symbol.lower() == 'quit':
                print("\n👋 Exiting program. Thank you!")
                break
            
            if ticker_symbol.lower() == 'clear':
                clear_cache()
                continue
            
            if not ticker_symbol:
                print("⚠️  Please enter a valid ticker symbol.")
                continue
            
            result = analyze_stock(ticker_symbol, use_alphavantage=use_alphavantage, alphavantage_key=av_key)
            
            if result:
                print(f"\n{'='*60}")
                print("📊 Displaying chart...")
                print(f"{'='*60}")
                
                plt.figure(result['figure'].number)
                plt.show()
                
                # Save the figure
                filename = f"{result['ticker']}_analysis.png"
                result['figure'].savefig(filename, dpi=300, bbox_inches='tight')
                print(f"\n✓ Analysis chart saved as: {filename}")
                
                # Close the figure to free memory
                plt.close(result['figure'])
            else:
                print("\n❌ Analysis failed.")
                print("\n💡 Quick fixes:")
                print("   1. Wait 10-15 minutes if you see rate limit errors")
                print("   2. Try 'clear' command to clear cache")
                print("   3. Get Alpha Vantage API key for alternative data source")
                print("   4. Try US stocks like AAPL, MSFT if Indian stocks fail")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program terminated by user.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'quit' to exit.")