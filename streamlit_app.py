import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests # New: Used for the KuCoin API call
import time

# --- CONFIGURATION (Default Values) ---
LOOKBACK_BARS = 200     
MA_FAST = 9
MA_SLOW = 20
RSI_PERIOD = 14
FIB_RETRACEMENTS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSIONS = [1.272, 1.618, 2.0]
DEFAULT_EXCHANGE = 'kucoin'

# List of top global coins to prioritize (ensures these are always at the top)
PRIORITY_COINS = [
    'BTC', 'ETH', 'DOGE', 'XRP', 'BNB, 'USDC', 'ADA', 'AVAX', 'SHIB', 'DOT', 'LINK', 
    'MATIC', 'LTC', 'TRX', 'ATOM', 'XLM', 'FIL', 'ETC', 'ZEC', 'BCH', 
    'XMR', 'WLD', 'GMT', 'PAXG', 'DASH', 'FLOW', 'ENJ', 'BAT', 'IOST', 'RVN', 
    'GTC', 'CVC', 'OMG', 'KCS', 'ICP',  'APT', 'NEAR', 'FET', 'RNDR', 
    'IMX', 'ARB', 'OP', 'ALGO', 'SAND', 'MANA', 'GALA', 'AXS', 'CHZ', 'APE', 
    'LDO', 'CRV', 'UNI', 'AAVE', 'MKR' , 'CC' , 'ALLO' , 'CVC'
]
# ----------------------------------------

# --- NEW/FIXED FUNCTION: GET TOP GAINERS VIA API ---

@st.cache_data(ttl=300) # Cache the result for 5 minutes
def get_top_gainers():
    """Fetches the top 20 gainers using KuCoin's public API."""
    try:
        url = "https://api.kucoin.com/api/v1/market/allTickers"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"KuCoin API request failed with status code: {response.status_code}")
        
        data = response.json()
        tickers = data['data']['ticker']  # List of dicts with symbol, last price, changeRate

        # Filter for USDT pairs and create DataFrame
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('-USDT') and t['changeRate'] is not None]
        df = pd.DataFrame(usdt_tickers)
        
        # Data Cleaning and Type Conversion
        df['Ticker'] = df['symbol'].str.split('-').str[0] # Extract base ticker (e.g., BTC from BTC-USDT)
        df['change_rate'] = pd.to_numeric(df['changeRate'], errors='coerce') * 100 # Convert 0.05 to 5.0%
        df['last_price'] = pd.to_numeric(df['last'], errors='coerce') # Current price
        
        # Filter gainers (positive change) and sort descending
        gainers = df[df['change_rate'] > 0].sort_values('change_rate', ascending=False).head(20)
        
        # Format for display
        gainers['24h Change %'] = gainers['change_rate'].apply(lambda x: f"+{x:.2f}%")
        gainers['Price (USDT)'] = gainers['last_price'].apply(lambda x: f"${x:,.4f}" if x > 0.01 else f"${x:,.8f}")
        
        return gainers[['Ticker', 'Price (USDT)', '24h Change %']].reset_index(drop=True).reset_index(names=['Rank'])

    except Exception as e:
        print(f"API Error fetching gainers: {e}")
        st.error("Failed to fetch Top Gainers via KuCoin API. Displaying static fallback data.")
        # Static Fallback Data
        return pd.DataFrame({
            'Rank': [1, 2, 3, 4, 5],
            'Ticker': ['BTC', 'ETH', 'SOL', 'LTC', 'XRP'],
            'Price (USDT)': ['$65,000.00', '$4,500.00', '$150.00', '$100.00', '$0.50'],
            '24h Change %': ['+5.00%', '+4.50%', '+3.50%', '+3.00%', '+2.50%']
        })

# --- EXISTING HELPER FUNCTIONS ---

@st.cache_data(ttl=43200) # Cache the list for 12 hours
def fetch_kucoin_symbols(exchange_id=DEFAULT_EXCHANGE):
    """Fetches available markets, prioritizes the list, and ranks the rest by volume."""
    try:
        exchange = getattr(ccxt, exchange_id)()
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
        
        guaranteed_pairs = []
        volume_ranked_pairs = []
        
        # 1. Build the list of guaranteed top pairs
        for symbol in PRIORITY_COINS:
            pair = f'{symbol}/USDT'
            if pair in markets and markets[pair]['active']:
                if pair not in guaranteed_pairs:
                    guaranteed_pairs.append(pair)
        
        # 2. Fetch all tickers for volume ranking (excluding already guaranteed ones)
        for symbol, ticker in tickers.items():
            if 'USDT' in symbol and symbol not in guaranteed_pairs:
                if ticker['baseVolume'] is not None and ticker['baseVolume'] > 0:
                    volume_ranked_pairs.append({
                        'symbol': symbol,
                        'volume': ticker['baseVolume']
                    })
        
        # 3. Sort the rest by volume
        volume_ranked_pairs.sort(key=lambda x: x['volume'], reverse=True)
        
        # 4. Combine and return the final list
        top_symbols = [item['symbol'] for item in volume_ranked_pairs]
        final_list = guaranteed_pairs + top_symbols
        
        return final_list[:200]
        
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return [f'{s}/USDT' for s in PRIORITY_COINS[:5]] 

@st.cache_data(ttl=60) # Cache data for 60 seconds
def fetch_ohlcv_data(symbol, timeframe, limit, exchange_id):
    """Fetches historical OHLCV data."""
    try:
        exchange = getattr(ccxt, exchange_id)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculates EMA and RSI."""
    df['EMA_Fast'] = df['close'].ewm(span=MA_FAST, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=MA_SLOW, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=RSI_PERIOD, adjust=False).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    return df

def find_swing_points(df, lookback_window=50):
    """Identifies recent Swing High and Swing Low points."""
    swing_high = df['high'].iloc[-lookback_window:].max()
    swing_low = df['low'].iloc[-lookback_window:].min()
    return swing_high, swing_low

def calculate_fib_levels(high, low):
    """Calculates all Fibonacci Retracement and Extension price levels."""
    diff = high - low
    levels = {}

    for level in FIB_RETRACEMENTS:
        price = high - (diff * level)
        levels[f'FIB_RET_{int(level*100)}'] = price
        
    levels['FIB_RET_100'] = low
    levels['FIB_RET_0'] = high

    for level in FIB_EXTENSIONS:
        price = high + (diff * (level - 1))
        levels[f'FIB_EXT_{str(level).replace(".", "_")}'] = price
        
    return levels

def generate_recommendation(df, symbol, timeframe):
    """Generates the trade setup."""
    
    df = calculate_indicators(df)
    last = df.iloc[-1]
    second_last = df.iloc[-2]
    high, low = find_swing_points(df)
    fib_levels = calculate_fib_levels(high, low)

    signal = 'HOLD'
    entry, tp1, sl, reason = None, None, None, "No active high-probability setup detected."

    # --- LONG (BUY) SIGNAL CHECK ---
    long_cross = (second_last['EMA_Fast'] < second_last['EMA_Slow']) and (last['EMA_Fast'] > last['EMA_Slow'])
    rsi_filter = last['RSI'] > 50

    if long_cross and rsi_filter:
        signal = 'LONG'
        entry = fib_levels['FIB_RET_38'] 
        tp1 = fib_levels['FIB_EXT_1_272']
        sl = fib_levels['FIB_RET_61']
        reason = f"Bullish Crossover (9EMA > 20EMA) with strong RSI ({last['RSI']:.2f})."
    
    # --- SHORT (SELL) SIGNAL CHECK ---
    short_cross = (second_last['EMA_Fast'] > second_last['EMA_Slow']) and (last['EMA_Fast'] < last['EMA_Slow'])
    rsi_filter = last['RSI'] < 50

    if short_cross and rsi_filter:
        signal = 'SHORT'
        entry = fib_levels['FIB_RET_61'] 
        tp1 = low - (high - low) * 0.272
        sl = fib_levels['FIB_RET_38']
        reason = f"Bearish Crossover (9EMA < 20EMA) with weak RSI ({last['RSI']:.2f})."

    # --- COMPILE RECOMMENDATION ---
    return {
        "Symbol": symbol,
        "Timeframe": timeframe,
        "Current_Price": f"${last['close']:,.2f}",
        "Signal": signal,
        "Entry_Level": entry,
        "Take_Profit_1": tp1,
        "Stop_Loss": sl,
        "Strategy_Reason": reason,
        "Fibonacci_Levels": fib_levels
    }


# --- STREAMLIT APP LAYOUT & EXECUTION ---

def main():
    st.set_page_config(page_title="Crypto Scalping Advisor", layout="wide")
    st.title("📈 Fibonacci Scalping Advisor")
    st.markdown("---")
    
    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis'] = False
        
    # --- TOP GAINERS DASHBOARD SECTION ---
    st.header("🔥 Market Momentum: Top Gainers (KuCoin)")
    
    # Get the gainers using the new reliable API function
    gainers_df = get_top_gainers()
    
    # Apply conditional formatting for gainers (green for positive change)
    def color_change(val):
        # We only care about coloring the Change % column
        if isinstance(val, str) and '+' in val:
            return 'color: green'
        return ''

    # The styling must be applied to the string column '24h Change %'
    styled_gainers = gainers_df.style.applymap(color_change, subset=['24h Change %'])
    
    st.dataframe(styled_gainers, hide_index=True, use_container_width=True)
    st.markdown("---")
    # -------------------------------------------
        
    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("Configuration")
        
        # --- DYNAMIC SYMBOL LIST FETCH ---
        symbol_list = fetch_kucoin_symbols(exchange_id=DEFAULT_EXCHANGE)
        
        # 1. TEXT INPUT FILTER 
        search_term = st.text_input("Search Ticker:", placeholder="Type BTC, FIL, etc...").upper()
        
        # Filter the list based on search term
        if search_term:
            filtered_symbols = [s for s in symbol_list if search_term in s]
            if not filtered_symbols:
                st.warning(f"No symbols found matching '{search_term}'.")
                filtered_symbols = symbol_list
        else:
            filtered_symbols = symbol_list
        
        # 2. SELECTBOX
        default_index = filtered_symbols.index('BTC/USDT') if 'BTC/USDT' in filtered_symbols else 0
        
        symbol = st.selectbox(
            "Select Symbol (Filtered List)", 
            options=filtered_symbols, 
            index=default_index
        )
        
        timeframe = st.selectbox("Select Timeframe", options=['5m', '15m'], index=0)
        exchange_id = st.text_input("Exchange ID", value=DEFAULT_EXCHANGE) 
        
        if st.button("Generate Trade Setup"):
            st.session_state['run_analysis'] = True
            
        st.markdown(f"###### Last Analysis Run: {pd.Timestamp.now().strftime('%H:%M:%S')}")
        st.warning("Data is cached for 60 seconds. Click 'Generate' to refresh.")

    # --- Main Analysis Logic ---
    if st.session_state.get('run_analysis'):
        st.subheader(f"Current Setup for **{symbol} ({timeframe})**")
        
        with st.spinner('Fetching and analyzing data...'):
            df = fetch_ohlcv_data(symbol, timeframe, LOOKBACK_BARS, exchange_id)

        if not df.empty:
            setup = generate_recommendation(df, symbol, timeframe)
            
            # --- Display Trade Signal ---
            if setup['Signal'] == 'LONG':
                st.success(f"**SIGNAL: {setup['Signal']}**")
            elif setup['Signal'] == 'SHORT':
                st.error(f"**SIGNAL: {setup['Signal']}**")
            else:
                st.info(f"**SIGNAL: {setup['Signal']}**")
                
            st.metric(label="Current Price", value=setup['Current_Price'])
            
            # Display the key trade parameters in columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Entry Level", f"${setup['Entry_Level']:,.2f}" if setup['Entry_Level'] else setup['Entry_Level'])
            col2.metric("Take Profit (TP1)", f"${setup['Take_Profit_1']:,.2f}" if setup['Take_Profit_1'] else setup['Take_Profit_1'])
            col3.metric("Stop Loss (S/L)", f"${setup['Stop_Loss']:,.2f}" if setup['Stop_Loss'] else setup['Stop_Loss'])
            
            st.markdown("---")
            st.markdown(f"**Strategy Reason:** *{setup['Strategy_Reason']}*")
            
            # --- Display All Fibonacci Levels ---
            st.subheader("All Calculated Fibonacci Levels") 
            
            
            # Convert Fib levels to a DataFrame for clean display
            fib_data = {
                level: f"${price:,.2f}" 
                for level, price in sorted(setup['Fibonacci_Levels'].items(), key=lambda item: item[1], reverse=True)
            }
            
            fib_df = pd.Series(fib_data).rename_axis('Level').to_frame(name='Price ($)')
            st.dataframe(fib_df, use_container_width=True)

        else:
            st.error(f"Could not fetch data for {symbol}. Check connection or try another exchange/symbol.")

if __name__ == "__main__":
    main()
