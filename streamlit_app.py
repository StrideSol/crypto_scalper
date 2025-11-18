import streamlit as st
import ccxt
import pandas as pd
import numpy as np

# --- CONFIGURATION (Default Values) ---
LOOKBACK_BARS = 200     
MA_FAST = 9
MA_SLOW = 20
RSI_PERIOD = 14
FIB_RETRACEMENTS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSIONS = [1.272, 1.618, 2.0]
DEFAULT_EXCHANGE = 'kucoin'
# ----------------------------------------

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=43200) # Cache the list for 12 hours
def fetch_kucoin_symbols(exchange_id=DEFAULT_EXCHANGE):
    """
    Fetches all active USDT markets, sorts them by 24h trading volume,
    and returns the top 200 pairs.
    """
    try:
        exchange = getattr(ccxt, exchange_id)()
        
        # 1. Fetch all market tickers (includes 24h volume)
        tickers = exchange.fetch_tickers()
        
        volume_ranked_pairs = []

        for symbol, ticker in tickers.items():
            # 2. Filter for USDT-quoted pairs that are active and have valid volume data
            if 'USDT' in symbol and ticker['baseVolume'] is not None and ticker['baseVolume'] > 0:
                volume_ranked_pairs.append({
                    'symbol': symbol,
                    'volume': ticker['baseVolume']
                })
        
        # 3. Sort the pairs by volume in descending order
        volume_ranked_pairs.sort(key=lambda x: x['volume'], reverse=True)
        
        # 4. Extract the symbols and take the top 200
        top_symbols = [item['symbol'] for item in volume_ranked_pairs[:200]]
        
        return top_symbols
        
    except Exception as e:
        # Fallback list if the API fails
        print(f"Error fetching symbols: {e}")
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

@st.cache_data(ttl=60) # Cache data for 60 seconds
def fetch_ohlcv_data(symbol, timeframe, limit, exchange_id):
    """Fetches historical OHLCV data from the specified exchange."""
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
    """Calculates EMA and RSI and adds them to the DataFrame."""
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
    """Generates the trade setup based on MA Crossover + RSI strategy and Fib levels."""
    
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
        
    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("Configuration")
        
        # --- DYNAMIC SYMBOL LIST FETCH ---
        # NOTE: Using the DEFAULT_EXCHANGE ('kucoin') for the symbol list fetch
        symbol_list = fetch_kucoin_symbols(exchange_id=DEFAULT_EXCHANGE)
        
        default_index = symbol_list.index('BTC/USDT') if 'BTC/USDT' in symbol_list else 0
        
        symbol = st.selectbox(
            "Select Symbol", 
            options=symbol_list, 
            index=default_index
        )
        # --- END DYNAMIC SYMBOL LIST ---
        
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
