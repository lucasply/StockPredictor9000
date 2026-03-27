import pandas as pd

def create_features(df):
    all_features = []

    # Ensure Close exists
    if 'Close' not in df.columns:
        close_col = [c for c in df.columns if c.lower() == 'close']
        if close_col:
            df = df.rename(columns={close_col[0]: 'Close'})
        else:
            raise ValueError("No 'Close' column found!")

    for ticker, g in df.groupby('Ticker'):
        if g.empty or 'Close' not in g.columns:
            continue

        g = g.copy()
        g['return'] = g['Close'].pct_change()
        g['ma5'] = g['Close'].rolling(5).mean()
        g['ma10'] = g['Close'].rolling(10).mean()
        g['ma20'] = g['Close'].rolling(20).mean()
        g['Return1'] = g['Close'].pct_change(1)
        g['Return3'] = g['Close'].pct_change(3)
        g['Volatility5'] = g['Close'].rolling(5).std()
        
        g = g.shift(1)  # avoid lookahead bias
        g['target'] = (g['Close'].shift(-1) > g['Close']).astype(int)
        g = g.dropna()
        all_features.append(g)

    return pd.concat(all_features)