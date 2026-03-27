import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

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

        # Basic returns
        g['return'] = g['Close'].pct_change()
        g['Return1'] = g['Close'].pct_change(1)
        g['Return3'] = g['Close'].pct_change(3)

        # Moving averages
        g['ma5'] = g['Close'].rolling(5).mean()
        g['ma10'] = g['Close'].rolling(10).mean()
        g['ma20'] = g['Close'].rolling(20).mean()
        g['ma50'] = g['Close'].rolling(50).mean()
        g['ma100'] = g['Close'].rolling(100).mean()
        g['ma200'] = g['Close'].rolling(200).mean()

        # Exponential moving averages
        g['ema12'] = g['Close'].ewm(span=12, adjust=False).mean()
        g['ema26'] = g['Close'].ewm(span=26, adjust=False).mean()

        # Momentum
        g['momentum5'] = g['Close'] - g['Close'].shift(5)
        g['momentum10'] = g['Close'] - g['Close'].shift(10)

        # Volatility
        g['Volatility5'] = g['Close'].rolling(5).std()
        g['Volatility10'] = g['Close'].rolling(10).std()
        g['range'] = g['High'] - g['Low']
        g['range_pct'] = g['range'] / g['Close']

        # Volume
        g['volume_ma5'] = g['Volume'].rolling(5).mean()
        g['volume_ma10'] = g['Volume'].rolling(10).mean()
        g['volume_change'] = g['Volume'].pct_change()

        # High-Low-Close features
        g['HL_pct'] = (g['High'] - g['Low']) / g['Close']
        g['OC_pct'] = (g['Close'] - g['Open']) / g['Open']

        # RSI
        delta = g['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        RS = roll_up / roll_down
        g['RSI14'] = 100 - (100 / (1 + RS))

        # MACD
        g['MACD'] = g['ema12'] - g['ema26']
        g['MACD_signal'] = g['MACD'].ewm(span=9, adjust=False).mean()

        # Lag features
        g['lag1'] = g['Close'].shift(1)
        g['lag2'] = g['Close'].shift(2)
        g['lag3'] = g['Close'].shift(3)

        # Shift features to avoid lookahead bias
        g = g.shift(1)

        # Target: next day's movement
        g['target'] = (g['Close'].shift(-1) > g['Close']).astype(int)

        # Drop NaNs from rolling features
        g = g.dropna()

        all_features.append(g)

    return pd.concat(all_features)


def prepare_features(df, drop_cols=['target', 'Date', 'Ticker']):
    """
    df: DataFrame returned from create_features
    drop_cols: columns to drop before training

    Returns:
        X: scaled and cleaned features
        y: target labels
    """

    # Separate target
    y = df['target']
    
    # Drop non-feature columns
    X = df.drop(columns=drop_cols, errors='ignore')

    # Optional: remove features with very low variance (likely uninformative)
    selector = VarianceThreshold(threshold=1e-5)
    X = selector.fit_transform(X)

    # Scale features to mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y