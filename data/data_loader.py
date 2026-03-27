import yfinance as yf
import pandas as pd
from config import TICKERS, START_DATE, END_DATE

def load_data():
    all_data = []

    for ticker in TICKERS:
        # Download data for the ticker
        df = yf.download([ticker], start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)

        if df.empty:
            print(f"Warning: No data for {ticker}, skipping.")
            continue

        # Flatten MultiIndex columns if present
        # Flatten MultiIndex correctly: keep the first level (OHLCV) only
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure we have 'Close' column
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df = df.rename(columns={'Adj Close': 'Close'})
            else:
                raise ValueError(f"No Close column found for {ticker}. Columns: {df.columns.tolist()}")

        df = df[['Open','High','Low','Close','Volume']]
        df['Ticker'] = ticker
        df = df.reset_index()  # make Date a column
        all_data.append(df)

    if not all_data:
        raise ValueError("No data was loaded for any ticker!")

    data = pd.concat(all_data, ignore_index=True)
    return data