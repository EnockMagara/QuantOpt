# portfolio_ml_ready_dataset.py

import yfinance as yf
import pandas as pd
from fredapi import Fred
import numpy as np

# =========================
# 1. Set up tickers & API
# =========================
stock_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
etf_tickers = ["SPY", "QQQ", "VTI"]
crypto_tickers = ["BTC-USD", "ETH-USD"]
bond_tickers = ["GS10", "GS2"]  # 10-year and 2-year Treasury yields

fred_api_key = "92ec5b276fe498a139f602b7f90eaf94"
fred = Fred(api_key=fred_api_key)

start_date = "2018-01-01"
end_date = "2025-01-01"

all_tickers = stock_tickers + etf_tickers + crypto_tickers

# =========================
# 2. Download Stock, ETF & Crypto Data
# =========================
data = yf.download(all_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
processed_data = []

for ticker in all_tickers:
    df = data[ticker].copy()
    df['Return'] = df['Close'].pct_change()
    
    # Rolling features (window = 21 trading days ≈ 1 month)
    df['Volatility'] = df['Return'].rolling(window=21).std()
    df['Cumulative_Return'] = (1 + df['Return']).cumprod() - 1
    df['Rolling_Mean_Return'] = df['Return'].rolling(window=21).mean()
    df['Rolling_Sharpe'] = df['Rolling_Mean_Return'] / df['Volatility']
    
    # Lagged returns for RL agent
    for lag in range(1, 6):
        df[f'Return_Lag{lag}'] = df['Return'].shift(lag)
    
    df['Ticker'] = ticker
    df = df.dropna()
    processed_data.append(df.reset_index())

final_df = pd.concat(processed_data, ignore_index=True)
final_df.to_csv("ml_ready_assets.csv", index=False)
print("ML-ready asset data saved to ml_ready_assets.csv")

# =========================
# 3. Download Bond Data
# =========================
bond_data = []

for bond in bond_tickers:
    series = fred.get_series(bond, start_date)
    df_bond = pd.DataFrame(series, columns=['Yield'])
    df_bond['Date'] = df_bond.index
    df_bond['Ticker'] = bond
    # Lagged bond yield changes
    df_bond['Yield_Change'] = df_bond['Yield'].diff()
    for lag in range(1, 4):
        df_bond[f'Yield_Change_Lag{lag}'] = df_bond['Yield_Change'].shift(lag)
    df_bond = df_bond.dropna()
    bond_data.append(df_bond.reset_index(drop=True))

bond_df = pd.concat(bond_data, ignore_index=True)
bond_df.to_csv("ml_ready_bonds.csv", index=False)
print("ML-ready bond data saved to ml_ready_bonds.csv")