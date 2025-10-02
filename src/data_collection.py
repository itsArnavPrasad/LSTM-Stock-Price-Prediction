import os
import yfinance as yf
import pandas as pd

def download_stock_data(ticker="AAPL", start="2010-01-01", end="2025-01-01", save_path="../data/"):
    """
    Download historical stock data using yfinance and save to CSV.
    
    Args:
        ticker (str): Stock ticker symbol (default: AAPL).
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
        save_path (str): Directory to save CSV.
    """
    # Ensure save_path exists
    os.makedirs(save_path, exist_ok=True)

    # Download data
    print(f"Downloading {ticker} stock data from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end)

    # Save to CSV
    file_path = os.path.join(save_path, f"{ticker}.csv")
    data.to_csv(file_path)

    print(f"Saved {ticker} data to {file_path}")
    print("Data shape:", data.shape)
    print("Head:\n", data.head())

    return data

if __name__ == "__main__":
    # Default run: AAPL from 2010–2025
    download_stock_data()
