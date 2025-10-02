import pandas as pd
import os

def clean_stock_data(input_path="data/AAPL.csv", output_path="data/AAPL_clean.csv"):
    """
    Cleans raw stock CSV by fixing headers and formatting.
    
    Args:
        input_path (str): Path to raw stock data CSV.
        output_path (str): Path to save cleaned CSV.
    """
    # Read CSV, skipping the extra header rows
    df = pd.read_csv(input_path, skiprows=2)

    # Rename columns
    df.rename(columns={
        "Price": "Close",  # This is your actual closing price
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Adj Close"  # The second 'Close' is probably Adj Close
    }, inplace=True)

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Reorder columns
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    # Sort by date just in case
    df.sort_values("Date", inplace=True)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to {output_path}")
    print("Shape:", df.shape)
    print(df.head())

    return df


if __name__ == "__main__":
    clean_stock_data()
