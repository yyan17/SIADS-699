import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna


def download_market_data(tickers: list) -> None:
    for ticker in tickers:
        md = yf.download(ticker, start="2008-06-02", end="2023-05-30")
        market_data_df = get_technical_indicators(md)
        market_data_df.to_csv(f"{ticker}.csv")


def get_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Clean NaN values
    df = dropna(df)

    # Add all ta features
    df_with_technical_indicators = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume")

    return df_with_technical_indicators


if __name__ == "__main__":
    tickers = ['^NSEI', '^GSPC', 'CL=F', 'HSI', 'SHA', '^STI', '^BSESN', 'TATAMOTORS.NS', 'SBIN.NS', 'IOC.NS',
               'RAJESHEXPO.NS', 'INR=X']
    download_market_data(tickers)





