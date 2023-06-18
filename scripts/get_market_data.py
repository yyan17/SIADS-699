import yfinance as yf


def download_market_data(tickers: list) -> None:
    for ticker in tickers:
        md = yf.download(ticker, start="2008-06-02", end="2023-05-30")
        md.to_csv(f"{ticker}.csv")


if __name__ == "__main__":
    tickers = ['^GSPC', 'CL=F', 'HSI', 'SHA', '^STI', '^BSESN', 'TATAMOTORS.NS', 'SBIN.NS', 'IOC.NS', 'RAJESHEXPO.NS']
    download_market_data(tickers)





