import pandas as pd

import sys
sys.path.append('src/utils/')
from tickerutil import getTicker

path = "datasets/"
bse_500 = "S&P BSE 500index_Constituents.csv"
filepath = path + bse_500

# command to run this script
# python src/scripts/data_download/get_bse500_tickers.py

if __name__ == "__main__":    
    # read the BSE 500 companies excel list
    bse_500_df = pd.read_csv(filepath)
        
    # get company ticker using clean company names using yfinance package  
    bse_500_df['TICKER'] = bse_500_df['ISIN No.'].apply(lambda company: getTicker(company))
    bse_500_df.to_csv(path + 'ticker.csv', index=False)