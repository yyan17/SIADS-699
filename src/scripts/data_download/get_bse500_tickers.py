import pandas as pd

import sys
sys.path.append('../')
from utils import tickerutil

path = "../../datasets/"
bse_500 = "S&P BSE 500index_Constituents.csv"
filepath = path + bse_500

if __name__ == "__main__":
    
    # read the BSE 500 companies excel list
    bse_500_df = pd.read_csv(filepath)
    
    # clean the company names
    bse_500_df = tickerutil.clean_company(bse_500_df)
    
    # get company ticker using clean company names using yfinance package  
    bse_500_df['TICKER'] = bse_500_df['CLEAN_COMPANY'].apply(lambda company: tickerutil.getTicker(company))
    bse_500_df.to_csv(path + 'ticker.csv')