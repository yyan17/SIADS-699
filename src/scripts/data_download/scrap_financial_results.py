from bs4 import BeautifulSoup as bs
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import requests

import sys
sys.path.append('src/utils/')
from scrap_financials import scrap_financial_results_placeholer, scrap_financial_result, combine_financial_result

# top 8 tickers with whole time series data
tickers = ['EIHOTEL.BO', 'ELGIEQUIP.BO', 'IPCALAB.BO', 'PGHL.BO', 'TV18BRDCST.BO']

baseuri = "https://www.bseindia.com/corporates/Comp_Results.aspx?"

# command to run this script
# python src/scripts/data_download/scrap_financial_results.py datasets/ticker.csv datasets/rawdata/financial_results/

if __name__ == "__main__":
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_PATH', help='path from where to read the tickers scrip code')    
    parser.add_argument('OUTPUT_PATH', help='path where to write the financial results dataframe')    
    
    print("script started")
    args = parser.parse_args()    
    
    # get the scrip/ticker names for which to scrap financial results
    tickers_df = pd.read_csv(args.INPUT_PATH)
    cond = np.where(tickers_df['TICKER'].isin(tickers))
    tickers_df = tickers_df.iloc[cond]
    
#     # get the scrips corresponding to ticker
    scrips = tickers_df['Scrip Code'].values.tolist()
    
    for scrip,ticker in zip(scrips, tickers):
        uri_param = "Code=" + str(scrip)
        uri = baseuri + uri_param
        
        # scrap financial resutls place holder webpage
        print(ticker, uri)
        qtr_results, yrly_results = scrap_financial_results_placeholer(uri)

        # scrap quarterly/yearly financial results from result webpage
        result_types = ['QTR', 'YRLY']
        for result, result_type  in zip([qtr_results, yrly_results], result_types[:1]):
            financial_df = combine_financial_result(result)
                                   
            # change columns names to reflect quarterly/yearly results 
            financial_df = financial_df.rename(columns = {col: col + '_' + result_type for col in financial_df.columns if col != 'date'})
                                                          
            # change the order of the columns to keep the date at first column
            new_col_order = ['date'] + [col for col in financial_df.columns.tolist() if col != 'date']
            financial_df = financial_df[new_col_order]
            
            filename = ticker + '_' + result_type +'.csv'
            financial_df.to_csv(args.OUTPUT_PATH + filename, index=False)
                                
    end_time = datetime.now()
    running_time = end_time - start_time
    print("Total running time for the job is:", running_time)

    