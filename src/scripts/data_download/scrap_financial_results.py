from bs4 import BeautifulSoup as bs
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import requests

import sys
sys.path.append('src/utils/')
from scrap_financials import scrap_financial_results_placeholer, scrap_financial_result, combine_financial_result

tickers = ['KSB3.DE', 'IPCALAB.NS',  'TTML.NS', 'DLF.NS', 'CHOLAFIN.NS']
scrips = [500249, 524494,  532371, 532868, 511243]

baseuri = "https://www.bseindia.com/corporates/Comp_Results.aspx?"

# command to run this script
# python src/scripts/data_download/scrap_financial_results.py datasets/rawdata/financial_results/

if __name__ == "__main__":
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('OUTPUT_PATH', help='path where to write the financial results dataframe')    
    
    print("script started")
    args = parser.parse_args()    
    
    for scrip,ticker in zip(scrips, tickers):
        uri_param = "Code=" + str(scrip)
        uri = baseuri + uri_param
        
#         if ticker == 'CHOLAFIN.NS':
#             uri = chola_uri        
        # scrap financial resutls place holder webpage
        print(ticker, uri)
        qtr_results, yrly_results = scrap_financial_results_placeholer(uri)

        # scrap quarterly/yearly financial results from result webpage
        qtry_financial_df = combine_financial_result(qtr_results)
        yrly_financial_df = combine_financial_result(yrly_results)
        
        # combine quarterly/yearly financial results dataframe
        financials_df = pd.concat([qtry_financial_df, yrly_financial_df])
        filename = ticker + '.csv'
        financials_df.to_csv(args.OUTPUT_PATH + filename, index=False)
        
    end_time = datetime.now()
    running_time = end_time - start_time
    print("Total running time for the job is:", running_time)

    