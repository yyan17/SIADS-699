from datetime import datetime
import pandas as pd
import numpy as np
import argparse

import sys
sys.path.append('src/utils/')
from scrap_financials import clean_financial_columns

filter_cols = ['Other Income', 'Net Profit (+)/ Loss (-) from Ordinary Activities after Tax', 'Tax', 'Profit (+)/ Loss (-) from Ordinary Activities before Tax', 'date', 'Expenditure', 'Net Profit', 'Profit after Interest but before Exceptional Items', 'Depreciation and amortisation expense', 'Finance Costs', 'Total Income', 'Employee benefit expense', 'Exceptional Item', 'Diluted for discontinued & continuing operation', 'Basic for discontinued & continuing operation', 'Diluted EPS for continuing operation', 'Basic EPS for continuing operation']

eps_cols = ['Basic for discontinued & continuing operation', 'Basic EPS for continuing operation', 
                'Diluted for discontinued & continuing operation', 'Diluted EPS for continuing operation']
tickers = ['KSB3.DE', 'IPCALAB.NS', 'CHOLAFIN.NS', 'TTML.NS', 'DLF.NS']

# after checking how the model performs with these features
cols_to_drop = ['Depreciation and amortisation expense', 'Exceptional Item', 'Finance Costs']

tickers = ['KSB3.DE', 'IPCALAB.NS',  'TTML.NS', 'DLF.NS', 'CHOLAFIN.NS']

# command to run this script
# python src/scripts/filter_financial_features.py datasets/raw/financial_results/ datasets/processed_data/financial_results/

if __name__ == "__main__":
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_PATH', help='path from where to read scraped financial resutls')    
    parser.add_argument('OUTPUT_PATH', help='path where to write the financial results filtered dataframe')    
    
    print("script started")
    args = parser.parse_args()    
    
    for indx, ticker in enumerate(tickers):
        # read the downloaded raw financial results
        fin_df = pd.read_csv(args.INPUT_PATH + tickers[indx] + '.csv', low_memory=False)
        fin_df = fin_df[filter_cols]
        fin_df['basic_eps'] = np.where(fin_df['Basic for discontinued & continuing operation'], fin_df['Basic for discontinued & continuing operation'],
                                                                                          fin_df['Basic EPS for continuing operation'])

        fin_df['diluted_eps'] = np.where(fin_df['Diluted for discontinued & continuing operation'], fin_df['Diluted for discontinued & continuing operation'],
                                                                                  fin_df['Diluted EPS for continuing operation'])    
        fin_df = fin_df.drop(columns=eps_cols)
        
        # clean and normalize the column names for financial result data
        new_columns = clean_financial_columns(fin_df.columns.tolist())
        fin_df.columns = new_columns
        
#         fin_df = fin_df.reindex(columns= (['date'] + list(col for col in new_columns if col != 'date')))
#         df.head()
        # write the filtered financial features in the privided output path 
        filename = ticker + '.csv'
        fin_df.to_csv(args.OUTPUT_PATH + filename, index=False)
        
    end_time = datetime.now()
    running_time = end_time - start_time
    print("Total running time for the job is:", running_time)

    