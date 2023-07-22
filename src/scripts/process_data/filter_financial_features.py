from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import os

import sys
sys.path.append('src/utils/')
from scrap_financials import clean_financial_columns

qtr_cols = ['Depreciation and amortisation expense_QTR', 'Employee Benefit Expenses_QTR', 'Employee benefit expense_QTR', 'Equity Capital_QTR', 'Exceptional Item_QTR', 'Expenditure_QTR', 'Face Value (in Rs)_QTR', 'Finance Costs_QTR', 'Interest_QTR', 'Net Profit (+)/ Loss (-) from Ordinary Activities after Tax_QTR', 'Net Profit_QTR', 'Number of Public Shareholding_QTR', 'Other Income_QTR', 'Percentage of Public Shareholding_QTR', 'Profit (+)/ Loss (-) from Ordinary Activities before Tax_QTR', 'Profit after Interest but before Exceptional Items_QTR', 'Profit before Interest and Exceptional Items_QTR', 'Profit from Operations before Other Income, Interest and Exceptional Items_QTR', 'Revenue from Operations_QTR', 'Tax_QTR', 'Total Income_QTR', 'basic_eps_QTR', 'date_QTR', 'diluted_eps_QTR']

yrly_cols = ['Depreciation and amortisation expense_YRLY', 'Employee benefit expense_YRLY', 'Exceptional Item_YRLY', 'Expenditure_YRLY', 'Finance Costs_YRLY', 'Net Profit (+)/ Loss (-) from Ordinary Activities after Tax_YRLY', 'Net Profit_YRLY', 'Other Income_YRLY', 'Profit (+)/ Loss (-) from Ordinary Activities before Tax_YRLY', 'Profit after Interest but before Exceptional Items_YRLY', 'Total Income_YRLY', 'basic_eps_YRLY', 'date_YRLY', 'diluted_eps_YRLY']

tickers = ['KSB3.DE', 'IPCALAB.NS', 'CHOLAFIN.NS', 'TTML.NS', 'DLF.NS']

# after checking how the model performs with these features
cols_to_drop = ['Depreciation and amortisation expense', 'Exceptional Item', 'Finance Costs']

tickers = ['KSB3.DE', 'IPCALAB.NS',  'TTML.NS', 'DLF.NS', 'CHOLAFIN.NS']

tickers = ['TTML.NS']
# command to run this script
# python src/scripts/process_data/filter_financial_features.py datasets/rawdata/financial_results/ datasets/processed_data/financial_results/

if __name__ == "__main__":
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_PATH', help='path from where to read scraped financial resutls')    
    parser.add_argument('OUTPUT_PATH', help='path where to write the financial results filtered dataframe')    
    
    print("script started")
    args = parser.parse_args()    
    result_files = os.listdir(args.INPUT_PATH)
    
    for indx, filename in enumerate(result_files):
        # read the downloaded raw financial results
        print(filename)
        fin_df = pd.read_csv(args.INPUT_PATH + filename, low_memory=False)
        
        # filter only predefined colums financial results dataframe
        if 'QTR' in filename:
            filter_cols = qtr_cols
        else:
            filter_cols = yrly_cols
        fin_df = fin_df[filter_cols]
                            
        # clean and normalize the column names for financial result data
        new_columns = clean_financial_columns(fin_df.columns.tolist())
        fin_df.columns = new_columns
        
        # write the filtered financial features in the privided output path 
        fin_df.to_csv(args.OUTPUT_PATH + filename, index=False)
        
    end_time = datetime.now()
    running_time = end_time - start_time
    print("Total running time for the job is:", running_time)

    