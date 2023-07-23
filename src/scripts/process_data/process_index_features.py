from datetime import datetime
import argparse
import pandas as pd
import numpy as np

import sys
sys.path.append('src/utils')
from data_wrangler import preprocess_index_features, combine_index_features

yesterday_index = ['INR=X', 'CL=F','Treasury_Yeild_10_Years', 'USDX-Index', '^NSEI', '^BSESN', '^GSPC']
todays_index = ['HSI', 'SHA', '^STI']

# command to run the script
# python src/scripts/process_data/process_index_features.py datasets/rawdata/market_data/ datasets/processed_data/index_features/

if __name__ == "__main__":
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_PATH', help='path for the articles to be read')    
    parser.add_argument('OUTPUT_PATH', help='path from where to read predicted sentiment scores')   

    args = parser.parse_args()
    combined_index_df = combine_index_features(args.INPUT_PATH, yesterday_index, todays_index)      
    
    # write all three kinds of sentiments scores in file
    combined_index_df.to_csv(args.OUTPUT_PATH + 'index_features.csv', index=False)
    
    end_time = datetime.now()
    running_time = end_time - start_time
    print("Total running time for the job is:", running_time)
    