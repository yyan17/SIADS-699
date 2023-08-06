from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
import mlflow.sklearn
import mlflow
import warnings
import pandas as pd
import numpy as np
import argparse
import sys

sys.path.append('src/utils/')
from data_wrangler import data_cleaning, create_rolling_features, timeseries_to_supervise, create_lags
from data_wrangler import read_sentiment_features, read_financial_features, data_cleaning_financial
from model_utils import evaluate_model, train_model, extract_shap_values, select_model

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
             
data_paths = {'RAW_DATA': 'datasets/rawdata/market_data/',
                 'FINANCIAL_RESULTS': 'datasets/processed_data/financial_results/',
                 'INDEX_FEATURES': 'datasets/processed_data/index_features/',
                 'AGG_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/agg_sentiment.csv',
                 'TOPIC_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/agg_sent_topic.csv',
                 'TICKER_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/ticker_news_sent.csv',
                 'TICKERS': ['EIHOTEL.BO', 'ELGIEQUIP.BO', 'IPCALAB.BO', 'PGHL.BO', 'TV18BRDCST.BO'],
                 'TOPIC_IDS': [33, 921, 495, 495, 385]
             }


train_size = 0.8  # 80% for training, 20% for testing
window_size = 10  # Number of past records to consider
cols_to_drop  = ['trend_psar_up', 'trend_psar_down']
window_lst = [5, 10, 20, 50, 100, 200]
target_price = 'high'
rolling_exclude_cols = ['open', 'date','close','adj close','volume', 'low', 'high']
# feature_exclude_cols = ['open', 'date','close','adj close','volume', 'low']

feature_exclude_cols = ['date']
seed= 42


# command to run the script
# python src/scripts/process_data/create_combined_features.py datasets/processed_data/combined_features/

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('OUTPUT_PATH', help='path where to write created combined features')

    args = parser.parse_args()
    
    for indx, ticker in enumerate(data_paths['TICKERS']):
        args.TICKER = ticker
        topic = data_paths['TOPIC_IDS'][indx]

        try:
            print("running experiment for ticker:", ticker, topic)
            tickers_df = pd.read_csv(data_paths['RAW_DATA'] + args.TICKER + '.csv')
        except FileNotFoundError as e:
            print("raw data file not found for ticker provided")        

        # data cleaning, handling missing/infinite values, dorp columns with mostly na values
        clean_df = data_cleaning(tickers_df, cols_to_drop)
        print("shape of tickers after data cleaning", clean_df.shape)
        
        # create rolling average feature over the timeframe of provide time frame(window list) list
        feature_df = create_rolling_features(clean_df, window_lst, rolling_exclude_cols)
        print("shape for features after creating rolling features", feature_df.shape)
        
        # combine all sentiment scores to create ready useable features
        sentiment_df = read_sentiment_features(data_paths, topic, ticker)

        # create ready usable financial features
        qtr_fin_df, yrly_fin_df = read_financial_features(data_paths, ticker)
        qtr_fin_df = data_cleaning_financial(qtr_fin_df)
        yrly_fin_df = data_cleaning_financial(yrly_fin_df)        
        print("shape of qtr/yrly financial results:", qtr_fin_df.shape, yrly_fin_df.shape)
        
        # read index features computed for foreign indexes like US, China, Singapore, inlcuding Dollar rate, INR rate, oil price etc.
        index_features_df = pd.read_csv(data_paths['INDEX_FEATURES'] + 'index_features.csv')


        # combine all the features 
        combined_df = feature_df.merge(qtr_fin_df, on='date', how='left')
        print("shape after merging qtr results", combined_df.shape)
        
        combined_df = combined_df.merge(yrly_fin_df, on='date', how='left')
        print("shape after merging yrly results", combined_df.shape)

        # combine sentiment scores features
        combined_df = combined_df.merge(sentiment_df, on='date', how='left')
        print("shape after merging sentiment features:", combined_df.shape)

        # combine other index features/oli price, Rs rate, interest rate etc
        combined_df = combined_df.merge(index_features_df, on='date', how='left')
        print("shape after merging index features:", combined_df.shape)
        
        # drop date columns to handle nan values
        combined_date_df = combined_df['date']
        
        # exclude columns, which can provide potential future information to the model 
        combined_df = combined_df.drop(columns=feature_exclude_cols)

        # finally fill any missing values
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
        combined_df = pd.concat([combined_date_df, combined_df], axis=1)
        print("shape for combined feature dataframe",combined_df.shape)        
        
        # write the combined features created so far 
        filename = ticker + '.csv'
        combined_df.to_csv(args.OUTPUT_PATH + filename, index=False)          
        print(combined_df.shape)
     

