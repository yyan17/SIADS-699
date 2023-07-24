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
from mlflow.tracking import MlflowClient

sys.path.append('src/utils/')
from data_wrangler import data_cleaning, create_rolling_features, timeseries_to_supervise, create_lags
from data_wrangler import read_sentiment_features, read_financial_features
from model_utils import evaluate_model, train_model

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
             
data_paths = {'RAW_DATA': 'datasets/rawdata/market_data/',
                 'FINANCIAL_RESULTS': 'datasets/processed_data/financial_results/',
                 'INDEX_FEATURES': 'datasets/processed_data/index_features/',
                 'AGG_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/agg_sentiment.csv',
                 'TOPIC_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/agg_sent_topic.csv',
                 'TICKER_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/ticker_news_sent.csv',
                 'TICKERS': ['TTML.NS', 'CHOLAFIN.NS', 'KSB3.DE', 'DLF.NS', 'IPCALAB.NS'],
                 'TOPIC_IDS': [50, 363, 921, 46, 495]}


train_size = 0.8  # 80% for training, 20% for testing
window_size = 10  # Number of past records to consider
cols_to_drop  = ['trend_psar_up', 'trend_psar_down']
window_lst = [5, 10, 20, 50, 100, 200]
lags = 5
target_price = 'high'
exclude_cols = ['open', 'date','close','adj close','volume']
ticker = data_paths['TICKERS'][0]
topic = data_paths['TOPIC_IDS'][0]   


# command to run the script
# python src/scripts/training_evaluation/train-wo_financial_results.py 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
#     parser.add_argument('TICKER', help='ticker for which to train the model')
#     parser.add_argument('TARGET_PRICE', help='provide which stock price to do the prediction for of open/high/low/close prices')

    args = parser.parse_args()
    
    for ticker in data_paths['TICKERS'][:1]:
        args.TICKER = ticker

        try:
            tickers_df = pd.read_csv(data_paths['RAW_DATA'] + args.TICKER + '.csv')
        except FileNotFoundError as e:
            print("raw data file not found for ticker provided")        

        # data cleaning, handling missing/infinite values, dorp columns with mostly na values
        clean_df = data_cleaning(tickers_df, cols_to_drop)

        # create time lags feature for target feature
        data_df = create_lags(clean_df, lags, target_price)

        # create rolling average feature over the timeframe of provide time frame(window list) list
        feature_df = create_rolling_features(clean_df, window_lst, exclude_cols)

        # combine all sentiment scores to create ready useable features
        sentiment_df = read_sentiment_features(data_paths, topic, ticker)

        # create ready usable financial features
#         qtr_fin_df, yrly_fin_df = read_financial_features(data_paths, ticker)

        # create index features
        index_features_df = pd.read_csv(data_paths['INDEX_FEATURES'] + 'index_features.csv')


        # combine all the features 
#         combined_df = feature_df.merge(qtr_fin_df, on='date', how='left')
#         combined_df = combined_df.merge(yrly_fin_df, on='date', how='left')

        # combine sentiment scores features
        combined_df = feature_df.merge(sentiment_df, on='date', how='left')

        # combine other index features/oli price, Rs rate, interest rate etc
        combined_df = combined_df.merge(index_features_df, on='date', how='left')
        combined_df = combined_df.drop(columns=['date'])

        # finally fill any missing values
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.fillna(method='ffill', axis=1).reset_index(drop=True)


        # do train/test split the data with shuffle = False
        train_data, test_data = train_test_split(combined_df, train_size=train_size, shuffle=False)

        scaler = MinMaxScaler()
        # convert timeseries to be used in supervise learning model
        X_train, y_train, indx_train = timeseries_to_supervise(train_data, window_size, target_price)  
        X_train = scaler.fit_transform(X_train)    

        # further split test set to have an hold out set to be used for backtesting
        dev_data, test_data = train_test_split(test_data, train_size=0.5, shuffle=False)

        # convert timeseries to be used in supervise learning model    
        X_test, y_test, indx_test = timeseries_to_supervise(dev_data, window_size, target_price)  
        X_test = scaler.transform(X_test)

        
        experiment_name = "without_financial_results"  # Replace with your desired experiment name

        # Create or get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Start an MLflow run with the specified experiment name
#         mlflow.start_run(experiment_id=experiment_id)
        with mlflow.start_run(experiment_id=experiment_id):
            # train the Random Forest model
            rf_model = train_model(X_train, y_train)    

            y_pred = rf_model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            print("mean absolute percentage error:", round(mape, 3))


            mlflow.log_param("ticker", args.TICKER)
            mlflow.log_param("target_price", target_price)
            mlflow.log_param("mape", mape)    
            signature = infer_signature(X_train, y_pred)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(
                    rf_model, "model", registered_model_name="RandomForestRegressor", signature=signature
                )
            else:
                mlflow.sklearn.log_model(rf_model, "model", signature=signature)        



