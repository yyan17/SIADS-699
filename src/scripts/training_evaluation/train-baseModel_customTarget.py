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
from data_wrangler import data_cleaning, create_rolling_features, timeseries_to_supervise, create_lags, create_custom_target
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
                 'TICKERS': ['EIHOTEL.BO', 'ELGIEQUIP.BO', 'IPCALAB.BO', 'PGHL.BO',  'TV18BRDCST.BO'],
                 'TOPIC_IDS': [33, 921, 495, 495, 921]
             }


train_size = 0.8  # 80% for training, 20% for testing
window_size = 10  # Number of past records to consider
cols_to_drop  = ['trend_psar_up', 'trend_psar_down']
window_lst = [5, 10, 20, 50, 100, 200]
target_price = 'ln_target'
rolling_exclude_cols = ['open', 'date','close','adj close','volume']
seed= 42
agg_topic_ticker_sent_cols = ['date', 'agg_polarity', 'agg_compound', 'topic_polarity', 'topic_compound', 'ticker_polarity', 'ticker_compound']


# command to run the script
# python src/scripts/training_evaluation/train-baseModel_customTarget.py RandomForest EXPERIMENT_NAME datasets/processed_data/feature_importance/RandomForest/ 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL_NAME', help='provide the ml model, for which we want to train/predict the data ')
    parser.add_argument('EXPERIMENT_NAME', help='provide the experiment name, for which to run the training')    
    parser.add_argument('OUTPUT_PATH', help='path where to write computed shape values for feature importance')
    

    args = parser.parse_args()
    for indx, ticker in enumerate(data_paths['TICKERS']):
        print(ticker)
        args.TICKER = ticker
        topic = data_paths['TOPIC_IDS'][indx]
        
        try:
            tickers_df = pd.read_csv(data_paths['RAW_DATA'] + args.TICKER + '.csv')
        except FileNotFoundError as e:
            print("raw data file not found for ticker provided")        

        # data cleaning, handling missing/infinite values, dorp columns with mostly na values
        clean_df = data_cleaning(tickers_df, cols_to_drop)
        
        # create rolling average feature over the timeframe of provide time frame(window list) list
        feature_df = create_rolling_features(clean_df, window_lst, rolling_exclude_cols)
        
        # create the custom target price using ln(high/yesterday_close)
        feature_df = create_custom_target(feature_df)
        
        # combine all sentiment scores to create ready useable features
        sentiment_df = read_sentiment_features(data_paths, topic, ticker)
        sentiment_df = sentiment_df[agg_topic_ticker_sent_cols]

        # create ready usable financial features
        qtr_fin_df, yrly_fin_df = read_financial_features(data_paths, ticker)
        qtr_fin_df = data_cleaning_financial(qtr_fin_df)
        yrly_fin_df = data_cleaning_financial(yrly_fin_df)        

        # create index features
        index_features_df = pd.read_csv(data_paths['INDEX_FEATURES'] + 'index_features.csv')

        # combine all the features 
        combined_df = feature_df.merge(qtr_fin_df, on='date', how='left')
        combined_df = combined_df.merge(yrly_fin_df, on='date', how='left')

        # combine sentiment scores features
        combined_df = combined_df.merge(sentiment_df, on='date', how='left')

        # combine other index features/oli price, Rs rate, interest rate etc
        combined_df = combined_df.merge(index_features_df, on='date', how='left')
        print("shape after combining index featuress results:", combined_df.shape)
        
        combined_date_df = combined_df['date']
        combined_df = combined_df.drop(columns='date')

#         combined_df = combined_df.loc[:, XGBoost_top_50_features]

        # finally fill any missing values
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
        
        # do train/test split the data with shuffle = False
        train_data, test_data = train_test_split(combined_df, train_size=train_size, shuffle=False)
        
        # convert timeseries to be used in supervise learning model
        X_train, y_train, indx_train = timeseries_to_supervise(train_data, window_size, target_price)  
        
        # further split test set to have an hold out set to be used for backtesting
        dev_data, test_data = train_test_split(test_data, train_size=0.5, shuffle=False)

        # convert timeseries to be used in supervise learning model    
        X_test, y_test, indx_test = timeseries_to_supervise(dev_data, window_size, target_price)  

        # set the experiment name to be run, can be seen in mlflow
        experiment_name = args.EXPERIMENT_NAME  # Replace with your desired experiment name

        # Create or get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Start an MLflow run with the specified experiment name
#         mlflow.start_run(experiment_id=experiment_id)
        with mlflow.start_run(experiment_id=experiment_id):
            # choose the model as per provided arguments
            model = select_model(args.MODEL_NAME, seed)
            
            # train the Random Forest model
            trained_model = train_model(model, X_train, y_train)

            # evaluate the fitted model using mape and rmse metrics
            y_pred, mape, rmse = evaluate_model(trained_model, window_size, dev_data, X_test, y_test)            
            print("for ticker {0} mean absolute percentage error: {1}, root_mean_square_error: {2}".format(args.TICKER, round(mape, 3), round(rmse, 3)))

            # compute the feature importance using TreeSHAP 
            feature_importance_df = extract_shap_values(trained_model, train_data, X_train, window_size)
            feature_importance_df.columns = ['shap_value' + '_' + ticker,  'feature']
            # write the feature importance tree shap values to the file
            filename = ticker + '.csv'
            feature_importance_df.to_csv(args.OUTPUT_PATH + filename, index=False)  
        
            mlflow.log_param("ticker", args.TICKER)
            mlflow.log_param("target_price", target_price)
            mlflow.log_param("mape", round(mape, 5))    
            mlflow.log_param("rmse", round(rmse, 5))                
            signature = infer_signature(X_train, y_pred)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.sklearn.log_model(
                    trained_model, "model", registered_model_name=args.MODEL_NAME, signature=signature
                )
            else:
                mlflow.sklearn.log_model(trained_model, "model", signature=signature)        

                