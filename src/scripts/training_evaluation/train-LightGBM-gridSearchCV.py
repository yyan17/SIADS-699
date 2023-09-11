from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
import mlflow.sklearn
import mlflow
import warnings
import pandas as pd
import numpy as np
import argparse
import os
import sys
from mlflow.tracking import MlflowClient
import joblib

sys.path.append('src/utils/')
from data_wrangler import data_cleaning, create_rolling_features, timeseries_to_supervise, create_lags, create_custom_target
from data_wrangler import read_sentiment_features, read_financial_features, data_cleaning_financial, fetch_topn_features, convert_custom_target_to_actual, create_all_features
from model_utils import evaluate_model, train_model, extract_shap_values, select_model

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
             
data_paths = {'RAW_DATA': 'datasets/rawdata/market_data/',
                 'FINANCIAL_RESULTS': 'datasets/processed_data/financial_results/',
                 'INDEX_FEATURES': 'datasets/processed_data/index_features/',
                 'COMBINED_FEATURES': 'datasets/processed_data/combined_features/',
                 'AGG_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/agg_sentiment.csv',
                 'TOPIC_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/agg_sent_topic.csv',
                 'TICKER_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/ticker_news_sent.csv',
                 'TICKERS': ['EIHOTEL.BO', 'ELGIEQUIP.BO', 'IPCALAB.BO', 'PGHL.BO',  'TV18BRDCST.BO'],
                 'TOPIC_IDS': [33, 921, 495, 495, 921]

             }

scoring = 'neg_mean_absolute_percentage_error'
train_size = 0.8  # 80% for training, 20% for testing
window_size = 10  # Number of past records to consider
target_price = 'ln_target'
topn_feature_count = 50
n_test_split = 5
seed= 42

# command to run the script
# python src/scripts/training_evaluation/train-LightGBM-gridSearchCV.py LightGBM gridSearch_withTop50featuresLightGBM  datasets/processed_data/model_predictions/LightGBM/ datasets/processed_data/feature_importance/LightGBM/ trained_models/LightGBM/

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL_NAME', help='provide the ml model, for which we want to train/predict the data ')
    parser.add_argument('EXPERIMENT_NAME', help='provide the experiment name, for which to run the training')
    parser.add_argument('MODEL_PREDICTIONS', help='path where to write computed shape values for feature importance')
    parser.add_argument('FEATURE_PATH', help='path where to write/read computed shape values for feature importance')    
    parser.add_argument('TRAINED_MODEL_PATH', help='path where to write computed shape values for feature importance')
    

    args = parser.parse_args()
    # fetch topn features as per feature importance
    topn_features_df = fetch_topn_features(args.FEATURE_PATH, topn_feature_count)
    topn_features = topn_features_df['feature'].values.tolist()
    topn_features = topn_features + ['yesterday_close', 'ln_target']        
    
    for indx, ticker in enumerate(data_paths['TICKERS']):
        topic_id = data_paths['TOPIC_IDS'][indx]
        
        path = data_paths['COMBINED_FEATURES'] + ticker + '.csv.gz'         
        if os.path.isfile(path):
               combined_df = pd.read_csv(path)
        else:            
        # create all the features
            combined_df = create_all_features(data_paths, ticker, topic_id)        
            combined_df.to_csv(path, index=False)

        combined_date_df = combined_df['date']
        combined_df = combined_df.drop(columns='date')        
                
        # do train/test split the data with shuffle = False
        train_data, test_data = train_test_split(combined_df.loc[:, topn_features], train_size=train_size, shuffle=False)
        train_date, test_date = train_test_split(combined_date_df, train_size=train_size, shuffle=False)
        
        # convert timeseries to be used in supervise learning model
        X_train, y_train, indx_train = timeseries_to_supervise(train_data, window_size, target_price)  

        
        # further split test set to have an hold out set to be used for backtesting
        dev_data, test_data = train_test_split(test_data, train_size=0.5, shuffle=False)
        dev_date, test_date = train_test_split(test_date, train_size=0.5, shuffle=False)

        # convert timeseries to be used in supervise learning model    
        X_test, y_test, indx_test = timeseries_to_supervise(test_data, window_size, target_price)  
                
        X_train, X_eval = train_test_split(X_train, train_size=0.7, shuffle=False)
        y_train, y_eval = train_test_split(y_train, train_size=0.7, shuffle=False)
        
        # create model based on params
        model = select_model(args.MODEL_NAME, seed)            
        tscv = TimeSeriesSplit(n_splits=n_test_split)
        
        param_grid = {
                'learning_rate': [0.1],
                'n_estimators': [100],
                'max_depth': [8],
                'num_leaves': [7, 8, 9],
                'boosting': ['gbdt', 'dart'],
                'min_data_in_leaf': [70],
                'reg_alpha': [0.2, 0.25],
                'reg_lambda': [0.2, 0.25],
                'objective': ['regression'],
                'early_stopping_rounds' : [10]
              }        
        
        grid_search = GridSearchCV(estimator=model, scoring=scoring, n_jobs=3, cv = tscv, verbose=1,
                          return_train_score=True, param_grid=param_grid)
        
        grid_search.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric="mape")
        model_path = args.TRAINED_MODEL_PATH 
        joblib.dump(grid_search.best_estimator_,  model_path + ticker + '.pkl')
        
        print("model fitted")
        result_df = pd.DataFrame(grid_search.cv_results_)
        result_df.to_csv(args.MODEL_PREDICTIONS + ticker + '_result.csv', index=False)
        print('Saved test scores to ' + args.MODEL_PREDICTIONS)
      
        path = args.TRAINED_MODEL_PATH + ticker + '.pkl'
        trained_model = joblib.load(path)
        
        experiment_name = args.EXPERIMENT_NAME  # Replace with your desired experiment name

        # Create or get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Start an MLflow run with the specified experiment name
        with mlflow.start_run(experiment_id=experiment_id):
            # choose the model as per provided arguments
#             model = select_model(args.MODEL_NAME, seed)
            
            # train the Random Forest model
            trained_model = grid_search.best_estimator_

            # evaluate the fitted model using mape and rmse metrics
            predictions_df, mape, rmse = evaluate_model(trained_model, window_size, dev_data, dev_date, X_test, y_test)          
            predictions_df.to_csv(args.MODEL_PREDICTIONS + ticker + '.csv', header=True, index=False)
        
            print("for ticker {0} mean absolute percentage error: {1}, root_mean_square_error: {2}".format(ticker, round(mape, 3), round(rmse, 3)))
        
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("target_price", target_price)
            mlflow.log_param("mape", round(mape, 5))    
            mlflow.log_param("rmse", round(rmse, 5))                
            signature = infer_signature(X_train, predictions_df)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(
                    trained_model, "model", registered_model_name=args.MODEL_NAME, signature=signature
                )
            else:
                mlflow.sklearn.log_model(trained_model, "model", signature=signature)        



