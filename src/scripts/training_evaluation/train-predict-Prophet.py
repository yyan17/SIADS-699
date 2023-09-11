from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
import joblib
import mlflow.sklearn
import mlflow
import os
import warnings
import pandas as pd
import numpy as np
import argparse
import sys
from mlflow.tracking import MlflowClient

sys.path.append('src/utils/')
from data_wrangler import create_all_features, fetch_topn_features
from prophet_util import prepare_data_for_training, create_model, evaluate_model, prepare_data_for_predictions

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
                 'TOPIC_IDS': [33, 921, 495, 495, 385]

             }

train_size = 0.8  # 80% for training, 20% for testing
window_size = 10  # Number of past records to consider = 50
topn_feature_count = 50
target_price = 'ln_target'
seed= 42

# command to run the script
# python src/scripts/training_evaluation/train-predict-Prophet.py Prophet customTargetProphetTop50 datasets/processed_data/feature_importance/LightGBM/ datasets/processed_data/model_predictions/Prophet/ trained_models/Prophet/

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL_NAME', help='provide the ml model, for which we want to train/predict the data ')
    parser.add_argument('EXPERIMENT_NAME', help='provide the experiment name, for which to run the training')   
    parser.add_argument('FEATURE_PATH', help='path where to write/read computed shape values for feature importance')
    parser.add_argument('MODEL_PREDICTIONS', help='path where to write model predictions')    
    parser.add_argument('TRAINED_MODEL_PATH', help='path form where to read pre-trained LightGBM model')    
    
    args = parser.parse_args()
    # fetch topn features as per feature importance
    topn_features_df = fetch_topn_features(args.FEATURE_PATH, topn_feature_count)
    regressor_cols = topn_features_df['feature'].values.tolist()
    prophet_train_cols = ['date', 'ln_target'] + regressor_cols
    for indx, ticker in enumerate(data_paths['TICKERS']):
        topic_id = data_paths['TOPIC_IDS'][indx]
        
        path = data_paths['COMBINED_FEATURES'] + ticker + '.csv'
        if os.path.isfile(path):
            combined_df = pd.read_csv(path)
        else:            
            # create all the features
            combined_df = create_all_features(data_paths, ticker, topic_id)                
            combined_df.to_csv(path, index=False)

        
        # extract date column for pre-processing
        combined_date_df = combined_df['date']
        
        # do train/test split the data with shuffle = False
        train_data, test_data = train_test_split(combined_df, train_size=train_size, shuffle=False)
        train_date, test_date = train_test_split(combined_date_df, train_size=train_size, shuffle=False)
        
        # prepare data for prophet model training/evaluation
        train_df = prepare_data_for_training(train_data, prophet_train_cols)
        test_df = prepare_data_for_training(test_data, prophet_train_cols)        
        
        experiment_name = args.EXPERIMENT_NAME  # Replace with your desired experiment name

        # Create or get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Start an MLflow run with the specified experiment name
        with mlflow.start_run(experiment_id=experiment_id):
            # create the model object as per provided arguments
            pf_model = create_model(regressor_cols)    
            
            # train the prophet model
            trained_model = pf_model.fit(train_df)            

#             model_path = args.TRAINED_MODEL_PATH + args.MODEL_NAME + '/'
            joblib.dump(trained_model,  args.TRAINED_MODEL_PATH + ticker + '.pkl')
            
            
            # make dataframe for future predictions for days of eval dataset    
            future = prepare_data_for_predictions(trained_model, train_df, train_date, test_df, test_date, regressor_cols)    
            
            # evaluate the fitted model using mape and rmse metrics
            predictions_df, mape, rmse = evaluate_model(trained_model, future, test_data, test_df)
#             pred_len = int(predictions_df.shape[0]/2)
            predictions_df.to_csv(args.MODEL_PREDICTIONS + ticker + '.csv', header=True, index=False)
            
            print("for ticker {0} mean absolute percentage error: {1}, root_mean_square_error: {2}".format(ticker, round(mape, 5), round(rmse, 5)))
        
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("target_price", target_price)
            mlflow.log_param("mape", round(mape, 5))    
            mlflow.log_param("rmse", round(rmse, 5))                
            signature = infer_signature(train_df, predictions_df)

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

