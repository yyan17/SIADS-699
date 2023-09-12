from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error 
from prophet import Prophet
import pandas as pd
import numpy as np

# load user defined libraries
import sys
sys.path.append('src/utils/')
from data_wrangler import create_all_features, fetch_topn_features, convert_custom_target_to_actual


def prepare_data_for_training(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Takes the input features dataframe and creates the input dataframe suitable for prophet model 
    """
    df = df.loc[:, cols]
    df = df.rename(columns={'date': 'ds', 'ln_target': 'y'})
    return(df)

def create_model(cols: list):
    # define a prophet model
    model = Prophet()
    
    # add country specific holidays, available in prophet model
#     model.add_country_holidays(country_name='IN')
    
    # add topn features as additional regressors
    for col in cols:
        model.add_regressor(col)
    return(model)

# make dataframe for future predictions for days of eval dataset
def prepare_data_for_predictions(model, train_df, train_date, eval_df, eval_date, regressor_cols):
    future = pd.concat([train_date, eval_date], axis=0)
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future[regressor_cols] =  pd.concat([train_df[regressor_cols], eval_df[regressor_cols]],axis=0).values
    return(future)

def evaluate_model(model, future: pd.DataFrame, eval_data: pd.DataFrame, eval_df: pd.DataFrame):
    # do target prediction using the provide model
    prediction_df = model.predict(future)

    # combines actual and predictions for computing actual target prices
    result_df = combine_actual_and_predictions(prediction_df, eval_data, eval_df)
    
    # convert predictions to actual prices
    result_df = convert_predicitons_to_actual(result_df)
    
    # compute regression metric - mape 
    mape = mean_absolute_percentage_error(result_df['high'], result_df['pred_high'])

    # compute rmse metric
    rmse = mean_squared_error(result_df['high'], result_df['pred_high'], squared=False)        
    return(result_df, mape, rmse)

def convert_custom_target_to_actual(df: pd.DataFrame, col: str) -> "pd.Series[int]":
    """
    this module converts custom target - ln(high/yesterday_close) to actual high price again
    """
    data_df = df.copy()
    
    # exclude first 10 rows of train/test data, as while us
    y = np.exp(data_df[col]) * data_df.loc[:, 'yesterday_close'].reset_index(drop=True)
    return(y) 


def combine_actual_and_predictions(prediction_df: pd.DataFrame, eval_data: pd.DataFrame, eval_df:pd.DataFrame) -> pd.DataFrame:
    """
    Combines the actual and prophet prediciton dataframes, which can be used to convert custom target/predicitons to actual prices
    """
    actual_cols = ['date', 'yesterday_close', 'high']
    log_predicted_cols = ['yhat_lower', 'yhat', 'yhat_upper']
    
    # choose only the predictions for future data
    result_df = prediction_df.tail(eval_df.shape[0])[log_predicted_cols].reset_index(drop=True)

    # concatenate actual data columns and prediction columns
    result_df = pd.concat([result_df, eval_data[actual_cols].reset_index(drop=True)], axis=1)
    return(result_df)

def convert_predicitons_to_actual(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    converts all predicitons like yhat_lower, yhat, yhat_upper to actual prices
    """
    prediction_cols =  ['date','high','pred_high_lower','pred_high_upper','pred_high']
    
    result_df['pred_high_lower'] = convert_custom_target_to_actual(result_df, 'yhat_lower').round(3)
    result_df['pred_high_upper'] = convert_custom_target_to_actual(result_df, 'yhat_upper').round(3)
    result_df['pred_high'] = convert_custom_target_to_actual(result_df, 'yhat').round(3)
    return(result_df.loc[:, prediction_cols])
