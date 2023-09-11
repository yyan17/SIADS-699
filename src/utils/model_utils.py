from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import xgboost as xgb
from prophet import Prophet
import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
seed= 42

# import user defined libraries
from data_wrangler import convert_custom_target_to_actual

def evaluate_model(model, window, dev_data, dev_date, X_test, y_test):
    # do target prediction using the provide model
    y_pred = model.predict(X_test)

    # convert back to original value, before computing mape            
    y_test = convert_custom_target_to_actual(dev_data, window, y_test)
    y_pred = convert_custom_target_to_actual(dev_data, window, y_pred)

    dev_dates = dev_date[window:].reset_index(drop=True)
    predictions_df = pd.DataFrame({'date': dev_dates, 'high': y_test, 'pred_high': y_pred})

    # compute regression metric - mape 
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # compute rmse metric
    rmse = mean_squared_error(y_test, y_pred, squared=False)        
    return(predictions_df, mape, rmse)


def train_model(model, X_train, y_train):
    start_time = datetime.now()

#     model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)  # You can adjust the hyperparameters as needed

    # fit the model 
    model.fit(X_train, y_train)
    end_time = datetime.now()
    train_time = end_time - start_time    
    print('Total trainging time: ', train_time)
    return(model)


def extract_shap_values(model, train_data, X_train, window_size):
    # use shapely to compute features SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)


    # convert shape values shape back to timeseries format to get feature importance
    reconstructed_shape_values = shap_values.reshape((X_train.shape[0], int(X_train.shape[1]/10), window_size))
    average_shap_val_reconstrcuted = np.mean(reconstructed_shape_values, axis=(2))
    average_shape_values = np.mean(average_shap_val_reconstrcuted, axis=0)
    
    feature_importance_df = pd.DataFrame(zip(average_shape_values, train_data.columns), columns=['shap_value', 'feature'])
    feature_importance_df = feature_importance_df.sort_values('shap_value', ascending=False).reset_index(drop=True)
    return(feature_importance_df)

def select_model(model_name, seed):
    if model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=seed, n_jobs=-1)  # You can adjust the hyperparameters as needed
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(random_state=seed, n_jobs=-1)
    elif model_name =='LightGBM':
        model = lgb.LGBMRegressor(random_state=seed, n_jobs=2, verbose=-1)
    elif model_name == 'Prophet':
        model = Prophet()
    return(model)
