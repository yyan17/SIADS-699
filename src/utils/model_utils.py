from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import pandas as pd
import numpy as np
import shap
seed= 42

def evaluate_model(model, X_test, y_test, y_pred):
    # do target prediction using the provide model
    y_pred = model.predict(X_test)
    
    # compute regression metric - mape 
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return(mape)


def train_model(X_train, y_train):
    start_time = datetime.now()

    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)  # You can adjust the hyperparameters as needed

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
