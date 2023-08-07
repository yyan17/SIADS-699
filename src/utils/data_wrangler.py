import pandas as pd
from datetime import datetime
import plotly.express as px
import numpy as np
import time
import os

cols_to_drop  = ['trend_psar_up', 'trend_psar_down']
window_lst = [5, 10, 20, 50, 100, 200]
rolling_exclude_cols = ['open', 'date','close','adj close', 'volume', 'high', 'low']
agg_topic_ticker_sent_cols = ['date', 'agg_polarity', 'agg_compound', 'topic_polarity', 'topic_compound', 'ticker_polarity', 'ticker_compound']

missing_val_threshold_percent = 40
             
def load_rawdata(path, tickers_list):
    files = os.listdir(path)
    ticker_files = [file for ticker in tickers_list for file in files if ticker in file]
    
    tickers_df = pd.read_csv(path + ticker_files[0]).head(0)
    for ticker in ticker_files:
        ticker_df = pd.read_csv(path + ticker)
        
        tickers_df = pd.concat([tickers_df, ticker_df])
    return(tickers_df)

def find_topn_missing_val_cols(df, topn): 
    # make a copy of the dataframe, so as not to change the source dataframe
    data_df = df.copy()
    
    # compute the missing columns percentage
    mssing_col_df = pd.DataFrame(data_df.isnull().mean() * 100).reset_index().sort_values([0], ascending=False)
    mssing_col_df.columns = ['column', 'percent_missing']
    topn_missing_col_df = mssing_col_df.iloc[:topn, :].reset_index(drop=True)
    return(topn_missing_col_df)

def find_missing_val_cols_by_threshold(df, percent): 
    # find the missing values percentage for all the columns
    missing_df = find_topn_missing_val_cols(df, None)
    
    # find the columns which have missing values more than provided percent
    cond = np.where(missing_df['percent_missing'] > percent)
    missing_cols = missing_df.iloc[cond].column.tolist()
    return(missing_cols)


def find_all_missing_val_cols(path, results, usecols):
    missing_value_cols = set()
    for indx, result in enumerate(results):
        df = pd.read_csv(path + result, usecols=usecols)
        curr_missing_value_cols = find_missing_val_cols_by_threshold(df, missing_val_threshold_percent)
        missing_value_cols = missing_value_cols.union(curr_missing_value_cols)
    return(list(missing_value_cols))


def data_cleaning(df, drop_cols):
    data_df = df.copy()
    data_df = data_df.drop(columns = drop_cols)
    data_df.columns = [col.lower() for col in data_df.columns]
    data_df = data_df.replace([np.inf, -np.inf], np.nan)
#     data_df = data_df.fillna(method='ffill')
    data_df = data_df.convert_dtypes()
    return(data_df)

def create_lags(df, lags, target):
    # create the number of lags as per params for the target column
    data_df = df.copy()
    for lag in range(1, lags+1):
        data_df[f'lag_{lag}'] = data_df[target].shift(lag)  
    
    # drop NaN values for the lags column created
    data_df = data_df.dropna()    
    return(data_df)

def create_rolling_features(df, win_lst, exclude_cols):
    data_df = df.copy()
    
    # create the combined rolling dataframe, in which all rolling dataframe would be concatenated
    
    # dataframe for which to create rolling features
    base_data_df = data_df.drop(columns=exclude_cols).copy()
    
    # combined roling dataframe to be used
    combined_rolling_df = data_df.drop(columns=exclude_cols).copy()
    exclude_df = data_df.loc[:, exclude_cols]
    
    for window in win_lst:
        rolling_df = base_data_df.rolling(window, min_periods=1,axis=1).mean()
        rolling_df.columns = [col + '_' + str(window) for col in base_data_df.columns]
        
        # concatenate combined rolling dataframe and newly created rolling average dataframe
        
        combined_rolling_df = pd.concat([combined_rolling_df, rolling_df], axis=1)
    combined_rolling_df = pd.concat([exclude_df, combined_rolling_df], axis=1)
    return(combined_rolling_df)

def timeseries_to_supervise(df, window_size, target):    
    X = []
    y = []
    indx = []
    no_records = len(df)
    #     
    for i in range(window_size, no_records):
        X.append(df.iloc[i-window_size:i].drop(target, axis=1).values.flatten())  # Collect past records as a sequence
        y.append(df.iloc[i][target])  # Next record as target variable
        indx.append(np.arange(i-window_size, i))

    X = pd.DataFrame(X)
    y = pd.Series(y)
    return(X, y, indx)

def data_cleaning_financial(df):
    fin_df = df.copy()
    fin_df = fin_df.replace([np.inf, -np.inf], np.nan)
    fin_df = fin_df.fillna(method='ffill').fillna(method='bfill')    

    date_col = fin_df['date']    
    num_df = fin_df.drop(columns='date')
    num_df = num_df.replace({',': ''}, regex=True)
    num_df = num_df.apply(pd.to_numeric)

    fin_df = pd.concat([date_col, num_df], axis=1)
    return(fin_df)

def read_sentiment_features(data_paths, topic, ticker):
    agg_sentiment_df = pd.read_csv(data_paths['AGG_SENTIMENT'])

    topic_cols = ['date', 'topic_polarity', 'topic_subjectivity', 'topic_compound']
    agg_topic_sentiment_df = pd.read_csv(data_paths['TOPIC_SENTIMENT'])
    cond = np.where(agg_topic_sentiment_df['topic_topic_id'] == topic)
    agg_topic_sentiment_df = agg_topic_sentiment_df.iloc[cond].reset_index(drop=True)[topic_cols]


    ticker_cols = ['date', 'ticker_compound', 'ticker_polarity']
    ticker_sentiment_df = pd.read_csv(data_paths['TICKER_SENTIMENT'])
    cond = np.where(ticker_sentiment_df['ticker_ticker'] == ticker)
    ticker_sentiment_df = ticker_sentiment_df.iloc[cond].reset_index(drop=True)[ticker_cols]

    sentiment_df = agg_sentiment_df.merge(agg_topic_sentiment_df, on='date', how='outer').merge(ticker_sentiment_df, on='date', how='outer')
    sentiment_df = sentiment_df.fillna(0)
    return(sentiment_df)


def read_financial_features(data_paths, ticker):
    qtr_fin_df = pd.read_csv(data_paths['FINANCIAL_RESULTS'] + ticker + '_QTR.csv', low_memory=False)
    yrly_fin_df = pd.read_csv(data_paths['FINANCIAL_RESULTS'] + ticker + '_YRLY.csv', low_memory=False)

    qtr_fin_df = qtr_fin_df.rename(columns={'date_qtr': 'date'})
    yrly_fin_df = yrly_fin_df.rename(columns={'date_yrly': 'date'})
    return(qtr_fin_df, yrly_fin_df)


def preprocess_index_features(path, ticker, day='today'):
    ticker_df = pd.read_csv(path + ticker + '.csv')
    ticker_df['yesterday_Close'] = ticker_df['Close'].shift(+1)
    # for eastern countries, market is not yet close today, it is opened few hours before Indian market, so we use their todays' Open and yesterday Close price to compute the percent change in the market today
    ticker_df[ticker + '_' + 'PERCENT_CHANGE'] = (ticker_df['Open'] - ticker_df['yesterday_Close'])/ticker_df['yesterday_Close']

    if day == 'yesterday':
    # for western markets, they have already closed and are one day behind, hence we compute the percent change in Close from yesterday Close and move it one day ahead, when it can be used for Indian markets
        ticker_df[ticker + '_' + 'PERCENT_CHANGE'] = (ticker_df['Close'] - ticker_df['yesterday_Close'])/ticker_df['yesterday_Close']
        ticker_df[ticker + '_' + 'PERCENT_CHANGE'] = ticker_df[ticker + '_' + 'PERCENT_CHANGE'].shift(+1)
    return(ticker_df[['Date', ticker + '_' + 'PERCENT_CHANGE']])


def combine_index_features(path, yesterday_index, todays_index):
#     combined_index_df = pd.DataFrame()
    for indx, ticker in enumerate(yesterday_index):
        curr_index_df = preprocess_index_features(path, ticker, 'yesterday')
        if indx == 0:
            combined_index_df = curr_index_df.copy()
        else:            
            combined_index_df = combined_index_df.merge(curr_index_df, on='Date', how="outer")
            
    for indx, ticker in enumerate(todays_index):
        curr_index_df = preprocess_index_features(path, ticker, 'today')
        combined_index_df = combined_index_df.merge(curr_index_df, on='Date', how="outer")
    combined_index_df.columns = [col.lower() for col in combined_index_df.columns]
    return(combined_index_df)

def create_index_features(data_paths):
    file = os.listdir(data_paths['INDEX_FEATURES'])
    index_features_df = pd.read_csv(data_paths['INDEX_FEATURES'] + file[0])
    index_features_df = index_features_df.fillna(method='ffill')
    return(index_features_df)

def create_custom_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates the custom target price, which is computed as ln(high/yesterday_close)
    """    
    # make a copy of the dataframe so as not to change the original dataframe
    data_df = df.copy()
    
    # create yesterday_close feature as
    data_df['yesterday_close'] = data_df['close'].shift(1)
    
    # create custom target price to predict, computing  ln(high/yesterday_close)
    data_df['ln_target'] = np.log(data_df['high'] / data_df['yesterday_close'])
    
    # as yesterday close would not be available for first day, 
    # we would not have custom target price for that day, which needs to be excluded 
    return(data_df.iloc[1:, ])
    
    
def convert_custom_target_to_actual_for_supervise(df: pd.DataFrame, window: int, y: "pd.Series[int]") -> "pd.Series[int]":
    """
    this module converts custom target - ln(high/yesterday_close) to actual high price again for timeseries converted data using rolling         window of size 10
    """
    data_df = df.copy()
    
    # exclude first 10 rows of train/test data, as while us
    y = np.exp(y) * data_df.loc[data_df.index[window:], 'yesterday_close'].reset_index(drop=True)
    return(y)    

def convert_custom_target_to_actual(df: pd.DataFrame, col: str) -> "pd.Series[int]":
    """
    this module converts custom target - ln(high/yesterday_close) to actual high price again
    """
    data_df = df.copy()
    
    # exclude first 10 rows of train/test data, as while us
    y = np.exp(data_df[col]) * data_df.loc[:, 'yesterday_close'].reset_index(drop=True)
    return(y)    

def fetch_topn_features(path: str, topn: int) -> list:  
    shap_files = os.listdir(path)  
    shap_files = [file for file in shap_files if '.csv' in file]
    for indx, shap_file in enumerate(shap_files):
        feat_df = pd.read_csv(path + shap_file)
        if indx == 0:
            whole_feat_df = feat_df
        else:
            whole_feat_df = whole_feat_df.merge(feat_df, on='feature', how='inner')  
            
    whole_feat_df['avg_shap_value'] = whole_feat_df.loc[:, whole_feat_df.columns != 'feature'].sum(axis=1)
    whole_feat_df = whole_feat_df.sort_values('avg_shap_value', ascending=False).reset_index(drop=True)
    
    topn_features = whole_feat_df['feature'].values.tolist()[:topn]
    return(topn_features)

           
def create_all_features(data_paths: str, ticker: str, topic_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    this module accepts path for data, ticker and topic_id for the industry of ticker and create all the features from them
    """    
    try:
        tickers_df = pd.read_csv(data_paths['RAW_DATA'] + ticker + '.csv')
    except FileNotFoundError as e:
        print("raw data file not found for ticker provided")
        return(None, None)

    # data cleaning, handling missing/infinite values, dorp columns with mostly na values
    clean_df = data_cleaning(tickers_df, cols_to_drop)

    # create rolling average feature over the timeframe of provide time frame(window list) list
    rolling_feature_df = create_rolling_features(clean_df, window_lst, rolling_exclude_cols)

    # combine all sentiment scores to create ready useable features
    sentiment_df = read_sentiment_features(data_paths, topic_id, ticker)
    sentiment_df = sentiment_df[agg_topic_ticker_sent_cols]

    # create ready usable financial features
    qtr_fin_df, yrly_fin_df = read_financial_features(data_paths, ticker)
    qtr_fin_df = data_cleaning_financial(qtr_fin_df)
    yrly_fin_df = data_cleaning_financial(yrly_fin_df)        

    # create index features
    index_features_df = pd.read_csv(data_paths['INDEX_FEATURES'] + 'index_features.csv')

    
    # combine all the features 
    combined_df = rolling_feature_df.merge(qtr_fin_df, on='date', how='left')
    combined_df = combined_df.merge(yrly_fin_df, on='date', how='left')

     # combine sentiment scores features
    combined_df = combined_df.merge(sentiment_df, on='date', how='left')

     # combine other index features/oli price, Rs rate, interest rate etc
    combined_df = combined_df.merge(index_features_df, on='date', how='left')
    print("shape after combining index featuress results:", combined_df.shape)

    # create the custom target price using ln(high/yesterday_close)
    combined_df = create_custom_target(combined_df)

    combined_date_df = combined_df['date']
    combined_df = combined_df.drop(columns='date')

    # combined_df = combined_df.loc[:, lightGBM_top_200_features]

    # finally fill any missing values
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    combined_df = pd.concat([combined_date_df, combined_df], axis=1)
    return(combined_df)


# utility for preparig the data for prophet
def prepare_data_for_prophet(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    prophet_cols = ['ds', 'y']
    df = df.loc[:, cols]
    df.columns = prophet_cols
    return(df)

    