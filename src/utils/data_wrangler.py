import pandas as pd
from datetime import datetime
import plotly.express as px
import numpy as np
import time
import os

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
