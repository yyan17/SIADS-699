from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from PyEMD import EMD, CEEMDAN
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import numpy as np
import math
import os

seed=42

def load_data(basepath):
    # list all the raw data files in the base path
    stocks_files = os.listdir(basepath)

    # create column names based on the ticker name
    colnames = [stock.replace('.csv', '') for stock in stocks_files]
    
    frames = []
    # read the raw data for all the tickers and combine to create a dataframe
    for file in stocks_files:
        filepath = basepath + file
        
        # choose the Date and Adj Close column data from the raw data
        df = pd.read_csv(filepath, usecols=[0, 5])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        frames.append(df)
    df = pd.concat(frames, axis=1)
    df.columns = colnames
    df = df.reset_index()
    return(df)

def preprocess(df, start_date):
    # choose the data as per start date provided and handle missing values in the data
    cond = np.where(df['Date'] > start_date)
    
    # choose 120-150 days of data
    data = df.iloc[cond].copy()
    
    # drop ticker column, which has only null values 
    data = data.dropna(axis=1, how='all')
    
    # drop row, which has only null values for all the tickers
    data = data.dropna(axis=0, how='all')
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    data = data.set_index('Date')
    return(data)    

def scaling(df, scaling):
    # scale the data with minmax/standard scaling based on option
    if scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    data_df = df.copy()
    scaled_df = scaler.fit_transform(data_df)
    scaled_df = pd.DataFrame(scaled_df)
    scaled_df.columns = data_df.columns
    return(scaled_df)

def log_returns(df):
    # preprocess the stocks data as log returns 
    data_df = df.copy()
    log_df = np.log(data_df)
    log_scaled = np.diff(log_df, axis=0)
    log_scaled = pd.DataFrame(log_scaled, columns=data_df.columns)
    return(log_scaled)

def kelbow_visualizer(X):
    kmeans = TimeSeriesKMeans(n_jobs=-1, metric='dtw', random_state=seed)    
    visualizer = KElbowVisualizer(kmeans, k=(5, 25), timing=False)
    visualizer.fit(X)
    visualizer.show()    
    
def extract_imfsby_ceemdan(signal):
    ceemdan = CEEMDAN()
    imfs = ceemdan(signal)    
    return(imfs)            

def ceemdan_feature(df):
    data_df = df.copy()
    ticker_list = data_df.columns

    preprocessed_dataset = []
    for ticker in tqdm(ticker_list):
        signal = data_df[ticker].values
        imfs = extract_imfsby_ceemdan(signal)
        
        # choosing top three imfs as feature
        preprocessed_dataset.append(np.concatenate(imfs[:3]))    
    return(preprocessed_dataset)


def plot_clusters(df, cluster_labels):
    plot_count = math.ceil(math.sqrt(len(set(cluster_labels))))
    fig, axis = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    row = 0
    col = 0

    for cluster_label in set(cluster_labels):
        cluster_cond = np.where(cluster_labels == cluster_label)[0].tolist()
        cluster_df = df.iloc[:, cluster_cond].copy()
        ticker_list = cluster_df.columns.tolist()
        cur_cluster = []

        for ticker in ticker_list:
            axis[row, col].plot(cluster_df[ticker], c='gray', alpha=0.4)
            cur_cluster.append(cluster_df[ticker])

        if len(ticker_list) > 0:
            axis[row, col].plot(np.average(np.vstack(cur_cluster), axis=0), c='green')  
            axis[row, col].plot(dtw_barycenter_averaging(np.vstack(cur_cluster)), c='red')  
            axis[row, col].set_title('Cluster ' + str(cluster_label))
            axis[row,col].title.set_size(20)
        col += 1

        if col % plot_count == 0:
            row += 1
            col = 0
            
            
# Compute dissimilarity matrix between stocs across clusters            
def extract_max_dis_pairs(ds_df, cluster_labels):
    df = pd.DataFrame(ds_df.idxmax()).reset_index()
    df.columns = ['row', 'col']

    # get max dissimilarity values across different stocks pairs
    df['max_dis_val'] = ds_df.max()
    
    # get cluster labels
    df['cluster_src'] = cluster_labels
    df['cluster_target'] = df['col'].apply(lambda x: cluster_labels[x])
    df = df.sort_values(['max_dis_val'], ascending=[False]).reset_index(drop=True)
    return(df)

def get_max_dis_stocks(df, cluster_labels, stocks_df):
    # create a copy of source dataframe to avoid source being modified in the function
    data_df = df.copy()
    
    # creta an empty dataframe
    columns = ['ticker_indx', 'max_dissimilarity_distance', 'cluster_source', 'ticker']
    df = pd.DataFrame(columns=columns)

    tickers_list = []
    for label in set(cluster_labels):
        # for every cluster label, get the stocks pair with max dissimilarity metric value
        cond = np.where(data_df['cluster_src'] == label)
        
        ticker_df = data_df.iloc[cond][['col', 'max_dis_val']].head(1)
        # check if at least one ticker is available for the cluster
        if len(ticker_df) > 0:
            ticker_details = data_df.iloc[cond][['row', 'max_dis_val', 'cluster_src']].head(1).values.tolist()[0]
            # get the ticker index
            stocks_idx = ticker_details[0]
            ticker = stocks_df.columns[int(stocks_idx)]

            # check if data available for ticker full time series
            ticker_details.append(ticker)
            tickers_list.append(ticker_details)        
            
    # sort the tickers dataframe by max dissimilarity distance
    tickers_df = pd.DataFrame(tickers_list, columns=columns).sort_values('max_dissimilarity_distance', ascending=False)
    tickers_df = tickers_df.reset_index(drop=True)
    return(tickers_df) 


# Extract Ticker Info
def get_ticker_info(stocks_list):
    # declare the ticker info initial dictionary
    ticker_info = {'symbol': [], 'industry': [], 'sector': [], 'marketCap': [], 'shortName': [], 'revenuePerShare': [],   
                       'currentPrice': [], 'totalRevenue': [], 'revenueGrowth': [], 'operatingMargins': [], 'longBusinessSummary': []}
    
    # declare the ticker columns for ticker info to be extracted
    ticker_cols = ['symbol', 'industry', 'sector', 'marketCap', 'shortName', 'revenuePerShare', 'currentPrice', 'totalRevenue',
                         'revenueGrowth', 'operatingMargins', 'longBusinessSummary']
#     stocks_list = ['KSB3']
    for ticker in stocks_list:
        # extract ticker info by using yfinance api
        ticker_details = yf.Ticker(ticker)
        try:
            ticker_details.info
        except:
            print('details not available for ticker: {0}'.format(ticker))
        for col in ticker_cols:
            try:
                ticker_info[col].append(ticker_details.info[col])
            except:
                print('for ticker: {0} {1} detail is not available'.format(ticker, col))
                ticker_info[col].append('NA')
    ticker_info_df = pd.DataFrame(ticker_info)
    return(ticker_info_df)