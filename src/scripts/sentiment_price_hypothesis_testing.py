import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

sys.path.append('src/utils')


def scale_data(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    data_df = df.copy()

    # scale all the columns leaving date column
    scaled_df = pd.DataFrame(scaler.fit_transform(data_df.iloc[:, 1:]))

    # assign column names
    scaled_df.columns = data_df.columns[1:]

    # get the date column
    scaled_df['date'] = data_df['Date']
    return (scaled_df)


data_path = {'TICKER_LIST': ['EIHOTEL.BO', 'ELGIEQUIP.BO', 'IPCALAB.BO', 'PGHL.BO', 'TV18BRDCST.BO'],
             'TOPIC_LIST': [33, 921, 495, 495, 385],
             'MARKET_DATA_PATH': 'datasets/rawdata/rawdata_bse_500/',
             'TARGET_PRICE': ['High']
             }

sentiment_path = {'AGG_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/agg_sentiment.csv',
                  'TOPIC_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/agg_sent_topic.csv',
                  'TICKER_SENTIMENT': 'datasets/processed_data/agg_sentiment_scores/ticker_news_sent.csv',
                  }

# command to run this script
# python src/scripts/sentiment_price_hypothesis_testing.py

if __name__ == "__main__":
    columns = ['ticker', 'price', 'sentiment_type', 'statistic', 'pvalue', 'hypothesis_result']
    hypothesis_df = pd.DataFrame(columns=columns)
    for ticker, topic_id in zip(data_path['TICKER_LIST'], data_path['TOPIC_LIST']):
        ticker_df = pd.read_csv(data_path['MARKET_DATA_PATH'] + ticker + '.csv')
        scaled_df = scale_data(ticker_df)

        for sentiment_type in sentiment_path.keys():
            sentiment_df = pd.read_csv(sentiment_path[sentiment_type])

            # shift sentiment by one day behind
            sentiment_df = sentiment_df.set_index('date').shift(-1).reset_index()

            # merge the ticker data and sentiment dataframe
            combined_df = scaled_df.merge(sentiment_df, on='date', how='inner')
            combined_df = combined_df.dropna()

            if 'TOPIC' in sentiment_type:
                cond = np.where(combined_df['topic' + '_' + 'topic_id'] == topic_id)

            for target_price in data_path['TARGET_PRICE']:
                x = combined_df.loc[:, target_price].values
                sentiment_col = sentiment_type.split('_')[0] + '_' + 'compound'
                y = combined_df.loc[:, sentiment_col.lower()]

                result = pearsonr(x, y)
                statistic, pvalue = round(result.statistic, 4), round(result.pvalue, 15)
                data = [ticker, target_price, sentiment_type, statistic, pvalue, result]
                data_df = pd.DataFrame(data).T
                data_df.columns = hypothesis_df.columns
                hypothesis_df = pd.concat([hypothesis_df, data_df])
    print(hypothesis_df)
    # write the hypothesis result dataframe to file             
    hypothesis_df.to_csv('datasets/hypothesis_result.csv', index=False)
