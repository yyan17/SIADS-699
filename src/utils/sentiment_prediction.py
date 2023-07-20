from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import numpy as np

# declare user defined packages
import sys
sys.path.append('src/utils')
import text_preprocessing, data_load

tickers = ['TTML.NS', 'CHOLAFIN.NS', 'KSB3.DE', 'ZENSARTECH.NS', 'OFSS.NS', 'LXCHEM.NS', 'VMART.NS', 'DLF.NS', 'IPCALAB.NS', 'ZFCVINDIA.NS']
company_names = ['Tata Teleservices', 'Cholamandalam Investment & Finance Company', 'KSB', 'Zensar Tech', 
                     'Oracle Financial Services Software','Laxmi Organic', 'V-Mart', 'DLF', 'Ipca Laboratories', 'ZF Commercial Vehicle']
ticker_dic = dict(zip(tickers, company_names))

num_cols = ['polarity', 'subjectivity', 'compound']


def get_vader_sentiment(row):
    sent_score = sia.polarity_scores(row['clean_text'])['compound']
    score = round(sent_score, 2)
    return(score)

def get_textblob_sentiment(row):
    sent_score = TextBlob(row['clean_text']).sentiment
    scores = sent_score.polarity, sent_score.subjectivity 
    scores = [round(score, 2) for score in scores]
    return(scores)

# topic wise sentiment aggregation
def agg_topic_sentiment(sentiment_df, topic_df):
    sent_topic_df = sentiment_df.merge(topic_df, on=['id', 'date'], how='inner')
    sent_topic_df['date'] = pd.to_datetime(sent_topic_df['date'])
    agg_sent_topic_df = sent_topic_df.groupby(['date', 'topic_id'])[num_cols].mean().reset_index()
    agg_sent_topic_df = agg_sent_topic_df.round(3)
    return(agg_sent_topic_df)

def scale_data(df):
    scaler  = MinMaxScaler(feature_range=(-1, 1))
    
    data_df = df.copy()
    
    # scale all the columns leaving date column
    scaled_df = pd.DataFrame(scaler.fit_transform(data_df.iloc[:, 1:]))
    
    # assign column names
    scaled_df.columns = data_df.columns[1:] 
    
    # get the date column
    scaled_df['date'] = data_df['Date']
    return(scaled_df)


def get_ticker_news_sentiment(row):
    for ticker, company_name in ticker_dic.items():
        if company_name in row['article']:
            compound = sia.polarity_scores(row['clean_text'])['compound'] 
            polarity = TextBlob(row['clean_text']).sentiment.polarity
            company = ticker
            break
        else:
            compound, polarity = 0, 0
            company = np.nan
    return(row['date'], company, row['article'],round(compound, 3), round(polarity, 3))

# ticker news sentiment
def agg_ticker_news_sentiment(sent_df, art_df):    
#     sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
#     # merge articles and sendiment dataframe

#     articles_sent_df = sentiment_df.merge(articles_df, on=['id', 'date'], how='inner')    

    sentiment_df = sent_df.copy()
    articles_df = art_df.copy()
    
    articles_df['article'] = articles_df['article'].astype('str')
    articles_df['clean_text'] = articles_df['article'].apply(lambda text : 
                                    text_preprocessing.preprocess_text(text, flg_stemm=False, flg_lemm = True, 
                                                            lst_stopwords = text_preprocessing.stop_words))

    ticker_sent_df = articles_df.apply(get_ticker_news_sentiment, axis=1, result_type="expand")  
    ticker_sent_df.columns = ['date', 'ticker', 'article', 'compound', 'polarity']
    ticker_sent_df = ticker_sent_df.dropna()
    return(ticker_sent_df)

sia = SentimentIntensityAnalyzer()

