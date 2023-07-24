from nltk.corpus import stopwords
import pandas as pd
from datetime import datetime
import argparse
import os
import warnings
# warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
seed=699

stop_words = stopwords.words('english')

# declare user defined packages
import sys
sys.path.append('src/utils')
import text_preprocessing, data_load
from sentiment_prediction import get_textblob_sentiment, get_vader_sentiment, agg_topic_sentiment, get_ticker_news_sentiment, agg_ticker_news_sentiment


agg_sent_cols = ['date', 'polarity', 'subjectivity', 'compound']
agg_topic_sent_cols = ['date', 'topic_id', 'polarity', 'subjectivity', 'compound']
ticker_sent_cols = ['date', 'ticker', 'article', 'compound', 'polarity']

# command to run the script 
# python src/scripts/process_data/agg_predicted_sentiment.py datasets/rawdata/articles/  datasets/processed_data/sentiment_scores/ datasets/processed_data/topic_labels/ datasets/processed_data/agg_sentiment_scores/

if __name__ == '__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('ARTICLES_PATH', help='path for the articles to be read')    
    parser.add_argument('SENTIMENT_PATH', help='path from where to read predicted sentiment scores')   
    parser.add_argument('TOPIC_PATH', help='path from where to read predicted topic labels')       
    parser.add_argument('OUTPUT_PATH', help='path where to write the aggregate sentiment dataframes')       
    
    args = parser.parse_args()    

    articles_gen = data_load.load_articles(args.ARTICLES_PATH)
    sentiment_files = os.listdir(args.SENTIMENT_PATH)    
    topic_files = os.listdir(args.TOPIC_PATH)
    
    
    out_agg_sentiment_df = pd.DataFrame(columns=agg_sent_cols)
    out_topic_sentiment_df = pd.DataFrame(columns=agg_topic_sent_cols)    
    out_ticker_sentiment_df = pd.DataFrame(columns=ticker_sent_cols)
    
    for sent_file, topic_file in zip(sentiment_files, topic_files):
        # get next year and articles file through the generator
        year, articles_df = next(articles_gen)

        # used for stub testing
#         articles_df = articles_df.head(10)

        # preprocess the articles to get clean text
        articles_df.loc[:, 'clean_text'] = articles_df.loc[:, 'article'].apply(lambda text : 
                            text_preprocessing.preprocess_text(text, flg_stemm=False, flg_lemm=True,lst_stopwords=stop_words))

        # 1. read the predicted sentiment scores file        
        sentiment_df = pd.read_csv(args.SENTIMENT_PATH + sent_file)
        
        # read the predicted topic labels file
        topic_df = pd.read_csv(args.TOPIC_PATH + topic_file)

        # compute aggregate daily sentiment
        num_cols = ['polarity', 'subjectivity', 'compound']
        agg_sentiment_df = sentiment_df.groupby('date')[num_cols].mean().reset_index().round(3)
        out_agg_sentiment_df = pd.concat([out_agg_sentiment_df, agg_sentiment_df])

        # compute aggregate topic wise sentiment score
        agg_sent_topic_df = agg_topic_sentiment(sentiment_df, topic_df) 
        out_topic_sentiment_df = pd.concat([out_topic_sentiment_df, agg_sent_topic_df])

        # compute ticker news sentiment score
        ticker_news_sent_df = agg_ticker_news_sentiment(sentiment_df, articles_df)
        out_ticker_sentiment_df = pd.concat([out_ticker_sentiment_df, ticker_news_sent_df])
   

    # change sentiment column names to relfect sentiment type
    out_agg_sentiment_df.columns = ['date'] + ['agg_' + col for col in out_agg_sentiment_df.columns.tolist() if col != 'date']
    out_topic_sentiment_df.columns = ['date'] + ['topic_' + col for col in out_topic_sentiment_df.columns.tolist() if col != 'date']
    out_ticker_sentiment_df.columns = ['date'] + ['ticker_' + col for col in out_ticker_sentiment_df.columns.tolist() if col != 'date']
    
    # write all three kinds of sentiments scores in file
    out_agg_sentiment_df.to_csv(args.OUTPUT_PATH + 'agg_sentiment.csv', index=False)
    out_topic_sentiment_df.to_csv(args.OUTPUT_PATH + 'agg_sent_topic.csv', index=False)
    out_ticker_sentiment_df.to_csv(args.OUTPUT_PATH + 'ticker_news_sent.csv', index=False)
    
    end_time = datetime.now()
    running_time = end_time - start_time
    print("Total running time for the job is:", running_time)
        
    

    