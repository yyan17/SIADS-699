from nltk.corpus import stopwords
from bertopic import BERTopic
import torch
import pandas as pd
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
from topic_exploration import topic_inference

# articles_path = 'datasets/articles/'
# topic_path = 'datasets/topic_labels/'
# modelpath = 'trained_models/bertopic_model_3pc'

# command to run the script 
# python src/scripts/process_data/predict_topic.py datasets/rawdata/articles/  trained_models/bertopic_model_10pc  datasets/processed_data/topic_labels/ 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ARTICLES_PATH', help='path for the articles to be read')    
    parser.add_argument('MODEL_PATH', help='path for trained BERTopic model to be provided')      
    parser.add_argument('TOPIC_PATH', help='path for predicted topic labels to be written')   
    
    args = parser.parse_args()    

    # load the pre-trained BERTopic model from the provided path
    topic_model = BERTopic.load(args.MODEL_PATH)
    
    # load the news articles from the provided articles path
    articles_gen = data_load.load_articles(args.ARTICLES_PATH)
    
    
    columns = ['id', 'date', 'topic_id', 'topic_prob']
    while True:
        try:
            # get next year and articles file through the generator
            year, articles_df = next(articles_gen)
            
            # used for stub testing
            articles_df = articles_df.sample(100)
            
            # preprocess the articles to get clean text
            articles_df['clean_text'] = articles_df['article'].apply(lambda text : 
                                text_preprocessing.preprocess_text(text, flg_stemm=False, flg_lemm=True,lst_stopwords=stop_words))

            # predict sentimnent scores using textblob
            topics_df = topic_inference(articles_df, args.TOPIC_PATH, topic_model)

            # choose only selected columns to write to file
            topics_df = topics_df.loc[:, columns]

            # write the topic labels to the file
            filename = 'topic_labels' + '_' + (year) + '.csv'
            topics_df.to_csv(args.TOPIC_PATH + filename, index=False)  
#             break
        
        # checks for the end of generator
        except StopIteration:
            print("Sentiment scores predicted for all the articles")
            break
            