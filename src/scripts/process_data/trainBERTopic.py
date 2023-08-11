from collections import Counter
from bertopic import BERTopic
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import os

import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
seed=699

import sys
sys.path.append('src/utils')
import topic_exploration, text_preprocessing


# articles_path = 'datasets/articles/'
# model_path = 'trained_models/'

# define path for writing processed date(clean text)
processed_path = 'datasets/processed_articles/'

# command to be run
# python src/scripts/trainBERTopic.py datasets/articles/  trained_models/ 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ARTICLES_PATH', help='path of the news articles to be read')    
    parser.add_argument('MODEL_PATH', help='path where trained topic model needs to be written')        
    parser.add_argument('TRAIN_PERCENT', help='percentage of the data to train on')      
    args = parser.parse_args()                
    
    start_time = datetime.now()
    # define the model name to be saved and its path    
    model_name = 'bertopic_model_' + args.TRAIN_PERCENT + 'pc'
    modelpath = args.MODEL_PATH + model_name


    # split the news articles in train/test
    # this train dataset would be used to train the BERTopic model from which topic inference can be done    
    train_df = text_preprocessing.split_data(args.ARTICLES_PATH, int(args.TRAIN_PERCENT))
    
    # do minimal pre-processing of the data, as all parts of the docs are important to understand the topic as per documentation
    train_df['clean_text'] = train_df['article'].apply(lambda text : 
                        text_preprocessing.preprocess_text(text, flg_stemm=False, flg_lemm=True, 
                                           lst_stopwords =  text_preprocessing.stop_words))

        
    texts = list(train_df['clean_text'].values)
    
    # used for stub testing
#     texts = texts[:1000]
    
    # train the BERTTopic model with default params and number of topics as per args
    topic_model = BERTopic(language='english', calculate_probabilities=False,verbose=True)
    topics, probs = topic_model.fit_transform(texts)

    # save the trained model for future use
    BERTopic.save(topic_model, modelpath)
    
    # save the processed trained data for future use
    filename = 'train_df_' + args.TRAIN_PERCENT + 'pc.csv'
    processed_data_path = processed_path + filename
    train_df.to_csv(processed_data_path)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))





