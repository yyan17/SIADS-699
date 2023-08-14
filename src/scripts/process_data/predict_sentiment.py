import argparse
import warnings

from nltk.corpus import stopwords

# warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
seed = 699

stop_words = stopwords.words('english')

# declare user defined packages
import sys

sys.path.append('src/utils')
import text_preprocessing, data_load
from sentiment_prediction import get_textblob_sentiment, get_vader_sentiment

# articles_path = 'datasets/articles/'
# sentiment_path = 'datasets/sentiment_scores/'

# command to run the script 
# python src/scripts/process_data/predict_sentiment.py datasets/rawdata/articles/  datasets/processed_data/sentiment_scores/

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ARTICLES_PATH', help='path for the articles to be read')
    parser.add_argument('SENTIMENT_PATH', help='path for predicted sentiments to be written')

    args = parser.parse_args()

    articles_gen = data_load.load_articles(args.ARTICLES_PATH)

    columns = ['id', 'date', 'polarity', 'subjectivity', 'compound']
    while True:
        try:
            # get next year and articles file through the generator
            year, articles_df = next(articles_gen)

            # used for stub testing
            articles_df = articles_df.sample(100)

            # preprocess the articles to get clean text
            articles_df['clean_text'] = articles_df['article'].apply(lambda text:
                                                                     text_preprocessing.preprocess_text(text,
                                                                                                        flg_stemm=False,
                                                                                                        flg_lemm=True,
                                                                                                        lst_stopwords=stop_words))

            # predict sentimnent scores using textblob
            articles_df[['polarity', 'subjectivity']] = articles_df.apply(get_textblob_sentiment, axis=1,
                                                                          result_type="expand")

            # predict sentimnent scores using vader sentiment analyzer
            articles_df['compound'] = articles_df.apply(get_vader_sentiment, axis=1, result_type="expand")

            # choose only selected columns to write to file
            articles_df = articles_df.loc[:, columns]

            # wrote the sentiment scores to the file
            filename = 'sentiment_scores' + '_' + (year) + '.csv'
            articles_df.to_csv(args.SENTIMENT_PATH + filename, index=False)

            # checks for the end of generator
        except StopIteration:
            print("Sentiment scores predicted for all the articles")
            break
