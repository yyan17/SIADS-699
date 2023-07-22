from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import os


allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
stop_words = stopwords.words('english')


def preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    lst_text = text.split()
    ## choose only alphbetical words and filter words less than 3 chars
    lst_text = [token.lower() for token in lst_text if token.isalpha() and len(token) >=3]      
#     text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
#     lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
    
    lst_text = [token for token in lst_text if len(token) >=3]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

def lemmatization(texts, allowed_postags=allowed_postags):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return(texts_out)


# divides the articles in to train/test by splitting on year/month basis 
def split_data(path, sample_size, random_state=699):
    # list all the files in diectory
    files = sorted(os.listdir(path))
    
    # select only articles data file *.jl
    files = [file for file in files if ".jl" in file]
    
    columns = ['id', 'date','article']
    train_df = pd.DataFrame({'id': [], 'date': [],'article': []}) 
    for indx, file in enumerate(files):
            file_path = path + file
            print(file_path)
            articles_df = pd.read_json(file_path, lines=True)
            
            # choose only limited columns to process
            articles_df = articles_df.loc[:, columns]
            
            # drop any rows having null for article
            articles_df = articles_df.dropna()                
            articles_df['date'] = pd.to_datetime(articles_df['date'])
            articles_df['month'] = articles_df['date'].dt.month
            num_months = 12
            yr_sample_df = pd.DataFrame({'id': [], 'date': [], 'article': []})
            for month in range(num_months):                
                month_df = articles_df.loc[articles_df['month'] == month + 1]
                size = round(len(month_df) * sample_size/100)  
                month_df = month_df.sample(size, random_state=random_state)

                # combine the monthly dataframe with yearly dataframe as per sample size
                yr_sample_df = pd.concat([yr_sample_df, month_df], ignore_index=False)
            
            print(len(yr_sample_df),  len(articles_df))
            
            train_df = pd.concat([train_df, yr_sample_df], ignore_index=True) 
            train_df = train_df.dropna()
    return(train_df)