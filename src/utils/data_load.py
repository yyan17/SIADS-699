import pandas as pd
import os

def load_articles(path):
    # list all the files in diectory
    files = sorted(os.listdir(path))
    
    # select only articles data file *.jl
    files = [file for file in files if ".jl" in file]
    
    columns = ['id', 'date', 'article']
    for indx, file in enumerate(files):
            
            # get filename and year from the read articles file
            filename = file.split('.')[0]
            year = filename.split('_')[1]
            
            # creates filepath for articles file to be read
            file_path = path + file
            print(file_path)
            article_df = pd.read_json(file_path, lines=True)
            
            # choose only limited columns to process
            article_df = article_df.loc[:, columns]
            
            # drop any rows having null for article
            article_df = article_df.dropna() 
            yield year, article_df