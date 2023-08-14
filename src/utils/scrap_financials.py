import re
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs


def scrap_financial_results_placeholer(uri):
    base_uri = "https://www.bseindia.com/corporates/"
    response = requests.get(uri, headers = {'User-Agent':'Mozilla/5.0'})

    soup = bs(response.content, features="lxml")

    qtr_results = {'term': [], 'uri': []}
    yrly_results = {'term': [], 'uri': []}

    result = soup.find_all(class_='TTRow')
    for indx, item in enumerate(result):
        if indx% 8 > 0 and indx% 8 < 7 and indx% 8 != 5:        
            report = item.find(class_='tablebluelink')
            if report:
                year = int(report.text.split('-')[1])
                if indx %8 <5:
                    # choose the results only from 2008                    
                    if year >= 8:
                        qtr_results['term'].append(report.text)
                        qtr_results['uri'].append(report['href'])
        #             print(report)
                if indx%8 == 6:
                    # choose the results only from 2008
                    if year >= 8:
                        yrly_results['term'].append(report.text)
                        yrly_results['uri'].append(report['href'])

    # clean the result url for quarterly and yearly reports
    qtr_results['uri'] = [(base_uri + uri_param).replace(' ', '%20').replace("amp;", '') for uri_param in qtr_results['uri']]
    yrly_results['uri'] = [(base_uri + uri_param).replace(' ', '%20').replace("amp;", '') for uri_param in yrly_results['uri']]
    return(qtr_results, yrly_results)

def scrap_financial_result(uri):
    # handle requests failure
    while True:
        try:
            response = requests.get(uri, headers = {'User-Agent':'Mozilla/5.0'})
            break
        except:
            print("scraping failed for uri:", uri)
            time.sleep(10)
            
    soup = bs(response.content, features='lxml')
    columns = soup.find_all(class_='TTRow_left')
    values = soup.find_all(class_='TTRow_right')

    financial = {'column': [], 'value': []}
    for var, val in zip(columns, values):
            financial['column'].append(var.text)
            financial['value'].append(val.text)

    # extract report dates
    report_details = soup.find_all(class_='resultgreyhead')[2:]
    # if result is available get the result
    if len(report_details) >= 1:
        start_date = report_details[1].text
        end_date = report_details[3].text
        report_dates = {'start_date': start_date, 'end_date': end_date}
    else:
        return(None, None)
    return(financial, report_dates)

def combine_financial_result(results):
    # create an empty placeholder dataframe to combine all the financial dataframe
    for term, uri,indx in zip(results['term'], results['uri'], range(len(results['uri']))):
        print("scraping :", term, uri)
        # scrap the quarterly/yearly result from bse result page
        financial, report_dates   = scrap_financial_result(uri)
        # is result has been scraped create a dataframe for the same
        if financial:
            print("financial results are avialble for the scrip")
            financial_df = pd.DataFrame(financial).T.reset_index(drop=True)

            # use first row of the dataframe as columns of financial dataframe
            financial_df.columns = financial_df.iloc[0]

            # dorp the first row, which is column name for financial data values
            financial_df = financial_df.iloc[1:, ]
                        
            # select one of the eps features from many a names available
            financial_df = filter_eps_cols(financial_df)
            
            # create date range for which report would be valid
            date_range = pd.date_range(start=report_dates['start_date'], end=report_dates['end_date'])

            financial_df= pd.concat([financial_df]*len(date_range), ignore_index=True)
            financial_df['date'] = date_range                   
            if indx == 0:
                financials_df = financial_df.copy()                
            else:
                financial_df = financial_df.loc[~financial_df.index.duplicated(keep='first')].copy()

                # choose unique column names to extracted from financial results
                columns2 = list(set(financial_df.columns.tolist()))
        
                financial_df = financial_df.loc[:, columns2].copy()
                try:
                    financials_df = pd.concat([financials_df, financial_df], axis=0)
                    print("financials_df columns length:", len(financials_df.columns.tolist()))                    
                except:
                    print("dataframe concatenation failed for term:", term, uri)
    return(financials_df)

def clean_financial_columns(columns):
    for indx, column in enumerate(columns):
        columns[indx] = re.sub(r'[^\w\s]', '', str(column).lower().strip())
        columns[indx] = columns[indx].split()
        columns[indx] = '_'.join([word for word in columns[indx] if len(word) >= 2])
    return(columns)

# data cleaning for eps columns
# often eps feature names are different across companies/quarterly/yearly reports
# this method is to provide one common name to eps feature
def filter_eps_cols(df):
    fin_df = df.copy()
    basic_eps_cols = [col for col in fin_df.columns.tolist() if 'Basic' in col]
    diluted_eps_cols = [col for col in fin_df.columns.tolist() if 'Diluted' in col]
    eps_cols = basic_eps_cols + diluted_eps_cols

    if len(basic_eps_cols) > 1:
            fin_df['basic_eps'] = np.where(fin_df[basic_eps_cols[0]], fin_df[basic_eps_cols[0]], fin_df[basic_eps_cols[1]])
    elif len(basic_eps_cols) == 1:
        fin_df['basic_eps'] = fin_df[basic_eps_cols[0]]
    else:
        fin_df['basic_eps'] = None
    if len(diluted_eps_cols) > 1:
        fin_df['diluted_eps'] = np.where(fin_df[diluted_eps_cols[0]], fin_df[diluted_eps_cols[0]], fin_df[diluted_eps_cols[1]])
    elif len(diluted_eps_cols) == 1:
        fin_df['diluted_eps'] = fin_df[diluted_eps_cols[0]]
    else:
        # if diluted eps is not reported, consider it same as basic eps
        fin_df['diluted_eps'] = fin_df['basic_eps']
    return(fin_df)
    
# this method reads all the scraped raw financial resutls and gets result column, which is reported across all tickers
# this is to have consistency across all tickers and maintain the consistent shape for all tickers
def get_common_columns(path, tickers, result_type):
    for indx, ticker in enumerate(tickers):
        ticker_df = pd.read_csv(path + ticker + '_' + result_type + '.csv')
        curr_cols = set(ticker_df.columns.tolist())
        if indx == 0:
            ticker_cols = curr_cols
        else:
            # get common columns across all tickers
            ticker_cols = ticker_cols.intersection(curr_cols)

    ticker_cols = list(ticker_cols)    
    # remove the blank column scraped from site
    ticker_cols.remove('_' + result_type)
    
    ticker_cols = ['date'] + [ticker for ticker in ticker_cols if ticker != 'date']
    return(ticker_cols)
    