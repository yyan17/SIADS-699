import yfinance as yf
import requests
import pandas as pd

# https://gist.github.com/bruhbruhroblox
# adopted from above git code
def getTicker(company_name: str) -> str:   
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()
    
    if len(data['quotes']) >0:
        company_code = data['quotes'][0]['symbol']
    else:
        print('failed to get ticker for company:', company_name)
        company_code = 'NA'
    return company_code

def clean_company(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # change company names to lower
    data['CLEAN_COMPANY'] = data['COMPANY'].str.lower()
    
    # remove the ltd/LTD/Ltd from the company names, as having these causes problem in getting ticket from yfinance
    data['CLEAN_COMPANY'] = data['CLEAN_COMPANY'].apply(lambda company: company.split('ltd')[0].strip())  
    
    # remove the "&" from the company names as these causes problem in getting ticker
    data['CLEAN_COMPANY'] = data['CLEAN_COMPANY'].apply(lambda company: company.split('&')[0].strip())      

    # remove the "&" from the company names as these causes problem in getting ticker
    data['CLEAN_COMPANY'] = data['CLEAN_COMPANY'].apply(lambda company: company.split('and')[0].strip())
    return(data)