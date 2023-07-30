import yfinance as yf
import requests
import pandas as pd

# https://gist.github.com/bruhbruhroblox
# adopted from above git code
def getTicker(company_code: str) -> str:   
    yfinance_uri = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_code, "quotes_count": 2, "country": "India"}
    try:
        res = requests.get(url=yfinance_uri, params=params, headers={'User-Agent': user_agent})
        data = res.json()
        if len(data['quotes'])  ==  1:
            ticker = data['quotes'][0]['symbol']
        else:
            if '.BO' in data['quotes'][0]['symbol']:
                ticker = data['quotes'][0]['symbol']
            else:
                ticker = data['quotes'][1]['symbol']
        # change NSE ticker name to BSE ticker name, as we are working on BSE 500 constituent stocks
        ticker = ticker.replace('.NS', '.BO')
    except:
        print('failed to get ticker for company:', company_code)
        ticker = None
    return ticker
