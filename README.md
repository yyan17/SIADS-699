# Indian Stock Market Prediction

## Introduction

With a market capitalization of 3.2 trillion USD, India’s stock market is the world’s fifth-largest stock market. Yet while much literature is available on the analysis of stock prices of the USA, the same cannot be said about the Indian stock market. This project aims to fill this gap by providing a robust analysis using different statistical techniques to predict stock prices.

## Key Features

**- Data Scraping:** We scraped 1.5 million financial news articles from the popular English financial news website, [Economic Times](https://economictimes.indiatimes.com/), using Scrapy. We also collected financial results of companies from https://www.bseindia.com/.  
**- Topic Modeling:** BERTopic was used for topic modeling of financial news, enabling us to better understand market sentiment and its impact on stock prices.   
**- Feature Importance:** We used TreeSHAP to compute feature importance, enabling us to select the most influential features for our prediction models.  
**- Hyperparameter Tuning:** GridSearchCV was employed to perform hyperparameter tuning, optimizing the performance of our models.  
**- Experiment Tracking:** MLflow was used to track and manage machine learning experiments, facilitating better reproducibility and collaboration.  
**- Visualization:** We utilized Plotly and Altair for interactive and insightful visualizations, aiding us in understanding data patterns and model results.  

## Data Sources

**- Yahoo Finance:** We collected historical stock data from [Yahoo Finance](https://github.com/ranaroussi/yfinance), including price, volume, and other relevant information for various Indian stocks.  
**- Economic Times:** We scraped financial news articles from the Economic Times to extract sentiment and topic information for market sentiment analysis.  
**- BSE India:** We scraped financial results of companies from [BSE India](https://www.bseindia.com/), enriching our dataset with valuable fundamental information.  

## Tools Used

**- Scrapy:** For web scraping financial news articles from Economic Times.  
**- BERTopic:** To perform topic modeling on the financial news data, helping us understand key trends and topics in the market.  
**- TreeSHAP:** For computing feature importance and selecting the most relevant features for our prediction models.  
**- MLflow:** For experiment tracking, enabling better management and reproducibility of machine learning experiments.  
**- GridSearchCV:** To perform hyperparameter tuning, optimizing model performance.  
**- Plotly and Altair:** For interactive and insightful visualizations, aiding us in understanding data patterns and model results.  

## Market Sentiment Analysis

In this project, we take sentiment analysis to the next level by first performing topic modeling on financial news and then using the sentiment score only for the particular industry to which the stock belongs. This approach aims to provide a more accurate reflection of market sentiment and aid in better prediction of stock prices.

### Experiment with Sentiment Features 
**- Sentiment Correlation Hypothesis Testing:** We conducted hypothesis testing to assess the correlation between stock prices and various sentiment types   (daily/topic/ticker news sentiment). Our findings showed:  
Moderate positive correlation for CHOLAFIN and IPCALAB tickers.
Weak positive correlation for TTML and DLF tickers.
Weak negative correlation for one ticker.  
**- Stock Price Prediction (With/Wo Sentiment Scores):** In the second experiment, we ran baseline RandomForest Regression for five tickers with and without   sentiment scores while keeping all other features constant. Surprisingly, we observed:
Improved performance using only technical features.
Decreased importance of sentiment features (feature importance score of 100+) when including other features like rolling features and financial results.  

## Model Training

We employed a diverse range of models for stock price prediction:

**- Statistical Models:** We utilized AR, ARIMA, and SARIMA to capture time series patterns and trends in stock prices.
**- Supervised Machine Learning Models:** For regression tasks, we explored Linear Regression, RandomForest, and LightGBM to capture complex relationships in the data.
**- Deep Learning-Based Models:** LSTM and GRU were used for sequence modeling, allowing us to capture long-term dependencies in stock price patterns.

## Results

We analyze the predictive accuracy and robustness of each model, highlight significant findings, and discuss the influence of market sentiment scores on prediction results.

## Discussion：
As we are still in the midst of running the models and analyzing the results, we would update this section, once we have some validated results.  

## Packages Used:
Python implementation: CPython
Python version       : 3.10.12
IPython version      : 8.12.0
pandas    : 2.0.3
numpy     : 1.24.4
scrapy    : 2.6.2
matplotlib: 3.7.1
missingno : 0.4.2
altair    : 5.0.1
shap      : 0.42.0
sklearn   : 1.3.0
bertopic  : 0.15.0
plotly    : 5.15.0
mlflow    : 2.4.2
bs4       : 4.12.2
Compiler    : GCC 11.2.0
OS          : Linux
Release     : 5.4.72-microsoft-standard-WSL2
Machine     : x86_64
Processor   : x86_64
CPU cores   : 12
Architecture: 64bit

---
