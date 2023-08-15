# Indian Stock Market Prediction

## Getting Started

- Clone the repository: <br/>
<code>git clone https://github.com/yyan17/SIADS-699.git</code>
- Move to the project folder:<br/>
<code>cd SIADS-699</code>
- Set up the necessary dependencies:<br/>
<code>pip install -r requirements.txt</code>
- Launch Jupyter notebooks via the command prompt:<br/>
<code>cd path/to/target/notebook</code> <br/>
<code>jupyter <notebook_name></code>
- Or run python script via the command: <br/>
<code>cd path/to/target/script</code> <br/>
<code>python3 <script_name></code>
- Jupyter notebooks for Random Forest: <br/>
  - notebooks/random_forest/random_forest_EIHOTEL.ipynb
  - notebooks/random_forest/random_forest_ELGIEQUIP.ipynb
  - notebooks/random_forest/random_forest_PGHL.ipynb
  - notebooks/random_forest/random_forest_TV18BRDCST.ipynb
  - notebooks/random_forest/random_forest_IPCALAB.ipynb
- Jupyter notebooks for Linear Regression: <br/>
  - notebooks/linear_regression/linear_regression_EIHOTEL.ipynb
  - notebooks/linear_regression/linear_regression_ELGIEQUIP.ipynb
  - notebooks/linear_regression/linear_regression_PGHL.ipynb
  - notebooks/linear_regression/linear_regression_TV18BRDCST.ipynb
  - notebooks/linear_regression/linear_regression_IPCALAB.ipynb
- Jupyter notebooks for LSTM: <br/>
  - notebooks/LSTM_EIHOTEL_log_return.ipynb
  - notebooks/LSTM_ELGIEQUIP_log_return.ipynb
  - notebooks/LSTM_IPCALAB_log_return.ipynb
  - notebooks/LSTM_PGHL_log_return.ipynb
  - notebooks/LSTM_TV18BRDCST_log_return.ipynb
  - notebooks/EIHOTEL_outcome_visualizations.ipynb
  - notebooks/ELGIEQUIP_outcome_visualizations.ipynb
  - notebooks/IPCALAB_outcome_visualizations.ipynb
  - notebooks/PGHL_outcome_visualizations.ipynb
  - notebooks/TV18BRDCST_outcome_visulaizations.ipynb
- Python script for Prophet: <br/>
  - src/scripts/training_evaluation/train-predict-Prophet.py
- Python script for LightGBM: <br/>
  - src/scripts/training_evaluation/predict_byLightGBM.py
- Combined datasets for all 5 tickers can be found at: <br/>
<code>datasets/processed_data/combined_features</code>

## Data Access Statement
The data sets available in this GitHub repository were sourced from Yahoo Finance, The Economic Times, and BSE India. To the best of our understanding, their utilization is open and unrestricted.

We advise that any copying, dissemination, or re-hosting of this data should be undertaken through its original source.
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
**- Sentiment Correlation Hypothesis Testing:** In the first experiment, we did hypothesis testing to find if there is a correlation between stock price and different kinds of news sentiment(daily/topci/ticker news sentiment). 
We found that while four of the stocks have moderate positive correlation one had weak negative correlation(TV18BRDCST). Weak negative correlation may be attributed to the fact that if there is any positive news for a stock, stock rallies that day and it often goes down the next day owing to profit booking by traders.

**- Stock Price Prediction (With/Wo Sentiment Scores):** In the second experiment, we ran baseline RandomForest Regression for five tickers with and without   sentiment scores while keeping all other features constant. Surprisingly, we observed:
Improved performance using only technical features.
Decreased importance of sentiment features (feature importance score of 100+) when including other features like rolling features and financial results.  

## Model Training

We employed a diverse range of models for stock price prediction:

**- Statistical Models:** We utilized Prophet to capture time series patterns and trends in stock prices. <br/>
**- Supervised Machine Learning Models:** For regression tasks, we explored Linear Regression, RandomForest, and LightGBM to capture complex relationships in the data. <br/>
**- Deep Learning-Based Models:** LSTM were used for sequence modeling, allowing us to capture long-term dependencies in stock price patterns. <br/>

## Results

In the academic setting of three month time and limited resources, we were able to achieve the best performance by a model(Prophet) 0.67% to 2.35%. For three stocks, we achieved the MAPE score of either <1% or close to 1%. It shows that given the required resources good performance can be achieved.

Further results underscore that the influence of sentiment features on stock price prediction varies depending on the stock and the model employed. In certain instances, the incorporation of sentiment features can bolster prediction accuracy, while in others, it might dampen performance. This emphasizes the significance of meticulous investigation when selecting features and models.
Additionally, we are using sentiment scores aggregated on a daily basis as we are doing daily predictions. However, emotional reactions to them happen during the trading day as and when news comes. Potentially changing the frequency of the stock market prediction from daily to lower frequency like - min, hourly might be one of the options for further investigation.

For supervised learning models, to provide the temporal sense, we used a 10-day window. Further experiments can be done to show how other rolling window sizes like 5, 20, 30, 50 days impact the performance of the model.


## Environment Variables:
Python implementation: CPython <br/>
Python version       : 3.10.12 <br/>
IPython version      : 8.12.0 <br/>
Compiler    : GCC 11.2.0 <br/>
OS          : Linux <br/>
Release     : 5.4.72-microsoft-standard-WSL2 <br/>
Machine     : x86_64 <br/>
Processor   : x86_64 <br/>
CPU cores   : 12 <br/>
Architecture: 64bit <br/>

---
