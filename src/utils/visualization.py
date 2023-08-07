import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

# plot chart for actual and predicted values with including predicted values range(highe/lower)
def plot_prediction_range(df: pd.DataFrame, ticker: str) -> None:
    fig = go.Figure([
        go.Scatter(
            name='Actual Price',
            x=df['date'],
            y=df['high'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Predicted Price',
            x=df['date'],
            y=df['pred_high'],
            mode='lines',
            line=dict(color='rgb(255,140,0)'),
        ),    
        go.Scatter(
            name='Upper Bound',
            x=df['date'],
            y=df['pred_high_upper'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=df['date'],
            y=df['pred_high_lower'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Stock Price(Rs)',
        title=f'Stock Price Prediction for {ticker}',
        hovermode="x"
    )
    fig.update_layout(template="simple_white")
    fig.show()
    
def plot_predictions(df: pd.DataFrame, ticker: str) -> None:
    fig = go.Figure([
        go.Scatter(
            name='Actual Price',
            x=df['date'],
            y=df['y_test'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Predicted Price',
            x=df['date'],
            y=df['y_pred'],
            mode='lines',
            line=dict(color='rgb(255,140,0)'),
        )
    ])
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Stock Price(Rs)',
        title=f'Stock Price Prediction for {ticker}',
        hovermode="x"
    )
    fig.update_layout(template="simple_white")
    fig.show()    