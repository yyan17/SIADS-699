import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def select_features(df, ticker_ds):
    # If the dataframe has less than 3 columns, return all columns
    if df.shape[1] <= 3:
        return df.columns.tolist()

    # If 'Date' column exists, exclude it from the features
    features = df.drop('Date', axis=1).columns if 'Date' in df.columns else df.columns

    # If the dataframe has 3 or 4 columns, return all columns except 'Date'
    if len(features) <= 3:
        return features.tolist()

    # If the dataframe has more than 4 columns, select 3 best features
    selector = SelectKBest(score_func=f_regression, k=3)
    X = df[features]
    y = ticker_ds['Close']

    # # Drop cols with NaN values
    X = X.dropna(axis=1, how='all')
    X.fillna(method='ffill', inplace=True)

    # Drop columns where any value is NaN
    X = X.dropna(axis=1, how='any')

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    selector.fit(X, y)
    return X.columns[selector.get_support()].tolist()


def combine_datasets(file_path, new_filename):
    # Load CHOLAFIN.NS data
    ticker_ds = pd.read_csv(file_path)
    ticker_ds['Date'] = pd.to_datetime(ticker_ds['Date'])
    ticker_ds.set_index('Date', inplace=True)

    # List of data file paths
    file_paths = [
        r'datasets\raw\market_data\BSESN.csv',
        r'datasets\raw\market_data\interest_rate.csv',
        r'datasets\raw\market_data\CL=F.csv',
        r'datasets\raw\market_data\INR=X.csv',
        r'datasets\raw\market_data\Treasury_Yeild_10_Years.csv',
        r'datasets\raw\market_data\GSPC.csv',
        r'datasets\raw\market_data\USDX-Index.csv'
    ]

    # Merge selected features from each file into CHOLAFIN.NS data
    for file_path in file_paths:
        # Load data
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Get filename for column renaming
        filename = os.path.splitext(os.path.basename(file_path))[0]

        # Select features
        features = select_features(df, ticker_ds)

        # Rename selected features
        renamed_features = [f"{feature}_{filename}".lower() for feature in features]
        df.rename(columns=dict(zip(features, renamed_features)), inplace=True)

        # Merge data
        ticker_ds = ticker_ds.merge(df[renamed_features], how='left', left_index=True, right_index=True)

    # Save the combined data to a new file
    ticker_ds.to_csv(new_filename)


if __name__ == "__main__":
    file_path = r'<your_local_workspace_path>\SIADS-699\datasets\raw\market_data\KSB3.DE.csv'
    new_filename = r'<your_local_workspace_path>\SIADS-699\datasets\raw\market_data\KSB3.DE_combined.csv'
    combine_datasets(file_path, new_filename)
