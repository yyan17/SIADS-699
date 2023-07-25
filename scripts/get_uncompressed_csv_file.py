import pandas as pd


def main(target_file, shap_file, feature_file):
    # Load the shap values
    shap_values = pd.read_csv(shap_file)

    # Get the top 53 features
    top_50_features = shap_values.head(53)['feature'].tolist()

    # Ensure 'Date' is always included
    if 'date' not in top_50_features:
        top_50_features.append('date')

    # Load the features data
    features_data = pd.read_csv(feature_file)

    # Select the top 53 features from the features data
    features_data_top_50 = features_data[top_50_features]

    # Make 'Date' the first column
    columns = features_data_top_50.columns.tolist()
    columns.insert(0, columns.pop(columns.index('date')))
    features_data_top_50 = features_data_top_50.reindex(columns=columns)

    # Save the top 53 features data
    features_data_top_50.to_csv(target_file, index=False)


# Specify the directory where your .csv.gz files are located
shap_file = r'C:\Users\yangy\Desktop\workspace\SIADS-699\datasets\processed_data\feature_importance\TTML.NS_shap_values.csv'
feature_file = r'C:\Users\yangy\Desktop\workspace\SIADS-699\datasets\processed_data\combined_features\TTML.NS_combined.csv'
target_file = r"C:\Users\yangy\Desktop\workspace\SIADS-699\datasets\processed_data\combined_features\TTML.NS_features_top_53_with_sentiment.csv"
main(target_file, shap_file, feature_file)
