import pandas as pd


def main(target_file, shap_file, feature_file):
    # Load the shap values
    shap_values = pd.read_csv(shap_file)

    # Get the top 50 features
    top_53_features = shap_values.head(53)['feature'].tolist()

    # Load the features data
    features_data = pd.read_csv(feature_file, compression='gzip')

    # Select the top 50 features from the features data
    features_data_top_50 = features_data[top_53_features]

    # Save the top 50 features data
    features_data_top_50.to_csv(target_file, index=False)


# Specify the directory where your .csv.gz files are located
shap_file = r'C:\Users\yangy\Desktop\workspace\SIADS-699\datasets\processed_data\feature_importance\TTML.NS_shap_values.csv'
feature_file = r'C:\Users\yangy\Desktop\workspace\SIADS-699\datasets\processed_data\combined_features\TTML.NS_features.csv.gz'
target_file = r"C:\Users\yangy\Desktop\workspace\SIADS-699\datasets\processed_data\combined_features\TTML.NS_features_top_53_with_sentiment.csv"
main(target_file, shap_file, feature_file)
