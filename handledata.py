import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
def load_data(filepath):
    return pd.read_csv('retractions35215.csv')

# Drop rows with missing values in specific columns
def drop_missing_data(data, columns):
    return data.dropna(subset=columns)

# Fill missing data with a specified value or statistic
def fill_missing_data(data, fill_strategy):
    for column, strategy in fill_strategy.items():
        if strategy == 'mean':
            data[column].fillna(data[column].mean(), inplace=True)
        elif strategy == 'median':
            data[column].fillna(data[column].median(), inplace=True)
        elif strategy == 'mode':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(strategy, inplace=True)
    return data

# Main function to clean data
def clean_data(filepath):
    data = load_data(filepath)

    # Columns to check for missing data that you want to drop
    columns_to_check = ['RetractionDate', 'OriginalPaperDate']

    # Dropping missing data
    data_cleaned = drop_missing_data(data, columns_to_check)

    # Filling missing data strategy
    fill_strategy = {
        'Institution': 'Unknown',  # For categorical data
        'Author': 'Unknown',       # For categorical data
        'RetractionPubMedID': 0,   # For numerical data
        'CitationCount': 'mean'    # Could be 'mean', 'median', or 'mode'
    }
    
    data_cleaned = fill_missing_data(data_cleaned, fill_strategy)

    return data_cleaned

# Save cleaned data to a new CSV file
def save_data(data, filepath):
    data.to_csv(filepath, index=False)

if __name__ == "__main__":
    input_filepath = 'path_to_your_data.csv'
    output_filepath = 'path_to_save_cleaned_data.csv'

    cleaned_data = clean_data(input_filepath)
    save_data(cleaned_data, output_filepath)
    print("Data cleaning completed. Cleaned data saved to:", output_filepath)
