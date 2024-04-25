import pandas as pd

# Load the dataset
file_path = 'retractions35215.csv'
data = pd.read_csv(file_path)

# Dropping unnecessary columns
columns_to_drop = ['URLS', 'Notes', 'RetractionDOI', 'OriginalPaperDOI']
data.drop(columns=columns_to_drop, inplace=True)

# Handling missing values
# Assuming 'Institution' and 'Author' are critical, we replace missing values with 'Unknown'
data['Institution'].fillna('Unknown', inplace=True)
data['Author'].fillna('Unknown', inplace=True)
# For numerical IDs where 0 can indicate missing, we check if 0 is an appropriate placeholder
data['RetractionPubMedID'].fillna(0, inplace=True)
data['OriginalPaperPubMedID'].fillna(0, inplace=True)

# Convert dates to datetime objects
# Convert dates to datetime objects with a specified format
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], format='%d/%m/%Y', errors='coerce')
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], format='%d/%m/%Y', errors='coerce')


# Normalize text data: stripping extra spaces, converting to lowercase
text_columns = ['Title', 'Reason']
for col in text_columns:
    data[col] = data[col].str.lower().str.strip()

# Saving the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_retractions.csv'
data.to_csv(cleaned_file_path, index=False)

print("Data cleaning completed. Cleaned data saved to:", cleaned_file_path)
