import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("retractions35215.csv")

# Display basic information about the dataset
print(data.info())

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Check for outliers
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[col])
    plt.title(col)
    plt.show()

# Summary statistics
print("Summary Statistics:")
print(data.describe())

# Identify common reasons for retractions
common_reasons = data['Reason'].value_counts()
print("Common Reasons for Retractions:")
print(common_reasons)

# Temporal analysis
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'])
data['Year'] = data['RetractionDate'].dt.year
retraction_trend = data.groupby('Year').size()
plt.figure(figsize=(10, 6))
retraction_trend.plot(kind='bar')
plt.title('Retraction Trend Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.show()

# Geospatial analysis
countries = data['Country'].str.split(';', expand=True).stack().str.strip().value_counts()
top_countries = countries.head(10)
print("Top 10 Countries with Most Retractions:")
print(top_countries)

# Author and Institutional Analysis
authors = data['Author'].str.split(';', expand=True).stack().str.strip().value_counts()
top_authors = authors.head(10)
print("Top 10 Authors with Most Retractions:")
print(top_authors)

institutions = data['Institution'].str.split(';', expand=True).stack().str.strip().value_counts()
top_institutions = institutions.head(10)
print("Top 10 Institutions with Most Retractions:")
print(top_institutions)

# Predictive Modeling (if desired)
# You can use machine learning algorithms to predict future retractions based on historical data

# Stakeholder Engagement
# Collaborate with stakeholders to share insights and recommendations

#This code will help you perform exploratory data analysis, identify missing values, outliers, common reasons for retractions, analyze temporal and geospatial trends, and identify top authors and institutions associated with retractions. You can further extend this code to include predictive modeling and stakeholder engagement as needed.