# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Dataset
df = pd.read_csv("retractions35215.csv")

# Step 3: Basic Data Exploration
# Check the first few rows
print(df.head())

# Check dimensions
print("Dimensions of the dataset:", df.shape)

# Check column names
print("Column names:", df.columns)

# Check data types
print("Data types:\n", df.dtypes)

# Summary statistics
print("Summary statistics:\n", df.describe())

# Step 4: Data Cleaning (if necessary)
# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Step 5: Exploratory Data Analysis (EDA)
# Visualize the distribution of retraction dates
plt.figure(figsize=(10, 6))
sns.histplot(df['RetractionDate'])
plt.title('Distribution of Retraction Dates')
plt.xlabel('Retraction Date')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Visualize the distribution of citation counts
plt.figure(figsize=(10, 6))
sns.histplot(df['CitationCount'], bins=20)
plt.title('Distribution of Citation Counts')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.show()

# Explore relationships between variables (e.g., CitationCount vs. ArticleType)
plt.figure(figsize=(10, 6))
sns.boxplot(x='ArticleType', y='CitationCount', data=df)
plt.title('Citation Count by Article Type')
plt.xlabel('Article Type')
plt.ylabel('Citation Count')
plt.xticks(rotation=45)
plt.show()

# Explore correlations between numerical variables
correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

