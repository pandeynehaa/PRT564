# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import warnings

# Step 2: Load the Dataset
df = pd.read_csv("D:\\CDU sem 2\\Data analyst\\Assignment2\\retractions35215.csv")
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], errors='coerce')

# Step 3: Basic Data Exploration
print(df.head())
print("Dimensions of the dataset:", df.shape)
print("Column names:", df.columns)
df.dtypes  # Confirm data types, especially after conversion

# Step 4: Data Cleaning
print("Missing values:\n", df.isnull().sum())
df.fillna({
    'numeric': df.select_dtypes(include=['number']).median(),
    'object': df.select_dtypes(include=['object']).mode().iloc[0]
}, inplace=True)
df.drop_duplicates(inplace=True)


# Suppress specific seaborn tight_layout warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")

# Step 5: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['RetractionDate'].dt.year.dropna())
plt.title('Distribution of Retraction Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)  # Adjust layout parameters
plt.show()
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

#Visualize the distribution of citation counts
plt.figure(figsize=(10, 6))
sns.histplot(df['CitationCount'], bins=30)
plt.title('Distribution of Citation Counts')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.show()

# Explore relationships between variables (e.g., CitationCount vs. ArticleType)
plt.figure(figsize=(10, 6))
sns.boxplot(x='ArticleType', y='CitationCount', data=df, showfliers=False)
plt.title('Citation Count by Article Type')
plt.xlabel('Article Type')
plt.ylabel('Citation Count')
plt.xticks(rotation=45)
plt.show()

# Adjusting pair plot title
g = sns.pairplot(df.select_dtypes(include=['number']))
g.fig.suptitle('Pair Plot of Numerical Variables', y=1.02)  # Adjust title placement
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
