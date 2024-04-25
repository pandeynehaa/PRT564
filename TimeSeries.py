import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('D:\\CDU sem 2\\Data analyst\\Assignment2\\retractions35215.csv')

# Convert the RetractionDate to datetime format with the correct date format
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], format='%d/%m/%Y')

# Extract year and month from RetractionDate for further analysis
data['Year'] = data['RetractionDate'].dt.year
data['Month'] = data['RetractionDate'].dt.month

# Yearly Trends: Number of Retractions Per Year
yearly_counts = data['Year'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-')
plt.title('Number of Retractions Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Retractions')
plt.grid(True)
plt.show()

# Month/Season Analysis: Number of Retractions by Month
monthly_counts = data['Month'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.bar(monthly_counts.index, monthly_counts.values, color='skyblue')
plt.title('Number of Retractions Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Retractions')
plt.xticks(monthly_counts.index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(axis='y')
plt.show()