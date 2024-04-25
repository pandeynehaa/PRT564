import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('D:\\CDU sem 2\\Data analyst\\Assignment2\\retractions35215.csv')

# Retraction by Journal: Bar Chart
top_journals = data['Journal'].value_counts().head(12)
plt.figure(figsize=(12, 6))
top_journals.plot(kind='bar', color='teal')
plt.title('Top 10 Journals by Number of Retractions')
plt.xlabel('Journal')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Retraction by Subject: Pie Chart
top_subjects = data['Subject'].value_counts().head(12)
plt.figure(figsize=(12, 8))
top_subjects.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Retractions Across Top 10 Subjects')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Retraction by Country: Bar Chart
# Splitting the 'Country' column which contains semicolon-separated countries, and then exploding it into separate rows
country_data = data['Country'].str.split(';').explode()
top_countries = country_data.value_counts().head(12)
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar', color='coral')
plt.title('Top 10 Countries by Number of Retractions')
plt.xlabel('Country')
plt.ylabel('Number of Retractions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
