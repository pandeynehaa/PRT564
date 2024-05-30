import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv('D:\\CDU sem 2\\Data analyst\\Assignment2\\retractions35215.csv')

# Split the 'Reasons' column which contains multiple reasons separated by semicolons, then explode it into separate rows
reason_data = data['Reason'].str.split('; ').explode()
reason_counts = reason_data.value_counts().nlargest(10)

# Adjust the figure size for the bar chart and apply constrained layout for better space management
plt.figure(figsize=(16, 12))  # Enlarged figure size
reason_counts.plot(kind='barh', color='cadetblue')
plt.title('Top 10 Common Reasons for Retractions', fontsize=14)
plt.xlabel('Frequency', fontsize=12)
plt.ylabel('Reasons', fontsize=12)
plt.xticks(fontsize=10)  # Smaller font size for ticks
plt.yticks(fontsize=10)
plt.show()

# Next, create a word cloud from the reasons field.
# Join all reasons into a single string, separating them with spaces
reasons_text = ' '.join(reason for reason in reason_data.dropna())

# Create the word cloud with specified dimensions and background color
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reasons_text)

# Display the word cloud
plt.figure(figsize=(14, 7))  # Adjusted figure size for word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove the axis
plt.title('Word Cloud for Reasons of Retraction', fontsize=14)
plt.show()

