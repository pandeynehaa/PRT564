import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('D:\\CDU sem 2\\Data analyst\\Assignment2\\retractions35215.csv')

# Select features for clustering
features = ['CitationCount', 'RetractionPubMedID', 'OriginalPaperPubMedID']

# Check for NaN values and drop rows with NaNs in the selected features
data = data.dropna(subset=features)

# Extract the features
X = data[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3,n_init='auto', random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA results and cluster labels
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters

# Plotting the scatter plot with clusters
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Clusters on PCA-reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# For the biplot, let's plot the principal components and annotate them with the feature names
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', alpha=0.5)
feature_vectors = pca.components_.T
arrow_size, text_pos = 2.5, 2.8
for i, v in enumerate(feature_vectors):
    plt.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], width=0.1, alpha=0.5, color='red')
    plt.text(v[0]*text_pos, v[1]*text_pos, features[i], color='black')
plt.title('PCA Biplot with Cluster Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()