import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import PowerTransformer
import time

# Start the timer
start_time = time.time()

# Function to run RM KMeans with KMeans++ initialization and Power Transformation
def rm_kmeans_kplusplus(df, n_clusters):
    # Step 1: Apply Power Transformation (Box-Cox for positive values)
    pt = PowerTransformer(method='box-cox')  # Box-Cox can handle positive values only
    df[['Transformed R score', 'Transformed F score', 'Transformed M score']] = pt.fit_transform(
        df[['R score', 'F score', 'M score']]
    )

    # Step 2: Initialize KMeans++ with more initializations to ensure centroid spread
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,  # Increase the number of initializations for better centroid spread
        max_iter=300,
        random_state=42
    )
    kmeans.fit(df[['Transformed R score', 'Transformed F score', 'Transformed M score']])

    # Step 3: Add the cluster labels to the original DataFrame
    df['Cluster'] = kmeans.labels_

    # Calculate silhouette score
    silhouette_avg = silhouette_score(
        df[['Transformed R score', 'Transformed F score', 'Transformed M score']],
        kmeans.labels_
    )

    return kmeans, df, silhouette_avg, pt

# Function to plot clusters in 3D (non-interactive) with size proportional to cluster size
def plot_clusters_3d(df, kmeans, silhouette_avg, pt):
    # Prepare the centroids in the original scale
    transformed_centroids = pt.inverse_transform(kmeans.cluster_centers_)

    # Count the number of records in each cluster
    cluster_sizes = df['Cluster'].value_counts().sort_index()

    # Scale factor for point sizes (proportional to the number of records in the cluster)
    scale_factor = 5  # Reduced this factor for smaller points
    sizes = df['Cluster'].map(cluster_sizes) * scale_factor

    # Create a non-interactive 3D scatter plot using Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the color map
    cmap = plt.cm.get_cmap('tab10', len(df['Cluster'].unique()))

    # Plot the clusters
    scatter = ax.scatter(
        df['R score'], df['F score'], df['M score'],
        c=df['Cluster'], cmap=cmap, s=sizes, alpha=0.7
    )

    # Plot the centroids with the same color as the cluster points
    for i, centroid in enumerate(transformed_centroids):
        ax.scatter(
            centroid[0], centroid[1], centroid[2],
            s=300, c=[cmap(i)], marker='X', label=f'Centroid {i}'
        )

    # Labels and title
    ax.set_xlabel('R score')
    ax.set_ylabel('F score')
    ax.set_zlabel('M score')
    ax.set_title(f'3D Cluster Visualization (Silhouette Score: {silhouette_avg:.2f})')

    # Create color legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    # Show the legend for centroids
    ax.legend()

    # Display the plot
    plt.show()

# Load the preprocessed dataset
df = pd.read_csv('/content/drive/MyDrive/Online Retail Preprocessed.csv')

# Print the column names to debug
print("Columns in the DataFrame:", df.columns)

# Check the number of records
num_records = df.shape[0]
print(f'Number of records used in the clustering: {num_records}')

# Define the number of clusters
n_clusters = 10  # You can adjust this value


# Run RM K-Means with KMeans++
kmeans_model, clustered_df, silhouette_avg, pt = rm_kmeans_kplusplus(df, n_clusters)

# End the timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

# Display the silhouette score
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Plot the clusters in 3D with centroids properly placed and matching colors
plot_clusters_3d(clustered_df, kmeans_model, silhouette_avg, pt)



