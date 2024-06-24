import faiss
import numpy as np
import matplotlib.pyplot as plt

# Load the FAISS index and metadata
index = faiss.read_index('index/image_features.index')
metadata = np.load('index/image_metadata.npy', allow_pickle=True)

# Extract feature vectors from the FAISS index
features = index.reconstruct_n(0, index.ntotal)

# Normalize the feature vectors
def normalize_vector(vec):
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / norm

features = normalize_vector(features)

# Compute the pairwise dot products
dot_products = np.dot(features, features.T)

# Flatten the upper triangle of the dot product matrix to get all pairwise similarities
upper_triangle_indices = np.triu_indices_from(dot_products, k=1)
similarities = dot_products[upper_triangle_indices]

# Plot the distribution of dot products
plt.hist(similarities, bins=50)
plt.xlabel('Dot Product (Cosine Similarity)')
plt.ylabel('Frequency')
plt.title('Distribution of Dot Products between Image Pairs')
plt.show()

# Print some statistics
print(f"Mean: {np.mean(similarities)}")
print(f"Median: {np.median(similarities)}")
print(f"90th Percentile: {np.percentile(similarities, 90)}")
print(f"95th Percentile: {np.percentile(similarities, 95)}")
print(f"99th Percentile: {np.percentile(similarities, 99)}")
