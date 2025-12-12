# Spatial Autocorrelation K-Means Clustering

A Python package for optimal K-Means clustering based on spatial autocorrelation analysis using Moran's I.

## Features
- Find optimal number of clusters (k) based on spatial autocorrelation
- Combine Moran's I spatial statistics with within-cluster sum of squares (WCSS)
- Parallel computation for efficient processing
- Returns trained K-Means model with optimal k

## Installation

### System Requirements
- Python 3.9 or higher
- pip package manager

### Install Required Packages

Run these commands in your terminal/command prompt:

```bash
# Core data science libraries
pip install numpy pandas scikit-learn

# Geospatial libraries
pip install geopandas shapely pyproj

# Spatial statistics
pip install libpysal esda

# Parallel processing
pip install joblib

# Visualization (optional, if you want to plot results)
pip install matplotlib
```

## Execute
```python
Save the script as autocorrkmeans.py in your working directory. Then, you can import and run the main calculation function as follows:

import pandas as pd
import numpy as np
from autocorrkmeans import kmeans_autocorr

# 'data': The features used for clustering (e.g., rainfall, temperature)
data = np.load("my_data_features.npy") 

# 'df': DataFrame containing strictly 'latitude' and 'longitude' columns
df = pd.read_csv("coordinates.csv") 

# 'latent_space_pca': Spatial features/embeddings (e.g., from a spatial PCA)
latent_space_pca = np.load("spatial_embeddings.npy")

# Run the Algorithm
# This calculates the optimal k and returns the trained model
best_model = kmeans_autocorr(
    data=data, 
    df=df, 
    latent_space_pca=latent_space_pca, 
    k_range=range(2, 11),  # Test k from 2 to 10
    alpha=0.5              # Weight for Spatial Score (0.5 = balanced)
)

print(f"Optimal clusters: {best_model.n_clusters}")
labels = best_model.labels_
```
