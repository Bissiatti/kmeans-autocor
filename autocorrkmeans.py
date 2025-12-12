# --- Standard Library ---
import gc
import sqlite3
import json

# --- Data Manipulation & Math ---
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# --- Visualization ---
import matplotlib.pyplot as plt

# --- Machine Learning (Scikit-Learn) & Utilities ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator

# --- Geospatial & Spatial Statistics ---
import geopandas as gpd
from libpysal.weights import DistanceBand, KNN, Queen
from esda.moran import Moran


from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score, 
    mutual_info_score
)

def calculate_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[labels == i]
        centroid = np.mean(cluster_points, axis=0)
        centroids[i] = centroid
    return centroids

lat_long_uf = pd.read_csv("lat_long_bacias.csv")
lat_long = lat_long_uf[['latitude', 'longitude']]

def normalize_custom(arr, vmin, vmax):
    """Normalize array between vmin and vmax."""
    arr = np.clip(arr, vmin, vmax)
    return (arr - vmin) / (vmax - vmin) if vmax != vmin else arr * 0

def calculate_moran_for_cluster(cluster, gdf, spei_columns, col='KMeans_k', k=24):
    """Calculate Moran's I for a specific cluster."""
    gdf_cluster = gdf[gdf[col] == cluster]
    
    if len(gdf_cluster) < 9:
        return (cluster, np.nan)

    # coords = np.array([[p.x, p.y] for p in gdf_cluster.geometry])  # UNUSED VARIABLE
    w = KNN.from_dataframe(gdf_cluster, k=k)
    w.transform = "r"

    moran_values = []
    for column in spei_columns:
        if column in gdf_cluster.columns:
            values = gdf_cluster[column].values
            if len(np.unique(values)) > 1:  # Check variability
                try:
                    moran = Moran(values, w)
                    moran_values.append(moran.I)
                except:
                    continue

    return (cluster, np.mean(moran_values) if moran_values else np.nan)

def calculate_wcss_for_cluster(cluster, gdf, features, col='KMeans_k'):
    """Calculate Within-Cluster Sum of Squares for a specific cluster."""
    gdf_cluster = gdf[gdf[col] == cluster]
    if len(gdf_cluster) < 2:
        return (cluster, np.nan)

    X = gdf_cluster[features].values
    centroid = np.mean(X, axis=0)
    wcss = np.sum((X - centroid) ** 2)
    return (cluster, wcss)



def total_sum_of_squares(X):
    """Calculate total sum of squares for a dataset."""
    # X: numpy array shape (n_samples, n_features)
    X = np.asarray(X)
    mean = np.mean(X, axis=0)
    return np.sum((X - mean)**2)

def compute_cluster_wcss(X, labels, k):
    """Calculate WCSS for each cluster in a clustering result."""
    wcss_per_cluster = []
    sizes = []
    for i in range(k):
        mask = (labels == i)
        Xi = X[mask]
        sizes.append(len(Xi))
        if len(Xi) == 0:
            wcss_per_cluster.append(0.0)
        else:
            cen = Xi.mean(axis=0)
            wcss_per_cluster.append(np.sum((Xi - cen)**2))
    return np.array(wcss_per_cluster), np.array(sizes)

def moran_to_score(moran_vals):
    """Convert Moran's I values to scores where lower is better in range [0,1]."""
    # moran_vals in [-1, 1] -> want "lower is better" in [0,1]
    # moran_scaled = (I + 1) / 2 in [0,1], then invert:
    return (1 - (moran_vals + 1) / 2)  # = (1 - I)/2

def calculate_moran_wcss_score_for_k_parallel(
    gdf, 
    k, 
    features, 
    alpha=0.3, 
    col='KMeans_k', 
    k_moran=8
):
    """
    Calculate Moran's I and WCSS scores for all clusters in parallel.
    Returns tuple of (moran_values, wcss_values) for all clusters.
    """
    clusters = list(range(k))
    beta = 1 - alpha

    moran_results = Parallel(n_jobs=-1)(
        delayed(calculate_moran_for_cluster)(ck, gdf, features, col, k_moran) for ck in clusters
    )
    wcss_results = Parallel(n_jobs=-1)(
        delayed(calculate_wcss_for_cluster)(ck, gdf, features, col) for ck in clusters
    )

    # Remove invalid values
    moran_vals = np.array([v for _, v in moran_results])
    wcss_vals = np.array([v for _, v in wcss_results])
    total_ss = total_sum_of_squares(gdf[features])
    if total_ss <= 0:
        total_ss = 1.0

    # WCSS normalization by total SS -> [0,1]
    wcss_vals = wcss_vals / total_ss

    if len(moran_vals) == 0 or len(wcss_vals) == 0:
        return np.nan

    # Invert Moran's I since lower autocorrelation is better for clustering
    moran_vals = -moran_vals

    # Alternative: use log transformation for WCSS (commented out)
    # wcss_vals = np.log1p(wcss_vals)

    return moran_vals, wcss_vals

def normalize(vs):
    """Normalize values to range [0,1]."""
    vs = np.array(vs)
    if vs.max() - vs.min() < 1e-6:
        return np.ones_like(vs)
    return (vs - vs.min()) / (vs.max() - vs.min())


def save_metrics_sqlite(db_path, state, metrics_data, alpha):
    """
    Save clustering metrics to SQLite.
    
    Parameters
    ----------
    db_path : str
        Path to sqlite file.
    state : str
        Name/state identifier.
    metrics_data : list of tuples
        Each tuple = (method_name, errors_list, optimal_k)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clustering_metrics_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT,
            method_name TEXT,
            errors_list TEXT,
            optimal_k INTEGER,
            alpha REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert data
    for method_name, errors_list, optimal_k in metrics_data:
        cur.execute("""
            INSERT INTO clustering_metrics_v2 (state, method_name, errors_list, optimal_k, alpha)
            VALUES (?, ?, ?, ?, ?)
        """, (state, method_name, json.dumps(errors_list), optimal_k, alpha))

    conn.commit()
    conn.close()


def evaluate_and_validate_k_optimized(data, df, latent_space_pca, k_range=range(2, 11), model=KMeans, 
                                     plot=False, validate_spatial=True, validate_spatial_moran=True, state='MG', alpha=0.5):
    """
    Optimized function combining evaluate_k and validate_k.
    
    Parameters:
    - data: data for clustering
    - df: DataFrame with latitude and longitude
    - latent_space_pca: PCA-transformed latent space representations
    - k_range: range of k values to test
    - model: clustering model (default KMeans)
    - plot: whether to plot results
    - validate_spatial: whether to calculate spatial autocorrelation
    - validate_spatial_moran: UNUSED PARAMETER - whether to validate spatial Moran's I
    - state: name of state/region for identification
    - alpha: weight parameter for Moran's I in combined score
    
    Returns:
    - dict with optimal k for each metric
    - DataFrame with cluster labels
    - autocorr: autocorrelation scores for each k
    """
    
    # Traditional metrics
    inertia = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    # Spatial autocorrelation metrics
    autocorr = [] if validate_spatial else None
    labels_k = {}
    k_values_computed = []  # Track which k values were successfully computed
    global_moran = []
    global_wcss = []
    # Normalize data
    data_normalized = data.copy()
    
    # Prepare DataFrame for spatial validation
    if validate_spatial:
        df_const = pd.DataFrame(latent_space_pca, columns=[f'const{i+1}' for i in range(latent_space_pca.shape[1])], index=df.index)
        df_extended = pd.concat([df.copy(), df_const], axis=1)
        const_columns = list(df_const.columns)

    for k in k_range:
            # Fit model
            model_t = model(n_clusters=k, random_state=42)
            model_t.fit(data_normalized)
            labels = model_t.labels_
            
            # Check if we have valid clusters
            if len(np.unique(labels)) < 2:
                print(f"Warning: k={k} resulted in insufficient clusters")
                continue
                
            labels_k[k] = labels
            k_values_computed.append(k)

            # Calculate traditional metrics
            if hasattr(model_t, 'inertia_'):
                inertia_k = model_t.inertia_
            else:
                inertia_k = np.sum([np.sum((data_normalized[labels == i] - calculate_centroids(data_normalized, labels, k)[i]) ** 2) 
                                  for i in range(k) if np.sum(labels == i) > 0])

            inertia.append(inertia_k)

            # Calculate quality metrics
            try:
                silhouette_scores.append(silhouette_score(data_normalized, labels))
            except:
                silhouette_scores.append(np.nan)
                
            try:
                calinski_scores.append(calinski_harabasz_score(data_normalized, labels))
            except:
                calinski_scores.append(np.nan)
                
            try:
                davies_scores.append(davies_bouldin_score(data_normalized, labels))
            except:
                davies_scores.append(np.nan)
                
            # Calculate spatial autocorrelation if requested
            if validate_spatial:
                    df_temp = df_extended.copy()
                    df_temp['KMeans_k'] = labels

                    gdf = gpd.GeoDataFrame(
                        df_temp,
                        geometry=gpd.points_from_xy(df_temp.longitude, df_temp.latitude),
                        crs="EPSG:4326"
                    ).to_crs("EPSG:5880")

                    gdf['KMeans_k'] = labels_k[k]  # saved model result
                    morans, wcss = calculate_moran_wcss_score_for_k_parallel(
                        gdf,
                        k,
                        features=const_columns,
                    )
                    global_moran += list(morans) 
                    global_wcss += list(wcss)   

                    # Calculate average of valid Moran's I values
                    mean_moran = np.mean(morans) 

    # Check if we have enough data
    if len(k_values_computed) == 0:
        print("Error: No k values were successfully calculated")
        return {}, df.copy(), []
        
    if len(k_values_computed) != len(inertia):
        print(f"Warning: Dimension mismatch. k_values: {len(k_values_computed)}, inertia: {len(inertia)}")

    # Find optimal k for each metric using only computed values
    try:
        if len(inertia) > 1:
            kneedle = KneeLocator(k_values_computed, inertia, curve='convex', direction='decreasing')
            optimal_k_elbow = kneedle.elbow
        else:
            optimal_k_elbow = k_values_computed[0] if k_values_computed else None
    except:
        optimal_k_elbow = k_values_computed[0] if k_values_computed else None
        
    # For other metrics, use only valid values
    valid_silhouette = [(k, s) for k, s in zip(k_values_computed, silhouette_scores) if not np.isnan(s)]
    valid_calinski = [(k, s) for k, s in zip(k_values_computed, calinski_scores) if not np.isnan(s)]
    valid_davies = [(k, s) for k, s in zip(k_values_computed, davies_scores) if not np.isnan(s)]
    
    optimal_k_silhouette = max(valid_silhouette, key=lambda x: x[1])[0] if valid_silhouette else None
    optimal_k_davies = min(valid_davies, key=lambda x: x[1])[0] if valid_davies else None

    global_moran = normalize(global_moran)
    global_wcss = normalize(global_wcss)

    print(f"Number of Moran values: {len(global_moran)}")
    print(f"k_range: {list(k_range)}")
    
    # Calculate autocorrelation scores for each k
    if validate_spatial:
        k0 = 0
        beta = 1 - alpha
        for k in k_range:
            moran_norm = global_moran[k0:k0+k]
            wcss_norm = global_wcss[k0:k0+k]

            moran_norm = np.clip(moran_norm, 1e-6, None)
            wcss_norm = np.clip(wcss_norm, 1e-6, None)
            
            # Current: weighted arithmetic mean
            scores = (moran_norm * alpha) + (wcss_norm * beta)

            autocorr.append(np.max(scores))
            k0 += k
    
    optimal_k_autocorr = None
    if validate_spatial:
        valid_autocorr = [(k, s) for k, s in zip(k_values_computed, autocorr) if not np.isnan(s)]
        optimal_k_autocorr = min(valid_autocorr, key=lambda x: x[1])[0] if valid_autocorr else None
        

    # Add labels to DataFrame
    df_result = df.copy()
    if optimal_k_elbow and optimal_k_elbow in labels_k:
        df_result[f"{model.__name__}_k_elbow"] = labels_k[optimal_k_elbow]
    if optimal_k_silhouette and optimal_k_silhouette in labels_k:
        df_result[f"{model.__name__}_k_silhouette"] = labels_k[optimal_k_silhouette]
    if optimal_k_davies and optimal_k_davies in labels_k:
        df_result[f"{model.__name__}_k_davies"] = labels_k[optimal_k_davies]
    if optimal_k_autocorr and optimal_k_autocorr in labels_k:
        df_result[f"{model.__name__}_k_autocorr"] = labels_k[optimal_k_autocorr]

    # Prepare metrics dictionary with "lower is better" meaning
    metrics_dict = {
        "silhouette": [-s if not np.isnan(s) else np.nan for s in silhouette_scores],  # invert since higher is better
        "davies": davies_scores,
    }

    if validate_spatial and autocorr:
        metrics_dict["autocorr"] = autocorr

    # Create matrix with valid metrics for mix
    valid_metrics = []
    metric_names = []

    for name, list_ in metrics_dict.items():
        if list_ and len(list_) == len(k_values_computed):
            if not all(np.isnan(list_)):
                valid_metrics.append(list_)
                metric_names.append(name)

    # Calculate mixed scores
    if valid_metrics:
        metrics_matrix = np.array(valid_metrics).T  # shape (n_k, n_metrics)
        scaler = MinMaxScaler()
        normalized_matrix = scaler.fit_transform(metrics_matrix)

        mix_scores = np.mean(normalized_matrix, axis=1)
        optimal_k_mix = k_values_computed[np.argmin(mix_scores)]
    else:
        mix_scores = [np.nan] * len(k_values_computed)
        optimal_k_mix = None

    # Add mixed score labels to DataFrame
    if optimal_k_mix and optimal_k_mix in labels_k:
        df_result[f"{model.__name__}_k_mix"] = labels_k[optimal_k_mix]

    # Plotting
    if plot and len(k_values_computed) > 1:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            # Elbow plot
            axes[0].plot(k_values_computed, inertia, 'bo-', label='Elbow')
            if optimal_k_elbow:
                axes[0].axvline(optimal_k_elbow, color='red', linestyle='--',
                                label=f'Optimal k = {optimal_k_elbow}')
            axes[0].set_title('Elbow Method')
            axes[0].set_xlabel('k')
            axes[0].set_ylabel('Score')
            axes[0].grid(True)
            axes[0].legend()

            # Silhouette plot
            axes[1].plot(k_values_computed, silhouette_scores, 'go-', label='Silhouette')
            if optimal_k_silhouette:
                axes[1].axvline(optimal_k_silhouette, color='red', linestyle='--',
                                label=f'Optimal k = {optimal_k_silhouette}')
            axes[1].set_title('Silhouette Score')
            axes[1].set_xlabel('k')
            axes[1].set_ylabel('Score')
            axes[1].grid(True)
            axes[1].legend()

            # Davies-Bouldin plot
            axes[2].plot(k_values_computed, davies_scores, 'ro-', label='Davies-Bouldin')
            if optimal_k_davies:
                axes[2].axvline(optimal_k_davies, color='red', linestyle='--',
                                label=f'Optimal k = {optimal_k_davies}')
            axes[2].set_title('Davies-Bouldin Score')
            axes[2].set_xlabel('k')
            axes[2].set_ylabel('Score (lower is better)')
            axes[2].grid(True)
            axes[2].legend()

            # Spatial Autocorrelation plot
            if validate_spatial and autocorr:
                axes[3].plot(k_values_computed, autocorr, 'yo-', label="Moran's I")
                if optimal_k_autocorr:
                    axes[3].axvline(optimal_k_autocorr, color='red', linestyle='--',
                                    label=f'Optimal k = {optimal_k_autocorr}')
                axes[3].set_title("Spatial Autocorrelation (Moran's I)")
                axes[3].set_xlabel('k')
                axes[3].set_ylabel("Moran's I mean")
                axes[3].grid(True)
                axes[3].legend()
            else:
                axes[3].axis('off')

            # Adjust layout and overall title
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.suptitle(f'Clustering Validation Metrics - {state}',
                        fontsize=16, fontweight='bold')

            plt.savefig(f"{state}_{alpha}.png")

        except Exception as e:
            print(f"Error plotting: {e}")
            
    # Save metrics to SQLite database
    metrics_data = [
        ("Elbow", inertia, optimal_k_elbow),
        ("Silhouette", silhouette_scores, optimal_k_silhouette),
        ("Davies-Bouldin", davies_scores, optimal_k_davies),
    ]

    if validate_spatial and autocorr:
        metrics_data.append(("Autocorr", autocorr, optimal_k_autocorr))

    save_metrics_sqlite("clustering_metrics_2.sqlite", state, metrics_data, alpha)

    # Prepare final results
    result = {
        "optimal_k_elbow": optimal_k_elbow,
        "optimal_k_silhouette": optimal_k_silhouette,
        "optimal_k_davies": optimal_k_davies,
        "optimal_k_mix": optimal_k_mix,
    }

    if validate_spatial:
        result["optimal_k_autocorr"] = optimal_k_autocorr
        print(f"Spatial autocorrelation: {autocorr}")

    print(f"Results for {state}:")
    for metric, k_val in result.items():
        print(f"{metric}: {k_val}")

    return result, df_result, autocorr
    
def kmeans_state(state, lat_long_scale, latent_representations_np, latent_space_pca, col='bacia', model=KMeans, plot=False, range_k=range(2, 10, 1), alpha=0.5):
    """
    Perform K-means clustering for a specific state/region.
    
    Parameters:
    - state: state/region identifier
    - lat_long_scale: UNUSED PARAMETER - latitude/longitude scaling factor
    - latent_representations_np: latent space representations
    - latent_space_pca: PCA-transformed latent space
    - col: column name for filtering state data
    - model: clustering model to use
    - plot: whether to plot results
    - range_k: range of k values to test
    - alpha: weight parameter for Moran's I in combined score
    
    Returns:
    - best_k: dictionary of optimal k values for different metrics
    - df: DataFrame with cluster labels
    - autocorr: autocorrelation scores for each k
    """
    print(f"Processing state: {state}")
    state_index = list(lat_long_uf[lat_long_uf[col] == state].index)
    state_train = latent_representations_np[state_index, :]
    gc.collect()
    state_train = np.nan_to_num(state_train)
    state_lat_long = lat_long.loc[state_index]
    latent_space_pca = latent_space_pca[state_index, :]
    state_lat_long = pd.DataFrame(state_lat_long, columns=['latitude', 'longitude'])
    best_k, df, autocorr = evaluate_and_validate_k_optimized(state_train, state_lat_long, latent_space_pca, range_k, model, plot=plot, state=state, alpha=alpha)
    return best_k, df, autocorr

def kmeans_autocorr(data, df, latent_space_pca, k_range=range(2, 11), alpha=0.5):
    """
    Calculates the optimal k using only the spatial autocorrelation method (Moran's I + WCSS)
    and returns the optimized KMeans model.

    Parameters:
    - data: Data for clustering (numpy array).
    - df: DataFrame containing 'latitude' and 'longitude'.
    - latent_space_pca: PCA-transformed latent space representations (for spatial features).
    - k_range: Range of k values to test.
    - alpha: Weight parameter for Moran's I in the combined score (0 to 1).

    Returns:
    - best_model: The KMeans model trained with the optimal k.
    """
    
    # Normalize data
    data_normalized = data.copy()
    
    # Prepare spatial features DataFrame (from PCA)
    df_const = pd.DataFrame(latent_space_pca, columns=[f'const{i+1}' for i in range(latent_space_pca.shape[1])], index=df.index)
    df_extended = pd.concat([df.copy(), df_const], axis=1)
    const_columns = list(df_const.columns)

    # Containers for global normalization logic
    global_moran = []
    global_wcss = []
    k_values_computed = []

    # Loop through k_range to collect metrics
    for k in k_range:
        # Fit KMeans
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data_normalized)
        labels = model.labels_

        # Validate cluster count
        if len(np.unique(labels)) < 2:
            print(f"Warning: k={k} resulted in insufficient clusters")
            continue
        
        k_values_computed.append(k)

        # Prepare GeoDataFrame for spatial calculation
        df_temp = df_extended.copy()
        df_temp['KMeans_k'] = labels
        
        gdf = gpd.GeoDataFrame(
            df_temp,
            geometry=gpd.points_from_xy(df_temp.longitude, df_temp.latitude),
            crs="EPSG:4326"
        ).to_crs("EPSG:5880")

        # Calculate Moran's I and WCSS for this k
        morans, wcss = calculate_moran_wcss_score_for_k_parallel(
            gdf,
            k,
            features=const_columns
        )
        
        # Accumulate raw values for global normalization
        global_moran.extend(list(morans))
        global_wcss.extend(list(wcss))

    if not k_values_computed:
        print("Error: No valid k values computed.")
        return None

    # Normalize metrics globally (essential for the logic consistency)
    global_moran = normalize(global_moran)
    global_wcss = normalize(global_wcss)

    # Calculate scores per k
    autocorr_scores = []
    beta = 1 - alpha
    current_idx = 0

    for k in k_values_computed:
        end_idx = current_idx + k
        moran_k = global_moran[current_idx:end_idx]
        wcss_k = global_wcss[current_idx:end_idx]
        
        # Clip to ensure stability
        moran_k = np.clip(moran_k, 1e-6, None)
        wcss_k = np.clip(wcss_k, 1e-6, None)

        # Calculate Combined Score (Lower is better in this logic)
        score = np.max((moran_k * alpha) + (wcss_k * beta))
        autocorr_scores.append((k, score))
        
        current_idx = end_idx

    # Finds the k with the minimum score
    best_k, best_score = min(autocorr_scores, key=lambda x: x[1])

    print(f"Optimal k found (Autocorr method): {best_k} (Score: {best_score:.4f})")

    best_model = KMeans(n_clusters=best_k, random_state=42)
    best_model.fit(data_normalized)

    return best_model