import pandas as pd
import polars as pl
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colormaps
import os
import sys


def preprocess_consumption(consumption, n_hours=3):
    """
    Preprocess the consumption data by aggregating hourly data into chunks of n_hours,
    calculating daily consumption percentages, and pivoting to a wide format.
    """
    print("### Preprocessing Consumption Data ###")
    consumption = consumption.copy(deep=True)

    consumption['daily_consumption'] = consumption.groupby(['postalcode', 'date'])['consumption_filtered'].transform(
        lambda x: 'NaN' if x.isna().any() else x.mean() * 24
        #lambda x: x.mean() * 24
    )
    consumption['hour'] = (np.floor(consumption['localtime'].dt.hour / n_hours) * n_hours).astype(int)

    # Group again to calculate the mean of 'consumption_filtered' by hour
    def mean_with_nan(x):
        if x.isna().any():
            return 'NaN'
        else:
            return x.mean()

    consumption_long = consumption.groupby(['postalcode', 'hour', 'date', 'daily_consumption'], as_index=False).agg(
        consumption_filtered_mean=('consumption_filtered', mean_with_nan)
    )

    #  Multiply by n_hours
    consumption_long['consumption_filtered'] = consumption_long['consumption_filtered_mean'].apply(
        lambda x: 'NaN' if x == 'NaN' else x * n_hours
    )
    
    # Calculate the consumption percentage relative to daily consumption
    consumption_long['consumption'] = consumption_long.apply(
        lambda row: 'NaN' if row['daily_consumption'] == 'NaN' else (row['consumption_filtered'] * 100) / row['daily_consumption'],
        axis=1
    )
    consumption_long.drop(columns=['consumption_filtered_mean'], inplace=True)

    #export to check data
    #consumption_long.to_csv('consumption_long.csv')
    consumption_long = consumption_long.map(lambda x: np.nan if x == "NaN" else x)
    
    # Ensure aggregation per hour i'ts fine
    consumption_wide = consumption_long.pivot_table(
        index=["postalcode", "date"], columns="hour", values="consumption", aggfunc="mean"
    )

    consumption_wide = consumption_wide.sort_index(axis=1, ascending=True)
    
    return consumption_wide

def scale_data(consumption_wide, scaling_method):
    """
    Apply the selected scaling method to the data: no scaling, Min-Max scaling, or Z-normalization.
    """
    print(f"### Scaling Data using {scaling_method} ###")

    if scaling_method == "no_scaling":
        return consumption_wide

    elif scaling_method == "min_max_scaling":
        return pd.DataFrame(
            MinMaxScaler().fit_transform(consumption_wide),
            columns=consumption_wide.columns,
            index=consumption_wide.index,
        )

    elif scaling_method == "z_norm_scaling":
        return pd.DataFrame(
            StandardScaler().fit_transform(consumption_wide),
            columns=consumption_wide.columns,
            index=consumption_wide.index,
        )

    else:
        raise ValueError("Invalid scaling method. Choose from 'no_scaling', 'min_max_scaling', or 'z_norm_scaling'.")


def save_plot_dendrogram(model, filepath, **kwargs):
    """
    Create and save a dendrogram plot for hierarchical clustering.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    plt.figure(figsize=(15, 8))
  
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the dendrogram
    dendrogram(linkage_matrix, **kwargs)
    

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Number of points in node (or index if no parenthesis).")
    plt.ylabel("Distance (Ward’s)")
    plt.savefig(filepath)
    plt.close()


def save_daily_load_curves(consumption, cluster_labels, index_X, scaling_type, filepath):
    """
    Save daily load curves with centroids for each cluster to a PDF file.
    """
    print("### Saving Daily Load Curves ###")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    consumption = pl.from_pandas(consumption)

    df=(consumption.select(pl.all().exclude("cluster")).
        join(
            consumption.group_by(["postalcode", "date"]).agg(
                (pl.col("consumption_filtered").mean() * 24).alias("daily_consumption")
            ),
            on=["postalcode", "date"]).
        with_columns(
            (pl.col("consumption_filtered") * 100 / pl.col("daily_consumption")).alias("consumption_filtered")
        ).
        join(
            pl.DataFrame(
                pd.concat([
                    index_X.reset_index(drop=True),  # Ensure no index issues
                    pd.DataFrame(cluster_labels, columns=["cluster"])],
                    axis=1
                )).
            with_columns(pl.col("date").cast(pl.Date)),
        on=["postalcode","date"]))

    if "cluster" not in df.columns:
        df = df.with_columns(pl.lit(1).alias("cluster"))

    # Ensure the DataFrame is sorted by date and hour
    df = df.sort(["date", "postalcode", "hour"])

    # Group by 'date' and 'cluster', and collect consumption data for each day
    daily_curves = df.group_by(["date", "postalcode", "cluster"]).agg(pl.col("consumption_filtered"))

    # Create a centroid DataFrame by averaging consumption for each hour and cluster
    centroids = (
        df.group_by(["hour", "cluster"]).
        agg(pl.col("consumption_filtered").drop_nans().mean().alias("consumption_filtered")).
        sort(["hour","cluster"]).
        pivot(index=["cluster"], on="hour")
    )

    # Convert to pandas for easy plotting with matplotlib
    daily_curves_pandas = daily_curves.to_pandas()
    centroids_pandas = centroids.to_pandas()

    with PdfPages(filepath) as pdf:
        plt.figure(figsize=(10, 6))
        unique_clusters = df["cluster"].unique().to_list()
        colors = colormaps["tab10"]  # Use the updated way to access colormaps
        for _, row in daily_curves_pandas.iterrows():
            date = row["date"]
            cluster = row["cluster"]
            consumption = row["consumption_filtered"]
            # Plot individual daily load curves with low opacity
            plt.plot(range(len(consumption)), consumption, alpha=0.1, lw=0.005, color=colors(cluster / len(unique_clusters)),
                     label=None)
        # Plot centroids with bold lines
        for cluster in unique_clusters:
            centroid = centroids_pandas[centroids_pandas["cluster"] == cluster].drop("cluster", axis=1).iloc[0].to_numpy()
            plt.plot(range(len(centroid)), centroid, linewidth=2.5, label=f"Cluster {cluster}",
                     color=colors(cluster / len(unique_clusters)))
        # Add labels and legend
        plt.xlabel("Hour of Day")
        plt.ylabel("Consumption")
        plt.title(f"Daily Load Curves {scaling_type} Clustering")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Save the current figure to the PDF
        pdf.savefig()
        plt.close()

        print(f"### Daily Load Curves saved to {filepath} ###")
        
def perform_clustering(consumption_wide_scaled, consumption, n_clusters, scaling_type="no_scale"):
    """
    Perform hierarchical clustering with the specified number of clusters and plot results.
    """
    print("### Performing Clustering ###")

    clustering_X = consumption_wide_scaled.dropna(axis=0)
    len(clustering_X)
    
    print(f"Clustering data with {n_clusters} clusters")
    model = AgglomerativeClustering(
        n_clusters=n_clusters, compute_distances=True, linkage="ward"
    )
    
    index_X = clustering_X.index.to_frame(index=False)
    cluster_model = model.fit(clustering_X)
    cluster_labels = model.fit_predict(clustering_X)
    

    # Save dendrogram
    save_plot_dendrogram(
        cluster_model,
        filepath=f"plots/hierarchical_dendrogram_{n_clusters}_{scaling_type}.pdf",
        truncate_mode="lastp",
        p=n_clusters
    )

    # Save daily load curves for clusters
    save_daily_load_curves(
        consumption, 
        cluster_labels,
        index_X,
        scaling_type=scaling_type, 
        filepath=f"plots/daily_load_curves_cluster_{n_clusters}_{scaling_type}.pdf"
    )

def identify_load_curves(consumption, scaling_method="no_scaling", n_clusters=3):
    """
    Main function to identify daily load curves using clustering.
    """
    print("### IDENTIFY LOAD CURVES ###")
    n_hours = 3

    # Preprocess the data
    consumption_wide = preprocess_consumption(consumption, n_hours=n_hours)

    # Scale the data
    consumption_wide_scaled = scale_data(consumption_wide, scaling_method)

    # Perform clustering
    perform_clustering(consumption_wide_scaled, consumption, n_clusters, scaling_type=scaling_method)

def main(consumption, scaling_method="no_scaling", n_clusters=3):
    identify_load_curves(consumption, scaling_method=scaling_method, n_clusters=n_clusters)
