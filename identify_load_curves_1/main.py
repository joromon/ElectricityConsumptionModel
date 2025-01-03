import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import os
import sys


def preprocess_consumption(consumption, n_hours=3):
    """
    Preprocess the consumption data by aggregating hourly data into chunks of n_hours,
    calculating daily consumption percentages, and pivoting to a wide format.
    """

    # Ensure 'localtime' is of datetime type, if it's not already
    consumption['localtime'] = pd.to_datetime(consumption['localtime'])
    
    # Calculate daily consumption by postalcode and date
    consumption['hour'] = consumption['localtime'].dt.hour
    consumption['day'] = consumption['localtime'].dt.date
    
    # Group by postalcode and date and calculate daily consumption
    daily_consumption = consumption.groupby(['postalcode', 'day'])['consumption_filtered'].mean() * 24
    consumption = consumption.merge(daily_consumption, on=['postalcode', 'day'], suffixes=('', '_daily'))
    
    # Group by postalcode, adjusted hour (to 'n_hours'), date, and daily_consumption, 
    # and calculate the intraday consumption
    consumption['hour_group'] = (consumption['hour'] // n_hours) * n_hours
    consumption_grouped = consumption.groupby(['postalcode', 'hour_group', 'day', 'consumption_filtered_daily']).agg({'consumption_filtered': 'mean'}).reset_index()
    
    # Calculate the consumption percentage
    consumption_grouped['consumption_percentage'] = (consumption_grouped['consumption_filtered'] * 100) / consumption_grouped['consumption_filtered_daily']
    
    # Change format to long (one row per combination of postalcode, hour, and date)
    consumption_long = consumption_grouped.rename(columns={'hour_group': 'hour'})

    print(consumption_long)
    sys.exit()
    
    # Transform the dataset into a wide format (one column per hour)
    consumption_wide = consumption_long.pivot_table(index=['postalcode', 'day'], columns='hour', values='consumption_percentage', aggfunc='first')
    
    # Reset the index to make postalcode and day regular columns again
    consumption_wide.reset_index(inplace=True)
    
    return consumption_wide

'''
def preprocess_consumption(consumption, n_hours=3):
    print(consumption)
    sys.exit()
    """
    Preprocess the consumption data by aggregating hourly data into chunks of n_hours,
    calculating daily consumption percentages, and pivoting to a wide format.
    """
    print("### Preprocessing Consumption Data ###")

    # Calcular la 'daily_consumption' como la media de 'consumption_filtered' * 24
    consumption["daily_consumption"] = consumption.groupby(
        ["postalcode", "date"]
    )["consumption_filtered"].transform("mean") * 24

    # Ajustar la hora para agrupar en bloques de n_hours
    consumption["hour"] = (consumption["localtime"].dt.hour // n_hours) * n_hours

    # Agregar por 'hour' y calcular el porcentaje respecto a 'daily_consumption'
    consumption_long = consumption.groupby(
        ["postalcode", "hour", "date", "daily_consumption"]
    )["consumption_filtered"].mean().reset_index()
    
    consumption_long["consumption"] = (
        consumption_long["consumption_filtered"] * 100 / consumption_long["daily_consumption"]
    )

    # Asegurarse de que la agregación por 'hour' y el cálculo del porcentaje estén bien hechos
    consumption_wide = consumption_long.pivot_table(
        index=["postalcode", "date"], columns="hour", values="consumption", aggfunc="mean"
    ).fillna(0)

    # Ajustar el orden de las columnas para asegurarse de que los valores de las horas estén en el orden correcto
    consumption_wide = consumption_wide.sort_index(axis=1, ascending=True)

    return consumption_wide
'''

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
                current_count += 1  # Leaf node
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

def save_daily_load_curves(consumption, cluster_labels, scaling_type, n_clusters, filepath="plots"):
    """
    Save daily load curves grouped by clusters into PDF files.
    """
    print(f"### Saving Daily Load Curves for {scaling_type} with {n_clusters} Clusters ###")

    os.makedirs(filepath, exist_ok=True)

    # Add cluster labels to the consumption data
    consumption_with_clusters = consumption.copy()
    
    if len(cluster_labels) != len(consumption_with_clusters.index):
        consumption_with_clusters = consumption_with_clusters.iloc[:len(cluster_labels)]

    consumption_with_clusters["cluster"] = cluster_labels

    for cluster in range(n_clusters):
        cluster_data = consumption_with_clusters[consumption_with_clusters["cluster"] == cluster]

        # Plot daily load curves for the cluster
        plt.figure(figsize=(12, 6))
        for _, row in cluster_data.iterrows():
            plt.plot(row.index[2:], row.values[2:], alpha=0.5)
        
        plt.title(f"Cluster {cluster} - {scaling_type}")
        plt.xlabel("Hour")
        plt.ylabel("Consumption (%)")
        plt.grid()
        plt.savefig(f"{filepath}/daily_load_curves_cluster_{cluster}.pdf")
        plt.close()

def perform_clustering(consumption_wide_scaled, original_data, n_clusters, scaling_type="no_scale"):
    """
    Perform hierarchical clustering with the specified number of clusters and plot results.
    """
    print("### Performing Clustering ###")

    print(consumption_wide_scaled)
    sys.exit()

    clustering_X = consumption_wide_scaled.dropna(axis=0)
    
    print(f"Clustering data with {n_clusters} clusters")
    model = AgglomerativeClustering(
        n_clusters=n_clusters, compute_distances=True, linkage="ward"
    )
    cluster_labels = model.fit_predict(clustering_X)

    # Save dendrogram
    save_plot_dendrogram(
        model,
        filepath=f"plots/hierarchical_dendrogram_{n_clusters}_{scaling_type}.pdf",
        truncate_mode="lastp",
        p=n_clusters
    )

    '''
    # Save daily load curves for clusters
    save_daily_load_curves(
        original_data, 
        cluster_labels, 
        scaling_type=scaling_type, 
        n_clusters=n_clusters,
        filepath=f"plots"
    )
    '''

def identify_load_curves(consumption, scaling_method="no_scaling", n_clusters=3):
    """
    Main function to identify daily load curves using clustering.
    """
    print("### IDENTIFY LOAD CURVES ###")
    n_hours = 3

    # Preprocess the data
    consumption_wide = preprocess_consumption(consumption, n_hours=n_hours)

    print(consumption_wide)
    sys.exit()
    # Scale the data
    consumption_wide_scaled = scale_data(consumption_wide, scaling_method)

    # Perform clustering
    perform_clustering(consumption_wide_scaled, consumption, n_clusters, scaling_type=scaling_method)

def main(consumption, scaling_method="no_scaling", n_clusters=3):
    identify_load_curves(consumption, scaling_method=scaling_method, n_clusters=n_clusters)
